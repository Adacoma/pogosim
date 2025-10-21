#include "pogobase.h"
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <strings.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "pogo-utils/kinematics.h"     // DDK (PID + wall avoidance handled internally)
#include "pogo-utils/version.h"

/** \file main_with_stats.c
 *  \brief Continuous-time Vicsek controller with additional local stats + push-sum consensus.
 *
 *  This file extends the previous Vicsek example by computing and aggregating, via
 *  push-sum consensus, several local statistics in addition to the Rayleigh-corrected
 *  local polarization:
 *
 *  1) Wall/cluster occupancy ratio: fraction of time spent either in wall-avoidance
 *     behavior (as reported by the DDK) or within a cluster U-turn window.
 *  2) Neighbor persistence: for currently visible neighbors, the mean normalized age
 *     since first sighting (clamped by `neighbor_persist_norm_ms`).
 *  3) Neighbor number: instantaneous number of distinct neighbors (excluding self).
 *
 *  All stats use a **single push-sum weight** per message to limit payload bloat.
 *  Each local stat s_i is multiplied by the same weight base (N_eff) before sending,
 *  and a single weight w is sent; thus remote nodes can form consensus estimates as
 *  s_i_total / w_total.
 *
 *  **C11** code style with opening brackets on the same line.
 */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// -----------------------------------------------------------------------------
// Radio beacons
// -----------------------------------------------------------------------------
#define BEACON_HZ          10u
#define BEACON_PERIOD_MS  (1000u / BEACON_HZ)

// -----------------------------------------------------------------------------
// Tunables (configurable via YAML in simulator)
// -----------------------------------------------------------------------------
uint32_t max_age                   = 600;     /* ms: neighbor expiration for headings */
double   vicsek_beta_rad_per_s     = 3.0;     /* dθ/dt = β sin(θ̄ - θ) + σ ξ */
double   cont_noise_sigma_rad      = 0.0;     /* rad/sqrt(s), diffusion-like noise on heading */
double   cont_max_dt_s             = 0.05;    /* clamp dt for stability */
float    base_speed_ratio          = 0.70f;   /* [0..1] duty-cycle for forward motion */

double   vicsek_turn_gain_clip_deg = 30.0;    /* scaling in deg for turning normalization */
bool     include_self_in_avg       = true;    /* for local mean direction + polarization */

// Photogeometry (passed to the internal HD)
double   alpha_deg                 = 40.0;
double   robot_radius              = 0.0265;  /* 26.5 mm */
heading_chirality_t heading_chiralty_enum = HEADING_CW;

// === CLUSTER U-TURN ===
uint32_t cluster_u_turn_duration_ms = 1500;
float    phi_rad_min = (float)M_PI;           // default: U-turn
float    phi_rad_max = (float)M_PI;

// --- Consensus smoothing ---
double   ewma_alpha_polarization   = 0.20;    /* EWMA smoothing of global consensus [0..1] */
double   ewma_alpha_wallratio      = 0.10;    /* EWMA for wall/cluster ratio consensus */
double   ewma_alpha_persistence    = 0.10;    /* EWMA for neighbor persistence consensus */
double   ewma_alpha_nb             = 0.10;    /* EWMA for neighbor number consensus */

// --- Local-window tunables ---
uint32_t neighbor_persist_norm_ms  = 10000;   /* normalization horizon for persistence */
uint8_t  neighbor_norm_max         = 12;      /* used to normalize LED mapping */

// -----------------------------------------------------------------------------
// LEDs
// -----------------------------------------------------------------------------
typedef enum {
    SHOW_STATE,
    SHOW_ANGLE,
    SHOW_POLARIZATION,
    SHOW_STATS_RGB  // R: wall/cluster ratio, G: neighbor persistence, B: neighbor count (normalized)
} main_led_display_type_t;
main_led_display_type_t main_led_display_enum = SHOW_STATS_RGB;

#define SCALE_0_255_TO_0_25(x)   (uint8_t)((x) * (25.0f / 255.0f) + 0.5f)

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------
static inline double wrap_pi(double a){ while(a> M_PI)a-=2.0*M_PI; while(a<-M_PI)a+=2.0*M_PI; return a; }
static inline int16_t rad_to_mrad(double a){ a=wrap_pi(a); long v=lround(a*1000.0); if(v>32767)v=32767; if(v<-32768)v=-32768; return (int16_t)v; }
static inline double  mrad_to_rad(int16_t m){ return ((double)m)/1000.0; }
static inline double  rand_uniform(double a, double b){ double u=(double)rand()/(double)RAND_MAX; return a + (b - a) * u; }

// -----------------------------------------------------------------------------
// Push-sum fixed-point packing (Q15)
// -----------------------------------------------------------------------------
static inline int16_t q15_pack_double(double x, double xmin, double xmax){
    if (x < xmin) x = xmin;
    if (x > xmax) x = xmax;
    double xn = (x - xmin) / (xmax - xmin); // [0..1]
    return (int16_t)lround((xn * 2.0 - 1.0) * 32767.0); // map to [-1,1] -> Q15
}
static inline double q15_unpack_double(int16_t q, double xmin, double xmax){
    double xn = ((double)q) / 32767.0;   // [-1,1]
    double t  = (xn + 1.0) * 0.5;        // [0,1]
    return xmin + t * (xmax - xmin);
}
static inline uint16_t q15_pack_w(double w){
    if (w < 0.0) w = 0.0;
    if (w > 2.0) w = 2.0;
    return (uint16_t)lround(w / 2.0 * 65535.0);
}
static inline double q15_unpack_w(uint16_t qw){
    return ((double)qw) / 65535.0 * 2.0;
}

// -----------------------------------------------------------------------------
// Neighbor data
// -----------------------------------------------------------------------------
#define MAX_NEIGHBORS  20u

typedef struct {
    uint16_t id;
    uint32_t last_seen_ms;
    uint32_t first_seen_ms;   /*!< \brief Timestamp when this neighbor was first observed. */
    int16_t  theta_mrad;      /*!< \brief Neighbor heading command (Vicsek intent). */
} neighbor_t;

static neighbor_t* upsert_neighbor(neighbor_t* arr, uint8_t* nbn, uint16_t id){
    uint32_t now = current_time_milliseconds();
    for(uint8_t i=0;i<*nbn;++i) if(arr[i].id==id) return &arr[i];
    if(*nbn>=MAX_NEIGHBORS) return NULL;
    neighbor_t* n=&arr[(*nbn)++];
    n->id=id; n->theta_mrad=0; n->last_seen_ms=now; n->first_seen_ms=now; return n;
}

static void purge_old_neighbors(neighbor_t* arr, uint8_t* nbn){
    uint32_t now=current_time_milliseconds();
    for(int i=(int)*nbn-1;i>=0;--i){
        if(now - arr[i].last_seen_ms > (int32_t)max_age){
            arr[i]=arr[*nbn-1];
            (*nbn)--;
        }
    }
}

// -----------------------------------------------------------------------------
// Cluster U-turn message + Push-sum piggyback
// -----------------------------------------------------------------------------
/** Additional stats carried in the radio packet. We keep a single weight `w` and
 *  four scalar "masses" s_i: polarization, wall ratio, neighbor persistence, and neighbor count. */

enum : uint8_t { VMSGF_CLUSTER_UTURN = 0x01 };

typedef struct __attribute__((__packed__)) {
    uint16_t sender_id;
    int16_t  theta_mrad;            // sender heading estimate
    uint8_t  flags;                 // bit0: cluster U-turn active
    int16_t  cluster_target_mrad;   // valid iff flags & VMSGF_CLUSTER_UTURN
    uint32_t cluster_wall_t0_ms;    // originator timestamp
    uint16_t cluster_msg_uid;       // de-dup / freshness id

    // --- push-sum on multiple stats (s_i, shared w) in Q15 ---
    int16_t  ps_s_pol_q15;          // polarization * weight
    int16_t  ps_s_wall_q15;         // wall/cluster ratio * weight
    int16_t  ps_s_pers_q15;         // neighbor persistence * weight
    int16_t  ps_s_nb_q15;           // neighbor count * weight (not normalized)
    uint16_t ps_w_q15;              // shared weight in [0..2]
} vicsek_msg_t;

#define MSG_SIZE ((uint16_t)sizeof(vicsek_msg_t))

// -----------------------------------------------------------------------------
// USERDATA
// -----------------------------------------------------------------------------
/** \brief Runtime state and statistics.
 *
 *  Fields prefixed `cs_` are consensus (global) EWMA estimates derived from push-sum.
 *  Fields prefixed `ls_` are local, instantaneous values before consensus.
 */
typedef struct {
    // Kinematics + Photostart
    ddk_t         ddk;
    photostart_t  ps;

    // Messaging
    uint32_t last_beacon_ms;

    // Neighbors
    neighbor_t neighbors[MAX_NEIGHBORS];
    uint8_t    nb_neighbors;

    // Vicsek state
    double     theta_cmd_rad;           // current command (cluster may override)
    uint32_t   last_update_ms;

    // Cluster U-turn
    bool       cluster_turn_active;
    double     cluster_target_rad;
    uint32_t   cluster_wall_t0_ms;
    uint32_t   cluster_active_until_ms;
    uint16_t   cluster_msg_uid;
    uint16_t   last_seen_cluster_uid;
    bool       have_seen_cluster_uid;

    // Behavior tracking for cluster rising-edge
    ddk_behavior_t prev_behavior;

    // --- Local instantaneous stats (pre push-sum) ---
    double     ls_pol_norm;             // [0..1] Rayleigh-corrected local polarization
    double     ls_wall_ratio;           // [0..1] fraction of time in avoidance or cluster
    double     ls_neighbor_persist;     // [0..1] mean normalized persistence
    double     ls_neighbor_count;       // raw count (>=0)

    // Integrators for wall/cluster occupancy
    double     accum_total_ms;
    double     accum_wc_ms;             // wall or cluster ms

    // --- Push-sum accumulators (single shared weight) ---
    double     ps_w;                    // shared weight
    double     ps_s_pol;
    double     ps_s_wall;
    double     ps_s_pers;
    double     ps_s_nb;

    // --- Consensus EWMA ---
    double     cs_pol_ewma;
    double     cs_wall_ewma;
    double     cs_pers_ewma;
    double     cs_nb_ewma;

} USERDATA;

DECLARE_USERDATA(USERDATA);
REGISTER_USERDATA(USERDATA);

// -----------------------------------------------------------------------------
// Utilities
// -----------------------------------------------------------------------------
static inline bool cluster_window_active(uint32_t now){
    return mydata->cluster_turn_active && (now < mydata->cluster_active_until_ms);
}

static void cluster_adopt_and_activate(int16_t tgt_mrad, uint32_t t0_ms, uint16_t uid){
    bool newer = (!mydata->cluster_turn_active) ||
                 (t0_ms > mydata->cluster_wall_t0_ms) ||
                 (!mydata->have_seen_cluster_uid) ||
                 (uid != mydata->last_seen_cluster_uid);
    if(newer){
        mydata->cluster_target_rad      = mrad_to_rad(tgt_mrad);
        mydata->cluster_wall_t0_ms      = t0_ms;
        mydata->cluster_active_until_ms = t0_ms + cluster_u_turn_duration_ms;
        mydata->cluster_turn_active     = true;
        mydata->last_seen_cluster_uid   = uid;
        mydata->have_seen_cluster_uid   = true;
    }
}

// -----------------------------------------------------------------------------
// Messaging
// -----------------------------------------------------------------------------
/** \brief Pack and broadcast the current push-sum masses.
 *
 *  We keep half of our mass and split the other half equally between neighbors.
 *  A single shared weight `w` is used for all stats to keep the packet compact.
 */
static bool send_message(void){
    uint32_t now=current_time_milliseconds();
    if (now - mydata->last_beacon_ms < BEACON_PERIOD_MS) return false;

    // self keeps 1/2 mass; other 1/2 split among neighbors
    double self_keep = 0.5;
    double to_split  = 0.5;
    uint32_t deg = (uint32_t)mydata->nb_neighbors;
    double share_w = (deg>0) ? (to_split*mydata->ps_w/deg) : 0.0;
    double share_pol  = (deg>0) ? (to_split*mydata->ps_s_pol/deg)  : 0.0;
    double share_wall = (deg>0) ? (to_split*mydata->ps_s_wall/deg) : 0.0;
    double share_pers = (deg>0) ? (to_split*mydata->ps_s_pers/deg) : 0.0;
    double share_nb   = (deg>0) ? (to_split*mydata->ps_s_nb/deg)   : 0.0;

    double heading_now = heading_detection_estimate(&mydata->ddk.hd);
    vicsek_msg_t m = {
        .sender_id = pogobot_helper_getid(),
        .theta_mrad= rad_to_mrad(heading_now),
        .flags     = 0u,
        .cluster_target_mrad = 0,
        .cluster_wall_t0_ms  = 0u,
        .cluster_msg_uid     = 0u,
        .ps_s_pol_q15        = q15_pack_double(share_pol,  0.0, (double)(MAX_NEIGHBORS+1)),
        .ps_s_wall_q15       = q15_pack_double(share_wall, 0.0, (double)(MAX_NEIGHBORS+1)),
        .ps_s_pers_q15       = q15_pack_double(share_pers, 0.0, (double)(MAX_NEIGHBORS+1)),
        .ps_s_nb_q15         = q15_pack_double(share_nb,   0.0, (double)(MAX_NEIGHBORS+1)),
        .ps_w_q15            = q15_pack_w(share_w)
    };

    if (cluster_window_active(now)) {
        m.flags               |= VMSGF_CLUSTER_UTURN;
        m.cluster_target_mrad  = rad_to_mrad(mydata->cluster_target_rad);
        m.cluster_wall_t0_ms   = mydata->cluster_wall_t0_ms;
        m.cluster_msg_uid      = mydata->cluster_msg_uid;
    }

    // Keep our retained self share
    mydata->ps_w      = self_keep * mydata->ps_w;
    mydata->ps_s_pol  = self_keep * mydata->ps_s_pol;
    mydata->ps_s_wall = self_keep * mydata->ps_s_wall;
    mydata->ps_s_pers = self_keep * mydata->ps_s_pers;
    mydata->ps_s_nb   = self_keep * mydata->ps_s_nb;

    mydata->last_beacon_ms = now;
    return pogobot_infrared_sendShortMessage_omni((uint8_t*)&m, MSG_SIZE);
}

static void process_message(message_t* mr){
    // Let DDK's avoidance consume wall messages first
    if (diff_drive_kin_process_message(&mydata->ddk, mr)) {
        return;
    }

    if (mr->header.payload_length < MSG_SIZE) return;
    vicsek_msg_t const* m=(vicsek_msg_t const*)mr->payload;
    if (m->sender_id==pogobot_helper_getid()) return;

    // Neighbor headings (for Vicsek + local polarization)
    neighbor_t* n = upsert_neighbor(mydata->neighbors, &mydata->nb_neighbors, m->sender_id);
    if(!n) return;
    n->theta_mrad    = m->theta_mrad;  // neighbor's measured heading
    n->last_seen_ms  = current_time_milliseconds();

    // Cluster flood
    if (m->flags & VMSGF_CLUSTER_UTURN) {
        cluster_adopt_and_activate(m->cluster_target_mrad, m->cluster_wall_t0_ms, m->cluster_msg_uid);
    }

    // Push-sum accumulation (shared weight)
    mydata->ps_s_pol  += q15_unpack_double(m->ps_s_pol_q15,  0.0, 1.0);
    mydata->ps_s_wall += q15_unpack_double(m->ps_s_wall_q15, 0.0, 1.0);
    mydata->ps_s_pers += q15_unpack_double(m->ps_s_pers_q15, 0.0, 1.0);
    mydata->ps_s_nb   += q15_unpack_double(m->ps_s_nb_q15,   0.0, 1.0);
    mydata->ps_w      += q15_unpack_w(m->ps_w_q15);
}

// -----------------------------------------------------------------------------
// Local polarization with Rayleigh correction
// -----------------------------------------------------------------------------
static double compute_local_polarization_norm(double self_heading, uint32_t *N_eff_out) {
    double sx = 0.0, sy = 0.0;
    uint32_t N = 0u;

    if (include_self_in_avg){
        sx += cos(self_heading);
        sy += sin(self_heading);
        N++;
    }
    for(uint8_t i=0;i<mydata->nb_neighbors;++i){
        double th = mrad_to_rad(mydata->neighbors[i].theta_mrad);
        sx += cos(th);
        sy += sin(th);
        N++;
    }
    if (N_eff_out) *N_eff_out = N;
    if (N == 0u) return 0.0;

    double R_bar = sqrt(sx*sx + sy*sy) / (double)N;
    double R0    = sqrt(M_PI) / (2.0 * sqrt((double)N));  // Rayleigh baseline
    if (R0 > 0.999) R0 = 0.999;                           // guard

    double P = (R_bar - R0) / (1.0 - R0);
    if (P < 0.0) P = 0.0;
    if (P > 1.0) P = 1.0;
    return P;
}

// -----------------------------------------------------------------------------
// LED
// -----------------------------------------------------------------------------
static void update_main_led(double heading){
    if (main_led_display_enum == SHOW_STATE){
        switch (diff_drive_kin_get_behavior(&mydata->ddk)) {
            case DDK_BEHAVIOR_AVOIDANCE:    pogobot_led_setColor(25,0,0);    break; // red
            case DDK_BEHAVIOR_NORMAL:       pogobot_led_setColor(0,25,0);    break; // green
            case DDK_BEHAVIOR_PID_DISABLED: pogobot_led_setColor(25,12,0);   break; // orange
            default:                        pogobot_led_setColor(6,6,6);     break;
        }
    } else if (main_led_display_enum == SHOW_ANGLE){
        if (heading < 0.0) heading += 2.0*M_PI;
        float hue_deg = (float)(heading * 180.0/M_PI);
        uint8_t r8,g8,b8; hsv_to_rgb(hue_deg, 1.0f, 1.0f, &r8,&g8,&b8);
        r8 = SCALE_0_255_TO_0_25(r8); g8=SCALE_0_255_TO_0_25(g8); b8=SCALE_0_255_TO_0_25(b8);
        if (r8==0&&g8==0&&b8==0) r8=1;
        pogobot_led_setColor(r8,g8,b8);
    } else if (main_led_display_enum == SHOW_POLARIZATION){
        // Map consensus polarization to hue [0..360)
        float hue_deg = (float)(mydata->cs_pol_ewma * 360.0);
        uint8_t r8,g8,b8; hsv_to_rgb(hue_deg, 1.0f, 1.0f, &r8,&g8,&b8);
        r8 = SCALE_0_255_TO_0_25(r8); g8=SCALE_0_255_TO_0_25(g8); b8=SCALE_0_255_TO_0_25(b8);
        if (r8==0&&g8==0&&b8==0) r8=1;
        pogobot_led_setColor(r8,g8,b8);
    } else if (main_led_display_enum == SHOW_STATS_RGB){
        // R: wall/cluster ratio; G: neighbor persistence; B: neighbor count (normalized)
        float nb_norm = (neighbor_norm_max>0) ? (float)fmin((double)mydata->cs_nb_ewma / (double)neighbor_norm_max, 1.0) : 0.0f;
        uint8_t r = (uint8_t)lround(fmin(fmax(mydata->cs_wall_ewma, 0.0),1.0)*25.0);
        uint8_t g = (uint8_t)lround(fmin(fmax(mydata->cs_pers_ewma, 0.0),1.0)*25.0);
        uint8_t b = (uint8_t)lround(nb_norm*25.0f);
        if (r==0&&g==0&&b==0) r=1;
        pogobot_led_setColor(r,g,b);
    }
}

// -----------------------------------------------------------------------------
// User init / step
// -----------------------------------------------------------------------------
static void user_init(void){
    srand(pogobot_helper_getRandSeed());
    memset(mydata, 0, sizeof(*mydata));

    main_loop_hz = 60;                                // tighter loop
    mydata->last_update_ms = current_time_milliseconds();
    max_nb_processed_msg_per_tick = 100;
    msg_rx_fn = process_message;
    msg_tx_fn = send_message;
    error_codes_led_idx = 3;

    // Init DDK (PID ON, avoidance ON by default)
    diff_drive_kin_init_default(&mydata->ddk);

    // Pass photostart into DDK so heading normalization can use calibrated sensors
    photostart_init(&mydata->ps);
    photostart_set_ewma_alpha(&mydata->ps, 0.30);
    diff_drive_kin_set_photostart(&mydata->ddk, &mydata->ps);

    // Heading detection geometry inside DDK
    heading_detection_set_geometry(&mydata->ddk.hd, alpha_deg, robot_radius);
    heading_detection_set_chirality(&mydata->ddk.hd, heading_chiralty_enum);

    // Behavior track
    mydata->prev_behavior = diff_drive_kin_get_behavior(&mydata->ddk);

    // Cluster defaults
    mydata->cluster_turn_active     = false;
    mydata->cluster_target_rad      = 0.0;
    mydata->cluster_wall_t0_ms      = 0u;
    mydata->cluster_active_until_ms = 0u;
    mydata->cluster_msg_uid         = 0u;
    mydata->have_seen_cluster_uid   = false;

    // Initial heading
    mydata->theta_cmd_rad = heading_detection_estimate(&mydata->ddk.hd);

    // Stats init
    mydata->accum_total_ms = 1e-9; // avoid div-by-zero
    mydata->accum_wc_ms    = 0.0;

    // Consensus init
    mydata->ps_w = 1.0;           // shared weight
    mydata->cs_pol_ewma  = 0.0;
    mydata->cs_wall_ewma = 0.0;
    mydata->cs_pers_ewma = 0.0;
    mydata->cs_nb_ewma   = 0.0;

#ifdef SIMULATOR
    printf("Vicsek (continuous) + Stats(wall, persist, nb) + Push-sum(shared w)\n");
    printf("  beta=%.3f rad/s, sigma=%.3f, max_dt=%.3f s, v=%.2f\n",
           vicsek_beta_rad_per_s, cont_noise_sigma_rad, cont_max_dt_s, base_speed_ratio);
#endif
}

static void user_step(void){
    // Photostart gate
    if (!photostart_step(&mydata->ps)) {
        pogobot_led_setColors(20,0,20,0); // purple while waiting
        pogobot_motor_set(motorL, motorStop);
        pogobot_motor_set(motorR, motorStop);
        return;
    }

    uint32_t now  = current_time_milliseconds();
    double dt_s   = (now - mydata->last_update_ms) * 1e-3;
    if (dt_s < 0.0) dt_s = 0.0;
    if (dt_s > cont_max_dt_s) dt_s = cont_max_dt_s;
    mydata->last_update_ms = now;

    // Current heading
    double heading = heading_detection_estimate(&mydata->ddk.hd);

    // Cluster rising-edge detection: NORMAL -> AVOIDANCE => originate a cluster instruction
    ddk_behavior_t beh = diff_drive_kin_get_behavior(&mydata->ddk);
    if (mydata->prev_behavior != DDK_BEHAVIOR_AVOIDANCE && beh == DDK_BEHAVIOR_AVOIDANCE) {
        double phi = rand_uniform((double)phi_rad_min, (double)phi_rad_max);
        mydata->cluster_target_rad      = wrap_pi(heading + phi);
        mydata->cluster_wall_t0_ms      = now;
        mydata->cluster_active_until_ms = now + cluster_u_turn_duration_ms;
        mydata->cluster_turn_active     = true;
        mydata->cluster_msg_uid         = (uint16_t)(rand() & 0xFFFF);
        mydata->last_seen_cluster_uid   = mydata->cluster_msg_uid;
        mydata->have_seen_cluster_uid   = true;
    }
    mydata->prev_behavior = beh;

    // Vicsek: compute mean neighbor direction (for command)
    purge_old_neighbors(mydata->neighbors, &mydata->nb_neighbors);
    double sx=0.0, sy=0.0;
    if (include_self_in_avg){ sx+=cos(heading); sy+=sin(heading); }
    for (uint8_t i=0; i<mydata->nb_neighbors; ++i){
        double th = mrad_to_rad(mydata->neighbors[i].theta_mrad);
        sx += cos(th); sy += sin(th);
    }
    double theta_mean = (sx==0.0 && sy==0.0) ? heading : atan2(sy, sx);

    // Command selection (cluster override window)
    double theta_cmd = cluster_window_active(now) ? mydata->cluster_target_rad : theta_mean;
    mydata->theta_cmd_rad = theta_cmd;

    // Continuous-time Vicsek controller -> desired dtheta/dt
    double err = wrap_pi(theta_cmd - heading);
    double dtheta = vicsek_beta_rad_per_s * sin(err) * dt_s;

    // Add diffusion-like noise on angle
    if (cont_noise_sigma_rad > 0.0){
        double u1 = (rand()+1.0)/(RAND_MAX+2.0);
        double u2 = (rand()+1.0)/(RAND_MAX+2.0);
        double z  = sqrt(-2.0*log(u1)) * cos(2.0*M_PI*u2);
        dtheta += cont_noise_sigma_rad * sqrt(dt_s) * z;
    }

    // --- Local stats ---
    // Polarization
    uint32_t N_eff = 0;
    mydata->ls_pol_norm = compute_local_polarization_norm(heading, &N_eff);

    // Wall/cluster ratio integrators
    mydata->accum_total_ms += dt_s * 1000.0;
    if (beh == DDK_BEHAVIOR_AVOIDANCE || cluster_window_active(now)) {
        mydata->accum_wc_ms += dt_s * 1000.0;
    }
    mydata->ls_wall_ratio = fmin(fmax(mydata->accum_wc_ms / mydata->accum_total_ms, 0.0), 1.0);

    // Neighbor persistence: mean normalized (clamped) age of currently visible neighbors
    double sum_norm_age = 0.0;
    for (uint8_t i=0; i<mydata->nb_neighbors; ++i){
        double age_ms = (double)(now - mydata->neighbors[i].first_seen_ms);
        double norm = (neighbor_persist_norm_ms>0) ? fmin(age_ms / (double)neighbor_persist_norm_ms, 1.0) : 0.0;
        sum_norm_age += norm;
    }
    mydata->ls_neighbor_persist = (mydata->nb_neighbors>0) ? (sum_norm_age / (double)mydata->nb_neighbors) : 0.0;

    // Neighbor count (raw, exclude self)
    mydata->ls_neighbor_count = (double)mydata->nb_neighbors;

    // --- Push-sum update (single shared weight) ---
    double w0 = (double)N_eff;           // shared weight for all stats
    if (N_eff < 2) w0 = 0.0;             // if isolated, contribute nothing

    mydata->ps_w      = w0;
    mydata->ps_s_pol  = mydata->ls_pol_norm        * w0;
    mydata->ps_s_wall = mydata->ls_wall_ratio      * w0;
    mydata->ps_s_pers = mydata->ls_neighbor_persist* w0;
    mydata->ps_s_nb   = mydata->ls_neighbor_count  * w0; // average neighbor count weighted by N_eff

    // --- Consensus instantaneous estimates ---
    double denom = (mydata->ps_w > 1e-12) ? mydata->ps_w : 1.0;
    double c_pol  = fmin(fmax(mydata->ps_s_pol  / denom, 0.0), 1.0);
    double c_wall = fmin(fmax(mydata->ps_s_wall / denom, 0.0), 1.0);
    double c_pers = fmin(fmax(mydata->ps_s_pers / denom, 0.0), 1.0);
    double c_nb   = fmax(mydata->ps_s_nb   / denom, 0.0);

    // EWMA smoothing
    mydata->cs_pol_ewma  = (1.0 - ewma_alpha_polarization) * mydata->cs_pol_ewma  + ewma_alpha_polarization * c_pol;
    mydata->cs_wall_ewma = (1.0 - ewma_alpha_wallratio)   * mydata->cs_wall_ewma + ewma_alpha_wallratio    * c_wall;
    mydata->cs_pers_ewma = (1.0 - ewma_alpha_persistence) * mydata->cs_pers_ewma + ewma_alpha_persistence  * c_pers;
    mydata->cs_nb_ewma   = (1.0 - ewma_alpha_nb)          * mydata->cs_nb_ewma   + ewma_alpha_nb           * c_nb;

    // --- Drive the DDK ---
    float  v_cmd      = base_speed_ratio;
    double dtheta_inc = dtheta;
    diff_drive_kin_step(&mydata->ddk, v_cmd, dtheta_inc, heading);

    // --- LED ---
    update_main_led(heading);

    // Debug printouts
    if (pogobot_ticks % 1000 == 0) {
        printf("consensus: pol=%.3f wall=%.3f persist=%.3f nb=%.2f | local: N=%u\n",
               mydata->cs_pol_ewma, mydata->cs_wall_ewma, mydata->cs_pers_ewma, mydata->cs_nb_ewma,
               (unsigned)mydata->nb_neighbors);
    }
}

// -----------------------------------------------------------------------------
// Simulator helpers
// -----------------------------------------------------------------------------
#ifdef SIMULATOR
static void create_data_schema(void){
    data_add_column_int8("nb_neighbors");
    data_add_column_double("theta_cmd_rad");

    data_add_column_double("pol_local_norm");
    data_add_column_double("wall_ratio_local");
    data_add_column_double("neighbor_persist_local");
    data_add_column_double("neighbor_count_local");

    data_add_column_double("pol_consensus_ewma");
    data_add_column_double("wall_consensus_ewma");
    data_add_column_double("persist_consensus_ewma");
    data_add_column_double("nb_consensus_ewma");

    data_add_column_int8("cluster_active");
    data_add_column_double("cluster_target_rad");
    data_add_column_int32("cluster_t0_ms");
    data_add_column_int32("cluster_until_ms");
}
static void export_data(void){
    data_set_value_int8("nb_neighbors", (int8_t)mydata->nb_neighbors);
    data_set_value_double("theta_cmd_rad", mydata->theta_cmd_rad);

    data_set_value_double("pol_local_norm",        mydata->ls_pol_norm);
    data_set_value_double("wall_ratio_local",      mydata->ls_wall_ratio);
    data_set_value_double("neighbor_persist_local",mydata->ls_neighbor_persist);
    data_set_value_double("neighbor_count_local",  mydata->ls_neighbor_count);

    data_set_value_double("pol_consensus_ewma",    mydata->cs_pol_ewma);
    data_set_value_double("wall_consensus_ewma",   mydata->cs_wall_ewma);
    data_set_value_double("persist_consensus_ewma",mydata->cs_pers_ewma);
    data_set_value_double("nb_consensus_ewma",     mydata->cs_nb_ewma);

    data_set_value_int8("cluster_active", (int8_t)(mydata->cluster_turn_active?1:0));
    data_set_value_double("cluster_target_rad", mydata->cluster_target_rad);
    data_set_value_int32("cluster_t0_ms", (int32_t)mydata->cluster_wall_t0_ms);
    data_set_value_int32("cluster_until_ms", (int32_t)mydata->cluster_active_until_ms);
}
static void global_setup(void){
    init_from_configuration(max_age);
    init_from_configuration(vicsek_beta_rad_per_s);
    init_from_configuration(cont_noise_sigma_rad);
    init_from_configuration(cont_max_dt_s);
    init_from_configuration(base_speed_ratio);
    init_from_configuration(vicsek_turn_gain_clip_deg);
    init_from_configuration(include_self_in_avg);

    init_from_configuration(ewma_alpha_polarization);
    init_from_configuration(ewma_alpha_wallratio);
    init_from_configuration(ewma_alpha_persistence);
    init_from_configuration(ewma_alpha_nb);

    init_from_configuration(neighbor_persist_norm_ms);
    init_from_configuration(neighbor_norm_max);

    init_from_configuration(alpha_deg);
    init_from_configuration(robot_radius);

    char heading_chiralty[128] = "CW";
    init_array_from_configuration(heading_chiralty);
    if (strcasecmp(heading_chiralty, "cw") == 0) {
        heading_chiralty_enum = HEADING_CW;
    } else if (strcasecmp(heading_chiralty, "ccw") == 0) {
        heading_chiralty_enum = HEADING_CCW;
    } else {
        printf("ERROR: unknown heading_chiralty: '%s' (use 'cw' or 'ccw').\n", heading_chiralty);
        exit(1);
    }

    init_from_configuration(cluster_u_turn_duration_ms);
    init_from_configuration(phi_rad_min);
    init_from_configuration(phi_rad_max);
    if (phi_rad_min > phi_rad_max){ float t=phi_rad_min; phi_rad_min=phi_rad_max; phi_rad_max=t; }

    char main_led_display[128] = "stats";
    init_array_from_configuration(main_led_display);
    if (strcasecmp(main_led_display, "state") == 0) {
        main_led_display_enum = SHOW_STATE;
    } else if (strcasecmp(main_led_display, "angle") == 0) {
        main_led_display_enum = SHOW_ANGLE;
    } else if (strcasecmp(main_led_display, "polarization") == 0) {
        main_led_display_enum = SHOW_POLARIZATION;
    } else if (strcasecmp(main_led_display, "stats") == 0 || strcasecmp(main_led_display, "rgb") == 0) {
        main_led_display_enum = SHOW_STATS_RGB;
    } else {
        printf("ERROR: unknown main_led_display: '%s' (use 'state'|'angle'|'polarization'|'stats').\n", main_led_display);
        exit(1);
    }
}
#endif

int main(void){
    pogobot_init();
    pogobot_start(user_init, user_step);
#ifdef SIMULATOR
    SET_CALLBACK(callback_global_setup,       global_setup);
    SET_CALLBACK(callback_create_data_schema, create_data_schema);
    SET_CALLBACK(callback_export_data,        export_data);
#endif
    // Walls app stays enabled—DDK listens and handles avoidance internally.
    pogobot_start(default_walls_user_init, default_walls_user_step, "walls");
    return 0;
}

