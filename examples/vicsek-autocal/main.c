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

// Optional: we rely on DDK's internal heading detection (hd) and photostart
// (Photostart struct is registered into the DDK so that heading normalization can use it)

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
double   vicsek_beta_rad_per_s     = 3.0;     /* continuous Vicsek: dθ/dt = β sin(θ̄ - θ) + σ ξ */
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

// --- Polarization consensus ---
double   pol_ewma_alpha            = 0.20;    /* EWMA smoothing of global consensus [0..1] */

// -----------------------------------------------------------------------------
// LEDs
// -----------------------------------------------------------------------------
typedef enum {
    SHOW_STATE,
    SHOW_ANGLE,
    SHOW_POLARIZATION  // HSV hue encodes polarization in [0..1] (same HSV mapping as SHOW_ANGLE)
} main_led_display_type_t;
main_led_display_type_t main_led_display_enum = SHOW_STATE;

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

// For weights w we keep [0..2] range
static inline uint16_t q15_pack_w(double w){
    if (w < 0.0) w = 0.0;
    if (w > 2.0) w = 2.0;
    return (uint16_t)lround(w / 2.0 * 65535.0);
}
static inline double q15_unpack_w(uint16_t qw){
    return ((double)qw) / 65535.0 * 2.0;
}

// -----------------------------------------------------------------------------
// Neighbor data (only heading mean is needed locally; push-sum carried in beacons)
// -----------------------------------------------------------------------------
#define MAX_NEIGHBORS  20u

typedef struct {
    uint16_t id;
    uint32_t last_seen_ms;
    int16_t  theta_mrad;  // sender heading command (Vicsek intent)
} neighbor_t;

static neighbor_t* upsert_neighbor(neighbor_t* arr, uint8_t* nbn, uint16_t id){
    for(uint8_t i=0;i<*nbn;++i) if(arr[i].id==id) return &arr[i];
    if(*nbn>=MAX_NEIGHBORS) return NULL;
    neighbor_t* n=&arr[(*nbn)++];
    n->id=id; n->theta_mrad=0; n->last_seen_ms=0; return n;
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
enum : uint8_t { VMSGF_CLUSTER_UTURN = 0x01 };

typedef struct __attribute__((__packed__)) {
    uint16_t sender_id;
    int16_t  theta_mrad;            // commanded direction of sender (Vicsek)
    uint8_t  flags;                 // bit0: cluster U-turn active
    int16_t  cluster_target_mrad;   // valid if flags & VMSGF_CLUSTER_UTURN
    uint32_t cluster_wall_t0_ms;    // originator timestamp
    uint16_t cluster_msg_uid;       // de-dup / freshness id

    // --- push-sum on polarization (s,w) in Q15 ---
    int16_t  ps_s_q15;              // packs s in [-1..+1], but we only use [0..1]
    uint16_t ps_w_q15;              // packs w in [0..2]
} vicsek_msg_t;

#define MSG_SIZE ((uint16_t)sizeof(vicsek_msg_t))

// -----------------------------------------------------------------------------
// USERDATA
// -----------------------------------------------------------------------------
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

    // Consensus (push-sum) over polarization
    double     pol_local_norm;          // [0..1] Rayleigh-corrected local polarization
    double     ps_s;                    // push-sum scalar
    double     ps_w;                    // push-sum weight
    double     pol_consensus_ewma;      // EWMA smoothed consensus

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
static bool send_message(void){
    uint32_t now=current_time_milliseconds();
    if (now - mydata->last_beacon_ms < BEACON_PERIOD_MS) return false;

//    // Split push-sum mass among (deg_out + 1) where deg_out ≈ nb_neighbors
//    uint32_t k = (uint32_t)mydata->nb_neighbors + 1u;
//    double share_s = mydata->ps_s / (double)k;
//    double share_w = mydata->ps_w / (double)k;
    // self keeps 1/2 mass; other 1/2 split among neighbors
    double self_keep = 0.5;
    double to_split  = 0.5;
    uint32_t deg = (uint32_t)mydata->nb_neighbors;
    double share_s = (deg>0) ? (to_split*mydata->ps_s/deg) : 0.0;
    double share_w = (deg>0) ? (to_split*mydata->ps_w/deg) : 0.0;

    double heading_now = heading_detection_estimate(&mydata->ddk.hd);
    vicsek_msg_t m = {
        .sender_id = pogobot_helper_getid(),
        .theta_mrad= rad_to_mrad(heading_now),
        .flags     = 0u,
        .cluster_target_mrad = 0,
        .cluster_wall_t0_ms  = 0u,
        .cluster_msg_uid     = 0u,
        .ps_s_q15            = q15_pack_double(share_s, 0.0, /*xmax*/ (double)(MAX_NEIGHBORS+1)),
        .ps_w_q15            = q15_pack_w(share_w)
    };

    if (cluster_window_active(now)) {
        m.flags               |= VMSGF_CLUSTER_UTURN;
        m.cluster_target_mrad  = rad_to_mrad(mydata->cluster_target_rad);
        m.cluster_wall_t0_ms   = mydata->cluster_wall_t0_ms;
        m.cluster_msg_uid      = mydata->cluster_msg_uid;
    }

    // Keep our retained self share
//    mydata->ps_s = share_s;
//    mydata->ps_w = share_w;
    mydata->ps_s = self_keep * share_s;
    mydata->ps_w = self_keep * share_w;

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
    n->theta_mrad    = m->theta_mrad;  // <-- neighbor's measured heading
    n->last_seen_ms  = current_time_milliseconds();

    // Cluster flood
    if (m->flags & VMSGF_CLUSTER_UTURN) {
        cluster_adopt_and_activate(m->cluster_target_mrad, m->cluster_wall_t0_ms, m->cluster_msg_uid);
    }

    // Push-sum accumulation
    double rx_s = q15_unpack_double(m->ps_s_q15, 0.0, 1.0);
    double rx_w = q15_unpack_w(m->ps_w_q15);
    mydata->ps_s += rx_s;
    mydata->ps_w += rx_w;
}

// -----------------------------------------------------------------------------
// Local polarization with Rayleigh correction
//   R_bar = |sum e^{i θ}| / N
//   R0(N) ≈ sqrt(pi)/(2*sqrt(N))  (expected for random uniform)
//   P_norm = clamp( (R_bar - R0)/(1 - R0), 0..1 )
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
static void update_main_led(double heading, double pol_consensus){
    if (main_led_display_enum == SHOW_STATE){
        switch (diff_drive_kin_get_behavior(&mydata->ddk)) {
            case DDK_BEHAVIOR_AVOIDANCE:    pogobot_led_setColor(255,0,0);   break; // red
            case DDK_BEHAVIOR_NORMAL:       pogobot_led_setColor(0,255,0);   break; // green
            case DDK_BEHAVIOR_PID_DISABLED: pogobot_led_setColor(255,128,0); break; // orange
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
        // Use the same HSV mapping as SHOW_ANGLE: map scalar p∈[0,1] to hue∈[0,360)
        float hue_deg = (float)(pol_consensus * 360.0);
        uint8_t r8,g8,b8; hsv_to_rgb(hue_deg, 1.0f, 1.0f, &r8,&g8,&b8);
        r8 = SCALE_0_255_TO_0_25(r8); g8=SCALE_0_255_TO_0_25(g8); b8=SCALE_0_255_TO_0_25(b8);
        if (r8==0&&g8==0&&b8==0) r8=1;
        pogobot_led_setColor(r8,g8,b8);
    }
}

// -----------------------------------------------------------------------------
// User init / step
// -----------------------------------------------------------------------------
static void user_init(void){
    srand(pogobot_helper_getRandSeed());
    memset(mydata, 0, sizeof(*mydata));

    main_loop_hz = 60;                                // tighter loop like example
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

    // Initial heading (use DDK's heading detector)
    double hd0 = heading_detection_estimate(&mydata->ddk.hd);
    mydata->theta_cmd_rad = hd0;

    // Consensus init
    mydata->pol_local_norm     = 0.0;
    mydata->ps_s               = 0.0;
    mydata->ps_w               = 1.0;
    mydata->pol_consensus_ewma = 0.0;

#ifdef SIMULATOR
    printf("Vicsek (continuous) + DDK + Photostart + Push-sum-Pol\n");
    printf("  beta=%.3f rad/s, sigma=%.3f, max_dt=%.3f s, v=%.2f\n",
           vicsek_beta_rad_per_s, cont_noise_sigma_rad, cont_max_dt_s, base_speed_ratio);
#endif
}

static void user_step(void){
    // Photostart gate: keep still until flash sequence complete
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

    // Current heading from DDK's heading detector
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

    // --- Vicsek: compute mean neighbor direction (for command) ---
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

    // --- Local polarization (Rayleigh corrected) ---
    uint32_t N_eff = 0;
    mydata->pol_local_norm = compute_local_polarization_norm(heading, &N_eff);

    // Weight by N_eff so sparse nodes don't dominate
    double s0 = mydata->pol_local_norm * (double)N_eff;
    double w0 = (double)N_eff;

    // If totally isolated, contribute ~nothing
    if (N_eff < 2) { s0 = 0.0; w0 = 0.0; }

    // --- Push-sum update (broadcast done in send_message) ---
    mydata->ps_s = s0;
    mydata->ps_w = w0;

    // Current ratio (= instantaneous consensus estimate before EWMA)
    double pol_consensus = (mydata->ps_w > 1e-12) ? (mydata->ps_s / mydata->ps_w) : 0.0;
    if (pol_consensus < 0.0) pol_consensus = 0.0;
    if (pol_consensus > 1.0) pol_consensus = 1.0;

    // EWMA smoothing for stability
    mydata->pol_consensus_ewma = (1.0 - pol_ewma_alpha) * mydata->pol_consensus_ewma
                                 + pol_ewma_alpha * pol_consensus;

    // --- Drive the DDK ---
    // Speed is constant ratio; angular increment is dtheta (per tick)
    float  v_cmd      = base_speed_ratio;
    double dtheta_inc = dtheta;

    // Provide heading explicitly (Option B in example) to keep everything consistent
    diff_drive_kin_step(&mydata->ddk, v_cmd, dtheta_inc, heading);

    // --- LED ---
    update_main_led(heading, mydata->pol_consensus_ewma);

    // Debug
    if (pogobot_ticks % 1000 == 0) {
        printf("pol_consensus_ewma=%f\n", mydata->pol_consensus_ewma);
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
    data_add_column_double("pol_consensus_ewma");
    data_add_column_int8("cluster_active");
    data_add_column_double("cluster_target_rad");
    data_add_column_int32("cluster_t0_ms");
    data_add_column_int32("cluster_until_ms");
}
static void export_data(void){
    data_set_value_int8("nb_neighbors", (int8_t)mydata->nb_neighbors);
    data_set_value_double("theta_cmd_rad", mydata->theta_cmd_rad);
    data_set_value_double("pol_local_norm", mydata->pol_local_norm);
    data_set_value_double("pol_consensus_ewma", mydata->pol_consensus_ewma);
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
    init_from_configuration(pol_ewma_alpha);

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

    char main_led_display[128] = "state";
    init_array_from_configuration(main_led_display);
    if (strcasecmp(main_led_display, "state") == 0) {
        main_led_display_enum = SHOW_STATE;
    } else if (strcasecmp(main_led_display, "angle") == 0) {
        main_led_display_enum = SHOW_ANGLE;
    } else if (strcasecmp(main_led_display, "polarization") == 0) {
        main_led_display_enum = SHOW_POLARIZATION;
    } else {
        printf("ERROR: unknown main_led_display: '%s' (use 'state'|'angle'|'polarization').\n", main_led_display);
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

