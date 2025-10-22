/* vicsek-autocal.c
 *
 * Vicsek controller with online autocalibration:
 * - Multi-stat push-sum (polarization, wall ratio, neighbor persistence, neighbor count)
 * - Per-cluster leader election (min-ID flooding)
 * - Leader runs windowed 1+1-ES on (beta, sigma, base_speed_ratio) to match target scores
 * - Leaders broadcast parameters; followers adopt
 * - Leader-only LED shows current loss; others black
 *
 * C11, same brace style as your codebase.
 */
#include "pogobase.h"
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <strings.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "pogo-utils/kinematics.h"
#include "pogo-utils/oneplusone_es.h"   // 1+1-ES (as in example main.c) :contentReference[oaicite:2]{index=2}
#include "pogo-utils/version.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---------- Tunables (YAML) ---------- */
uint32_t max_age                   = 600;
double   vicsek_beta_rad_per_s     = 3.0;
double   cont_noise_sigma_rad      = 0.10;
double   cont_max_dt_s             = 0.05;
float    base_speed_ratio          = 0.70f;

double   alpha_deg                 = 40.0;
double   robot_radius              = 0.0265;
heading_chirality_t heading_chiralty_enum = HEADING_CW;

/* Cluster U-turn */
uint32_t cluster_u_turn_duration_ms = 1500;
float    phi_rad_min = (float)M_PI;
float    phi_rad_max = (float)M_PI;

/* Push-sum EWMA */
double   ewma_alpha_polarization   = 0.20;
double   ewma_alpha_wallratio      = 0.10;
double   ewma_alpha_persistence    = 0.10;
double   ewma_alpha_nb             = 0.10;

/* Local windows */
uint32_t neighbor_persist_norm_ms  = 10000;
uint8_t  neighbor_norm_max         = 12;

/* Targets (user-provided) */
double   target_pol   = 0.90;
double   target_wall  = 0.10;
double   target_pers  = 0.15;
double   target_nb    = 6.0;

/* Loss weights */
double   w_pol  = 1.0;
double   w_wall = 0.5;
double   w_pers = 0.5;
double   w_nb   = 0.25;

/* ES evaluation schedule (leader only) */
uint32_t es_eval_window_ms = 5000;   /* window length per evaluation */
uint32_t es_eval_quiet_ms  = 200;    /* minimal guard before sampling (settling) */

/* ES bounds and params */
float    es_lo_beta   = -15.0f, es_hi_beta = 15.0f;  /* rad/s */
float    es_lo_sigma  = 0.00f,  es_hi_sigma = 0.80f; /* rad/sqrt(s) */
float    es_lo_speed  = 0.10f,  es_hi_speed = 1.00f; /* [0..1] */

float    es_sigma0    = 0.20f;
float    es_sigma_min = 1e-4f;
float    es_sigma_max = 0.8f;
float    es_s_target  = 0.20f;
float    es_s_alpha   = 0.20f;
float    es_c_sigma   = 0.0f;        /* 0 => auto = 0.6/sqrt(n) */
int      es_evals_per_tick = 1;

/* LED modes */
typedef enum {
    SHOW_LOSS_LEADER,   /* leader shows loss, followers off */
    SHOW_ANGLE,
    SHOW_POLARIZATION
} main_led_display_type_t;
main_led_display_type_t main_led_display_enum = SHOW_LOSS_LEADER;

/* Radio beacons */
#define BEACON_HZ          10u
#define BEACON_PERIOD_MS  (1000u / BEACON_HZ)

/* Helpers */
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
static inline double wrap_pi(double a){ while(a> M_PI)a-=2.0*M_PI; while(a<-M_PI)a+=2.0*M_PI; return a; }
static inline int16_t rad_to_mrad(double a){ a=wrap_pi(a); long v=lround(a*1000.0); if(v>32767)v=32767; if(v<-32768)v=-32768; return (int16_t)v; }
static inline double  mrad_to_rad(int16_t m){ return ((double)m)/1000.0; }
static inline double  rand_uniform(double a, double b){ double u=(double)rand()/(double)RAND_MAX; return a + (b - a) * u; }
#define SCALE_0_255_TO_0_25(x)   (uint8_t)((x) * (25.0f / 255.0f) + 0.5f)

/* Q15 pack for push-sum */
static inline int16_t q15_pack_double(double x, double xmin, double xmax){
    if (x < xmin) x = xmin;
    if (x > xmax) x = xmax;
    double xn = (x - xmin) / (xmax - xmin);
    return (int16_t)lround((xn * 2.0 - 1.0) * 32767.0);
}
static inline double q15_unpack_double(int16_t q, double xmin, double xmax){
    double xn = ((double)q) / 32767.0;
    double t  = (xn + 1.0) * 0.5;
    return xmin + t * (xmax - xmin);
}
static inline uint16_t q15_pack_w(double w){
    if (w < 0.0) w = 0.0;
    if (w > 2.0) w = 2.0;
    return (uint16_t)lround(w / 2.0 * 65535.0);
}
static inline double q15_unpack_w(uint16_t qw){ return ((double)qw) / 65535.0 * 2.0; }

/* Neighbors */
#define MAX_NEIGHBORS  20u
typedef struct {
    uint16_t id;
    uint32_t last_seen_ms;
    uint32_t first_seen_ms;
    int16_t  theta_mrad;
    uint16_t leader_id_hint; /* optional 1-hop hint */
} neighbor_t;

static neighbor_t* upsert_neighbor(neighbor_t* arr, uint8_t* nbn, uint16_t id){
    uint32_t now = current_time_milliseconds();
    for(uint8_t i=0;i<*nbn;++i) if(arr[i].id==id) return &arr[i];
    if(*nbn>=MAX_NEIGHBORS) return NULL;
    neighbor_t* n=&arr[(*nbn)++];
    n->id=id; n->theta_mrad=0; n->last_seen_ms=now; n->first_seen_ms=now; n->leader_id_hint=0;
    return n;
}
static void purge_old_neighbors(neighbor_t* arr, uint8_t* nbn){
    uint32_t now=current_time_milliseconds();
    for(int i=(int)*nbn-1;i>=0;--i){
        if(now - arr[i].last_seen_ms > (int32_t)max_age){
            arr[i]=arr[*nbn-1]; (*nbn)--;
        }
    }
}

/* Messages: push-sum + cluster + leader + params */
enum : uint8_t { VMSGF_CLUSTER_UTURN = 0x01, VMSGF_LEADER = 0x02 };

typedef struct __attribute__((__packed__)) {
    uint16_t sender_id;
    int16_t  theta_mrad;
    uint8_t  flags;
    /* Cluster */
    int16_t  cluster_target_mrad;
    uint32_t cluster_wall_t0_ms;
    uint16_t cluster_msg_uid;
    /* Push-sum (shared w) */
    int16_t  ps_s_pol_q15;
    int16_t  ps_s_wall_q15;
    int16_t  ps_s_pers_q15;
    int16_t  ps_s_nb_q15;
    uint16_t ps_w_q15;
    /* Leader election + params flood */
    uint16_t leader_id;
    float    par_beta;
    float    par_sigma;
    float    par_speed;
    uint32_t par_epoch;
} vicsek_msg_t;
#define MSG_SIZE ((uint16_t)sizeof(vicsek_msg_t))

/* USERDATA */
typedef struct {
    /* DDK + photostart */
    ddk_t        ddk;
    photostart_t ps;

    /* Messaging */
    uint32_t last_beacon_ms;

    /* Neighbors */
    neighbor_t neighbors[MAX_NEIGHBORS];
    uint8_t    nb_neighbors;

    /* Vicsek */
    double     theta_cmd_rad;
    uint32_t   last_update_ms;

    /* Cluster U-turn state */
    bool       cluster_turn_active;
    double     cluster_target_rad;
    uint32_t   cluster_wall_t0_ms;
    uint32_t   cluster_active_until_ms;
    uint16_t   cluster_msg_uid;
    uint16_t   last_seen_cluster_uid;
    bool       have_seen_cluster_uid;
    ddk_behavior_t prev_behavior;

    /* Local stats */
    double     ls_pol_norm;
    double     ls_wall_ratio;
    double     ls_neighbor_persist;
    double     ls_neighbor_count;
    double     accum_total_ms;
    double     accum_wc_ms;

    /* Push-sum accumulators (shared weight) */
    double     ps_w;
    double     ps_s_pol;
    double     ps_s_wall;
    double     ps_s_pers;
    double     ps_s_nb;

    /* Consensus EWMA */
    double     cs_pol_ewma;
    double     cs_wall_ewma;
    double     cs_pers_ewma;
    double     cs_nb_ewma;

    /* Leader election */
    uint16_t   leader_id;          /* our current view of cluster leader (min-ID) */
    uint32_t   leader_last_update_ms;
    uint32_t par_epoch_local;                  // leader's own counter
    uint32_t par_epoch_seen;                   // best epoch we’ve adopted/forward

    /* Param triple currently applied (controller) */
    float      cur_beta, cur_sigma, cur_speed;

    /* ES (leader only) */
    es1p1_t    es;
    float      es_x[3], es_x_try[3], es_lo[3], es_hi[3];
    uint32_t   es_window_t0_ms;
    double     es_accum_loss;
    double     es_accum_w;
    bool       es_has_candidate;  /* true if we've asked ES and are evaluating x_try */


    /* For LED */
    double     last_loss_for_led;
} USERDATA;

DECLARE_USERDATA(USERDATA);
REGISTER_USERDATA(USERDATA)

/* --- Polarization (Rayleigh-corrected) --- */
static double compute_local_polarization_norm(double self_heading, uint32_t *N_eff_out){
    double sx=0.0, sy=0.0; uint32_t N=0;
    sx += cos(self_heading); sy += sin(self_heading); N++; /* include self */
    for(uint8_t i=0;i<mydata->nb_neighbors;++i){ double th=mrad_to_rad(mydata->neighbors[i].theta_mrad); sx+=cos(th); sy+=sin(th); N++; }
    if (N_eff_out) *N_eff_out = N;
    if (N==0) return 0.0;
    double R_bar = sqrt(sx*sx + sy*sy) / (double)N;
    double R0    = sqrt(M_PI) / (2.0 * sqrt((double)N));
    if (R0 > 0.999) R0=0.999;
    double P = (R_bar - R0) / (1.0 - R0);
    if (P<0.0) P=0.0; if (P>1.0) P=1.0; return P;
}

/* --- Cluster helpers --- */
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

/* --- Leader election (min-ID flooding) --- */
static void leader_election_tick(void){
    uint16_t old = mydata->leader_id;
    uint16_t min_id = pogobot_helper_getid();
    for (uint8_t i=0;i<mydata->nb_neighbors;++i)
        if (mydata->neighbors[i].leader_id_hint && mydata->neighbors[i].leader_id_hint < min_id)
            min_id = mydata->neighbors[i].leader_id_hint;
        else if (mydata->neighbors[i].id < min_id)
            min_id = mydata->neighbors[i].id;

    if (min_id != old){
        mydata->leader_id = min_id;
        mydata->leader_last_update_ms = current_time_milliseconds();
        if (pogobot_helper_getid() != mydata->leader_id){
            // We’re now a follower of a (possibly) different leader:
            mydata->par_epoch_seen = 0;  // adopt on first packet we hear
        } else {
            // We became the leader: keep our local epoch
            mydata->par_epoch_seen  = mydata->par_epoch_local;
        }
    }
}

/* --- Loss (targets vs consensus) --- */
static inline double clamp01(double x){ if (x<0.0) return 0.0; if (x>1.0) return 1.0; return x; }
static double compute_loss_from_consensus(double pol, double wall, double pers, double nb){
    double e_pol  = pol  - target_pol;
    double e_wall = wall - target_wall;
    double e_pers = pers - target_pers;
    double e_nb   = nb   - target_nb;
    /* Optional: normalize neighbor error by a scale to keep magnitudes similar */
    double nb_scale = (neighbor_norm_max>0) ? (double)neighbor_norm_max : 8.0;
    e_nb = e_nb / nb_scale;
    return w_pol*e_pol*e_pol + w_wall*e_wall*e_wall + w_pers*e_pers*e_pers + w_nb*e_nb*e_nb;
}

/* --- Messaging --- */
static bool send_message(void){
    uint32_t now=current_time_milliseconds();
    if (now - mydata->last_beacon_ms < BEACON_PERIOD_MS) return false;

    /* Push-sum split: keep half, split half to neighbors */
    double self_keep = 0.5, to_split = 0.5;
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
        .ps_w_q15            = q15_pack_w(share_w),
        .leader_id           = mydata->leader_id,
        .par_beta            = mydata->cur_beta,
        .par_sigma           = mydata->cur_sigma,
        .par_speed           = mydata->cur_speed,
        .par_epoch           = mydata->par_epoch_seen
    };

    if (cluster_window_active(now)) {
        m.flags               |= VMSGF_CLUSTER_UTURN;
        m.cluster_target_mrad  = rad_to_mrad(mydata->cluster_target_rad);
        m.cluster_wall_t0_ms   = mydata->cluster_wall_t0_ms;
        m.cluster_msg_uid      = mydata->cluster_msg_uid;
    }
    if (pogobot_helper_getid() == mydata->leader_id) {
        m.flags |= VMSGF_LEADER;
        // For a leader, par_epoch_seen should equal par_epoch_local already
        m.par_epoch = mydata->par_epoch_local;
    }

    /* Keep retained self share */
    mydata->ps_w      = self_keep * mydata->ps_w;
    mydata->ps_s_pol  = self_keep * mydata->ps_s_pol;
    mydata->ps_s_wall = self_keep * mydata->ps_s_wall;
    mydata->ps_s_pers = self_keep * mydata->ps_s_pers;
    mydata->ps_s_nb   = self_keep * mydata->ps_s_nb;

    mydata->last_beacon_ms = now;
    return pogobot_infrared_sendShortMessage_omni((uint8_t*)&m, MSG_SIZE);
}

static void process_message(message_t* mr){
    if (diff_drive_kin_process_message(&mydata->ddk, mr)) return;
    if (mr->header.payload_length < MSG_SIZE) return;

    vicsek_msg_t const* m=(vicsek_msg_t const*)mr->payload;
    if (m->sender_id == pogobot_helper_getid()) return;

    /* Neighbor table and headings */
    neighbor_t* n = upsert_neighbor(mydata->neighbors, &mydata->nb_neighbors, m->sender_id);
    if(!n) return;
    n->theta_mrad    = m->theta_mrad;
    n->last_seen_ms  = current_time_milliseconds();
    n->leader_id_hint= m->leader_id;

    /* Cluster relay */
    if (m->flags & VMSGF_CLUSTER_UTURN) {
        cluster_adopt_and_activate(m->cluster_target_mrad, m->cluster_wall_t0_ms, m->cluster_msg_uid);
    }

    /* Push-sum masses */
    mydata->ps_s_pol  += q15_unpack_double(m->ps_s_pol_q15,  0.0, (double)(MAX_NEIGHBORS+1));
    mydata->ps_s_wall += q15_unpack_double(m->ps_s_wall_q15, 0.0, (double)(MAX_NEIGHBORS+1));
    mydata->ps_s_pers += q15_unpack_double(m->ps_s_pers_q15, 0.0, (double)(MAX_NEIGHBORS+1));
    mydata->ps_s_nb   += q15_unpack_double(m->ps_s_nb_q15,   0.0, (double)(MAX_NEIGHBORS+1));
    mydata->ps_w      += q15_unpack_w(m->ps_w_q15);

    // ---- Parameter adoption by epoch ----
//    // Only consider params that correspond to *our* current cluster leader id.
//    if (m->leader_id == mydata->leader_id && m->par_epoch > mydata->par_epoch_seen){
//        mydata->cur_beta   = m->par_beta;
//        mydata->cur_sigma  = m->par_sigma;
//        mydata->cur_speed  = m->par_speed;
//        mydata->par_epoch_seen = m->par_epoch;
//    }

    bool from_leader = (m->flags & VMSGF_LEADER) != 0;
    bool newer_epoch = (m->par_epoch > mydata->par_epoch_seen);
    bool equal_epoch_from_leader = from_leader && (m->par_epoch == mydata->par_epoch_seen);

    // also allow if we disagree on leader_id but the packet is flagged as leader:
    bool leader_identity_ok = from_leader || (m->leader_id == mydata->leader_id);

    if (leader_identity_ok && (newer_epoch || equal_epoch_from_leader)) {
        mydata->cur_beta  = m->par_beta;
        mydata->cur_sigma = m->par_sigma;
        mydata->cur_speed = m->par_speed;
        mydata->par_epoch_seen = m->par_epoch;
    }
}

/* --- LED --- */
static void led_update(double heading){
    bool i_am_leader = (pogobot_helper_getid() == mydata->leader_id);
    if (main_led_display_enum == SHOW_LOSS_LEADER){
        if (!i_am_leader) { pogobot_led_setColor(0,0,0); return; }
        /* Map loss to hue: low loss = green (~120°), high = red (~0°) */
        double L = mydata->last_loss_for_led;
        if (L < 0.0) L = 0.0;
        /* crude scaling: assume 0..1 useful range */
        double t = L; if (t>1.0) t=1.0;
        float hue = (float)((1.0 - t) * 120.0); /* 0=red,120=green */
        uint8_t r8,g8,b8; hsv_to_rgb(hue, 1.0f, 1.0f, &r8,&g8,&b8);
        r8 = SCALE_0_255_TO_0_25(r8); g8=SCALE_0_255_TO_0_25(g8); b8=SCALE_0_255_TO_0_25(b8);
        if (r8==0&&g8==0&&b8==0) r8=1;
        pogobot_led_setColor(r8,g8,b8);
    } else if (main_led_display_enum == SHOW_ANGLE){
        if (heading < 0.0) heading += 2.0*M_PI;
        float hue_deg = (float)(heading * 180.0/M_PI);
        uint8_t r8,g8,b8; hsv_to_rgb(hue_deg, 1.0f, 1.0f, &r8,&g8,&b8);
        r8 = SCALE_0_255_TO_0_25(r8); g8=SCALE_0_255_TO_0_25(g8); b8=SCALE_0_255_TO_0_25(b8);
        if (r8==0&&g8==0&&b8==0) r8=1;
        pogobot_led_setColor(r8,g8,b8);
    } else if (main_led_display_enum == SHOW_POLARIZATION){
        float hue_deg = (float)(mydata->cs_pol_ewma * 360.0);
        uint8_t r8,g8,b8; hsv_to_rgb(hue_deg, 1.0f, 1.0f, &r8,&g8,&b8);
        r8 = SCALE_0_255_TO_0_25(r8); g8=SCALE_0_255_TO_0_25(g8); b8=SCALE_0_255_TO_0_25(b8);
        if (r8==0&&g8==0&&b8==0) r8=1;
        pogobot_led_setColor(r8,g8,b8);
    }
}

/* --- Init --- */
static void user_init(void){
    srand(pogobot_helper_getRandSeed());
    memset(mydata, 0, sizeof(*mydata));

    main_loop_hz = 60;
    mydata->last_update_ms = current_time_milliseconds();
    max_nb_processed_msg_per_tick = 100;
    msg_rx_fn = process_message;
    msg_tx_fn = send_message;
    error_codes_led_idx = 3;

    /* DDK */
    diff_drive_kin_init_default(&mydata->ddk);
    photostart_init(&mydata->ps);
    photostart_set_ewma_alpha(&mydata->ps, 0.30);
    diff_drive_kin_set_photostart(&mydata->ddk, &mydata->ps);
    heading_detection_set_geometry(&mydata->ddk.hd, alpha_deg, robot_radius);
    heading_detection_set_chirality(&mydata->ddk.hd, heading_chiralty_enum);
    mydata->prev_behavior = diff_drive_kin_get_behavior(&mydata->ddk);

    /* Cluster defaults */
    mydata->cluster_turn_active     = false;
    mydata->cluster_target_rad      = 0.0;
    mydata->cluster_wall_t0_ms      = 0u;
    mydata->cluster_active_until_ms = 0u;
    mydata->have_seen_cluster_uid   = false;

    /* Stats init */
    mydata->accum_total_ms = 1e-9;
    mydata->accum_wc_ms    = 0.0;

    /* Consensus init */
    mydata->ps_w = 1.0;
    mydata->cs_pol_ewma  = 0.0;
    mydata->cs_wall_ewma = 0.0;
    mydata->cs_pers_ewma = 0.0;
    mydata->cs_nb_ewma   = 0.0;

    /* Leader election */
    mydata->leader_id = pogobot_helper_getid();
    mydata->leader_last_update_ms = current_time_milliseconds();
    mydata->par_epoch_local = 1;
    mydata->par_epoch_seen  = mydata->par_epoch_local;

    /* Start with random parameters (within bounds) */
    mydata->cur_beta  = rand_uniform(es_lo_beta,  es_hi_beta);
    mydata->cur_sigma = rand_uniform(es_lo_sigma, es_hi_sigma);
    mydata->cur_speed = rand_uniform(es_lo_speed, es_hi_speed);

    /* ES init (leader will use it; followers keep cur_* from floods) */
    mydata->es_lo[0]=es_lo_beta;  mydata->es_hi[0]=es_hi_beta;
    mydata->es_lo[1]=es_lo_sigma; mydata->es_hi[1]=es_hi_sigma;
    mydata->es_lo[2]=es_lo_speed; mydata->es_hi[2]=es_hi_speed;

    mydata->es_x[0] = mydata->cur_beta;
    mydata->es_x[1] = mydata->cur_sigma;
    mydata->es_x[2] = mydata->cur_speed;

    es1p1_params_t p = {
        .mode = ES1P1_MINIMIZE,
        .sigma0 = es_sigma0,
        .sigma_min = es_sigma_min,
        .sigma_max = es_sigma_max,
        .s_target = es_s_target,
        .s_alpha = es_s_alpha,
        .c_sigma = es_c_sigma
    };
    es1p1_init(&mydata->es, 3,
               mydata->es_x, mydata->es_x_try,
               mydata->es_lo, mydata->es_hi, &p);
    mydata->es_window_t0_ms = current_time_milliseconds();
    mydata->es_accum_loss = 0.0;
    mydata->es_accum_w    = 0.0;
    mydata->last_loss_for_led   = 1.0f;
    mydata->es_has_candidate = false;
    es1p1_tell_initial(&mydata->es, 1000.0f);

#ifdef SIMULATOR
    printf("Vicsek + Stats + Push-sum + Leader(1+1-ES online)\n"); /* base features from vicsek-base.c kept. */ /* :contentReference[oaicite:3]{index=3} */
#endif
}

/* --- Step --- */
static void user_step(void){
    /* Photostart gate */
    if (!photostart_step(&mydata->ps)) {
        pogobot_led_setColors(20,0,20,0);
        pogobot_motor_set(motorL, motorStop);
        pogobot_motor_set(motorR, motorStop);
        return;
    }

    uint32_t now  = current_time_milliseconds();
    double dt_s   = (now - mydata->last_update_ms) * 1e-3;
    if (dt_s < 0.0) dt_s = 0.0;
    if (dt_s > cont_max_dt_s) dt_s = cont_max_dt_s;
    mydata->last_update_ms = now;

    /* Current heading */
    double heading = heading_detection_estimate(&mydata->ddk.hd);

    /* Cluster origin (rising edge of avoidance) */
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

    /* Leader election */
    purge_old_neighbors(mydata->neighbors, &mydata->nb_neighbors);
    leader_election_tick();
    bool i_am_leader = (pogobot_helper_getid() == mydata->leader_id);

    /* Vicsek mean */
    double sx=cos(heading), sy=sin(heading);
    for (uint8_t i=0; i<mydata->nb_neighbors; ++i){
        double th = mrad_to_rad(mydata->neighbors[i].theta_mrad);
        sx += cos(th); sy += sin(th);
    }
    double theta_mean = (sx==0.0 && sy==0.0) ? heading : atan2(sy, sx);
    double theta_cmd = cluster_window_active(now) ? mydata->cluster_target_rad : theta_mean;
    mydata->theta_cmd_rad = theta_cmd;

    /* dtheta from Vicsek */
    double err = wrap_pi(theta_cmd - heading);
    double dtheta = (mydata->cur_beta * 10.0) * sin(err) * dt_s;

    /* Angular diffusion */
    if (mydata->cur_sigma > 0.0){
        double u1 = (rand()+1.0)/(RAND_MAX+2.0);
        double u2 = (rand()+1.0)/(RAND_MAX+2.0);
        double z  = sqrt(-2.0*log(u1)) * cos(2.0*M_PI*u2);
        dtheta += (mydata->cur_sigma * 10.0) * sqrt(dt_s) * z;
    }

    /* Local stats */
    uint32_t N_eff = 0;
    mydata->ls_pol_norm = compute_local_polarization_norm(heading, &N_eff);

    mydata->accum_total_ms += dt_s * 1000.0;
    if (beh == DDK_BEHAVIOR_AVOIDANCE || cluster_window_active(now)) {
        mydata->accum_wc_ms += dt_s * 1000.0;
    }
    mydata->ls_wall_ratio = clamp01(mydata->accum_wc_ms / mydata->accum_total_ms);

    double sum_norm_age = 0.0;
    for (uint8_t i=0; i<mydata->nb_neighbors; ++i){
        double age_ms = (double)(now - mydata->neighbors[i].first_seen_ms);
        double norm = (neighbor_persist_norm_ms>0) ? fmin(age_ms / (double)neighbor_persist_norm_ms, 1.0) : 0.0;
        sum_norm_age += norm;
    }
    mydata->ls_neighbor_persist = (mydata->nb_neighbors>0) ? (sum_norm_age / (double)mydata->nb_neighbors) : 0.0;
    mydata->ls_neighbor_count = (double)mydata->nb_neighbors;

    /* Push-sum (shared w) */
    double w0 = (double)N_eff;
    if (N_eff < 2) w0 = 0.0; /* isolated => don't bias consensus */

    mydata->ps_w      = w0;
    mydata->ps_s_pol  = mydata->ls_pol_norm         * w0;
    mydata->ps_s_wall = mydata->ls_wall_ratio       * w0;
    mydata->ps_s_pers = mydata->ls_neighbor_persist * w0;
    mydata->ps_s_nb   = mydata->ls_neighbor_count   * w0;

    double denom = (mydata->ps_w > 1e-12) ? mydata->ps_w : 1.0;
    double c_pol  = clamp01(mydata->ps_s_pol  / denom);
    double c_wall = clamp01(mydata->ps_s_wall / denom);
    double c_pers = clamp01(mydata->ps_s_pers / denom);
    double c_nb   = fmax(mydata->ps_s_nb / denom, 0.0);

    mydata->cs_pol_ewma  = (1.0 - ewma_alpha_polarization) * mydata->cs_pol_ewma  + ewma_alpha_polarization * c_pol;
    mydata->cs_wall_ewma = (1.0 - ewma_alpha_wallratio)    * mydata->cs_wall_ewma + ewma_alpha_wallratio    * c_wall;
    mydata->cs_pers_ewma = (1.0 - ewma_alpha_persistence)  * mydata->cs_pers_ewma + ewma_alpha_persistence  * c_pers;
    mydata->cs_nb_ewma   = (1.0 - ewma_alpha_nb)           * mydata->cs_nb_ewma   + ewma_alpha_nb           * c_nb;

    /* Drive DDK with current triple */
    diff_drive_kin_step(&mydata->ddk, mydata->cur_speed, dtheta, heading);

    /* --- Leader: accumulate loss over window, then one ES step --- */
    if (i_am_leader){
        /* ignore first 'quiet' ms of the window to reduce transients */
        uint32_t t0 = mydata->es_window_t0_ms;
        if (now > t0 + es_eval_quiet_ms){
            double loss = compute_loss_from_consensus(mydata->cs_pol_ewma, mydata->cs_wall_ewma,
                                                      mydata->cs_pers_ewma, mydata->cs_nb_ewma);
            mydata->es_accum_loss += loss * (double)(now - mydata->last_update_ms); /* time-weighted */
            mydata->es_accum_w    += (double)(now - mydata->last_update_ms);
            mydata->last_loss_for_led = loss;
        }

        if (now - t0 >= es_eval_window_ms){
            /* finalize window */
            float window_loss = (mydata->es_accum_w > 1e-6)
                ? (float)(mydata->es_accum_loss / mydata->es_accum_w)
                : (float)mydata->last_loss_for_led;

            /* TELL the ES the result if we had a candidate */
            if (mydata->es_has_candidate) {
                (void)es1p1_tell(&mydata->es, window_loss);
            }

            /* ASK for new parameters for next window */
            const float *x_try = es1p1_ask(&mydata->es);
            mydata->es_has_candidate = (x_try != NULL);  // Set based on success

            if (x_try) {
                mydata->cur_beta  = x_try[0];
                mydata->cur_sigma = x_try[1];
                mydata->cur_speed = x_try[2];

                mydata->par_epoch_local += 1;
                mydata->par_epoch_seen   = mydata->par_epoch_local;
            }

            /* reset window */
            mydata->es_window_t0_ms = now;
            mydata->es_accum_loss   = 0.0;
            mydata->es_accum_w      = 0.0;
        }

    } else {
        /* followers: LED shows black unless you switch display mode */
        mydata->last_loss_for_led = compute_loss_from_consensus(mydata->cs_pol_ewma, mydata->cs_wall_ewma,
                                                                mydata->cs_pers_ewma, mydata->cs_nb_ewma);
    }

    /* LED */
    led_update(heading);

#ifdef SIMULATOR
    if (pogobot_ticks % 1200 == 0) {
        printf("[ID %u] lead=%u  pol=%.2f wall=%.2f pers=%.2f nb=%.2f  | beta=%.2f sigma=%.2f v=%.2f  | L~%.3f \n",
               pogobot_helper_getid(), mydata->leader_id,
               mydata->cs_pol_ewma, mydata->cs_wall_ewma, mydata->cs_pers_ewma, mydata->cs_nb_ewma,
               mydata->cur_beta, mydata->cur_sigma, mydata->cur_speed,
               mydata->last_loss_for_led);
    }
#endif
}

/* --- Simulator hooks & main --- */
#ifdef SIMULATOR
static void create_data_schema(void){
    data_add_column_int16("leader_id");
    data_add_column_double("loss_led");
    data_add_column_double("beta");
    data_add_column_double("sigma");
    data_add_column_double("speed");
    data_add_column_double("pol_consensus_ewma");
    data_add_column_double("wall_consensus_ewma");
    data_add_column_double("pers_consensus_ewma");
    data_add_column_double("nb_consensus_ewma");
}
static void export_data(void){
    data_set_value_int16("leader_id", (int16_t)mydata->leader_id);
    data_set_value_double("loss_led", mydata->last_loss_for_led);
    data_set_value_double("beta",  (double)mydata->cur_beta);
    data_set_value_double("sigma", (double)mydata->cur_sigma);
    data_set_value_double("speed", (double)mydata->cur_speed);
    data_set_value_double("pol_consensus_ewma",  mydata->cs_pol_ewma);
    data_set_value_double("wall_consensus_ewma", mydata->cs_wall_ewma);
    data_set_value_double("pers_consensus_ewma", mydata->cs_pers_ewma);
    data_set_value_double("nb_consensus_ewma",   mydata->cs_nb_ewma);
}
static void global_setup(void){
    /* Import shared tunables (same names as base file where possible) */
    init_from_configuration(max_age);
    init_from_configuration(vicsek_beta_rad_per_s);
    init_from_configuration(cont_noise_sigma_rad);
    init_from_configuration(cont_max_dt_s);
    init_from_configuration(base_speed_ratio);

    init_from_configuration(alpha_deg);
    init_from_configuration(robot_radius);
    char heading_chiralty[32] = "cw";
    init_array_from_configuration(heading_chiralty);
    heading_chiralty_enum = (strcasecmp(heading_chiralty,"ccw")==0)?HEADING_CCW:HEADING_CW;

    init_from_configuration(cluster_u_turn_duration_ms);
    init_from_configuration(phi_rad_min);
    init_from_configuration(phi_rad_max);
    if (phi_rad_min > phi_rad_max){ float t=phi_rad_min; phi_rad_min=phi_rad_max; phi_rad_max=t; }

    init_from_configuration(ewma_alpha_polarization);
    init_from_configuration(ewma_alpha_wallratio);
    init_from_configuration(ewma_alpha_persistence);
    init_from_configuration(ewma_alpha_nb);

    init_from_configuration(neighbor_persist_norm_ms);
    init_from_configuration(neighbor_norm_max);

    /* Targets + weights */
    init_from_configuration(target_pol);
    init_from_configuration(target_wall);
    init_from_configuration(target_pers);
    init_from_configuration(target_nb);
    init_from_configuration(w_pol);
    init_from_configuration(w_wall);
    init_from_configuration(w_pers);
    init_from_configuration(w_nb);

    /* ES scheduling + bounds */
    init_from_configuration(es_eval_window_ms);
    init_from_configuration(es_eval_quiet_ms);

    init_from_configuration(es_lo_beta);
    init_from_configuration(es_hi_beta);
    init_from_configuration(es_lo_sigma);
    init_from_configuration(es_hi_sigma);
    init_from_configuration(es_lo_speed);
    init_from_configuration(es_hi_speed);

    init_from_configuration(es_sigma0);
    init_from_configuration(es_sigma_min);
    init_from_configuration(es_sigma_max);
    init_from_configuration(es_s_target);
    init_from_configuration(es_s_alpha);
    init_from_configuration(es_c_sigma);
    init_from_configuration(es_evals_per_tick);

    char main_led_display[32] = "loss";
    init_array_from_configuration(main_led_display);
    if (strcasecmp(main_led_display,"loss")==0 || strcasecmp(main_led_display,"leader")==0) {
        main_led_display_enum = SHOW_LOSS_LEADER;
    } else if (strcasecmp(main_led_display,"angle")==0){
        main_led_display_enum = SHOW_ANGLE;
    } else if (strcasecmp(main_led_display,"polarization")==0){
        main_led_display_enum = SHOW_POLARIZATION;
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
    /* Walls app stays enabled—DDK listens and handles avoidance internally. */
    pogobot_start(default_walls_user_init, default_walls_user_step, "walls");
    return 0;
}

