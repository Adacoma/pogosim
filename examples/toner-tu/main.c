/**
 * @file toner_tu_example.c
 * @brief Pogosim example: Toner–Tu-like controller (pressure + novelty advection)
 *        + phi_rad-based "cluster U-turn" for synchronized wall avoidance (toggleable)
 *
 * Implements:
 *  - Continuous Vicsek heading core
 *  - TT-like pressure on speed (density-dependent damping)
 *  - Novelty-driven advection (steering toward accumulated novelty direction)
 *  - Cluster U-turn: when a robot hits a wall, it samples phi ~ U[min,max],
 *    sets target heading = current + phi, and floods this to neighbors for a
 *    duration window so the whole group turns in sync.
 *
 * The Cluster U-turn mechanism mirrors your Vicsek example (message flags,
 * UID freshness, rising-edge on wall entry, uniform phi sampling). See:
 *  - extended message format with VMSGF_CLUSTER_UTURN,
 *  - cluster_adopt_and_activate(), cluster_window_active(),
 *  - rising-edge detection around wall_avoidance_step(...).  
 */

#include "pogobase.h"
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <strings.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "pogo-utils/wall_avoidance.h"
#include "pogo-utils/heading_detection.h"
#include "pogo-utils/version.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ===== Messaging / Neighbors =====
#define MAX_NEIGHBORS      20u
#define BEACON_HZ          10u
#define BEACON_PERIOD_MS  (1000u / BEACON_HZ)

// ===== Motors =====
static int forward_speed_base = motorHalf;   // scaled by v_cmd in [v_min..v_max]

// ===== Neighbor/Vicsek base =====
uint32_t max_age             = 600;
uint32_t vicsek_period_ms    = 17;

double   noise_eta_rad       = 1.5;
double   align_gain          = 1.0;
bool     include_self_in_avg = true;

// Continuous heading core
bool   vicsek_time_continuous = true;
double vicsek_beta_rad_per_s  = 3.0;
double cont_noise_sigma_rad   = 0.0;
double cont_max_dt_s          = 0.05;

// ===== TT pressure (speed vs density) =====
double tt_alpha = 1.5;
double tt_beta  = 1.5;
double press_kappa = 0.8;
double rho_smooth_tau_s = 0.5;
double v_min = 0.25;
double v_max = 1.00;

// ===== Novelty advection =====
uint32_t novelty_window_ms = 1500;
double novelty_tau_s       = 1.0;
double novelty_dir_tau_s   = 2.0;
double novelty_speed_gain  = 0.25;
double lambda_adv_rad_per_s= 1.2;

// ===== Heading & wall avoidance =====
double alpha_deg    = 40.0;
double robot_radius = 0.0265;
heading_chirality_t heading_chiralty_enum = HEADING_CW;

uint32_t wall_avoidance_memory_ms         = 300;
uint32_t wall_avoidance_turn_duration_ms  = 300;
uint32_t wall_avoidance_forward_commit_ms = 300;
float    wall_avoidance_forward_speed_ratio = 0.5f;
wall_chirality_t wall_avoidance_chiralty_policy = WALL_MIN_TURN;

// ===== Cluster U-turn (phi_rad mechanism) =====
// Toggle
bool     enable_cluster_uturn = true;       // <— turn ON/OFF synchronized wall-avoid
bool     broadcast_angle_when_avoiding_walls = true; // mute beacons near walls unless advertising cluster

// Duration
uint32_t cluster_u_turn_duration_ms = 1500;

// Sampling range for phi (uniform)
float phi_rad_min = 0.2f;
float phi_rad_max = 0.2f;

// Sender–receiver alignment threshold (radians)
// Only adopt/relay if |theta_sender - theta_self| <= this value.
double   cluster_uturn_align_thresh_rad = 0.4;  // ≈ 23°

// LED display
typedef enum { SHOW_STATE, SHOW_ANGLE, SHOW_SPEED } main_led_display_type_t;
main_led_display_type_t main_led_display_enum = SHOW_STATE;

// === Cluster U-turn wire format (copied from Vicsek example) ===
enum : uint8_t {
    VMSGF_CLUSTER_UTURN = 0x01
};

typedef struct __attribute__((__packed__)) {
    uint16_t sender_id;
    int16_t  theta_mrad;            // commanded heading
    uint8_t  flags;                 // bit0: cluster U-turn active
    int16_t  cluster_target_mrad;   // valid if flags & VMSGF_CLUSTER_UTURN
    uint32_t cluster_wall_t0_ms;    // originator t0
    uint16_t cluster_msg_uid;       // de-dup/freshness
} tt_msg_t;

#define MSG_SIZE ((uint16_t)sizeof(tt_msg_t))

// === Neighbor entry ===
typedef struct {
    uint16_t id;
    uint32_t last_seen_ms;
    int16_t  theta_mrad;
} neighbor_t;

// === Internal state ===
typedef struct {
    // timing
    float   dt_s;
    uint32_t last_beacon_ms;
    uint32_t last_update_ms;

    // neighbors
    neighbor_t neighbors[MAX_NEIGHBORS];
    uint8_t nb_neighbors;

    // heading & speed command
    double theta_cmd_rad;
    double v_cmd;

    // sensors
    heading_detection_t heading_detection;
    double photo_heading_rad;

    // wall avoidance
    wall_avoidance_state_t wall_avoidance;
    bool doing_wall_avoidance;
    bool prev_doing_wall_avoidance; // for rising-edge detection

    // density / novelty
    double rho_ema;
    double rho_norm;

    double novelty_ema;
    double nov_vx, nov_vy;

    // cluster U-turn state
    bool     cluster_turn_active;
    double   cluster_target_rad;
    uint32_t cluster_wall_t0_ms;
    uint32_t cluster_active_until_ms;
    uint16_t cluster_msg_uid;
    uint16_t last_seen_cluster_uid;
    bool     have_seen_cluster_uid;
} USERDATA;

DECLARE_USERDATA(USERDATA);
REGISTER_USERDATA(USERDATA);

// ===== Utilities =====
static inline double wrap_pi(double a){ while(a> M_PI)a-=2.0*M_PI; while(a<-M_PI)a+=2.0*M_PI; return a; }
static inline int16_t rad_to_mrad(double a){ a=wrap_pi(a); long v=lround(a*1000.0); if(v>32767)v=32767; if(v<-32768)v=-32768; return (int16_t)v; }
static inline double  mrad_to_rad(int16_t m){ return ((double)m)/1000.0; }
static inline double  clamp(double x, double lo, double hi){ return (x<lo)?lo:((x>hi)?hi:x); }
static inline double  noise_uniform(double eta){ double u=(double)rand()/(double)RAND_MAX; return (u-0.5)*eta; }
static inline double  rand_uniform(double a, double b){ double u=(double)rand()/(double)RAND_MAX; return a+(b-a)*u; }

static inline void motor_set_signed(motor_id id, int spd_signed, uint8_t fwd_dir_mem){
    int mag = spd_signed >= 0 ? spd_signed : -spd_signed;
    if(mag > motorFull) mag = motorFull;
    uint8_t dir = (spd_signed >= 0) ? fwd_dir_mem : ((fwd_dir_mem==0)?1:0);
    pogobot_motor_dir_set(id, dir);
    pogobot_motor_set(id, mag);
}

// ===== Neighbor mgmt =====
static void purge_old_neighbors(void){
    uint32_t now = current_time_milliseconds();
    for(int i=(int)mydata->nb_neighbors-1;i>=0;--i){
        if(now - mydata->neighbors[i].last_seen_ms > max_age){
            mydata->neighbors[i] = mydata->neighbors[mydata->nb_neighbors-1];
            mydata->nb_neighbors--;
        }
    }
}
static neighbor_t* upsert_neighbor(uint16_t id){
    for(uint8_t i=0;i<mydata->nb_neighbors;++i)
        if(mydata->neighbors[i].id==id) return &mydata->neighbors[i];
    if(mydata->nb_neighbors>=MAX_NEIGHBORS) return NULL;
    neighbor_t* n=&mydata->neighbors[mydata->nb_neighbors++];
    n->id=id; n->theta_mrad=0; n->last_seen_ms=0; return n;
}

// ===== Cluster U-turn helpers (same behavior as in Vicsek example) =====
static inline bool cluster_window_active(uint32_t now){
    return mydata->cluster_turn_active && (now < mydata->cluster_active_until_ms);
}

static void cluster_adopt_and_activate(int16_t tgt_mrad, uint32_t t0_ms, uint16_t uid){
    bool newer = (!mydata->cluster_turn_active) ||
                 (t0_ms > mydata->cluster_wall_t0_ms) ||
                 (!mydata->have_seen_cluster_uid) ||
                 (uid != mydata->last_seen_cluster_uid);
    if (newer){
        mydata->cluster_target_rad      = mrad_to_rad(tgt_mrad);
        mydata->cluster_wall_t0_ms      = t0_ms;
        mydata->cluster_active_until_ms = t0_ms + cluster_u_turn_duration_ms;
        mydata->cluster_turn_active     = true;
        mydata->last_seen_cluster_uid   = uid;
        mydata->have_seen_cluster_uid   = true;
    }
}

// ===== Messaging =====
bool send_message(void){
    uint32_t now=current_time_milliseconds();
    if (now - mydata->last_beacon_ms < BEACON_PERIOD_MS) return false;

    bool advertise_cluster = enable_cluster_uturn && cluster_window_active(now);

    // respect "mute near walls" unless we must advertise the cluster
    if (!advertise_cluster && !broadcast_angle_when_avoiding_walls && mydata->doing_wall_avoidance) {
        return false;
    }

    tt_msg_t m = {
        .sender_id = pogobot_helper_getid(),
        .theta_mrad= rad_to_mrad(mydata->theta_cmd_rad),
        .flags     = 0u,
        .cluster_target_mrad = 0,
        .cluster_wall_t0_ms  = 0u,
        .cluster_msg_uid     = 0u
    };

    if (advertise_cluster){
        m.flags               |= VMSGF_CLUSTER_UTURN;
        m.cluster_target_mrad  = rad_to_mrad(mydata->cluster_target_rad);
        m.cluster_wall_t0_ms   = mydata->cluster_wall_t0_ms;
        m.cluster_msg_uid      = mydata->cluster_msg_uid;
    }

    mydata->last_beacon_ms = now;
    return pogobot_infrared_sendShortMessage_omni((uint8_t*)&m, MSG_SIZE);
}

void process_message(message_t* mr){
    if (wall_avoidance_process_message(&mydata->wall_avoidance, mr)) {
        return;
    }
    if(mr->header.payload_length < MSG_SIZE) return;
    tt_msg_t const* m=(tt_msg_t const*)mr->payload;
    if(m->sender_id==pogobot_helper_getid()) return;

    neighbor_t* n=upsert_neighbor(m->sender_id);
    if(!n) return;
    n->theta_mrad = m->theta_mrad;
    n->last_seen_ms = current_time_milliseconds();

    if (enable_cluster_uturn && (m->flags & VMSGF_CLUSTER_UTURN)) {
        // Accept only if aligned with the SENDER (not with the target)
        double theta_self   = mydata->photo_heading_rad;          // current measured heading
        double theta_sender = mrad_to_rad(m->theta_mrad);         // sender's current heading
        double d = fabs(wrap_pi(theta_sender - theta_self));

        if (d <= cluster_uturn_align_thresh_rad) {
            // adopt and (later) relay via our own beacon
            cluster_adopt_and_activate(m->cluster_target_mrad,
                                      m->cluster_wall_t0_ms,
                                      m->cluster_msg_uid);
        } else {
            // too different → ignore and DO NOT transfer
            // (no state change: we won't advertise in send_message())
        }
    }

}

// ===== TT updates =====
static void density_update(double dt_s){
    double gamma = (rho_smooth_tau_s > 0.0) ? (dt_s / fmax(1e-6, rho_smooth_tau_s)) : 1.0;
    if (gamma > 1.0) gamma = 1.0;
    double rho_inst = (double)mydata->nb_neighbors;
    mydata->rho_ema  = (1.0 - gamma) * mydata->rho_ema + gamma * rho_inst;
    mydata->rho_norm = clamp(mydata->rho_ema / (double)MAX_NEIGHBORS, 0.0, 1.0);
}

static void novelty_update(double dt_s, int novelty_events_this_step, double heading_now){
    double g = (novelty_tau_s > 0.0) ? (dt_s / fmax(1e-6, novelty_tau_s)) : 1.0;
    if (g > 1.0) g = 1.0;
    mydata->novelty_ema = (1.0 - g) * mydata->novelty_ema + g * (double)novelty_events_this_step;

    double gd = (novelty_dir_tau_s > 0.0) ? (dt_s / fmax(1e-6, novelty_dir_tau_s)) : 1.0;
    if (gd > 1.0) gd = 1.0;
    double imp = gd * (double)novelty_events_this_step;
    mydata->nov_vx = (1.0 - gd) * mydata->nov_vx + imp * cos(heading_now);
    mydata->nov_vy = (1.0 - gd) * mydata->nov_vy + imp * sin(heading_now);
}

static void build_vicsek_mean(double* out_theta_mean, double heading_now){
    double sx = 0.0, sy = 0.0;
    if (include_self_in_avg){ sx += cos(heading_now); sy += sin(heading_now); }
    for (uint8_t i=0; i<mydata->nb_neighbors; ++i){
        double th = mrad_to_rad(mydata->neighbors[i].theta_mrad);
        sx += cos(th); sy += sin(th);
    }
    *out_theta_mean = (sx==0.0 && sy==0.0) ? heading_now : atan2(sy, sx);
}

static void tt_heading_and_speed_update(uint32_t now_ms, uint32_t* last_update_ms_io){
    double dt_s = (now_ms - *last_update_ms_io) * 1e-3;
    if (dt_s < 0.0) dt_s = 0.0;
    if (dt_s > cont_max_dt_s) dt_s = cont_max_dt_s;
    *last_update_ms_io = now_ms;

    // rough novelty spike detector (cheap)
    int novelty_events = 0;
    double approx_prev_rho = mydata->rho_ema;
    double approx_now_rho  = (double)mydata->nb_neighbors;
    if (approx_now_rho > approx_prev_rho) {
        novelty_events = (int)lround(approx_now_rho - approx_prev_rho);
        if (novelty_events < 0) novelty_events = 0;
    }

    mydata->photo_heading_rad = heading_detection_estimate(&mydata->heading_detection);
    double heading = mydata->photo_heading_rad;

    density_update(dt_s);
    novelty_update(dt_s, novelty_events, heading);

    // novelty direction
    double psi_nov = atan2(mydata->nov_vy, mydata->nov_vx);
    bool   have_psi = (mydata->nov_vx*mydata->nov_vx + mydata->nov_vy*mydata->nov_vy) > 1e-8;

    // Vicsek mean
    double theta_mean = heading;
    build_vicsek_mean(&theta_mean, heading);

    // If cluster U-turn window is active, override heading setpoint
    double theta_cmd;
    if (enable_cluster_uturn && cluster_window_active(now_ms)) {
        theta_cmd = mydata->cluster_target_rad;  // synchronized turn target
        mydata->theta_cmd_rad = theta_cmd;
    } else {
        // continuous Vicsek + advection + noise
        double det = vicsek_beta_rad_per_s * sin(wrap_pi(theta_mean - heading));
        if (have_psi) {
            det += lambda_adv_rad_per_s * sin(wrap_pi(psi_nov - heading));
        }

        double stoch = 0.0;
        if (cont_noise_sigma_rad > 0.0){
            double u1 = (rand() + 1.0) / (RAND_MAX + 2.0);
            double u2 = (rand() + 1.0) / (RAND_MAX + 2.0);
            double z  = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2); // N(0,1)
            stoch = cont_noise_sigma_rad * sqrt(dt_s) * z;           // Euler–Maruyama increment
        }
        mydata->theta_cmd_rad = wrap_pi(heading + det * dt_s + stoch);
    }

    // Speed ODE
    double v = mydata->v_cmd;
    double dv = tt_alpha * v - tt_beta * v * v * v;
    dv += -press_kappa * mydata->rho_norm;
    dv +=  novelty_speed_gain * mydata->novelty_ema;
    v += dv * dt_s;
    mydata->v_cmd = clamp(v, v_min, v_max);
}

// ==== Control application ====
static inline int build_diff_from_heading(double theta_cmd, double heading_now, double turn_gain){
    double err = wrap_pi(theta_cmd - heading_now);
    const double err_norm = err / (30.0 * M_PI / 180.0);
    int diff = (int)lround(turn_gain * err_norm * (double)forward_speed_base);
    if (diff >  forward_speed_base) diff =  forward_speed_base;
    if (diff < -forward_speed_base) diff = -forward_speed_base;
    return diff;
}

#define SCALE_0_255_TO_0_25(x)   (uint8_t)((x) * (25.0f / 255.0f) + 0.5f)

static void update_main_led(void){
    if (main_led_display_enum == SHOW_STATE) {
        if (mydata->doing_wall_avoidance) {
            pogobot_led_setColor(255,0,0); // red
        } else if (mydata->nb_neighbors == 0) {
            pogobot_led_setColor(0,0,255); // blue
        } else if (enable_cluster_uturn && cluster_window_active(current_time_milliseconds())) {
            pogobot_led_setColor(0,255,255); // cyan: cluster U-turn active
        } else {
            pogobot_led_setColor(0,255,0); // green
        }
    } else if (main_led_display_enum == SHOW_ANGLE) {
        float angle = mydata->photo_heading_rad;
        if (angle < 0.0f) angle += 2.f * (float)M_PI;
        float hue_deg = angle * 180.0f / (float)M_PI;
        uint8_t r8,g8,b8; hsv_to_rgb(hue_deg, 1.0f, 1.0f, &r8,&g8,&b8);
        r8 = SCALE_0_255_TO_0_25(r8);
        g8 = SCALE_0_255_TO_0_25(g8);
        b8 = SCALE_0_255_TO_0_25(b8);
        if (r8==0 && g8==0 && b8==0) r8=1;
        pogobot_led_setColor(r8,g8,b8);
    } else { // SHOW_SPEED
        uint8_t v = (uint8_t)lround(255.0 * clamp((mydata->v_cmd - v_min)/(v_max - v_min + 1e-9), 0.0, 1.0));
        pogobot_led_setColor(SCALE_0_255_TO_0_25(v),
                             SCALE_0_255_TO_0_25(v),
                             SCALE_0_255_TO_0_25(255));
    }
}

// ==== Lifecycle ====
void user_init(void){
    srand(pogobot_helper_getRandSeed());
    memset(mydata,0,sizeof(*mydata));

    main_loop_hz = 30;
    mydata->dt_s = (main_loop_hz>0)? (1.f/(float)main_loop_hz) : (1.f/60.f);
    max_nb_processed_msg_per_tick=3;
    percent_msgs_sent_per_ticks=50;
    msg_rx_fn=process_message;
    msg_tx_fn=send_message;
    error_codes_led_idx=3;

    // Motors fwd dirs
    uint8_t dir_mem[3]={0,0,0};
    pogobot_motor_dir_mem_get(dir_mem);
    uint8_t motor_dir_right_fwd = dir_mem[0];
    uint8_t motor_dir_left_fwd  = dir_mem[1];

    // Wall avoidance
    motor_calibration_t motors = {
        .motor_left = motorFull, .dir_left = motor_dir_left_fwd,
        .motor_right = motorFull, .dir_right = motor_dir_right_fwd
    };
    wall_avoidance_config_t wa_cfg = {
        .wall_memory_ms = wall_avoidance_memory_ms,
        .turn_duration_ms = wall_avoidance_turn_duration_ms,
        .forward_commit_ms = wall_avoidance_forward_commit_ms,
        .forward_speed_ratio = wall_avoidance_forward_speed_ratio
    };
    wall_avoidance_init(&mydata->wall_avoidance, &wa_cfg, &motors);
    wall_avoidance_set_policy(&mydata->wall_avoidance, wall_avoidance_chiralty_policy, 0);

    // Heading
    heading_detection_init(&mydata->heading_detection);
    heading_detection_set_geometry(&mydata->heading_detection, alpha_deg, robot_radius);
    heading_detection_set_chirality(&mydata->heading_detection, heading_chiralty_enum);

    // TT/Novelty initial
    mydata->photo_heading_rad = heading_detection_estimate(&mydata->heading_detection);
    mydata->theta_cmd_rad     = mydata->photo_heading_rad;
    mydata->v_cmd             = 0.75;
    mydata->rho_ema           = 0.0;
    mydata->rho_norm          = 0.0;
    mydata->novelty_ema       = 0.0;
    mydata->nov_vx = mydata->nov_vy = 0.0;

    // Cluster defaults
    mydata->cluster_turn_active     = false;
    mydata->cluster_target_rad      = 0.0;
    mydata->cluster_wall_t0_ms      = 0u;
    mydata->cluster_active_until_ms = 0u;
    mydata->cluster_msg_uid         = 0u;
    mydata->have_seen_cluster_uid   = false;

    mydata->last_update_ms=current_time_milliseconds();
    mydata->last_beacon_ms=0;

    pogobot_led_setColor(0,0,255);
}

void user_step(void){
    uint32_t now = current_time_milliseconds();

    purge_old_neighbors();

    // wall avoidance first
    bool wa = wall_avoidance_step(&mydata->wall_avoidance, true);
    mydata->prev_doing_wall_avoidance = mydata->doing_wall_avoidance;
    mydata->doing_wall_avoidance = wa;

    // === Cluster U-turn: rising edge on wall entry -> originate instruction
    if (enable_cluster_uturn && !mydata->prev_doing_wall_avoidance && mydata->doing_wall_avoidance) {
        double phi_sample = rand_uniform(phi_rad_min, phi_rad_max);
        double target = wrap_pi(mydata->photo_heading_rad + phi_sample);
        mydata->cluster_target_rad      = target;
        mydata->cluster_wall_t0_ms      = now;
        mydata->cluster_active_until_ms = now + cluster_u_turn_duration_ms;
        mydata->cluster_turn_active     = true;
        mydata->cluster_msg_uid         = (uint16_t)(rand() & 0xFFFF);
        mydata->last_seen_cluster_uid   = mydata->cluster_msg_uid;
        mydata->have_seen_cluster_uid   = true;
    }

    // TT update
    tt_heading_and_speed_update(now, &mydata->last_update_ms);

    // apply motors unless wall module owns them
    if (!mydata->doing_wall_avoidance) {
        int diff_cmd = build_diff_from_heading(mydata->theta_cmd_rad, mydata->photo_heading_rad, /*turn_gain*/0.8);
        int base = (int)lround((double)forward_speed_base * mydata->v_cmd);
        uint8_t dir_mem[3]={0,0,0};
        pogobot_motor_dir_mem_get(dir_mem);
        uint8_t motor_dir_right_fwd = dir_mem[0];
        uint8_t motor_dir_left_fwd  = dir_mem[1];
        motor_set_signed(motorL, base - diff_cmd, motor_dir_left_fwd);
        motor_set_signed(motorR, base + diff_cmd, motor_dir_right_fwd);
    }

    update_main_led();
}

// ===== SIM-only telemetry & config =====
#ifdef SIMULATOR
static void create_data_schema(void){
    data_add_column_int8("nb_neighbors");
    data_add_column_double("theta_photo_rad");
    data_add_column_double("theta_cmd_rad");
    data_add_column_double("v_cmd");
    data_add_column_double("rho_norm");
    data_add_column_double("novelty");
    data_add_column_double("psi_nov_rad");
    data_add_column_int8("cluster_active");
    data_add_column_double("cluster_target_rad");
    data_add_column_int32("cluster_t0_ms");
    data_add_column_int32("cluster_until_ms");
}
static void export_data(void){
    data_set_value_int8("nb_neighbors", (int8_t)mydata->nb_neighbors);
    data_set_value_double("theta_photo_rad", mydata->photo_heading_rad);
    data_set_value_double("theta_cmd_rad", mydata->theta_cmd_rad);
    data_set_value_double("v_cmd", mydata->v_cmd);
    data_set_value_double("rho_norm", mydata->rho_norm);
    data_set_value_double("novelty", mydata->novelty_ema);
    double psi_nov = atan2(mydata->nov_vy, mydata->nov_vx);
    data_set_value_double("psi_nov_rad", psi_nov);
    data_set_value_int8("cluster_active", (int8_t)(enable_cluster_uturn && mydata->cluster_turn_active ? 1 : 0));
    data_set_value_double("cluster_target_rad", mydata->cluster_target_rad);
    data_set_value_int32("cluster_t0_ms", mydata->cluster_wall_t0_ms);
    data_set_value_int32("cluster_until_ms", mydata->cluster_active_until_ms);
}

static void global_setup(void){
    init_from_configuration(max_age);
    init_from_configuration(vicsek_period_ms);

    init_from_configuration(include_self_in_avg);
    init_from_configuration(noise_eta_rad);
    init_from_configuration(vicsek_time_continuous);
    init_from_configuration(vicsek_beta_rad_per_s);
    init_from_configuration(cont_noise_sigma_rad);
    init_from_configuration(cont_max_dt_s);

    // TT
    init_from_configuration(tt_alpha);
    init_from_configuration(tt_beta);
    init_from_configuration(press_kappa);
    init_from_configuration(rho_smooth_tau_s);
    init_from_configuration(v_min);
    init_from_configuration(v_max);

    // Novelty
    init_from_configuration(novelty_window_ms);
    init_from_configuration(novelty_tau_s);
    init_from_configuration(novelty_dir_tau_s);
    init_from_configuration(novelty_speed_gain);
    init_from_configuration(lambda_adv_rad_per_s);

    // Heading/geom
    init_from_configuration(alpha_deg);
    init_from_configuration(robot_radius);
    char heading_chiralty[128] = "cw";
    init_array_from_configuration(heading_chiralty);
    if (strcasecmp(heading_chiralty, "cw") == 0) heading_chiralty_enum = HEADING_CW;
    else if (strcasecmp(heading_chiralty, "ccw") == 0) heading_chiralty_enum = HEADING_CCW;
    else { printf("ERROR: heading_chiralty must be cw|ccw\n"); exit(1); }

    // Wall-avoidance
    init_from_configuration(wall_avoidance_memory_ms);
    init_from_configuration(wall_avoidance_turn_duration_ms);
    init_from_configuration(wall_avoidance_forward_commit_ms);
    init_from_configuration(wall_avoidance_forward_speed_ratio);
    char wall_avoidance_policy[128] = "min_turn";
    init_array_from_configuration(wall_avoidance_policy);
    if (strcasecmp(wall_avoidance_policy, "cw") == 0) wall_avoidance_chiralty_policy = WALL_CW;
    else if (strcasecmp(wall_avoidance_policy, "ccw") == 0) wall_avoidance_chiralty_policy = WALL_CCW;
    else if (strcasecmp(wall_avoidance_policy, "random") == 0) wall_avoidance_chiralty_policy = WALL_RANDOM;
    else if (strcasecmp(wall_avoidance_policy, "min_turn") == 0) wall_avoidance_chiralty_policy = WALL_MIN_TURN;
    else { printf("ERROR: wall_avoidance_policy must be cw|ccw|random|min_turn\n"); exit(1); }

    // Cluster U-turn config
    init_from_configuration(enable_cluster_uturn);
    init_from_configuration(broadcast_angle_when_avoiding_walls);
    init_from_configuration(cluster_u_turn_duration_ms);
    init_from_configuration(phi_rad_min);
    init_from_configuration(phi_rad_max);
    if (phi_rad_min > phi_rad_max){ float t=phi_rad_min; phi_rad_min=phi_rad_max; phi_rad_max=t; }
    init_from_configuration(cluster_uturn_align_thresh_rad);

    // LED
    char main_led_display[128] = "state";
    init_array_from_configuration(main_led_display);
    if (strcasecmp(main_led_display, "state") == 0)      main_led_display_enum = SHOW_STATE;
    else if (strcasecmp(main_led_display, "angle") == 0) main_led_display_enum = SHOW_ANGLE;
    else if (strcasecmp(main_led_display, "speed") == 0) main_led_display_enum = SHOW_SPEED;
    else { printf("ERROR: main_led_display must be state|angle|speed\n"); exit(1); }
}
#endif

int main(void){
    pogobot_init();
    pogobot_start(user_init, user_step);
    pogobot_start(default_walls_user_init, default_walls_user_step, "walls");
    SET_CALLBACK(callback_global_setup,       global_setup);
    SET_CALLBACK(callback_create_data_schema, create_data_schema);
    SET_CALLBACK(callback_export_data,        export_data);
    return 0;
}

