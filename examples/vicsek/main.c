#include "pogobase.h"
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <strings.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Wall avoidance routines
#include "pogo-utils/wall_avoidance.h"
#include "pogo-utils/version.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MAX_NEIGHBORS      20u
#define BEACON_HZ          10u
#define BEACON_PERIOD_MS  (1000u / BEACON_HZ)

// main motor speed (1023 is max)
static int forward_speed = motorHalf;

// parameters (configurable in YAML)
uint32_t max_age             = 600;          /* ms: neighbor expiration */
uint32_t vicsek_period_ms    = 17;           /* ms: Vicsek update period */

// vicsek noise
double   noise_eta_rad       = 1.5;          /* rad */

// alignment gain [0..1]
double   align_gain          = 1.0;

// include self heading in average?
bool     include_self_in_avg = true;
bool     broadcast_angle_when_avoiding_walls = true;

// conversion gain
double   vicsek_turn_gain    = 0.8;          /* typically 0.6–1.0 */

// --- Vicsek mode switch and continuous-time parameters ---
bool   vicsek_time_continuous = false;     // false = discrete (default), true = continuous
double vicsek_beta_rad_per_s  = 3.0;       // β in dθ/dt = β sin(θ̄ - θ) + noise
double cont_noise_sigma_rad   = 0.0;       // σ for diffusion-like noise on θ (rad/√s)
double cont_max_dt_s          = 0.05;      // clamp dt for stability

// geometry
double alpha_deg    = 40.0;
double robot_radius = 0.0265;  /* 26.5 mm */

// === CLUSTER U-TURN: configurable duration window (ms) ===
uint32_t cluster_u_turn_duration_ms = 1500;

// Wall avoidance policy
wall_chirality_t wall_avoidance_chiralty_policy = WALL_MIN_TURN;

// What main LEDs show
typedef enum {
    SHOW_STATE,
    SHOW_ANGLE
} main_led_display_type_t;
main_led_display_type_t main_led_display_enum = SHOW_STATE;

// === CLUSTER U-TURN message flag ===
enum : uint8_t {
    VMSGF_CLUSTER_UTURN = 0x01
};

// Extended Vicsek message (packed).
typedef struct __attribute__((__packed__)) {
    uint16_t sender_id;
    int16_t  theta_mrad;            // commanded direction of sender
    uint8_t  flags;                 // bit0: cluster U-turn active
    int16_t  cluster_target_mrad;   // valid if flags & VMSGF_CLUSTER_UTURN
    uint32_t cluster_wall_t0_ms;    // first wall time at the originator
    uint16_t cluster_msg_uid;       // de-dup / freshness id
} vicsek_msg_t;

#define MSG_SIZE ((uint16_t)sizeof(vicsek_msg_t))

// neighbor structure
typedef struct {
    uint16_t id;
    uint32_t last_seen_ms;
    int16_t  theta_mrad;
} neighbor_t;

// internal state
typedef struct {
    // timing
    float dt_s;

    neighbor_t neighbors[MAX_NEIGHBORS];
    uint8_t    nb_neighbors;
    uint32_t   last_beacon_ms;

    // vicsek internal state
    double     theta_cmd_rad;
    uint32_t   last_vicsek_update_ms;

    // current heading (photos)
    double     photo_heading_rad;

    // memorized direction
    int        diff_cmd;
    uint8_t    motor_dir_left_fwd;
    uint8_t    motor_dir_right_fwd;

    // Wall avoidance
    wall_avoidance_state_t wall_avoidance;
    bool doing_wall_avoidance;
    bool prev_doing_wall_avoidance;        // === CLUSTER U-TURN: rising-edge detection

    // === CLUSTER U-TURN: cluster override state ===
    bool     cluster_turn_active;
    double   cluster_target_rad;
    uint32_t cluster_wall_t0_ms;           // t0 at originator
    uint32_t cluster_active_until_ms;      // t0 + duration
    uint16_t cluster_msg_uid;              // uid that we originated (if any)
    uint16_t last_seen_cluster_uid;        // last uid we accepted
    bool     have_seen_cluster_uid;
} USERDATA;

DECLARE_USERDATA(USERDATA);
REGISTER_USERDATA(USERDATA);

// helpers
static inline double wrap_pi(double a){ while(a> M_PI)a-=2.0*M_PI; while(a<-M_PI)a+=2.0*M_PI; return a; }
static inline int16_t rad_to_mrad(double a){ a=wrap_pi(a); long v=lround(a*1000.0); if(v>32767)v=32767; if(v<-32768)v=-32768; return (int16_t)v; }
static inline double  mrad_to_rad(int16_t m){ return ((double)m)/1000.0; }
static inline double noise_uniform(double eta){ double u=(double)rand()/(double)RAND_MAX; return (u-0.5)*eta; }

static inline void motor_set_signed(motor_id id, int spd_signed, uint8_t fwd_dir_mem){
    int mag = spd_signed >= 0 ? spd_signed : -spd_signed;
    if(mag > motorFull) mag = motorFull;
    uint8_t dir = (spd_signed >= 0) ? fwd_dir_mem : ((fwd_dir_mem==0)?1:0);
    pogobot_motor_dir_set(id, dir);
    pogobot_motor_set(id, mag);
}

static void purge_old_neighbors(void){
    uint32_t now=current_time_milliseconds();
    for(int i=(int)mydata->nb_neighbors-1;i>=0;--i){
        if(now - mydata->neighbors[i].last_seen_ms > max_age){
            mydata->neighbors[i]=mydata->neighbors[mydata->nb_neighbors-1];
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

// === CLUSTER U-TURN: are we currently in the active window?
static inline bool cluster_window_active(uint32_t now){
    return mydata->cluster_turn_active && (now < mydata->cluster_active_until_ms);
}

// === CLUSTER U-TURN: adopt a new cluster instruction if "newer"
static void cluster_adopt_and_activate(int16_t tgt_mrad, uint32_t t0_ms, uint16_t uid){
    //uint32_t now = current_time_milliseconds();
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

// === CLUSTER U-TURN: compose and send a beacon (with cluster flag if active)
bool send_message(void){
    uint32_t now=current_time_milliseconds();
    if (now - mydata->last_beacon_ms < BEACON_PERIOD_MS) return false;

    // Determine whether we advertise the cluster U-turn in this beacon
    bool advertise_cluster = cluster_window_active(now);

    // Respect "mute on wall-avoidance" *unless* we must advertise a cluster event.
    if (!advertise_cluster && !broadcast_angle_when_avoiding_walls && mydata->doing_wall_avoidance) {
        return false;
    }

    vicsek_msg_t m = {
        .sender_id = pogobot_helper_getid(),
        .theta_mrad= rad_to_mrad(mydata->theta_cmd_rad),
        .flags     = 0u,
        .cluster_target_mrad = 0,
        .cluster_wall_t0_ms  = 0u,
        .cluster_msg_uid     = 0u
    };

    if (advertise_cluster) {
        m.flags               |= VMSGF_CLUSTER_UTURN;
        m.cluster_target_mrad  = rad_to_mrad(mydata->cluster_target_rad);
        m.cluster_wall_t0_ms   = mydata->cluster_wall_t0_ms;
        m.cluster_msg_uid      = mydata->cluster_msg_uid;  // 0 if we’re a relay, which is fine
    }

    mydata->last_beacon_ms = now;
    return pogobot_infrared_sendShortMessage_omni((uint8_t*)&m, MSG_SIZE);
}

/**
 * @brief Handle an incoming packet.
 */
void process_message(message_t* mr){
    // Let the wall-avoidance module eat its own messages first.
    if (wall_avoidance_process_message(&mydata->wall_avoidance, mr)) {
        return;
    }

    if(mr->header.payload_length < MSG_SIZE) return;
    vicsek_msg_t const* m=(vicsek_msg_t const*)mr->payload;
    if(m->sender_id==pogobot_helper_getid()) return;

    // Neighbor update (for Vicsek averaging)
    neighbor_t* n=upsert_neighbor(m->sender_id);
    if(!n) return;
    n->theta_mrad=m->theta_mrad;
    n->last_seen_ms=current_time_milliseconds();

    // === CLUSTER U-TURN: accept cluster instruction and re-flood
    if (m->flags & VMSGF_CLUSTER_UTURN) {
        cluster_adopt_and_activate(m->cluster_target_mrad, m->cluster_wall_t0_ms, m->cluster_msg_uid);
        // We don't need to queue a special immediate send; our periodic beacon will
        // include the flag for as long as the window is active, achieving a flood.
    }
}

static inline double estimate_heading_from_photos(void) {
    int16_t pA_raw = pogobot_photosensors_read(0);
    int16_t pB_raw = pogobot_photosensors_read(1);
    int16_t pC_raw = pogobot_photosensors_read(2);

    const double alpha = alpha_deg * M_PI / 180.0;
    const double r     = robot_radius;

    double D_BA = (double)pB_raw - (double)pA_raw;
    double D_CA = (double)pC_raw - (double)pA_raw;

    double s = sin(alpha), c = cos(alpha);

    double gx = (D_CA - D_BA) / (2.0 * r * s);
    double gy = (D_CA + D_BA) / (2.0 * r * (c + 1.0));

    double angle_rel = atan2(gx, gy);
    //double angle_rel = atan2(gy, gx);
    double photo_heading = -angle_rel;
    return wrap_pi(photo_heading);
}

// Build diff for motors (Vicsek + optional cluster override)
static void vicsek_update_and_build_diff(void){
    purge_old_neighbors();

    uint32_t now = current_time_milliseconds();
    double heading = mydata->photo_heading_rad;
    double theta_cmd;
    // compute dt (s) and clamp for continuous-time mode
    double dt_s = (now - mydata->last_vicsek_update_ms) * 1e-3;
    if (dt_s < 0.0) dt_s = 0.0;
    if (dt_s > cont_max_dt_s) dt_s = cont_max_dt_s;


    if (cluster_window_active(now)) {
        // === CLUSTER U-TURN override ===
        theta_cmd = mydata->cluster_target_rad;
    } else {
        // Normal Vicsek
        double sx = 0.0, sy = 0.0;
        if (include_self_in_avg){
            sx += cos(heading);
            sy += sin(heading);
        }
        for (uint8_t i=0; i<mydata->nb_neighbors; ++i){
            double th = mrad_to_rad(mydata->neighbors[i].theta_mrad);
            sx += cos(th);
            sy += sin(th);
        }

        double theta_mean = heading;
        if (!(sx == 0.0 && sy == 0.0)){
            theta_mean = atan2(sy, sx);
        }

        
        if (vicsek_time_continuous){
            // Continuous-time Vicsek: dθ/dt = β sin(θ̄ - θ) + σ ξ(t)
            //double dtheta = vicsek_beta_rad_per_s * sin(wrap_pi(theta_mean - heading)) * dt_s; // CW angle definition
            double dtheta = vicsek_beta_rad_per_s * sin(wrap_pi(heading - theta_mean)) * dt_s; // CCW angle definition

            if (cont_noise_sigma_rad > 0.0){
                // Gaussian(0,1) via Box-Muller
                double u1 = (rand() + 1.0) / (RAND_MAX + 2.0);
                double u2 = (rand() + 1.0) / (RAND_MAX + 2.0);
                double z  = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
                dtheta += cont_noise_sigma_rad * sqrt(dt_s) * z;
            }
            theta_cmd = wrap_pi(heading + dtheta);
            mydata->last_vicsek_update_ms = now;
        } else {
            double theta_blend = wrap_pi((1.0 - align_gain) * heading
                    +  align_gain        * theta_mean);

            theta_cmd = wrap_pi(theta_blend + noise_uniform(noise_eta_rad));
            mydata->cluster_turn_active = false;
        }
        // ensure flag goes low if window elapsed
    }

    double err = wrap_pi(theta_cmd - heading);
    const double err_norm = err / (30.0 * M_PI / 180.0);

    int diff = (int)lround(vicsek_turn_gain * err_norm * (double)forward_speed);
    if (diff >  forward_speed) diff =  forward_speed;
    if (diff < -forward_speed) diff = -forward_speed;

    mydata->theta_cmd_rad = theta_cmd;
    mydata->diff_cmd      = diff;
}

#define SCALE_0_255_TO_0_25(x)   (uint8_t)((x) * (25.0f / 255.0f) + 0.5f)

void update_main_led(void) {
    if (main_led_display_enum == SHOW_STATE) {
        if (mydata->doing_wall_avoidance) {
            pogobot_led_setColor(255,0,0); // red: wall avoidance
        } else if (mydata->nb_neighbors == 0) {
            pogobot_led_setColor(0,0,255); // blue: alone
        } else if (cluster_window_active(current_time_milliseconds())) {
            pogobot_led_setColor(0,255,255); // cyan: cluster U-turn active
        } else {
            pogobot_led_setColor(0,255,0); // green: normal group
        }
    } else if (main_led_display_enum == SHOW_ANGLE) {
        float angle = mydata->photo_heading_rad;
        if (angle < 0.0f) { angle += 2.0f * M_PI; }

        float hue_deg = angle * 180.0f / (float)M_PI;   // 0-360
        uint8_t r8, g8, b8;
        hsv_to_rgb(hue_deg, 1.0f, 1.0f, &r8, &g8, &b8);
        r8 = SCALE_0_255_TO_0_25(r8);
        g8 = SCALE_0_255_TO_0_25(g8);
        b8 = SCALE_0_255_TO_0_25(b8);
        if (r8 == 0 && g8 == 0 && b8 == 0) { r8 = 1; }
        pogobot_led_setColor(r8, g8, b8);
    } else {
        // ...
    }
}

void user_init(void){
    srand(pogobot_helper_getRandSeed());
    memset(mydata,0,sizeof(*mydata));

    main_loop_hz=30;
    mydata->dt_s = (main_loop_hz>0)? (1.f/(float)main_loop_hz) : (1.f/60.f);
    max_nb_processed_msg_per_tick=3;
    percent_msgs_sent_per_ticks=50;
    msg_rx_fn=process_message;
    msg_tx_fn=send_message;
    error_codes_led_idx=3;

#ifdef SIMULATOR
    printf("Vicsek mode: %s\n", vicsek_time_continuous ? "continuous (sine coupling)" : "discrete");
    if (vicsek_time_continuous){
        printf("  beta=%.3f rad/s, sigma=%.3f rad/sqrt(s), max_dt=%.3f s\n",
               vicsek_beta_rad_per_s, cont_noise_sigma_rad, cont_max_dt_s);
    } else {
        printf("  eta=%.1f deg, T=%ums, align=%.2f\n",
               noise_eta_rad*180.0/M_PI, vicsek_period_ms, align_gain);
    }
#endif

    uint8_t dir_mem[3]={0,0,0};
    pogobot_motor_dir_mem_get(dir_mem);
    mydata->motor_dir_right_fwd = dir_mem[0];
    mydata->motor_dir_left_fwd  = dir_mem[1];

    // Initialize wall avoidance (enabled by default)
    motor_calibration_t motors = {
        .motor_left = motorFull,
        .dir_left = mydata->motor_dir_left_fwd,
        .motor_right = motorFull,
        .dir_right = mydata->motor_dir_right_fwd
    };
    wall_avoidance_config_t default_config = {
        .wall_memory_ms = 900,
        .turn_duration_ms = 300,
        .forward_commit_ms = 600,
        .forward_speed_ratio = 0.8f
    };
    wall_avoidance_init(&mydata->wall_avoidance, &default_config, &motors);
    wall_avoidance_set_forward_speed(&mydata->wall_avoidance, 0.5f);
    wall_avoidance_set_policy(&mydata->wall_avoidance, wall_avoidance_chiralty_policy, 0);
    mydata->doing_wall_avoidance    = false;
    mydata->prev_doing_wall_avoidance = false;

    // Cluster U-turn defaults
    mydata->cluster_turn_active     = false;
    mydata->cluster_target_rad      = 0.0;
    mydata->cluster_wall_t0_ms      = 0u;
    mydata->cluster_active_until_ms = 0u;
    mydata->cluster_msg_uid         = 0u;
    mydata->have_seen_cluster_uid   = false;

    mydata->photo_heading_rad = estimate_heading_from_photos();
    mydata->theta_cmd_rad     = wrap_pi(mydata->photo_heading_rad + noise_uniform(noise_eta_rad));
    mydata->diff_cmd          = 0;

    mydata->last_vicsek_update_ms=current_time_milliseconds();
    mydata->last_beacon_ms=0;

#ifdef SIMULATOR
    printf("Vicsek+ClusterUturn: eta=%.1f deg, T=%ums, align=%.2f, window=%ums\n",
            noise_eta_rad*180.0/M_PI, vicsek_period_ms, align_gain, cluster_u_turn_duration_ms);
    printf("Motor dir fwd: L=%u R=%u, alpha=%.1f deg, r=%.1f mm\n",
            (unsigned)mydata->motor_dir_left_fwd, (unsigned)mydata->motor_dir_right_fwd,
            alpha_deg, robot_radius*1000.0);
#endif
    pogobot_led_setColor(0,0,255); // blue
}

void user_step(void){
    uint32_t now = current_time_milliseconds();

    // Run wall-avoidance controller (handles its own LEDs for wall beacons)
    bool wa = wall_avoidance_step(&mydata->wall_avoidance, true);
    mydata->prev_doing_wall_avoidance = mydata->doing_wall_avoidance;
    mydata->doing_wall_avoidance = wa;

    mydata->photo_heading_rad = estimate_heading_from_photos();

    // === CLUSTER U-TURN: rising edge => originate a cluster instruction
    if (!mydata->prev_doing_wall_avoidance && mydata->doing_wall_avoidance) {
        // Set cluster target = current heading + π (U-turn)
        double target = wrap_pi(mydata->photo_heading_rad + M_PI);
        mydata->cluster_target_rad      = target;
        mydata->cluster_wall_t0_ms      = now;
        mydata->cluster_active_until_ms = now + cluster_u_turn_duration_ms;
        mydata->cluster_turn_active     = true;
        mydata->cluster_msg_uid         = (uint16_t)(rand() & 0xFFFF); // new uid
        mydata->last_seen_cluster_uid   = mydata->cluster_msg_uid;
        mydata->have_seen_cluster_uid   = true;
    }

    if (vicsek_time_continuous) {
        vicsek_update_and_build_diff();
    } else if(now - mydata->last_vicsek_update_ms >= vicsek_period_ms){
        vicsek_update_and_build_diff();
        mydata->last_vicsek_update_ms=now;
    }

    // Apply motors unless wall-avoidance temporarily owns them
    if (!mydata->doing_wall_avoidance) {
        motor_set_signed(motorL, forward_speed - mydata->diff_cmd, mydata->motor_dir_left_fwd);
        motor_set_signed(motorR, forward_speed + mydata->diff_cmd, mydata->motor_dir_right_fwd);
    }

    update_main_led();
}

#ifdef SIMULATOR
static void create_data_schema(void){
    data_add_column_int8("nb_neighbors");
    data_add_column_double("theta_photo_rad");
    data_add_column_double("theta_cmd_rad");
    data_add_column_int16("diff_cmd");
    // === CLUSTER U-TURN telemetry ===
    data_add_column_int8("cluster_active");
    data_add_column_double("cluster_target_rad");
    data_add_column_int32("cluster_t0_ms");
    data_add_column_int32("cluster_until_ms");
}
static void export_data(void){
    data_set_value_int8("nb_neighbors", (int8_t)mydata->nb_neighbors);
    data_set_value_double("theta_photo_rad", mydata->photo_heading_rad);
    data_set_value_double("theta_cmd_rad", mydata->theta_cmd_rad);
    data_set_value_int16("diff_cmd", (int16_t)mydata->diff_cmd);
    // === CLUSTER U-TURN telemetry ===
    data_set_value_int8("cluster_active", (int8_t)(mydata->cluster_turn_active ? 1 : 0));
    data_set_value_double("cluster_target_rad", mydata->cluster_target_rad);
    data_set_value_int32("cluster_t0_ms", mydata->cluster_wall_t0_ms);
    data_set_value_int32("cluster_until_ms", mydata->cluster_active_until_ms);
}
static void global_setup(void){
    init_from_configuration(max_age);
    init_from_configuration(vicsek_period_ms);

    init_from_configuration(noise_eta_rad);
    init_from_configuration(include_self_in_avg);
    init_from_configuration(broadcast_angle_when_avoiding_walls);
    init_from_configuration(align_gain);
    init_from_configuration(vicsek_turn_gain);
    init_from_configuration(vicsek_time_continuous);
    init_from_configuration(vicsek_beta_rad_per_s);
    init_from_configuration(cont_noise_sigma_rad);
    init_from_configuration(cont_max_dt_s);


    init_from_configuration(alpha_deg);
    init_from_configuration(robot_radius);

    // Optional:
    // init_from_configuration(forward_speed);

    // === CLUSTER U-TURN param ===
    init_from_configuration(cluster_u_turn_duration_ms);

    char main_led_display[128] = "state";
    init_array_from_configuration(main_led_display);
    if (strcasecmp(main_led_display, "state") == 0) {
        main_led_display_enum = SHOW_STATE;
    } else if (strcasecmp(main_led_display, "angle") == 0) {
        main_led_display_enum = SHOW_ANGLE;
    } else {
        printf("ERROR: unknown main_led_display: '%s' (use 'state' or 'angle').\n", main_led_display);
        exit(1);
    }

    char wall_avoidance_policy[128] = "min_turn";
    init_array_from_configuration(wall_avoidance_policy);
    if (strcasecmp(wall_avoidance_policy, "cw") == 0) {
        wall_avoidance_chiralty_policy = WALL_CW;
    } else if (strcasecmp(wall_avoidance_policy, "ccw") == 0) {
        wall_avoidance_chiralty_policy = WALL_CCW;
    } else if (strcasecmp(wall_avoidance_policy, "random") == 0) {
        wall_avoidance_chiralty_policy = WALL_RANDOM;
    } else if (strcasecmp(wall_avoidance_policy, "min_turn") == 0) {
        wall_avoidance_chiralty_policy = WALL_MIN_TURN;
    } else {
        printf("ERROR: unknown main_led_display: '%s' (use 'cw', 'ccw', 'random', or 'min_turn').\n", wall_avoidance_policy);
        exit(1);
    }
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

