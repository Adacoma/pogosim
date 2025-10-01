#include "pogobase.h"
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

// Wall avoidance routines
#include "pogo-utils/wall_avoidance.h"
#include "pogo-utils/version.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MAX_NEIGHBORS      20u // limitation coming from neighbour counter program, to be adapted?
#define BEACON_HZ          10u // frequency of beacon transmission
#define BEACON_PERIOD_MS  (1000u / BEACON_HZ) // converted into period

// main motor speed (can be exposed as parameter if needed), 1023 is max (cf the API)
static int forward_speed = motorHalf;
//static int forward_speed = motorFull;

// parameters, can be defined in the YAML, inspired by neighbour counter 
uint32_t max_age             = 600;          /* ms: neighbor considered expired beyond this */
uint32_t vicsek_period_ms    = 17;           /* ms: Vicsek update period */
// not useful I think : uint8_t  vicsek_ir_power     = 5;           

// vicsek noise
double   noise_eta_rad       = 1.5;          /* rad */

// alignment gain, should be in [0..1], 
double   align_gain          = 1.0;

// should the robot’s own measured heading (from photodiodes) be included in the shared average?
bool     include_self_in_avg = true;

// conversion gain 
double   vicsek_turn_gain    = 0.8;          /* typically 0.6–1.0 */

// geometrical parameters of pogobots
double alpha_deg    = 40.0;
double robot_radius = 0.0265;  /* default 26.5 mm */


// compact vicsek message to broadcast
typedef struct __attribute__((__packed__)) {
    uint16_t sender_id;
    int16_t  theta_mrad;  // ! i choose to diffuse the direction commanded by the robot, to have a more stable behavior!!
} vicsek_msg_t;

#define MSG_SIZE ((uint16_t)sizeof(vicsek_msg_t))

// neighbor structure
typedef struct {
    uint16_t id;
    uint32_t last_seen_ms; // timestamp of last reception, can be removed because more useful if we want to display the neighbors
    int16_t  theta_mrad;   
} neighbor_t;

// intern state of each pogobot
typedef struct {
    // timing
    float dt_s;

    neighbor_t neighbors[MAX_NEIGHBORS];
    uint8_t    nb_neighbors;
    uint32_t   last_beacon_ms; // last time we difused our message

    // vicsek internal state
    double     theta_cmd_rad;         
    uint32_t   last_vicsek_update_ms;

    // current heading
    double     photo_heading_rad;

    // memorized direction
    int        diff_cmd;
    // motor direction as memorized by the firmware
    uint8_t    motor_dir_left_fwd;
    uint8_t    motor_dir_right_fwd;

    // Wall avoidance
    wall_avoidance_state_t wall_avoidance;
    bool doing_wall_avoidance;
} USERDATA;

DECLARE_USERDATA(USERDATA);
REGISTER_USERDATA(USERDATA);

// some one liners that are useful !! WARNING : they have been written by my best friend ChatGPT and they are working
//  however, maybe not the most efficient possible (or have some cases not handled)
static inline double wrap_pi(double a){ while(a> M_PI)a-=2.0*M_PI; while(a<-M_PI)a+=2.0*M_PI; return a; }
static inline int16_t rad_to_mrad(double a){ a=wrap_pi(a); long v=lround(a*1000.0); if(v>32767)v=32767; if(v<-32768)v=-32768; return (int16_t)v; }
static inline double  mrad_to_rad(int16_t m){ return ((double)m)/1000.0; }
static inline double noise_uniform(double eta){ double u=(double)rand()/(double)RAND_MAX; return (u-0.5)*eta; }


// to transform a signed speed into a motor command, using the memorized forward direction
static inline void motor_set_signed(motor_id id, int spd_signed, uint8_t fwd_dir_mem){
    int mag = spd_signed >= 0 ? spd_signed : -spd_signed; // absolute value
    if(mag > motorFull) mag = motorFull; // saturation if needed
    uint8_t dir = (spd_signed >= 0) ? fwd_dir_mem : ((fwd_dir_mem==0)?1:0); // direction
    pogobot_motor_dir_set(id, dir);
    pogobot_motor_set(id, mag);
}

// !! important (the old version was not working precisely because we did not use this function in the main loop)
//  entirely inspired by neighbor counter program
static void purge_old_neighbors(void){
    uint32_t now=current_time_milliseconds();
    for(int i=(int)mydata->nb_neighbors-1;i>=0;--i){
        if(now - mydata->neighbors[i].last_seen_ms > max_age){
            mydata->neighbors[i]=mydata->neighbors[mydata->nb_neighbors-1];
            mydata->nb_neighbors--;
        }
    }
}

// insert or update a neighbor in the list,
static neighbor_t* upsert_neighbor(uint16_t id){
    // run through the list to find it
    for(uint8_t i=0;i<mydata->nb_neighbors;++i)
        if(mydata->neighbors[i].id==id) return &mydata->neighbors[i];
    if(mydata->nb_neighbors>=MAX_NEIGHBORS) return NULL; // not sure if this limit is useful, to see with material limitations
    neighbor_t* n=&mydata->neighbors[mydata->nb_neighbors++]; 
    n->id=id; n->theta_mrad=0; n->last_seen_ms=0; return n;
}

bool send_message(void){
    uint32_t now=current_time_milliseconds();
    if (now - mydata->last_beacon_ms < BEACON_PERIOD_MS) return false; // to not talk too much...
                                                                      // structure defined above
    if (mydata->doing_wall_avoidance) return false; // Don't communicate if current doing wall avoidance

    vicsek_msg_t m={
        .sender_id = pogobot_helper_getid(),
        .theta_mrad= rad_to_mrad(mydata->theta_cmd_rad)  
    };
    // send it
    mydata->last_beacon_ms=now;
    return pogobot_infrared_sendShortMessage_omni((uint8_t*)&m, MSG_SIZE);
}


/**
 * @brief Handle an incoming packet.
 *
 * @param[in] mr Pointer to the message wrapper provided by the firmware.
 */
void process_message(message_t* mr){
    // Process wall message and return if it was one
    if (wall_avoidance_process_message(&mydata->wall_avoidance, mr)) {
        return;
    }
    // Not a wall message, handle other message types below

    // verify if the size is correct 
    if(mr->header.payload_length < MSG_SIZE) return;
    vicsek_msg_t const* m=(vicsek_msg_t const*)mr->payload;
    if(m->sender_id==pogobot_helper_getid()) return;

    neighbor_t* n=upsert_neighbor(m->sender_id);
    if(!n) return;
    n->theta_mrad=m->theta_mrad;
    n->last_seen_ms=current_time_milliseconds();
}

// classic désormais, I do not have to explain, n'est ce pas Léo ? 
static inline double estimate_heading_from_photos(void){
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
    double photo_heading = -angle_rel;
    return wrap_pi(photo_heading); // le one liner définit plus haut, pratique
}

// le coeur de la bête!
static void vicsek_update_and_build_diff(void){
    purge_old_neighbors();

    double heading = mydata->photo_heading_rad;

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

    // relaxation avec align_gain
    double theta_blend = wrap_pi((1.0 - align_gain) * heading
            +  align_gain        * theta_mean);

    double theta_cmd = wrap_pi(theta_blend + noise_uniform(noise_eta_rad));

    // erreur instantanée
    double err = wrap_pi(theta_cmd - heading);

    // normalisation à 30°
    const double err_norm = err / (30.0 * M_PI / 180.0);

    int diff = (int)lround(vicsek_turn_gain * err_norm * (double)forward_speed);
    if (diff >  forward_speed) diff =  forward_speed;
    if (diff < -forward_speed) diff = -forward_speed;

    mydata->theta_cmd_rad = theta_cmd;
    mydata->diff_cmd      = diff;
}


void user_init(void){
    // not sure if I needed that !
    srand(pogobot_helper_getRandSeed());

    // maybe done by default by the API, but we better be sure
    memset(mydata,0,sizeof(*mydata));

    // from neighbor counter program
    main_loop_hz=10;
    mydata->dt_s = (main_loop_hz>0)? (1.f/(float)main_loop_hz) : (1.f/60.f);
    max_nb_processed_msg_per_tick=3;
    percent_msgs_sent_per_ticks=50;
    msg_rx_fn=process_message;
    msg_tx_fn=send_message;
    error_codes_led_idx=3;


    uint8_t dir_mem[3]={0,0,0};
    pogobot_motor_dir_mem_get(dir_mem);
    // convention
    mydata->motor_dir_right_fwd = dir_mem[0];
    mydata->motor_dir_left_fwd  = dir_mem[1];

    // Initialize wall avoidance (enabled by default)
    motor_calibration_t motors = {
        .motor_left = motorFull,
        .dir_left = mydata->motor_dir_left_fwd,
        .motor_right = motorFull,
        .dir_right = mydata->motor_dir_right_fwd
    };
    wall_avoidance_init_default(&mydata->wall_avoidance, &motors);
    // Wall avoidance: work at half speed
    wall_avoidance_set_forward_speed(&mydata->wall_avoidance, 0.5f);
    mydata->doing_wall_avoidance = false;

    /// first heading
    mydata->photo_heading_rad = estimate_heading_from_photos();

    // initial direction = measured heading + noise
    mydata->theta_cmd_rad = wrap_pi(mydata->photo_heading_rad + noise_uniform(noise_eta_rad));
    mydata->diff_cmd      = 0;

    mydata->last_vicsek_update_ms=current_time_milliseconds();
    mydata->last_beacon_ms=0;
    // some debug info, not everytime useful, can be deleted!
#ifdef SIMULATOR
    printf("Discrete Vicsek (arc-drive): eta=%.1f deg, T=%ums, gain_align=%.2f\n",
            noise_eta_rad*180.0/M_PI, vicsek_period_ms, align_gain);
    printf("Motor dir fwd: L=%u R=%u, alpha=%.1f deg, r=%.1f mm\n",
            (unsigned)mydata->motor_dir_left_fwd, (unsigned)mydata->motor_dir_right_fwd,
            alpha_deg, robot_radius*1000.0);
#endif
    pogobot_led_setColor(0,0,255); // blue
}


void user_step(void){
    uint32_t now = current_time_milliseconds();
    // Wall avoidance takes control if needed (with LED updates)
    mydata->doing_wall_avoidance = wall_avoidance_step(&mydata->wall_avoidance, true);

    mydata->photo_heading_rad = estimate_heading_from_photos();

    if(now - mydata->last_vicsek_update_ms >= vicsek_period_ms){
        vicsek_update_and_build_diff();
        mydata->last_vicsek_update_ms=now;  
    }

    /* Motor application: */
    if (!mydata->doing_wall_avoidance) { // Don't change motor commands if we are avoiding walls
        /* Normal mode: constant forward + frozen differential */
        motor_set_signed(motorL, forward_speed - mydata->diff_cmd, mydata->motor_dir_left_fwd);
        motor_set_signed(motorR, forward_speed + mydata->diff_cmd, mydata->motor_dir_right_fwd);
    }

    // Update LED
    if (mydata->doing_wall_avoidance) {
        pogobot_led_setColor(255,0,0); // red, doing wall avoidance
    } else if (mydata->nb_neighbors == 0) {
        pogobot_led_setColor(0,0,255); // blue, alone
    } else {
        pogobot_led_setColor(0,255,0); // green, in group
    }
}


/* Simulator hooks => GENERES AUTOMATIQUEMENT PAR COPILOT, donc hésite pas à vérifier si ça a un sens Léo*/
#ifdef SIMULATOR
static void create_data_schema(void){
    data_add_column_int8("nb_neighbors");
    data_add_column_double("theta_photo_rad");
    data_add_column_double("theta_cmd_rad");
    data_add_column_int16("diff_cmd");
}
static void export_data(void){
    data_set_value_int8("nb_neighbors", (int8_t)mydata->nb_neighbors);
    data_set_value_double("theta_photo_rad", mydata->photo_heading_rad);
    data_set_value_double("theta_cmd_rad", mydata->theta_cmd_rad);
    data_set_value_int16("diff_cmd", (int16_t)mydata->diff_cmd);
}
static void global_setup(void){
    init_from_configuration(max_age);
    init_from_configuration(vicsek_period_ms);

    init_from_configuration(noise_eta_rad);
    init_from_configuration(include_self_in_avg);
    init_from_configuration(align_gain);
    init_from_configuration(vicsek_turn_gain);

    /* Orientation photos: configurable parameters */
    init_from_configuration(alpha_deg);
    init_from_configuration(robot_radius);

    /* Optional:
       init_from_configuration(forward_speed);
       */
}
#endif

int main(void){
    pogobot_init();

    // Start the robot's main loop with the defined user_init and user_step functions. Ignored by the pogowalls.
    pogobot_start(user_init, user_step);
    // Use robots category "robots" by default. Same behavior as: pogobot_start(user_init, user_step, "robots");
    //   --> Make sure that the category "robots" is used to declare the pogobots in your configuration file. Cf conf/test.yaml

    // Init and main loop functions for the walls (pogowalls). Ignored by the robots.
    // Use the default functions provided by Pogosim. Cf examples 'walls' to see how to declare custom wall user code functions.
    pogobot_start(default_walls_user_init, default_walls_user_step, "walls");

    // Specify the callback functions. Only called by the simulator.
    SET_CALLBACK(callback_global_setup,       global_setup);
    SET_CALLBACK(callback_create_data_schema, create_data_schema);
    SET_CALLBACK(callback_export_data,        export_data);
    return 0;
}

// MODELINE "{{{1
// vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
// vim:foldmethod=marker
