
// Main include for pogobots, both for real robots and for simulations
#include "pogobase.h"
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884197169399375105820974944
#endif

// Minimal ABP + wall-avoidance
// - ABP: constant speed, rotational diffusion on heading
// - Wall-avoidance: if IMU says "not moving", do a brief reverse+spin

#include "pogobase.h"
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

// -------------------------------
// Tunables (few on purpose)
// -------------------------------
float abp_speed_frac        = 0.30f;  // [0..1] forward throttle
float abp_rot_diffusion     = 1.40f;  // [rad^2/s] D_r
float abp_turn_gain         = 0.56f;  // omega [rad/s] -> differential fraction

// Wall-avoidance (IMU-based)
bool  enable_backward_dir   = true;  // if false, escape is disabled (remains ABP only)
float backoff_duration_s    = 0.80f;  // duration of reverse+spin
int   stuck_ticks_to_trigger= 4;      // consecutive ticks "not moving" before escape
float gyro_treshold         = 1.85f;  // IMU thresholds, as provided
float acc_treshold          = 1.80f;

// -------------------------------
// USERDATA (lean)
// -------------------------------
typedef struct {
    // timing
    time_reference_t timer_it;
    float dt_s;

    // ABP state
    float theta;         // [rad]
    float omega_cmd;     // [rad/s]
    float diff_frac;     // [-1..1]

    // IMU wall-escape
    int   still_ticks;   // consecutive "not moving" ticks
    bool  in_escape;     // currently reversing+spinning?
    float escape_left_s; // time left in escape
    int   spin_sign;     // +/-1

    // motor direction calib
    uint8_t motor_dir_left;
    uint8_t motor_dir_right;

    // RNG cache for Gaussian
    int   has_spare;
    float spare;
} USERDATA;

DECLARE_USERDATA(USERDATA);
REGISTER_USERDATA(USERDATA);

// -------------------------------
// Small helpers
// -------------------------------
static inline float clampf(float x, float a, float b){ return (x<a)?a: (x>b)?b:x; }
static inline float wrap_angle(float a){
    while(a <= -M_PI) a += 2.f*(float)M_PI;
    while(a >   M_PI) a -= 2.f*(float)M_PI;
    return a;
}
static float rand_normal(USERDATA *ud){
    if (ud->has_spare){ ud->has_spare=0; return ud->spare; }
    float u1,u2; do{ u1=(rand()+1.f)/((float)RAND_MAX+2.f);}while(u1<=0.f);
    u2=(rand()+1.f)/((float)RAND_MAX+2.f);
    float m=sqrtf(-2.f*logf(u1));
    float z0=m*cosf(2.f*(float)M_PI*u2), z1=m*sinf(2.f*(float)M_PI*u2);
    ud->spare=z1; ud->has_spare=1; return z0;
}
static inline void set_motor_frac(motor_id id, float frac){
    int cmd=(int)lroundf(clampf(frac,0.f,1.f)*(float)motorFull);
    pogobot_motor_set(id, cmd);
}
static void apply_robot_direction(bool backward){
    if(!enable_backward_dir){
        pogobot_motor_dir_set(motorL, mydata->motor_dir_left);
        pogobot_motor_dir_set(motorR, mydata->motor_dir_right);
        return;
    }
    if(backward){
        pogobot_motor_dir_set(motorL, (mydata->motor_dir_left==0)?1:0);
        pogobot_motor_dir_set(motorR, (mydata->motor_dir_right==0)?1:0);
    }else{
        pogobot_motor_dir_set(motorL, mydata->motor_dir_left);
        pogobot_motor_dir_set(motorR, mydata->motor_dir_right);
    }
}

// -------------------------------
// Init
// -------------------------------
void user_init(void){
#ifndef SIMULATOR
    printf("setup ok\n");
#endif
    srand(pogobot_helper_getRandSeed());

    main_loop_hz = 60;
    mydata->dt_s = (main_loop_hz>0)? (1.f/(float)main_loop_hz) : (1.f/60.f);

    max_nb_processed_msg_per_tick = 0;
    msg_rx_fn = NULL;
    msg_tx_fn = NULL;
    error_codes_led_idx = 3;

    uint8_t dir_mem[3];
    pogobot_motor_dir_mem_get(dir_mem);
    mydata->motor_dir_left  = dir_mem[1];
    mydata->motor_dir_right = dir_mem[0];

    // random heading in [-pi,pi)
    float u = rand()/(float)RAND_MAX;
    mydata->theta = u*2.f*(float)M_PI - (float)M_PI;

    mydata->omega_cmd = 0.f; mydata->diff_frac = 0.f;
    mydata->has_spare = 0;

    mydata->still_ticks = 0;
    mydata->in_escape = false;
    mydata->escape_left_s = 0.f;
    mydata->spin_sign = +1;

    pogobot_stopwatch_reset(&mydata->timer_it);
    apply_robot_direction(false);
    pogobot_led_setColor(0,0,255); // blue
}

// -------------------------------
// Step
// -------------------------------
void user_step(void){
    const float dt = mydata->dt_s;

    // --- Simple IMU "moving?" test
    float acc[3], gyro[3];
    pogobot_imu_read(acc, gyro);
    bool moving = (fabsf(gyro[0])>gyro_treshold) || (fabsf(gyro[1])>gyro_treshold) ||
                  (fabsf(gyro[2])>gyro_treshold) || (fabsf(acc[0])>acc_treshold)   ||
                  (fabsf(acc[1])>acc_treshold)   || (fabsf(acc[2])>acc_treshold);

    // --- Enter escape if not moving for a few ticks
    if(!mydata->in_escape){
        if(!moving){
            //printf("Not moving! %f, %f, %f \n", fabsf(gyro[2]), fabsf(acc[0]), fabsf(acc[2]));
            if(++mydata->still_ticks >= stuck_ticks_to_trigger && enable_backward_dir){
                mydata->still_ticks = 0;
                mydata->in_escape = true;
                mydata->escape_left_s = backoff_duration_s;
                mydata->spin_sign = (rand()&1) ? +1 : -1;   // random spin direction
                apply_robot_direction(true);                // reverse motors
                pogobot_led_setColor(255,0,0);              // red while escaping
            }
        }else{
            //printf("Moving! %f, %f, %f \n", fabsf(gyro[2]), fabsf(acc[0]), fabsf(acc[2]));
            mydata->still_ticks = 0;
        }
    }

    // --- ABP heading update (always running)
    // dtheta ~ N(0, 2 D_r dt)
    float sigma = sqrtf(fmaxf(0.f, 2.f*abp_rot_diffusion*dt));
    float dtheta = sigma * rand_normal(mydata);
    mydata->theta = wrap_angle(mydata->theta + dtheta);
    mydata->omega_cmd = dtheta / fmaxf(dt, 1e-6f);

    // --- Build motor commands
    float v = abp_speed_frac;
    float diff;

    if(mydata->in_escape){
        // Reverse + near in-place spin
        // Use the same speed; spin by saturating differential
        diff = mydata->spin_sign * (0.9f * v);
        mydata->escape_left_s -= dt;
        if(mydata->escape_left_s <= 0.f){
            mydata->in_escape = false;
            apply_robot_direction(false);     // forward again
            pogobot_led_setColor(0,0,255);    // back to blue
        }
    }else{
        // Normal ABP differential from omega
        diff = abp_turn_gain * mydata->omega_cmd;
        diff = clampf(diff, -v, v);
    }

    float uL = clampf(v - diff, 0.f, 1.f);
    float uR = clampf(v + diff, 0.f, 1.f);
    set_motor_frac(motorL, uL);
    set_motor_frac(motorR, uR);

    // Optional tiny debug (once a second on #0)
    if(pogobot_ticks % 60 == 0 && pogobot_helper_getid()==0){
        printf("[ABPmin] v=%.2f Dr=%.2f esc=%d uL=%.2f uR=%.2f\n",
               v, abp_rot_diffusion, (int)mydata->in_escape, uL, uR);
    }
}

#ifdef SIMULATOR
void global_setup(){
    init_from_configuration(abp_speed_frac);
    init_from_configuration(abp_rot_diffusion);
    init_from_configuration(abp_turn_gain);

    init_from_configuration(enable_backward_dir);
    init_from_configuration(backoff_duration_s);
    init_from_configuration(stuck_ticks_to_trigger);
    init_from_configuration(gyro_treshold);
    init_from_configuration(acc_treshold);
}
#endif

int main(void){
    pogobot_init();
#ifndef SIMULATOR
    printf("init ok\n");
#endif
    pogobot_start(user_init, user_step);
    SET_CALLBACK(callback_global_setup, global_setup);
    return 0;
}

// MODELINE "{{{1
// vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
// vim:foldmethod=marker
