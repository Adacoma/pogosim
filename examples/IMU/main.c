// main.c – run-and-tumble demo with IMU plausibility check
#include "pogobase.h"
#include <math.h>

typedef enum { PHASE_RUN, PHASE_TUMBLE } phase_state_t;

uint32_t run_duration_min    = 200;
uint32_t run_duration_max    = 1200;
uint32_t tumble_duration_min = 100;
uint32_t tumble_duration_max = 1100;
bool     enable_backward_dir = false;
float    test_vect[4]        = {1.f, 2.f, 3.f, 4.f};

/* ───────── per-robot data ───────── */
typedef struct {
    uint8_t  data_foo[8];
    time_reference_t timer_it;
    phase_state_t  phase;
    uint32_t phase_start_time;
    uint32_t phase_duration;
    uint8_t  tumble_direction;      // 0 = left, 1 = right
    uint8_t  motor_dir_left;
    uint8_t  motor_dir_right;
} USERDATA;

DECLARE_USERDATA(USERDATA);
REGISTER_USERDATA(USERDATA);

/* ───────── helpers ───────── */
static uint32_t get_run_duration(void) {
    return run_duration_min + (rand() %
        (run_duration_max - run_duration_min + 1));
}

static uint32_t get_tumble_duration(void) {
    return tumble_duration_min + (rand() %
        (tumble_duration_max - tumble_duration_min + 1)); 
}

static void set_robot_direction(bool backward) {
    if(!enable_backward_dir) return;
    if(backward) {
        pogobot_motor_dir_set(motorL, (mydata->motor_dir_left  ? 0 : 1));
        pogobot_motor_dir_set(motorR, (mydata->motor_dir_right ? 0 : 1));
    } else {
        pogobot_motor_dir_set(motorL, mydata->motor_dir_left);
        pogobot_motor_dir_set(motorR, mydata->motor_dir_right);
    }
}

/* print IMU & temperature, and say if it matches the *current* phase */
static const char *phase_name[] = {"RUN", "TUMBLE"};

static void log_imu_check_transition(phase_state_t from, phase_state_t to) {
    float acc[3], gyro[3];
    pogobot_imu_read(acc, gyro);
    float temp = pogobot_imu_readTemp();

    /* heuristic threshold: 2 rad s⁻¹ separates “still” from “turning” */
    const float thresh = 2.0f;
    bool ok = (from == PHASE_RUN) ?
              (fabsf(gyro[2]) < thresh) :   /* RUN should not spin */
              (fabsf(gyro[2]) > thresh);    /* TUMBLE should spin */

    printf("[Robot %u] %s → %s | "
           "gyro={%.2f,%.2f,%.2f} rad/s | "
           "acc={%.2f,%.2f,%.2f} m/s² | "
           "temp=%.1f °C | %s\n",
           pogobot_helper_getid(),
           phase_name[from], phase_name[to],
           gyro[0], gyro[1], gyro[2],
           acc[0],  acc[1],  acc[2],
           temp,
           ok ? "IMU OK" : "IMU MISMATCH");
}

/* ───────── user initialisation ───────── */
void user_init(void) {
    srand(pogobot_helper_getRandSeed());
    pogobot_stopwatch_reset(&mydata->timer_it);

    main_loop_hz                  = 60;
    max_nb_processed_msg_per_tick = 0;
    msg_rx_fn = NULL;
    msg_tx_fn = NULL;
    error_codes_led_idx           = 3;

    uint8_t dir_mem[3];
    pogobot_motor_dir_mem_get(dir_mem);
    mydata->motor_dir_left  = dir_mem[1];
    mydata->motor_dir_right = dir_mem[0];

    mydata->phase            = PHASE_TUMBLE;        // start tumbling
    mydata->phase_start_time = current_time_milliseconds();
    mydata->phase_duration   = get_tumble_duration();
    mydata->tumble_direction = rand() % 2;

    if(pogobot_helper_getid() == 0) {
        printf("Test global values: (%d,%d,%d,%d)\n",
               (int)test_vect[0], (int)test_vect[1],
               (int)test_vect[2], (int)test_vect[3]);
    }
}

/* ───────── user loop ───────── */
void user_step(void) {
    uint32_t now = current_time_milliseconds();

    if(now - mydata->phase_start_time >= mydata->phase_duration) {

        /* decide next phase FIRST */
        phase_state_t old_phase = mydata->phase;
        phase_state_t new_phase = (old_phase == PHASE_RUN)
                                    ? PHASE_TUMBLE
                                    : PHASE_RUN;

        /* log IMU against the phase we are LEAVING */
        if(pogobot_helper_getid() == 0)
            log_imu_check_transition(old_phase, new_phase);

        /* update phase bookkeeping */
        mydata->phase            = new_phase;
        mydata->phase_start_time = now;
        mydata->phase_duration   =
            (new_phase == PHASE_RUN) ? get_run_duration()
                                     : get_tumble_duration();

        if(new_phase == PHASE_TUMBLE)
            mydata->tumble_direction = rand() % 2;
        else
            set_robot_direction((rand() % 2) == 0);   // maybe backward
    }

    /* motor commands according to *current* phase */
    if(mydata->phase == PHASE_RUN) {
        pogobot_led_setColor(0,255,0);
        pogobot_motor_set(motorL, motorFull);
        pogobot_motor_set(motorR, motorFull);
    } else { /* TUMBLE */
        pogobot_led_setColor(255,0,0);
        if(mydata->tumble_direction == 0) {
            pogobot_motor_set(motorL, motorStop);
            pogobot_motor_set(motorR, motorFull);
        } else {
            pogobot_motor_set(motorL, motorFull);
            pogobot_motor_set(motorR, motorStop);
        }
    }
}

/* ───────── simulator YAML hook ───────── */
#ifdef SIMULATOR
void global_setup(void) {
    init_from_configuration(run_duration_min);
    init_from_configuration(run_duration_max);
    init_from_configuration(tumble_duration_min);
    init_from_configuration(tumble_duration_max);
    init_from_configuration(enable_backward_dir);
    init_array_from_configuration(test_vect);
}
#endif

int main(void) {
    pogobot_init();
    pogobot_start(user_init, user_step);
#ifdef SIMULATOR
    SET_CALLBACK(callback_global_setup, global_setup);
#endif
    return 0;
}

// MODELINE "{{{1
// vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
// vim:foldmethod=marker
