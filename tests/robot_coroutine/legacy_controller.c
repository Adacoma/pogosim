#include "pogobase.h"
#include <stdlib.h>
#include <string.h>

/* This controller uses ordinary legacy compiler/link flags, no Context headers,
 * and no -lboost_context. Its archive must provide the new runtime itself. */
typedef struct {
    uint16_t id;
    unsigned steps;
    time_reference_t lifetime;
} USERDATA;
DECLARE_USERDATA(USERDATA);
REGISTER_USERDATA(USERDATA);

static void require(int condition) {
    if (!condition) exit(23); /* Unlike assert(), also checked in Release. */
}

static void nested_sleep(uint16_t saved_id) {
    time_reference_t local;
    pogobot_stopwatch_reset(&local);
    msleep(1);
    require(saved_id == pogobot_helper_getid() && mydata->id == saved_id);
    require(pogobot_stopwatch_get_elapsed_microseconds(&local) >= 1000);
}

void user_init(void) {
    memset(mydata, 0, sizeof(*mydata));
    mydata->id = pogobot_helper_getid();
    main_loop_hz = 0;
    pogobot_stopwatch_reset(&mydata->lifetime);
}

void user_step(void) {
    ++mydata->steps;
    nested_sleep(mydata->id);
}

static void robot_end(void) {
    require(mydata->steps == 90);
    require(pogobot_stopwatch_get_elapsed_microseconds(&mydata->lifetime) >= 90000);
}

int main(void) {
    pogobot_init();
    pogobot_start(user_init, user_step);
    SET_CALLBACK(callback_robot_end, robot_end);
    return 0;
}
