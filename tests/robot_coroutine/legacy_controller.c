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

static void require(int condition, const char *message, long long actual, long long expected) {
    /* Keep exit 23 and Release checks, but identify the failing C/C++ invariant
     * and its observed value in native Windows and installed-consumer logs. */
    if (!condition) {
        fprintf(stderr, "Legacy controller: %s (robot=%u, actual=%lld, expected=%lld)\n",
                message, (unsigned)pogobot_helper_getid(), actual, expected);
        exit(23);
    }
}

static void nested_sleep(uint16_t saved_id) {
    time_reference_t local;
    pogobot_stopwatch_reset(&local);
    msleep(1);
    uint16_t resumed_id = pogobot_helper_getid();
    require(saved_id == resumed_id, "Robot identity changed across sleep", resumed_id, saved_id);
    require(mydata->id == saved_id, "USERDATA identity changed across sleep", mydata->id, saved_id);
    int32_t elapsed = pogobot_stopwatch_get_elapsed_microseconds(&local);
    require(elapsed >= 1000, "Nested stopwatch below minimum duration", elapsed, 1000);
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
    require(mydata->steps == 90, "Controller step count mismatch", mydata->steps, 90);
    int32_t elapsed = pogobot_stopwatch_get_elapsed_microseconds(&mydata->lifetime);
    require(elapsed >= 90000, "Lifetime stopwatch below minimum duration", elapsed, 90000);
}

int main(void) {
    pogobot_init();
    /* Exercise both public registration forms in C/C++ and installed builds. */
    if (pogobot_helper_getid() == 0) {
        pogobot_start(user_init, user_step);
    } else {
        pogobot_start(user_init, user_step, "robots");
    }
    SET_CALLBACK(callback_robot_end, robot_end);
    return 0;
}
