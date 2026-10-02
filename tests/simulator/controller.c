#include "pogobase.h"
#include <string.h>

/* Real C controller exercises the existing API; no test hooks enter the core. */
typedef struct { uint16_t id; uint32_t steps; uint32_t received; } USERDATA;
DECLARE_USERDATA(USERDATA);
REGISTER_USERDATA(USERDATA);
static uint32_t regression_mode = 0;

void regression_check(int, const char*);
void regression_register_robot(void);
void regression_initialize(uint32_t);
void regression_end(uint32_t);
void regression_schema(uint32_t);
void regression_export(uint32_t);

static void setup(void) {
    /* Keep configuration initialization in a callback, not user_init(). */
    init_from_configuration(regression_mode);
}

static void receive(message_t* message) {
    regression_check(message->header._sender_id == 0 &&
        message->header.payload_length == 2 && message->payload[0] == 0xab && message->payload[1] == 0xcd,
        "Message payload/header changed in transit");
    ++mydata->received;
}

static void initialize(void) {
    memset(mydata, 0, sizeof(*mydata));
    mydata->id = pogobot_helper_getid();
    main_loop_hz = 0;
    msg_rx_fn = receive;
    regression_initialize(regression_mode);
}

static void step(void) {
    regression_check(mydata->id == pogobot_helper_getid(), "Controller USERDATA was swapped");
    ++mydata->steps;
    if (mydata->id == 0 && mydata->steps == 2) {
        uint8_t payload[2] = {0xab, 0xcd};
        regression_check(pogobot_infrared_sendLongMessage_omniGen(payload, sizeof(payload)) == 0,
                         "C infrared send failed");
    }
}

static void create_schema(void) {
    regression_schema(regression_mode);
    data_add_column_int32("controller_steps");
    data_add_column_string("controller_tag");
}

static void export_data(void) {
    regression_export(regression_mode);
    data_set_value_int32("controller_steps", (int32_t)mydata->steps);
    data_set_value_string("controller_tag", mydata->id == 0 ? "source" : "receiver");
}

static void end(void) {
    regression_check(mydata->steps >= 25 && mydata->steps <= 32, "Controller did not run for the requested duration");
    if (mydata->id == 1) {
        if (regression_mode == 0 || regression_mode == 3)
            regression_check(mydata->received > 0, "Expected infrared delivery did not occur");
        if (regression_mode == 1 || regression_mode == 4)
            regression_check(mydata->received == 0, "Dropped/out-of-range message was delivered");
    }
    regression_end(regression_mode);
}

int main(void) {
    pogobot_init();
    regression_register_robot();
    /* Both fixture categories intentionally share this controller. */
    _pogobot_start(initialize, step, get_current_robot_category());
    SET_CALLBACK(callback_global_setup, setup);
    SET_CALLBACK(callback_robot_end, end);
    SET_CALLBACK(callback_create_data_schema, create_schema);
    SET_CALLBACK(callback_export_data, export_data);
    return 0;
}
