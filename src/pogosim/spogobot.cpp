
#include <iostream>
#include <thread>
#include <chrono>
#include <cstdarg>
#include <sstream>
#include <string>
#include <cstdint>
#include "SDL2_gfxPrimitives.h"

#include "spogobot.h"
#include "robot.h"
#include "pogosim.h"
#include "simulator.h"
#include "version.h"


/************* GLOBALS *************/ // {{{1


uint8_t const _selected_power = 1;

/************* time_reference_t *************/ // {{{1

void time_reference_t::reset() {
    enabled = false;
    if (current_robot != nullptr)
        current_robot->register_stop_watch(this);

    start_time = current_robot->current_time_microseconds;
    auto const duration = start_time - sim_starting_time_microseconds;
    hardware_value_at_time_origin = duration;
    elapsed_ms = 0;
}

void time_reference_t::enable() {
    enabled = true;
    start_time = current_robot->current_time_microseconds;
    //glogger->debug("ENABLE!! {}", start_time);
}

void time_reference_t::disable() {
    enabled = false;
    get_elapsed_microseconds();
    //glogger->debug("DISABLE!! {}", start_time);
}

uint32_t time_reference_t::get_elapsed_microseconds() {
    auto const duration = current_robot->current_time_microseconds - start_time;
    elapsed_ms += duration;
    return elapsed_ms;
}


void time_reference_t::add_elapsed_microseconds(uint64_t microseconds) {
    elapsed_ms += microseconds;
}

void time_reference_t::offset_origin_microseconds(uint64_t microseconds) {
    start_time -= microseconds;
}


// XXX Move to robot.cpp ?
uint64_t get_current_time_microseconds() {
//    return current_robot->current_time_microseconds;
    // Get the current time in microseconds since epoch
    auto const now = std::chrono::system_clock::now();
    auto const duration = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch());

    // Convert
    uint64_t microseconds = static_cast<uint64_t>(duration.count());
    return microseconds;
}



/************* SIMULATED POGOLIB *************/ // {{{1

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

std::string log_current_robot() {
    std::ostringstream oss;
    oss << "[ROBOT #" << current_robot->id << "] ";
    return oss.str();
}

void pogobot_init() {
    glogger->debug("{} Pogobot initialized successfully.", log_current_robot());
}


/**
 * ## Infrared communication API Functions
 */


void pogobot_infrared_ll_init( void ) {
    // Do nothing ...
}

void pogobot_infrared_update( void ) {
    // Do nothing ...
}

int pogobot_infrared_message_available( void ) {
    return current_robot->messages.size() > 0;
}

void pogobot_infrared_recover_next_message( message_t *mes ) {
    if (!pogobot_infrared_message_available()) {
        glogger->warn("The function 'pogobot_infrared_recover_next_message( message_t *mes )' should *only* be called if 'pogobot_infrared_message_available()' returns >0.. Ignoring.");
        return;
    }
    message_t m = current_robot->messages.front();
    current_robot->messages.pop();
    memcpy(mes, &m, sizeof(message_t));
}

void pogobot_infrared_clear_message_queue( void ) {
    // Clear the queue
    while (!current_robot->messages.empty()) {
        current_robot->messages.pop();
    }
}


void pogobot_infrared_set_power( uint8_t power ) {
    // Do nothing ...
}

uint32_t pogobot_infrared_sendRawLongMessage( message_t *const message ) {
    if ( message->header.payload_length > MAX_PAYLOAD_SIZE_BYTES ) {
        return 1;
    }

    message->header._packet_type = ir_t_user; // User packets have type 16.
    message->header._sender_id = pogobot_helper_getid();
    message->header._receiver_ir_index = 0;

    // Rmq from original pogolib:
    // TODO: this currently emits packet via several IR at the same time (or
    // does it?). What this should to is emit packet several times, via one IR
    // at a time, with field _emitting_power_list reflecting which TX is used.
    // This will be necessary later to have robots detect each other and their
    // relative orientations.

    ir_direction dir = ir_all;
    bool directions[4]{false};
    if (get_infrared_emitter_dir(ir_front, message->header._emitting_power_list) > 0) {
        directions[0] = true;
    }
    if (get_infrared_emitter_dir(ir_right, message->header._emitting_power_list) > 0) {
        directions[1] = true;
    }
    if (get_infrared_emitter_dir(ir_back, message->header._emitting_power_list) > 0) {
        directions[2] = true;
    }
    if (get_infrared_emitter_dir(ir_left, message->header._emitting_power_list) > 0) {
        directions[3] = true;
    }

    if (directions[0] && directions[1] && directions[2] && directions[3]) {
        dir = ir_all;
    } else if (directions[0]) {
        dir = ir_front;
    } else if (directions[1]) {
        dir = ir_right;
    } else if (directions[2]) {
        dir = ir_back;
    } else if (directions[3]) {
        dir = ir_left;
    }

    //glogger->debug("sendRawLongMessage: {} ({}, {}, {}, {})", (uint8_t)dir, directions[0], directions[1],directions[2],directions[3]);

    current_robot->send_to_neighbors(dir, message);
    return 0;
}

uint32_t pogobot_infrared_sendRawShortMessage( ir_direction dir, short_message_t *const message ) {
    if ( message->header.payload_length > MAX_PAYLOAD_SIZE_BYTES ) {
        return 1;
    }

    message->header._packet_type = ir_t_short; // User packets have type 3.

    current_robot->send_to_neighbors(dir, message);
    return 0;
}

uint32_t pogobot_infrared_sendLongMessage_uniSpe( ir_direction dir, uint8_t *message, uint16_t message_size ) {
    message_t m;
    m.header._emitting_power_list =
        _selected_power << ( pogobot_infrared_emitter_width_bits * dir );
    m.header.payload_length = message_size;
    m.header._sender_ir_index = dir;
    memcpy( m.payload, message, message_size );
    return pogobot_infrared_sendRawLongMessage( &m );
}

uint32_t pogobot_infrared_sendLongMessage_omniGen( uint8_t *message, uint16_t message_size ) {
    message_t m;
    m.header._emitting_power_list =
        pogobot_infrared_emitting_power_list(_selected_power, _selected_power, _selected_power, _selected_power);
    m.header.payload_length = message_size;
    m.header._sender_ir_index = ir_all;
    memcpy( m.payload, message, message_size );
    return pogobot_infrared_sendRawLongMessage( &m );
}

uint32_t pogobot_infrared_sendLongMessage_omniSpe( uint8_t *message, uint16_t message_size ) {
    int i = 0;
    int error = 0;
    message_t m;
    m.header.payload_length = message_size;
    memcpy( m.payload, message, message_size );

    for ( i = 0; i < IR_RX_COUNT; i++ ) {
        m.header._emitting_power_list =
            _selected_power << ( pogobot_infrared_emitter_width_bits * i );
        m.header._sender_ir_index = i;
        error += pogobot_infrared_sendRawLongMessage( &m );
    }

    return error;
}

uint32_t pogobot_infrared_sendShortMessage_uni( ir_direction dir, uint8_t *message, uint16_t message_size ) {
    short_message_t m;
    m.header.payload_length = message_size;
    memcpy( m.payload, message, message_size );
    return pogobot_infrared_sendRawShortMessage( dir, &m );
}

uint32_t pogobot_infrared_sendShortMessage_omni( uint8_t *message, uint16_t message_size ) {
    short_message_t m;
    m.header.payload_length = message_size;
    memcpy( m.payload, message, message_size );
    return pogobot_infrared_sendRawShortMessage( ir_all, &m );
}

void pogobot_infrared_reset_receiver_error_counter( void ) {
    // Do nothing ...
}


/**
 * ## RGB LED API
 */

void pogobot_led_setColor(const uint8_t r, const uint8_t g, const uint8_t b) {
    glogger->debug("{} LED Color {} set to R:{} G:{} B:{}", log_current_robot(), 0, r, g, b);
    current_robot->leds[0] = {.r=r, .g=g, .b=b};
}

void pogobot_led_setColors(const uint8_t r, const uint8_t g, const uint8_t b, uint8_t id) {
    glogger->debug("{} LED Color {} set to R:{} G:{} B:{}", log_current_robot(), id, r, g, b);
    current_robot->leds[id] = {.r=r, .g=g, .b=b};
}


/**
 * ## Photosensors API Values
 */

int16_t pogobot_photosensors_read( uint8_t sensor_number ) {
    //return simulation->get_current_light_value();
    b2Vec2 pos = current_robot->get_position();
    return simulation->get_light_map()->get_light_level_at(pos.x * VISUALIZATION_SCALE, pos.y * VISUALIZATION_SCALE);
}


/**
 * ## IMU API
 */

void pogobot_imu_read( float *acc, float *gyro ) {
    glogger->warn("Function 'pogobot_imu_read' is not implemented yet!");
}

float pogobot_imu_readTemp( void ) {
    glogger->warn("Function 'pogobot_imu_readTemp' is not implemented yet!");
    return 0.0f;
}



/**
 * ## Battery API
 */
int16_t pogobot_battery_voltage_read( void ) {
    // Assumes the battery is full
    return std::numeric_limits<int16_t>::max();
}



/**
 * ## Motors API Values
 */
void pogobot_motor_power_set( motor_id motor, uint16_t value ) {
    glogger->debug("{} Motor {} set to speed {}", log_current_robot(), static_cast<uint8_t>(motor), value);
    if (value > motorFull)
        value = motorFull;
    current_robot->set_motor(motor, value);
}


void pogobot_motor_set ( motor_id motor, uint16_t value ) {
    pogobot_motor_power_set(motor, value);
}

uint32_t pogobot_motor_dir_current_status( void ) {
    glogger->warn("Function 'pogobot_motor_dir_current_status' is not implemented yet!");
    uint32_t value = 0xFFFFFFFF;
    return value;
}

int8_t pogobot_motor_dir_mem_get( uint8_t *p_directions ) {
    std::memcpy(p_directions, &current_robot->motor_dir_mem, sizeof(current_robot->motor_dir_mem));
    return 0;
}

int8_t pogobot_motor_dir_mem_set( uint8_t *p_directions) {
    std::memcpy(&current_robot->motor_dir_mem, p_directions, sizeof(current_robot->motor_dir_mem));
    return 0;
}

void pogobot_motor_dir_set( motor_id motor, uint8_t value ) {
    if (motor > motorB)
        return;
    current_robot->motor_dir[motor] = value;
}

uint8_t pogobot_motor_power_mem_get( uint16_t *p_powers ) {
    std::memcpy(p_powers, &current_robot->motor_power_mem, sizeof(current_robot->motor_power_mem));
    return 0;
}

uint8_t pogobot_motor_power_mem_set( uint16_t *p_powers ) {
    std::memcpy(&current_robot->motor_power_mem, p_powers, sizeof(current_robot->motor_power_mem));
    return 0;
}



/**
 * ## Helper API
 */

uint16_t pogobot_helper_getid(void) {
    return current_robot->id;
}

int16_t pogobot_helper_getRandSeed( void ) {
    // Define the distribution for int16_t range
    std::uniform_int_distribution<int16_t> dis(std::numeric_limits<int16_t>::min(),
                                               std::numeric_limits<int16_t>::max());

    // Generate a random number of type int16_t
    return dis(rnd_gen);
}

void pogobot_helper_print_version( void ) {
    pogosim_printf(" Pogolib Version : %s\n", POGOLIB_RELEASE_VERSION);
}



/**
 * ## Time API
 */

void pli_timer_sleep_stopwatch_init( void ) {
    // Do nothing ...
}


void pogobot_stopwatch_reset(time_reference_t *stopwatch) {
    stopwatch->reset();
}

int32_t pogobot_stopwatch_lap( time_reference_t *stopwatch ) {
    auto const res = stopwatch->get_elapsed_microseconds();
    stopwatch->reset();
    return res;
}

int32_t pogobot_stopwatch_get_elapsed_microseconds(time_reference_t *stopwatch) {
    return stopwatch->get_elapsed_microseconds();
}

void pogobot_stopwatch_offset_origin_microseconds( time_reference_t *stopwatch, int32_t microseconds_offset ) {
    stopwatch->offset_origin_microseconds(microseconds_offset);
}

void pogobot_timer_init( time_reference_t *timer, int32_t microseconds_to_go ) {
    pogobot_stopwatch_reset( timer );
    pogobot_stopwatch_offset_origin_microseconds( timer, microseconds_to_go );
}

int32_t pogobot_timer_get_remaining_microseconds( time_reference_t *timer ) {
    uint32_t const now = current_robot->current_time_microseconds;
    int32_t const remain = now - timer->start_time;
    return remain;
}

bool pogobot_timer_has_expired( time_reference_t *timer ) {
    return pogobot_timer_get_remaining_microseconds(timer) < 0;
}

void pogobot_timer_wait_for_expiry( time_reference_t *timer ) {
    int64_t const remain = pogobot_timer_get_remaining_microseconds(timer);
    if (remain <= 0)
        return;
    // Simulate sleep
    current_robot->sleep_µs(remain);
}

void pogobot_timer_offset_origin_microseconds( time_reference_t *timer, int32_t microseconds_offset ) {
    timer->offset_origin_microseconds(microseconds_offset);
}

#pragma GCC diagnostic pop


/************* General API *************/ // {{{1

void msleep(int milliseconds) {
    //std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
    glogger->debug("{} Sleeping for {} ms", log_current_robot(), milliseconds);
    current_robot->sleep_µs(static_cast<uint64_t>(milliseconds) * 1000);
}

// Helper function to convert va_list arguments to a string
std::string _format_args_to_string(const char* format, va_list args) {
    va_list args_copy;
    va_copy(args_copy, args); // Copy the va_list to avoid modifying the original

    // Use vsnprintf to determine the size of the formatted string
    int size = std::vsnprintf(nullptr, 0, format, args_copy);
    va_end(args_copy);

    if (size < 0) {
        throw std::runtime_error("Error formatting string.");
    }

    // Create a string with the exact size needed
    std::string result(size, '\0');
    std::vsnprintf(&result[0], size + 1, format, args); // Format into the string

    return result;
}

void pogosim_printf(const char* format, ...) {
    // Static buffer to hold incomplete message parts across calls.
    static std::string message_buffer;

    va_list args;
    va_start(args, format);

    // Convert va_list arguments to a formatted string.
    std::string formatted_message = _format_args_to_string(format, args);
    va_end(args);

    // Append the new message fragment to the buffer.
    message_buffer += formatted_message;

    // Process the buffer: for every newline, log a complete line.
    size_t newline_pos = 0;
    while ((newline_pos = message_buffer.find('\n')) != std::string::npos) {
        // Extract the complete line (excluding the newline character)
        std::string line = message_buffer.substr(0, newline_pos);

        // Combine with the robot log prefix.
        std::string final_message = log_current_robot() + "[PRINTF] " + line;
        robotlogger->info(final_message);

        // Remove the logged line (and its trailing newline) from the buffer.
        message_buffer.erase(0, newline_pos + 1);
    }
}

int pogosim_putchar(int ch) {
    pogosim_printf("%c", ch);
    return ch;
}

void enable_data_export(void) {
    simulation->current_robot_enable_data_logging = true;
}

void disable_data_export(void) {
    simulation->current_robot_enable_data_logging = false;
}

void data_add_column_int8(char const* name) {
    simulation->get_data_logger()->add_field(name, arrow::int8());
}
void data_add_column_int16(char const* name) {
    simulation->get_data_logger()->add_field(name, arrow::int16());
}
void data_add_column_int32(char const* name) {
    simulation->get_data_logger()->add_field(name, arrow::int32());
}
void data_add_column_int64(char const* name) {
    simulation->get_data_logger()->add_field(name, arrow::int64());
}
void data_add_column_double(char const* name) {
    simulation->get_data_logger()->add_field(name, arrow::float64());
}
void data_add_column_string(char const* name) {
    simulation->get_data_logger()->add_field(name, arrow::utf8());
}
void data_add_column_bool(char const* name) {
    simulation->get_data_logger()->add_field(name, arrow::boolean());
}

void data_set_value_int8(char const* name, int8_t value) {
    simulation->get_data_logger()->set_value(name, value);
}
void data_set_value_int16(char const* name, int16_t value) {
    simulation->get_data_logger()->set_value(name, value);
}
void data_set_value_int32(char const* name, int32_t value) {
    simulation->get_data_logger()->set_value(name, value);
}
void data_set_value_int64(char const* name, int64_t value) {
    simulation->get_data_logger()->set_value(name, value);
}
void data_set_value_double(char const* name, double value) {
    simulation->get_data_logger()->set_value(name, value);
}
void data_set_value_string(char const* name, char const* value) {
    simulation->get_data_logger()->set_value(name, value);
}
void data_set_value_bool(char const* name, bool value) {
    simulation->get_data_logger()->set_value(name, value);
}

static std::string const parameters_config_key = "parameters";

void init_bool_from_configuration(bool* var, char const* name, bool const default_value) {
    *var = simulation->get_config()[parameters_config_key][name].get(default_value);
}

void init_double_from_configuration(double* var, char const* name, double const default_value) {
    *var = simulation->get_config()[parameters_config_key][name].get(default_value);
}

void init_float_from_configuration(float* var, char const* name, float const default_value) {
    //*var = std::stof(simulation->get_config()[name].get(std::to_string(default_value)));
    *var = simulation->get_config()[parameters_config_key][name].get(default_value);
}

void init_int32_from_configuration(int32_t* var, char const* name, int32_t const default_value) {
    *var = simulation->get_config()[parameters_config_key][name].get(default_value);
}

void init_uint32_from_configuration(uint32_t* var, char const* name, uint32_t const default_value) {
    *var = simulation->get_config()[parameters_config_key][name].get(default_value);
}

void init_int16_from_configuration(int16_t* var, char const* name, int16_t const default_value) {
    *var = simulation->get_config()[parameters_config_key][name].get(default_value);
}

void init_uint16_from_configuration(uint16_t* var, char const* name, uint16_t const default_value) {
    *var = simulation->get_config()[parameters_config_key][name].get(default_value);
}

void init_int8_from_configuration(int8_t* var, char const* name, int8_t const default_value) {
    *var = simulation->get_config()[parameters_config_key][name].get(default_value);
}

void init_uint8_from_configuration(uint8_t* var, char const* name, uint8_t const default_value) {
    *var = simulation->get_config()[parameters_config_key][name].get(default_value);
}

void init_float_array_from_configuration(float* var, char const* name, size_t const size) {
    auto key = simulation->get_config()[parameters_config_key][name];
    if (key.exists()) {
        std::vector<float> data = key.get<std::vector<float>>();
        if (data.size() == 0) {
            return;
        } else if (data.size() < size) {
            throw std::runtime_error("Configuration key '" + std::string(name) +
                    "' does not have a correct size (current size: " + std::to_string(data.size()) +", expected: " + std::to_string(size) + ")'");
        } else {
            float* data_array = data.data();
            std::memcpy(var, data_array, size * sizeof(float));
        }
    }
}

void init_double_array_from_configuration(double* var, char const* name, size_t const size) {
    auto key = simulation->get_config()[parameters_config_key][name];
    if (key.exists()) {
        std::vector<double> data = key.get<std::vector<double>>();
        if (data.size() == 0) {
            return;
        } else if (data.size() < size) {
            throw std::runtime_error("Configuration key '" + std::string(name) +
                    "' does not have a correct size (current size: " + std::to_string(data.size()) +", expected: " + std::to_string(size) + ")'");
        } else {
            double* data_array = data.data();
            std::memcpy(var, data_array, size * sizeof(double));
        }
    }
}


// MODELINE "{{{1
// vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
// vim:foldmethod=marker
