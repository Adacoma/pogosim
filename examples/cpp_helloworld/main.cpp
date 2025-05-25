// Main include for pogobots, both for real robots and for simulations
extern "C" {
#include "pogobase.h"
}
// Use C headers instead of C++ headers for embedded compatibility
#include <stdint.h>

// C++20 constexpr helper functions
constexpr uint32_t calculate_blink_period(uint32_t base_time) noexcept {
    return base_time / 10000;
}

constexpr bool should_turn_left(uint32_t time_value) noexcept {
    return (time_value % 2) == 0;
}

// Simple POD array wrapper for embedded use (compatible with C linkage)
template<typename T, size_t N>
struct PODArray {
    T data[N];
    
    // C++20 constexpr member functions
    constexpr T& operator[](size_t idx) noexcept { 
        return data[idx]; 
    }
    
    constexpr const T& operator[](size_t idx) const noexcept { 
        return data[idx]; 
    }
    
    constexpr size_t size() const noexcept { 
        return N; 
    }
};

// "Global" variables should be inserted within the USERDATA struct.
// /!\  In simulation, don't declare non-const global variables outside this struct, 
//      elsewise they will be shared among all agents (and this is not realistic).
// Note: Using simple C-compatible struct to avoid C++/C linkage issues
typedef struct {
    // Put all global variables you want here.
    uint8_t data_foo[8];  // Using plain C array for compatibility
    time_reference_t timer_it;
} USERDATA;

// Call this macro in the same file (.h or .c) as the declaration of USERDATA
DECLARE_USERDATA(USERDATA);

// Don't forget to call this macro in the main .c file of your project (only once!)
REGISTER_USERDATA(USERDATA);
// Now, members of the USERDATA struct can be accessed through mydata->MEMBER. E.g. mydata->data_foo

// C++20 designated initializers for configuration
struct RobotConfig {
    int main_loop_frequency = 60;
    int max_messages_per_tick = 0;
    int error_led_index = 3;
};

constexpr RobotConfig default_config{
    .main_loop_frequency = 60,
    .max_messages_per_tick = 0,
    .error_led_index = 3
};

// Template function for LED color setting with compile-time validation
template<uint8_t R, uint8_t G, uint8_t B>
constexpr void set_led_color_safe() noexcept {
    static_assert(R <= 255 && G <= 255 && B <= 255, "RGB values must be 0-255");
    pogobot_led_setColor(R, G, B);
}

// C++20 wrapper class for robot operations
class RobotController {
public:
    // C++20 constexpr static methods
    static void turn_left() noexcept {
        set_led_color_safe<0, 0, 255>();  // Blue LED
        pogobot_motor_set(motorL, motorFull);
        pogobot_motor_set(motorR, motorStop);
    }
    
    static void turn_right() noexcept {
        set_led_color_safe<255, 0, 0>();  // Red LED
        pogobot_motor_set(motorL, motorStop);
        pogobot_motor_set(motorR, motorFull);
    }
    
    static void execute_behavior(uint32_t current_time) noexcept {
        const auto time_period = calculate_blink_period(current_time);
        
        if (should_turn_left(time_period)) {
            turn_left();
        } else {
            turn_right();
        }
    }
};

// Init function. Called once at the beginning of the program (cf 'pogobot_start' call in main())
// Note: Remove extern "C" as the header already declares these functions
void user_init(void) {
#ifndef SIMULATOR
    printf("setup ok\n");
#endif

    // Init timer
    pogobot_stopwatch_reset(&mydata->timer_it);

    // Use our C++20 config struct
    constexpr auto config = default_config;
    
    // Set main loop frequency, message sending frequency, message processing frequency
    main_loop_hz = config.main_loop_frequency;  // Call the 'user_step' function 60 times per second
    max_nb_processed_msg_per_tick = config.max_messages_per_tick;
    
    // Specify functions to send/transmit messages. See the "hanabi" example to see message sending/processing in action!
    msg_rx_fn = nullptr;    // If nullptr, no reception of message
    msg_tx_fn = nullptr;    // If nullptr, don't send any message

    // Set led index to show error codes (e.g. time overflows)
    error_codes_led_idx = config.error_led_index; // Default value, negative values to disable
}

// Step function. Called continuously at each step of the pogobot main loop
void user_step(void) {
    // C++20 constexpr if for compile-time optimization
    if constexpr (true) {  // This will be optimized away
        if (pogobot_ticks % 1000 == 0 && pogobot_helper_getid() == 0) {     // Only print messages for robot 0
            printf(" HELLO WORLD C++20 !!!   Robot ID: %d   Current time: %lums  Timer: %luµs   pogobot_ticks: %lu\n",
                    pogobot_helper_getid(),
                    current_time_milliseconds(),
                    pogobot_stopwatch_get_elapsed_microseconds(&mydata->timer_it),
                    pogobot_ticks       // Increased by one at each execution of user_step
                    );
        }
    }

    // Use our C++20 robot controller class
    RobotController::execute_behavior(static_cast<uint32_t>(current_time_milliseconds()));

    // C array access (compatible with C linkage)
    mydata->data_foo[0] = 42;
    
    // Demonstrate C++20 features that work on embedded
    if constexpr (sizeof(uint8_t) == 1) {
        // This will always be true, but shows constexpr if usage
        mydata->data_foo[1] = static_cast<uint8_t>(pogobot_ticks & 0xFF);
    }
}

// Entrypoint of the program
extern "C" int main(void) {
    pogobot_init();     // Initialization routine for the robots
#ifndef SIMULATOR
    printf("init ok\n");
#endif

    // Specify the user_init and user_step functions
    pogobot_start(user_init, user_step);
    return 0;
}

// MODELINE "{{{1
// vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
// vim:foldmethod=marker
