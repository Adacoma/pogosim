/**
 * @file neighbor_counter.c
 * @brief Per‑direction neighbour counter for PogoBot (C99).
 *
 * Robots periodically broadcast a minimal *heartbeat* packet that contains only
 * their unique identifier. Each received packet is stored together with the
 * **infra‑red face** (0‥3) from which it originated and a timestamp. A sliding
 * window, controlled by the global @c max_age parameter (in milliseconds),
 * automatically removes stale entries, so every robot permanently knows:
 *
 * | Symbol                        | Description                                       |
 * |------------------------------ |-------------------------------------------------- |
 * | @c dir_counts[4]              | Live neighbours currently visible on each IR face |
 * | @c total_neighbors            | Sum of the four directional counts                |
 *
 * The overall crowd size is mapped to a body‑LED colour through the firmware
 * helper ::qualitative_colormap (10‑colour qualitative palette). The mapping is
 * direct: the neighbour count (mod 10) is passed as‑is to the helper and then
 * fed straight to ::pogobot_led_setColor without any brightness scaling.
 *
 */

#include "pogobase.h"
#include <stdint.h>
#include <string.h>
#include <stdbool.h>

/* -------------------------------------------------------------------------- */
/* External symbols                                                           */
/* -------------------------------------------------------------------------- */
/** Age threshold (ms) after which a neighbour entry is considered obsolete.  */
uint32_t max_age = 1200;

/** Set to true to have moving robots.                                       */
bool moving_robots = false;

/* -------------------------------------------------------------------------- */
/* Compile‑time parameters                                                    */
/* -------------------------------------------------------------------------- */
#define MAX_NEIGHBORS           20U    /**< Hard cap on neighbour table size  */
#define HEARTBEAT_HZ            10U    /**< Broadcast frequency (Hz).         */
#define HEARTBEAT_PERIOD_MS (1000U / HEARTBEAT_HZ) /**< Broadcast period.     */

/* -------------------------------------------------------------------------- */
/* Message definition                                                         */
/* -------------------------------------------------------------------------- */
/**
 * @struct heartbeat_t
 * @brief Wire format of a heartbeat message.
 *
 * Packed to guarantee byte‑exact transmission both on the real robots and in
 * the simulator.
 */
typedef struct __attribute__((__packed__)) {
    uint16_t sender_id;          /**< Unique identifier of the sender. */
} heartbeat_t;

/** Size of the on‑wire heartbeat message in bytes (compile‑time constant). */
#define MSG_SIZE ((uint16_t)sizeof(heartbeat_t))

/* -------------------------------------------------------------------------- */
/* Colour names (for the start‑up table)                                      */
/* -------------------------------------------------------------------------- */
static char const *const color_names[10] = {
    "Red", "Green", "Blue", "Yellow", "Magenta",
    "Cyan", "Purple", "Olive", "Teal", "Gray"
};

/* -------------------------------------------------------------------------- */
/* Data structures                                                            */
/* -------------------------------------------------------------------------- */
/**
 * @struct neighbor_t
 * @brief Runtime descriptor for a single neighbour.
 */
typedef struct {
    uint16_t id;                 /**< Neighbour UID.                    */
    uint32_t last_seen_ms;       /**< Timestamp of last packet.         */
    uint8_t  direction;          /**< IR face index (0‥3).              */
} neighbor_t;

/**
 * @struct USERDATA
 * @brief All mutable state for this robot.
 */
typedef struct {
    neighbor_t neighbors[MAX_NEIGHBORS]; /**< Neighbour table.            */
    uint8_t    nb_neighbors;             /**< Current table size.         */

    uint8_t dir_counts[IR_RX_COUNT];   /**< Per‑face neighbour counts.  */
    uint8_t total_neighbors;             /**< Sum of the four counts.     */

    uint32_t last_heartbeat_ms;          /**< Timestamp of last broadcast.*/
} USERDATA;

/* Make @p mydata globally available. */
DECLARE_USERDATA(USERDATA);
REGISTER_USERDATA(USERDATA);

/* -------------------------------------------------------------------------- */
/* Utility functions                                                          */
/* -------------------------------------------------------------------------- */

/**
 * @brief Drop entries older than ::max_age from the neighbour table.
 */
static void purge_old_neighbors(void) {
    uint32_t const now = current_time_milliseconds();
    for(int8_t i = (int8_t)mydata->nb_neighbors - 1; i >= 0; --i) {
        if(now - mydata->neighbors[i].last_seen_ms > max_age) {
            mydata->neighbors[i] = mydata->neighbors[mydata->nb_neighbors - 1];
            mydata->nb_neighbors--;
        }
    }
}

/**
 * @brief Refresh directional counts and the global total.
 */
static void recalc_dir_counts(void) {
    memset(mydata->dir_counts, 0, sizeof(mydata->dir_counts));

    for(uint8_t i = 0; i < mydata->nb_neighbors; ++i) {
        uint8_t d = mydata->neighbors[i].direction;
        if(d < IR_RX_COUNT) {
            mydata->dir_counts[d]++;
        }
    }

    mydata->total_neighbors = 0;
    for(uint8_t d = 0; d < IR_RX_COUNT; ++d) {
        mydata->total_neighbors += mydata->dir_counts[d];
    }
}

/* -------------------------------------------------------------------------- */
/* Messaging callbacks                                                        */
/* -------------------------------------------------------------------------- */
/**
 * @brief Periodically broadcast a heartbeat.
 *
 * Called by the PogoBot scheduler when it wants to send a packet. The function
 * ensures we transmit at most one heartbeat every ::HEARTBEAT_PERIOD_MS.
 *
 * @return @c true if a message has been queued for transmission, @c false
 *         otherwise.
 */
bool send_message(void) {
    uint32_t const now = current_time_milliseconds();
    if(now - mydata->last_heartbeat_ms < HEARTBEAT_PERIOD_MS) {
        return false; /* Too early */
    }

    heartbeat_t hb = { .sender_id = pogobot_helper_getid() };
    pogobot_infrared_sendShortMessage_omni((uint8_t *)&hb, MSG_SIZE);
    mydata->last_heartbeat_ms = now;
    return true;
}

/**
 * @brief Handle an incoming packet.
 *
 * @param[in] mr Pointer to the message wrapper provided by the firmware.
 */
void process_message(message_t *mr) {
    if(mr->header.payload_length < MSG_SIZE) {
        return; /* Not a heartbeat */
    }

    uint8_t dir = mr->header._receiver_ir_index;
    if(dir >= IR_RX_COUNT) {
        return; /* Invalid face index */
    }

    heartbeat_t const *hb = (heartbeat_t const *)mr->payload;
    uint16_t sender = hb->sender_id;
    if(sender == pogobot_helper_getid()) {
        return; /* Ignore own echo */
    }

    /* Search for an existing entry. */
    uint8_t idx;
    for(idx = 0; idx < mydata->nb_neighbors; ++idx) {
        if(mydata->neighbors[idx].id == sender) {
            break;
        }
    }

    /* Append if new and space available. */
    if(idx == mydata->nb_neighbors) {
        if(mydata->nb_neighbors >= MAX_NEIGHBORS) {
            return; /* Table full */
        }
        mydata->nb_neighbors++;
    }

    mydata->neighbors[idx].id           = sender;
    mydata->neighbors[idx].last_seen_ms = current_time_milliseconds();
    mydata->neighbors[idx].direction    = dir;
}

/** Helper: print the neighbour‑count→colour lookup table. */
static void print_colormap_table(void) {
    printf("\nNeighbour count → LED colour (qualitative_colormap)\n");
    printf("  Count | Colour  | RGB\n");
    printf(" -------+---------+-------------\n");
    for(uint8_t i = 0; i < 10; ++i) {
        uint8_t r, g, b;
        qualitative_colormap(i, &r, &g, &b);
        printf("  %5u | %-7s| (%3u,%3u,%3u)\n", i, color_names[i], r, g, b);
    }
    printf("\nCounts ≥10 repeat modulo 10.\n\n");
}

/* -------------------------------------------------------------------------- */
/* Control logic                                                              */
/* -------------------------------------------------------------------------- */
/** Initialise user state and configure the framework. */
void user_init(void) {
    memset(mydata, 0, sizeof(*mydata));

    /* Radio power, scheduler and callbacks. */
    pogobot_infrared_set_power(2);
    main_loop_hz                  = 60;
    max_nb_processed_msg_per_tick = 3;
    percent_msgs_sent_per_ticks   = 50;
    msg_rx_fn = process_message;
    msg_tx_fn = send_message;

    /* LED #3 reserved for runtime error codes (‑1 to disable). */
    error_codes_led_idx = 3;

#ifdef SIMULATOR
    if (pogobot_helper_getid() == 0) {
        print_colormap_table();
    }
#else
    print_colormap_table();
#endif
}


void tumbling_motion(void) {
    if ((uint32_t)(current_time_milliseconds() / 10000) % 2 == 0) {
        pogobot_motor_set(motorL, motorFull);
        pogobot_motor_set(motorR, motorStop);
    } else {
        pogobot_motor_set(motorL, motorStop);
        pogobot_motor_set(motorR, motorFull);
    }
}


/** Main loop executed at @c main_loop_hz. */
void user_step(void) {
    purge_old_neighbors();
    recalc_dir_counts();

    if (moving_robots)
        tumbling_motion();

    uint8_t r, g, b;
    uint8_t idx = (mydata->total_neighbors > 9) ? 9 : mydata->total_neighbors;
    qualitative_colormap(idx, &r, &g, &b);
    pogobot_led_setColors(r, g, b, 0);

    /* per-direction colours on LEDs 1-4 */
    for(uint8_t d = 0; d < IR_RX_COUNT; ++d) {
        uint8_t idx = (mydata->dir_counts[d] > 9U) ? 9U : mydata->dir_counts[d];
        qualitative_colormap(idx, &r, &g, &b);
        pogobot_led_setColors(r, g, b, d + 1);  /* LED id = face + 1 */
    }
}

/* -------------------------------------------------------------------------- */
/* Simulator hooks (optional)                                                 */
/* -------------------------------------------------------------------------- */
#ifdef SIMULATOR

// Function called once to initialize global values (e.g. configuration-specified constants)
void global_setup() {
    init_from_configuration(max_age);
    init_from_configuration(moving_robots);
}

/** Register custom data columns for CSV export. */
static void create_data_schema(void) {
    data_add_column_int8("dir0_neighbors");
    data_add_column_int8("dir1_neighbors");
    data_add_column_int8("dir2_neighbors");
    data_add_column_int8("dir3_neighbors");
    data_add_column_int8("total_neighbors");
}

/** Periodically push current telemetry to the simulator. */
static void export_data(void) {
    data_set_value_int8("dir0_neighbors", mydata->dir_counts[0]);
    data_set_value_int8("dir1_neighbors", mydata->dir_counts[1]);
    data_set_value_int8("dir2_neighbors", mydata->dir_counts[2]);
    data_set_value_int8("dir3_neighbors", mydata->dir_counts[3]);
    data_set_value_int8("total_neighbors", mydata->total_neighbors);
}
#endif /* SIMULATOR */

/* -------------------------------------------------------------------------- */
/* Entry point                                                                */
/* -------------------------------------------------------------------------- */
int main(void) {
    pogobot_init();     // Initialization routine for the robots
    // Specify the user_init and user_step functions
    pogobot_start(user_init, user_step);
    // Specify the callback functions. Only called by the simulator.
    //  In particular, they serve to add data fields to the exported data files
    SET_CALLBACK(callback_global_setup, global_setup);              // Called once to initialize global values (e.g. configuration-specified constants)
    SET_CALLBACK(callback_create_data_schema, create_data_schema);  // Called once on each robot to specify the data format
    SET_CALLBACK(callback_export_data, export_data);                // Called at each configuration-specified period (e.g. every second) on each robot to register exported data
    return 0;
}

