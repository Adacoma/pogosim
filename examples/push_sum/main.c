/**
* @file main.c
* @brief Neighbor-counted push-sum gossip controller for PogoBot swarms.
*
* @details
* This controller implements a local, decentralized variant of the push-sum
* gossip algorithm for estimating the average of scalar values initially held
* by a swarm of robots. Push-sum is a classical randomized gossip protocol for
* aggregate computation in distributed systems. It was introduced and analyzed
* by Kempe, Dobra and Gehrke in the context of gossip-based computation of
* aggregate information, where each node repeatedly splits and forwards a
* conserved quantity through local communication. Unlike standard average
* consensus schemes based on doubly-stochastic weights, push-sum is especially
* useful on directed or time-varying communication graphs, because each node
* only needs to ensure that the mass it sends is split according to its own
* outgoing communication structure. Related gossip and consensus algorithms are
* discussed, for instance, in Boyd et al. on randomized gossip algorithms and
* in later work on push-sum / subgradient-push methods for time-varying
* directed graphs.
*
* In push-sum, each robot i maintains two local variables:
*
* ```
  s_i : a scalar "mass" initially equal to the value held by robot i;
  ```
* ```
  w_i : a positive weight, initially equal to 1.
  ```
*
* The local estimate of the global average is then:
*
* ```
  estimate_i = s_i / w_i.
  ```
*
* If the algorithm preserves the total scalar mass sum_i s_i and the total
* weight mass sum_i w_i, and if the communication graph remains sufficiently
* connected over time, then all ratios s_i / w_i converge toward the global
* average of the initial scalar values:
*
* ```
  average = (sum_i s_i(0)) / (sum_i w_i(0)).
  ```
*
* In this example, the initial scalar value is chosen as the robot UID cast to
* a float:
*
* ```
  s_i(0) = robot_id;
  ```
* ```
  w_i(0) = 1.
  ```
*
* Therefore, if the swarm contains robots with IDs 0, 1, ..., N-1, the expected
* final estimate is approximately:
*
* ```
  (N - 1) / 2.
  ```
*
* A subtle but important implementation issue arises on PogoBot-like robots:
* the radio primitive used here is omnidirectional broadcast. If a robot simply
* broadcasts half of its current (s,w) pair while keeping the other half, then
* that same half-mass is duplicated once per receiver. With d neighbors, the
* sender keeps one half but d neighbors each receive another half, so the total
* mass is no longer conserved. This is not a valid push-sum update; it can make
* s and w grow exponentially, eventually producing infinities or NaNs.
*
* To avoid this, this controller first estimates the sender's current number of
* live neighbors using a lightweight heartbeat mechanism. Each robot
* periodically broadcasts a heartbeat packet containing only its UID. Received
* heartbeat packets are stored in a small neighbor table together with the time
* at which they were last observed and the IR direction from which they were
* received. At each control step, entries older than the configurable max_age
* window are removed. The remaining number of unique neighbor IDs is used as
* the current live neighbor count d_i.
*
* During a push-sum transmission, robot i then splits its current mass into
* d_i + 1 equal shares:
*
* ```
  share_s = s_i / (d_i + 1);
  ```
* ```
  share_w = w_i / (d_i + 1).
  ```
*
* The robot keeps one share locally:
*
* ```
  s_i <- share_s;
  ```
* ```
  w_i <- share_w;
  ```
*
* and broadcasts one identical share to its current neighbors. If the live
* neighbor count matches the actual number of robots that receive the push-sum
* packet, the update is mass-conserving:
*
* ```
  kept by sender:      1 share
  ```
* ```
  received by d_i:     d_i shares
  ```
* ```
  total:               d_i + 1 shares = original mass.
  ```
*
* This turns the broadcast primitive into an approximate implementation of a
* column-stochastic push-sum update. The estimate s_i / w_i can then converge
* across the swarm while avoiding artificial mass creation.
*
* The controller uses a unified packet format with a packet type field:
*
* ```
  PACKET_HEARTBEAT : neighbor-discovery packet containing the sender UID;
  ```
* ```
  PACKET_PUSH_SUM  : push-sum packet containing sender UID, round, s, w.
  ```
*
* Heartbeat packets maintain the sliding-window neighbor table. Push-sum
* packets add the received share to the local (s,w) variables. A small
* per-neighbor round cache is used to ignore duplicate push-sum packets from
* the same sender and round. This is useful because, depending on the simulator
* or hardware radio model, the same logical broadcast may be observed multiple
* times or through multiple IR faces.
*
* The implementation also includes basic numerical safety checks. Non-finite
* values are never sent, and received push-sum messages containing NaN, infinity
* or invalid weights are ignored. These checks are not part of the mathematical
* algorithm itself; they are defensive programming measures to prevent a single
* corrupted message from poisoning the whole swarm.
*
* LEDs provide a qualitative visualization of the current estimate. The estimate
* is mapped modulo 10 through the qualitative_colormap helper, allowing visual
* inspection of convergence during simulation or robot experiments. In the
* simulator, the controller exports the current s value, w value, estimate,
* push-sum round counter, neighbor counts and neighbor list for offline
* analysis.
*
* Important limitations:
*
* 1. Exact mass conservation assumes that the number of live neighbors counted
* during the heartbeat window is equal to the number of robots that actually
* receive the following push-sum packet.
*
* 2. On real robots, IR communication may be lossy, asymmetric or intermittent.
* If a robot counts a neighbor but that neighbor misses the push-sum packet,
* some mass is effectively lost. If a robot fails to count a receiver that
* nevertheless receives the packet, mass is duplicated. In simulation this
* approximation may be acceptable, but on physical robots exact average
* consensus would require acknowledgements, retransmission, pairwise
* communication, correction terms, or a gossip variant explicitly designed
* for packet loss.
*
* 3. The heartbeat window max_age trades off reactivity and stability. A short
* window tracks moving neighborhoods quickly but may underestimate neighbors
* when packets are missed. A long window is more stable but may overestimate
* neighbors after topology changes. Both cases affect mass conservation.
*
* 4. The algorithm is intended as a clear baseline demonstration of push-sum
* gossip on a local robot communication graph, not as a fully robust
* production-grade distributed averaging protocol for unreliable wireless
* networks.
*
* References:
*
* * D. Kempe, A. Dobra and J. Gehrke, "Gossip-Based Computation of Aggregate
* Information", Proceedings of the 44th IEEE Symposium on Foundations of
* Computer Science, 2003.
*
* * S. Boyd, A. Ghosh, B. Prabhakar and D. Shah, "Randomized Gossip
* Algorithms", IEEE Transactions on Information Theory, 52(6), 2006.
*
* * A. Nedic and A. Olshevsky, "Distributed Optimization over Time-Varying
* Directed Graphs", IEEE Transactions on Automatic Control, 60(3), 2015.
*
* * K. I. Tsianos, S. Lawlor and M. G. Rabbat, "Push-Sum Distributed Dual
* Averaging for Convex Optimization", IEEE Conference on Decision and
* Control, 2012.
*/



#include "pogobase.h"
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>

/* -------------------------------------------------------------------------- */
/* Global constants configurable from YAML via global_setup                   */
/* -------------------------------------------------------------------------- */

/** Push-sum broadcast frequency (Hz). */
uint32_t gossip_hz = 15;

/** Heartbeat broadcast frequency (Hz), used only for neighbour discovery. */
uint32_t heartbeat_hz = 30;

/** Age threshold (ms) after which a neighbour entry is considered obsolete. */
uint32_t max_age = 1000;

/** Set true to enable the run-and-tumble demo motion. */
bool moving_robots = false;

/* -------------------------------------------------------------------------- */
/* Compile-time parameters                                                    */
/* -------------------------------------------------------------------------- */

#define MAX_NEIGHBORS 40U
#define MIN_WEIGHT    1e-12f

#define MSG_TYPE_HEARTBEAT 1U
#define MSG_TYPE_PUSH_SUM  2U

/* -------------------------------------------------------------------------- */
/* Message definition                                                         */
/* -------------------------------------------------------------------------- */

typedef struct __attribute__((__packed__)) {
    uint8_t  type;       /**< MSG_TYPE_HEARTBEAT or MSG_TYPE_PUSH_SUM. */
    uint16_t sender_id;  /**< UID of the sender. */
    uint32_t round;      /**< Sender's push-sum round; 0 for heartbeat. */
    float    s;          /**< One push-sum share of s; 0 for heartbeat. */
    float    w;          /**< One push-sum share of w; 0 for heartbeat. */
} push_sum_packet_t;

#define MSG_SIZE ((uint16_t)sizeof(push_sum_packet_t))

/* -------------------------------------------------------------------------- */
/* Data structures                                                            */
/* -------------------------------------------------------------------------- */

typedef struct {
    uint16_t id;
    uint32_t last_seen_ms;
    uint8_t  direction;
    uint32_t last_push_round;  /**< Used to ignore duplicate push packets. */
} neighbor_t;

typedef struct {
    /* Neighbour counting state. */
    neighbor_t neighbors[MAX_NEIGHBORS];
    uint8_t    nb_neighbors;
    uint8_t    dir_counts[IR_RX_COUNT];
    uint8_t    total_neighbors;

    /* Push-sum state. */
    float    s;
    float    w;
    float    estimate;
    uint32_t round;

    /* Timers. */
    uint32_t last_heartbeat_ms;
    uint32_t last_push_ms;
} USERDATA;

DECLARE_USERDATA(USERDATA);
REGISTER_USERDATA(USERDATA);

/* -------------------------------------------------------------------------- */
/* Utility helpers                                                            */
/* -------------------------------------------------------------------------- */

static uint32_t period_ms_from_hz(uint32_t hz) {
    if(hz == 0U) {
        return UINT_MAX;
    }
    uint32_t const p = 1000U / hz;
    return (p == 0U) ? 1U : p;
}

static bool local_state_is_finite(void) {
    return isfinite(mydata->s) && isfinite(mydata->w) && mydata->w > MIN_WEIGHT;
}

/**
 * @brief Drop entries older than max_age from the neighbour table.
 */
static void purge_old_neighbors(void) {
    uint32_t const now = current_time_milliseconds();

    for(int8_t i = (int8_t)mydata->nb_neighbors - 1; i >= 0; --i) {
        if(now - mydata->neighbors[i].last_seen_ms > max_age) {
            mydata->neighbors[i] = mydata->neighbors[mydata->nb_neighbors - 1U];
            mydata->nb_neighbors--;
        }
    }
}

/**
 * @brief Refresh directional counts and total neighbour count.
 */
static void recalc_dir_counts(void) {
    memset(mydata->dir_counts, 0, sizeof(mydata->dir_counts));

    for(uint8_t i = 0; i < mydata->nb_neighbors; ++i) {
        uint8_t const d = mydata->neighbors[i].direction;
        if(d < IR_RX_COUNT) {
            mydata->dir_counts[d]++;
        }
    }

    mydata->total_neighbors = 0;
    for(uint8_t d = 0; d < IR_RX_COUNT; ++d) {
        mydata->total_neighbors += mydata->dir_counts[d];
    }
}

/**
 * @brief Find or add a neighbour entry, then refresh its timestamp/direction.
 *
 * @return Pointer to the entry, or NULL if the table is full.
 */
static neighbor_t *find_or_add_neighbor(uint16_t sender, uint8_t direction) {
    uint8_t idx;
    for(idx = 0; idx < mydata->nb_neighbors; ++idx) {
        if(mydata->neighbors[idx].id == sender) {
            break;
        }
    }

    if(idx == mydata->nb_neighbors) {
        if(mydata->nb_neighbors >= MAX_NEIGHBORS) {
            return NULL;
        }
        memset(&mydata->neighbors[idx], 0, sizeof(mydata->neighbors[idx]));
        mydata->neighbors[idx].id = sender;
        mydata->nb_neighbors++;
    }

    mydata->neighbors[idx].last_seen_ms = current_time_milliseconds();
    mydata->neighbors[idx].direction    = direction;
    return &mydata->neighbors[idx];
}

/**
 * @brief Map estimate (mod 10) to LED colour.
 */
static void set_led_from_estimate(float estimate) {
    uint8_t r, g, b;

    if(!isfinite(estimate)) {
        pogobot_led_setColors(255, 0, 0, 0);
        return;
    }

    uint8_t const idx = (uint8_t)fmodf(fabsf(estimate), 10.0f);
    qualitative_colormap(idx, &r, &g, &b);
    pogobot_led_setColors(r, g, b, 0);
}

/**
 * @brief Optional simple periodic tumbling motion.
 */
static void tumbling_motion(void) {
    if((current_time_milliseconds() / 10000U) % 2U == 0U) {
        pogobot_motor_set(motorL, motorFull);
        pogobot_motor_set(motorR, motorStop);
    } else {
        pogobot_motor_set(motorL, motorStop);
        pogobot_motor_set(motorR, motorFull);
    }
}

/* -------------------------------------------------------------------------- */
/* Messaging callbacks                                                        */
/* -------------------------------------------------------------------------- */

static bool send_heartbeat(uint32_t now) {
    push_sum_packet_t pkt = {
        .type      = MSG_TYPE_HEARTBEAT,
        .sender_id = pogobot_helper_getid(),
        .round     = 0U,
        .s         = 0.0f,
        .w         = 0.0f
    };

    pogobot_infrared_sendShortMessage_omni((uint8_t *)&pkt, MSG_SIZE);
    mydata->last_heartbeat_ms = now;
    return true;
}

/**
 * @brief Send one mass-conserving push-sum share.
 *
 * If d neighbours are currently known, split local mass into d+1 shares.
 * The sender keeps one share and broadcasts one share. If d == 0, no push-sum
 * packet is sent, because broadcasting while assuming zero receivers would
 * create mass whenever a robot unexpectedly receives the packet.
 */
static bool send_push_sum(uint32_t now) {
    mydata->last_push_ms = now;

    purge_old_neighbors();
    recalc_dir_counts();

    uint8_t const d = mydata->total_neighbors;
    if(d == 0U) {
        return false;
    }

    if(!local_state_is_finite()) {
        return false;
    }

    float const denom   = (float)d + 1.0f;
    float const share_s = mydata->s / denom;
    float const share_w = mydata->w / denom;

    if(!isfinite(share_s) || !isfinite(share_w) || share_w <= 0.0f) {
        return false;
    }

    /* Keep exactly one share. The d other shares are represented by the
     * single broadcast packet received once by each of the d neighbours. */
    mydata->s = share_s;
    mydata->w = share_w;
    mydata->round++;

    push_sum_packet_t pkt = {
        .type      = MSG_TYPE_PUSH_SUM,
        .sender_id = pogobot_helper_getid(),
        .round     = mydata->round,
        .s         = share_s,
        .w         = share_w
    };

    pogobot_infrared_sendShortMessage_omni((uint8_t *)&pkt, MSG_SIZE);
    return true;
}

/**
 * @brief Scheduler callback: interleave push-sum packets and heartbeats.
 */
bool send_message(void) {
    uint32_t const now = current_time_milliseconds();

    uint32_t const push_period = period_ms_from_hz(gossip_hz);
    uint32_t const hb_period   = period_ms_from_hz(heartbeat_hz);

    /* Prefer push-sum when it is due, but fall through to heartbeat if no
     * push-sum packet can be sent, e.g. because no neighbours are known yet. */
    if(now - mydata->last_push_ms >= push_period) {
        if(send_push_sum(now)) {
            return true;
        }
    }

    if(now - mydata->last_heartbeat_ms >= hb_period) {
        return send_heartbeat(now);
    }

    return false;
}

/**
 * @brief Receive heartbeat or push-sum packets.
 */
void process_message(message_t *mr) {
    if(mr->header.payload_length < MSG_SIZE) {
        return;
    }

    uint8_t const dir = mr->header._receiver_ir_index;
    if(dir >= IR_RX_COUNT) {
        return;
    }

    push_sum_packet_t const *pkt = (push_sum_packet_t const *)mr->payload;
    uint16_t const sender = pkt->sender_id;

    if(sender == pogobot_helper_getid()) {
        return;
    }

    if(pkt->type != MSG_TYPE_HEARTBEAT && pkt->type != MSG_TYPE_PUSH_SUM) {
        return;
    }

    neighbor_t *neighbor = find_or_add_neighbor(sender, dir);
    if(neighbor == NULL) {
        return;
    }

    if(pkt->type == MSG_TYPE_HEARTBEAT) {
        return;
    }

    /* Ignore duplicate receptions of the same sender/round. This protects the
     * mass balance if the same omni packet is delivered through multiple faces. */
    if(pkt->round == 0U || neighbor->last_push_round == pkt->round) {
        return;
    }
    neighbor->last_push_round = pkt->round;

    if(!isfinite(pkt->s) || !isfinite(pkt->w) || pkt->w <= 0.0f) {
        return;
    }

    mydata->s += pkt->s;
    mydata->w += pkt->w;
}

/* -------------------------------------------------------------------------- */
/* Controller logic                                                           */
/* -------------------------------------------------------------------------- */

void user_init(void) {
    srand(pogobot_helper_getRandSeed());

    memset(mydata, 0, sizeof(*mydata));

    mydata->s = (float)pogobot_helper_getid();
    mydata->w = 1.0f;
    mydata->estimate = mydata->s;
    mydata->round = 0U;

    pogobot_infrared_set_power(2);
    main_loop_hz                  = 60;
    max_nb_processed_msg_per_tick = 8;
    percent_msgs_sent_per_ticks   = 50;
    msg_rx_fn = process_message;
    msg_tx_fn = send_message;

    error_codes_led_idx = 3;
    set_led_from_estimate(mydata->estimate);
}

void user_step(void) {
    purge_old_neighbors();
    recalc_dir_counts();

    if(mydata->w > MIN_WEIGHT && isfinite(mydata->s) && isfinite(mydata->w)) {
        mydata->estimate = mydata->s / mydata->w;
    }

    set_led_from_estimate(mydata->estimate);

    /* Optional: show per-direction neighbour counts on LEDs 1..IR_RX_COUNT. */
    uint8_t r, g, b;
    for(uint8_t d = 0; d < IR_RX_COUNT; ++d) {
        uint8_t const idx = (mydata->dir_counts[d] > 9U) ? 9U : mydata->dir_counts[d];
        qualitative_colormap(idx, &r, &g, &b);
        pogobot_led_setColors(r, g, b, d + 1U);
    }

    if(moving_robots) {
        tumbling_motion();
    }

    if(pogobot_ticks % 1000U == 0U && pogobot_helper_getid() == 0U) {
        printf("[Round %lu] est = %.4f (s=%.6g, w=%.6g, neighbors=%u)\n",
               (unsigned long)mydata->round,
               mydata->estimate,
               mydata->s,
               mydata->w,
               mydata->total_neighbors);
    }
}

/* -------------------------------------------------------------------------- */
/* Simulator hooks                                                            */
/* -------------------------------------------------------------------------- */
#ifdef SIMULATOR

void global_setup(void) {
    init_from_configuration(gossip_hz);
    init_from_configuration(heartbeat_hz);
    init_from_configuration(max_age);
    init_from_configuration(moving_robots);
}

char *get_neighbors_ids_string(void) {
    static char buf[MAX_NEIGHBORS * 6U] = {0};
    size_t pos = 0;

    buf[0] = '\0';

    for(uint8_t i = 0; i < mydata->nb_neighbors; ++i) {
        int const n = snprintf(&buf[pos], sizeof(buf) - pos, "%u", mydata->neighbors[i].id);
        if(n < 0 || (size_t)n >= sizeof(buf) - pos) {
            buf[sizeof(buf) - 1U] = '\0';
            break;
        }
        pos += (size_t)n;

        if(i < mydata->nb_neighbors - 1U && pos < sizeof(buf) - 1U) {
            buf[pos++] = ',';
            buf[pos] = '\0';
        }
    }

    return buf;
}

static void create_data_schema(void) {
    data_add_column_double("s_value");
    data_add_column_double("w_value");
    data_add_column_double("estimate");
    data_add_column_int32("round");
    data_add_column_int8("dir0_neighbors");
    data_add_column_int8("dir1_neighbors");
    data_add_column_int8("dir2_neighbors");
    data_add_column_int8("dir3_neighbors");
    data_add_column_int8("total_neighbors");
    data_add_column_string("neighbors_list");
}

static void export_data(void) {
    data_set_value_double("s_value", mydata->s);
    data_set_value_double("w_value", mydata->w);
    data_set_value_double("estimate", mydata->estimate);
    data_set_value_int32("round", mydata->round);
    data_set_value_int8("dir0_neighbors", mydata->dir_counts[0]);
    data_set_value_int8("dir1_neighbors", mydata->dir_counts[1]);
    data_set_value_int8("dir2_neighbors", mydata->dir_counts[2]);
    data_set_value_int8("dir3_neighbors", mydata->dir_counts[3]);
    data_set_value_int8("total_neighbors", mydata->total_neighbors);
    data_set_value_string("neighbors_list", get_neighbors_ids_string());
}

#endif /* SIMULATOR */

/* -------------------------------------------------------------------------- */
/* Entry point                                                                */
/* -------------------------------------------------------------------------- */

int main(void) {
    pogobot_init();
    pogobot_start(user_init, user_step);

    SET_CALLBACK(callback_global_setup,       global_setup);
    SET_CALLBACK(callback_create_data_schema, create_data_schema);
    SET_CALLBACK(callback_export_data,        export_data);

    return 0;
}
