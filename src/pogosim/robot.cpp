
#include <iostream>
#include <thread>
#include <chrono>
#include <cstdarg>
#include <sstream>
#include <string>
#include <cstdint>
#include <cmath>
#include "SDL2_gfxPrimitives.h"
#include <unordered_map>
#include <bit>
#include <ranges>


#include "robot.h"
#include "spogobot.h"
#include "pogosim.h"
#include "simulator.h"
#include "colormaps.h"


/************* GLOBALS *************/ // {{{1

//Robot* current_robot;
PogobotObject* current_robot;

//std::chrono::time_point<std::chrono::system_clock> sim_starting_time;
uint64_t sim_starting_time_microseconds;


// Inline function to calculate the normalized color values
inline uint8_t adjust_color(uint8_t const value) {
    if (value == 0) {
        return 0; // Use 0 for value 0
    } else if (value <= 25) {
        // Map values from 1-25 to 100-210
        return static_cast<uint8_t>(100 + (static_cast<float>(value - 1) / 24.0f * 110.0f));
    } else {
        return 210; // Use 210 for values > 25
    }
}


/************* MsgSuccessRate *************/ // {{{1

DynamicMsgSuccessRate::DynamicMsgSuccessRate(double alpha, double beta, double gamma, double delta)
    : alpha_(alpha), beta_(beta), gamma_(gamma), delta_(delta) {}

double DynamicMsgSuccessRate::operator()(double msg_size, double p_send, double cluster_size) const {
    return 1.0 / (1.0 + (alpha_ * std::pow(msg_size, beta_) *
                                  std::pow(p_send, gamma_) *
                                  std::pow(cluster_size, delta_)));
}

ConstMsgSuccessRate::ConstMsgSuccessRate(double value)
    : const_value_(value) {}

double ConstMsgSuccessRate::operator()(double /*msg_size*/, double /*p_send*/, double /*cluster_size*/) const {
    return const_value_;
}


MsgSuccessRate* msg_success_rate_factory(Configuration const& config) {
    std::string type = to_lowercase(config["type"].get(std::string("realistic")));
    if (type == "realistic") {
        float const alpha  = 0.000004f;
        float const beta   = 2.8096f;
        float const gamma  = 2.3807f;
        float const delta  = 1.2457f;
        return new DynamicMsgSuccessRate(alpha, beta, gamma, delta);
    } else if (type == "static") {
        float const rate = config["rate"].get(0.9);
        return new ConstMsgSuccessRate(rate);
    } else if (type == "dynamic") {
        float const alpha  = config["alpha"].get(0.000004f);
        float const beta   = config["beta"].get(2.8096f);
        float const gamma  = config["gamma"].get(2.3807f);
        float const delta  = config["delta"].get(1.2457f);
        return new DynamicMsgSuccessRate(alpha, beta, gamma, delta);
    } else {
        throw std::runtime_error("Unknown msg_success_rate type '" + type + "'.");
    }
}



/************* Pogobot Objects *************/ // {{{1


PogobotObject::PogobotObject(uint16_t _id, float _x, float _y,
       ObjectGeometry& geom, b2WorldId world_id,
       size_t _userdatasize,
       float _communication_radius,
       std::unique_ptr<MsgSuccessRate> _msg_success_rate,
       float _temporal_noise_stddev,
       float _linear_damping, float _angular_damping,
       float _density, float _friction, float _restitution,
       float _max_linear_speed, float _max_angular_speed,
       float _linear_noise_stddev, float _angular_noise_stddev,
       std::string const& _category)
    : PhysicalObject(_id, _x, _y, geom, world_id,
      _linear_damping, _angular_damping,
      _density, _friction, _restitution, _category),
    communication_radius(_communication_radius), msg_success_rate(std::move(_msg_success_rate)),
    temporal_noise_stddev(_temporal_noise_stddev),
    max_linear_speed(_max_linear_speed), max_angular_speed(_max_angular_speed),
    linear_noise_stddev(_linear_noise_stddev), angular_noise_stddev(_angular_noise_stddev) {
    data = malloc(_userdatasize);
    initialize_time();
    create_robot_body(world_id);
}

PogobotObject::PogobotObject(Simulation* simulation, uint16_t _id, float _x, float _y,
       b2WorldId world_id, size_t _userdatasize, Configuration const& config,
       std::string const& _category)
    : PhysicalObject(simulation, _id, _x, _y, world_id, config, _category) {
    parse_configuration(config, simulation);
    data = malloc(_userdatasize);
    initialize_time();
    create_robot_body(world_id);
}

void PogobotObject::parse_configuration(Configuration const& config, Simulation* simulation) {
    PhysicalObject::parse_configuration(config, simulation);
    msg_success_rate.reset(msg_success_rate_factory(config["msg_success_rate"]));
    communication_radius  = config["communication_radius"].get(80.0f);
    temporal_noise_stddev = config["temporal_noise_stddev"].get(0.0f);
    max_linear_speed      = config["max_linear_speed"].get(100.0f);
    max_angular_speed     = config["max_angular_speed"].get(1.0f);
    linear_noise_stddev   = config["linear_noise_stddev"].get(0.0f);
    angular_noise_stddev  = config["angular_noise_stddev"].get(0.0f);
}


void PogobotObject::create_robot_body([[maybe_unused]] b2WorldId world_id) {
    // Assign an initial velocity.
    b2Vec2 velocity = { 0.0f, 0.0f };
    b2Body_SetLinearVelocity(body_id, velocity);
    b2Body_SetAngularVelocity(body_id, 0.0f);

    // Extract radius from geometry
    radius = geom->compute_bounding_disk().radius;
}


void PogobotObject::launch_user_step([[maybe_unused]] float t) {
    update_time();
    enable_stop_watches();
    //user_step();
    pogo_main_loop_step(user_step);
    disable_stop_watches();
}

void PogobotObject::register_stop_watch(time_reference_t* sw) {
    stop_watches.insert(sw);
}

void PogobotObject::enable_stop_watches() {
    for (auto* sw : stop_watches) {
        sw->enable();
    }
}

void PogobotObject::disable_stop_watches() {
    for (auto* sw : stop_watches) {
        sw->disable();
    }
}


void PogobotObject::render(SDL_Renderer* renderer, [[maybe_unused]] b2WorldId worldId) const {
    // Get robot's position in the physics world
    b2Vec2 position = b2Body_GetPosition(body_id);

    // Convert to screen coordinates
    float screenX = position.x * VISUALIZATION_SCALE;
    float screenY = position.y * VISUALIZATION_SCALE;
    auto const circle_pos = visualization_position(screenX, screenY);

    // Get the robot's orientation as a rotation (cosine/sine pair)
    b2Rot rotation = b2Body_GetRotation(body_id);
    float cosAngle = rotation.c;
    float sinAngle = rotation.s;

    // Define relative positions for LEDs around the robot based on orientation.
    // For the lateral LEDs we want them exactly on the border, so use full 'radius'.
    std::vector<b2Vec2> ledOffsets;
    if (show_lateral_leds) {
        ledOffsets = {
            {0, 0},            // Center LED remains at the center
            {0, -radius},      // Top (in simulation units)
            {radius, 0},       // Right
            {0, radius},       // Bottom
            {-radius, 0}       // Left
        };

        // Apply a 45° clockwise rotation to the lateral LEDs (skip index 0)
        const float angle = M_PI / 4;  // -45 degrees in radians
        const float cos45 = cos(angle);
        const float sin45 = sin(angle);
        for (size_t i = 1; i < ledOffsets.size(); ++i) {
            float originalX = ledOffsets[i].x;
            float originalY = ledOffsets[i].y;
            ledOffsets[i].x = originalX * cos45 + originalY * sin45;
            ledOffsets[i].y = -originalX * sin45 + originalY * cos45;
        }
    } else {
        ledOffsets = {
            {0, 0}             // Only the center LED is used
        };
    }

    // Rotate LED offsets based on robot orientation
    std::vector<b2Vec2> rotatedLedOffsets;
    for (const auto& offset : ledOffsets) {
        float rotatedX = cosAngle * offset.x - sinAngle * offset.y;
        float rotatedY = sinAngle * offset.x + cosAngle * offset.y;
        rotatedLedOffsets.push_back({rotatedX, rotatedY});
    }

    // Draw each LED
    for (size_t i = 0; i < leds.size() && i < rotatedLedOffsets.size(); ++i) {
        color_t const& ledColor = leds[i]; // Get LED color
        uint8_t const r = adjust_color(ledColor.r);
        uint8_t const g = adjust_color(ledColor.g);
        uint8_t const b = adjust_color(ledColor.b);

        // Calculate screen coordinates for the LED.
        // Here, we scale the rotated offset from simulation units to pixels.
        float ledScreenX = screenX + rotatedLedOffsets[i].x  * 0.95;
        float ledScreenY = screenY + rotatedLedOffsets[i].y  * 0.95;

        auto const led_pos = visualization_position(ledScreenX, ledScreenY);
        if (i == 0) {
            // Center LED is drawn as a full circle.
            filledCircleRGBA(renderer, led_pos.x, led_pos.y, (radius - 2) * mm_to_pixels, r, g, b, 255);
        } else {
            // For lateral LEDs, only draw the half-disk that lies inside the robot's body.
            // Compute the angle from the LED center toward the robot center.
            float angleToCenter = atan2(screenY - ledScreenY, screenX - ledScreenX);
            // Convert the angle to degrees (SDL2_gfx expects degrees)
            float angleDeg = angleToCenter * 180.0f / M_PI;
            float startAngle = angleDeg - 90;  // start of 180° arc
            float endAngle   = angleDeg + 90;  // end of 180° arc

            // Draw the half-disk border
            filledPieRGBA(renderer, led_pos.x, led_pos.y,
                          (radius / 2.5 + 2) * mm_to_pixels, startAngle, endAngle,
                          255, 255, 255, 150);

            // Draw a half-disk for the LED.
            // (Assumes SDL2_gfx's filledPieRGBA is available.)
            filledPieRGBA(renderer, led_pos.x, led_pos.y,
                          (radius / 2.5) * mm_to_pixels, startAngle, endAngle,
                          r, g, b, 255);
        }
    }

    // Draw the main robot body (outline)
    circleRGBA(renderer, circle_pos.x, circle_pos.y, radius * mm_to_pixels, 0, 0, 0, 255);

    // Draw arrow indicating orientation
    float arrowLength = radius * 1.0;
    float arrowHeadSize = radius * 0.4;

    // Get opposite color of main LED for arrow
    uint8_t arrowR = 255, arrowG = 255, arrowB = 160;  // Default yellow arrow
    if (!leds.empty()) {
        color_t const& mainLed = leds[0];
        arrowR = 255 - adjust_color(mainLed.r);
        arrowG = 255 - adjust_color(mainLed.g);
        arrowB = 255 - adjust_color(mainLed.b);
        if (arrowR < 100 && arrowG < 100 && arrowB < 100) {
            arrowR = std::min(255, arrowR + 100);
            arrowG = std::min(255, arrowG + 100);
            arrowB = std::min(255, arrowB + 100);
        }
    }

    float endX = screenX + cosAngle * arrowLength;
    float endY = screenY + sinAngle * arrowLength;
    auto const end_pos = visualization_position(endX, endY);
    float const dist_arrow = sqrt(pow(circle_pos.x - end_pos.x, 2.0) + pow(circle_pos.y - end_pos.y, 2.0));
    if (dist_arrow > 5) {
        // Only draw arrow if the agent visual representation is large enough
        thickLineRGBA(renderer, circle_pos.x, circle_pos.y, end_pos.x, end_pos.y, 4, arrowR, arrowG, arrowB, 255);
    }

    float arrowLeft = endX - arrowHeadSize * (cosAngle * 0.7 + sinAngle * 0.5);
    float arrowRight = endX - arrowHeadSize * (cosAngle * 0.7 - sinAngle * 0.5);
    float arrowTopY = endY - arrowHeadSize * (sinAngle * 0.7 - cosAngle * 0.5);
    float arrowBottomY = endY - arrowHeadSize * (sinAngle * 0.7 + cosAngle * 0.5);
    auto const arrow_left = visualization_position(arrowLeft, arrowTopY);
    auto const arrow_right = visualization_position(arrowRight, arrowBottomY);
    filledTrigonRGBA(renderer, end_pos.x, end_pos.y,
                      arrow_left.x, arrow_left.y,
                      arrow_right.x, arrow_right.y,
                      arrowR, arrowG, arrowB, 255);
}


constexpr float arrow_head_size_world = 0.08f;

void PogobotObject::render_communication_channels(SDL_Renderer* renderer, [[maybe_unused]] b2WorldId worldId) const {
    /* 1.  Aggregate which channels each neighbour is on ------------------- */
    std::unordered_map<PogobotObject*, std::uint32_t> neighbour_mask;
    for (int ch = 0; ch < IR_RX_COUNT; ++ch) {
        for (PogobotObject* nbr : neighbors[ch]) {
            neighbour_mask[nbr] |= 1u << ch;
        }
        glogger->debug("objectId:{}  ch:{}  nb_neighbors:{}", id, ch, neighbors[ch].size() );
    }

    /* 2.  Pre-compute our own positions (world + screen) ------------------ */
    b2Vec2 const self_w = get_position();
    auto  const self_s = visualization_position(self_w.x * VISUALIZATION_SCALE,
                                                self_w.y * VISUALIZATION_SCALE);

    std::uint32_t const all_bits = (1u << IR_RX_COUNT) - 1u;

    /* 3.  Draw one segment per neighbour ---------------------------------- */
    for (auto const& [nbr, mask] : neighbour_mask) {
        /* ----- colour ------------------------------------------------------ */
        uint8_t r, g, b;
        if (mask == all_bits) {
            r = g = b = 0;                       // black → heard on every channel
        } else {
            int const repr_ch = std::countr_zero(mask);
            qualitative_colormap(repr_ch, &r, &g, &b);
        }

        /* ----- neighbour positions ----------------------------------------- */
        b2Vec2 const nbr_w = nbr->get_position();
        auto  const nbr_s = visualization_position(nbr_w.x * VISUALIZATION_SCALE,
                                                   nbr_w.y * VISUALIZATION_SCALE);

        thickLineRGBA(renderer,
                      self_s.x, self_s.y,
                      nbr_s.x,  nbr_s.y,
                      4,
                      r, g, b,
                      255);

        /* ----- bidirectional? --------------------------------------------- */
        bool const they_see_us =
            std::ranges::any_of(std::views::iota(0, IR_RX_COUNT),
                                [&](int ch) {
                                    auto const& their_vec = nbr->neighbors[ch];
                                    return std::ranges::find(their_vec, this) != their_vec.end();
                                });

        /* ----- arrow-head(s) – computed in world space -------------------- */
        auto draw_arrow = [&](b2Vec2 const& tip_w,        // arrow tip (world)
                              bool back_to_self) {
            float const dx_w = back_to_self ? self_w.x - nbr_w.x
                                            : nbr_w.x  - self_w.x;
            float const dy_w = back_to_self ? self_w.y - nbr_w.y
                                            : nbr_w.y  - self_w.y;

            float const len = std::sqrt(dx_w * dx_w + dy_w * dy_w);
            if (len == 0.0f) { return; }                  // Guard: same position.

            float const cos_a = dx_w / len;
            float const sin_a = dy_w / len;

            float const arrow_left_x  = tip_w.x - arrow_head_size_world * (cos_a * 0.7f + sin_a * 0.5f);
            float const arrow_left_y  = tip_w.y - arrow_head_size_world * (sin_a * 0.7f - cos_a * 0.5f);
            float const arrow_right_x = tip_w.x - arrow_head_size_world * (cos_a * 0.7f - sin_a * 0.5f);
            float const arrow_right_y = tip_w.y - arrow_head_size_world * (sin_a * 0.7f + cos_a * 0.5f);

            auto const tip_s        = visualization_position(tip_w.x  * VISUALIZATION_SCALE,
                                                             tip_w.y  * VISUALIZATION_SCALE);
            auto const arrow_left_s = visualization_position(arrow_left_x  * VISUALIZATION_SCALE,
                                                             arrow_left_y  * VISUALIZATION_SCALE);
            auto const arrow_right_s= visualization_position(arrow_right_x * VISUALIZATION_SCALE,
                                                             arrow_right_y * VISUALIZATION_SCALE);

            filledTrigonRGBA(renderer,
                             tip_s.x,        tip_s.y,
                             arrow_left_s.x, arrow_left_s.y,
                             arrow_right_s.x,arrow_right_s.y,
                             r, g, b, 255);
        };

        /* one-way → arrow only at neighbour end; two-way → both ends -------- */
        draw_arrow(nbr_w, false);          // always draw arrow pointing *towards* nbr
        if (they_see_us) {
            draw_arrow(self_w, true);      // reciprocal ⇒ arrow back to us
        }
        //else {
        //    glogger->info("UNIDIRECTIONAL communication channel DETECTED");
        //}
    }
}


void PogobotObject::set_motor(motor_id motor, int speed) {
    // Update motor speeds
    if (motor == motorL) {
        left_motor_speed = speed * (motor_dir[motorL] == 0 ? -1.f : 1.f);
    } else if (motor == motorR) {
        right_motor_speed = speed * (motor_dir[motorR] == 0 ? 1.f : -1.f);
    }
    // glogger->debug("set motor: {} {}", left_motor_speed, right_motor_speed);

    // Set damping values using those provided during construction.
    b2Body_SetLinearDamping(body_id, linear_damping);
    b2Body_SetAngularDamping(body_id, angular_damping);

    // Compute the desired linear velocity based on motor speeds.
    b2Rot const rot = b2Body_GetRotation(body_id);
    float const v = (max_linear_speed / VISUALIZATION_SCALE) * (left_motor_speed / static_cast<float>(motorFull) +
                            right_motor_speed / static_cast<float>(motorFull)) / 2.0f;
    b2Vec2 linear_velocity = {rot.c * v, rot.s * v};

    // Add Gaussian noise to linear velocity if the standard deviation is greater than 0.0.
    if (linear_noise_stddev > 0.0f) {
        // Use a static generator so that it persists across calls.
        std::normal_distribution<float> dist(0.0f, linear_noise_stddev);
        linear_velocity.x += dist(rnd_gen);
        linear_velocity.y += dist(rnd_gen);
    }
    b2Body_SetLinearVelocity(body_id, linear_velocity);

    // Compute the desired angular velocity based on motor speed difference.
    float angular_velocity = (max_angular_speed) / (static_cast<float>(motorFull) * 0.5f) *
                             (left_motor_speed - right_motor_speed);

    // Add Gaussian noise to angular velocity if the standard deviation is greater than 0.0.
    if (angular_noise_stddev > 0.0f) {
        std::normal_distribution<float> dist(0.0f, angular_noise_stddev);
        angular_velocity += dist(rnd_gen);
    }
    b2Body_SetAngularVelocity(body_id, angular_velocity);

    //glogger->debug("velocity: lin={},{}  ang={}  noise={},{}", linear_velocity.x, linear_velocity.y, angular_velocity, linear_noise_stddev, angular_noise_stddev);
}


b2Vec2 PogobotObject::get_IR_emitter_position(ir_direction dir) const {
    b2Vec2 pos = b2Body_GetPosition(body_id);

    // Get the robot's orientation as a rotation (cosine/sine pair)
    b2Rot rotation = b2Body_GetRotation(body_id);
    float cosAngle = rotation.c;
    float sinAngle = rotation.s;

    std::vector<b2Vec2> irOffsets = {
        {0, -radius},      // Front (in simulation units)
        {radius, 0},       // Right
        {0, radius},       // Back
        {-radius, 0},      // Left
        {0, 0}             // Middle
    };

    // Rotate IR offset based on robot orientation
    auto const& offset = irOffsets[dir];
    pos.x += cosAngle * offset.x / VISUALIZATION_SCALE - sinAngle * offset.y / VISUALIZATION_SCALE;
    pos.y += sinAngle * offset.x / VISUALIZATION_SCALE + cosAngle * offset.y / VISUALIZATION_SCALE;
    return pos;
}


float PogobotObject::get_IR_emitter_angle(ir_direction dir) const {
    /* body orientation as (cos θ, sin θ) -------------------------------- */
    b2Rot rot   = b2Body_GetRotation(body_id);
    float cos_t = rot.c;
    float sin_t = rot.s;

    /* local unit vectors for each emitter (robot frame) ----------------- */
    /*  front = ( 0,-1), right = ( 1,0), back = (0, 1), left = (-1,0)      */
    static constexpr b2Vec2 local_dirs[5] = {
        { 0.0f, -1.0f},     /* FRONT  */
        { 1.0f,  0.0f},     /* RIGHT  */
        { 0.0f,  1.0f},     /* BACK   */
        {-1.0f,  0.0f},     /* LEFT   */
        { 0.0f,  0.0f}      /* MIDDLE */
    };

    b2Vec2 v = local_dirs[dir];

    /* Special-case the middle LED: point along robot’s forward axis ----- */
    if (v.x == 0.0f && v.y == 0.0f) {
        /* forward in local frame is (0,-1) — same as FRONT                */
        v.x = 0.0f;
        v.y = -1.0f; 
    }

    /* rotate into world frame: R(θ)·v ----------------------------------- */
    float vx =  cos_t * v.x - sin_t * v.y;
    float vy =  sin_t * v.x + cos_t * v.y;

    return std::atan2(vy, vx);          /* bearing in radians              */
}

void PogobotObject::send_to_neighbors(ir_direction dir, short_message_t *const message) {
    // Reconstruct a long message from the short message
    message_t m;
    m.header._packet_type = message->header._packet_type;
    m.header._emitting_power_list = 0; // power all to 0 shouldn't emit something
    m.header._sender_id = 0xFF;
    m.header._sender_ir_index = 0xF; // index not possible
    m.header._receiver_ir_index = 0;
    m.header.payload_length = message->header.payload_length;
    memcpy( m.payload, message->payload, m.header.payload_length);

    send_to_neighbors(dir, &m);
}

void PogobotObject::send_to_neighbors(ir_direction dir, message_t *const message) {
    // Define a uniform real distribution between 0.0 and 1.0
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    double const payload_size = static_cast<double>(message->header.payload_length);
    double const msg_size = payload_size 
        + (message->header._packet_type == ir_t_short ? sizeof(message_short_header_t) : sizeof(message_header_t)) // Payload
        + 1 + 1 // Start and End bytes
        + 4 // CRC
        ;
    double const p_send = static_cast<double>(percent_msgs_sent_per_ticks) / 100.0;
    double const cluster_size = static_cast<double>(neighbors[dir].size() + 1);

    for (PogobotObject* robot : neighbors[dir]) {
        float const prob = dis(rnd_gen);
        //glogger->debug("MESSAGE !! with prob {} / {}: {} -> {}", prob, msg_success_rate, message->header._sender_id, robot->id);
        if (prob <= (*msg_success_rate)(msg_size, p_send, cluster_size)) {
            robot->receive_message(message, this);
        }
    }
}

void PogobotObject::receive_message(message_t *const message, PogobotObject* source) {
    if (messages.size() > 100) {
        // Overflow, ignore message
        return;
    }

    // Copy original message
    message_t m;
    memcpy(&m, message, sizeof(message_t));

    // Check which IR receptors can receive messages from the source
    std::vector<bool> possible_directions;
    for (uint8_t dir = ir_front; dir < ir_all; dir++) {
        possible_directions.push_back(contains(neighbors[dir], source));
    }

    // Check whether all directions are possible
    if (all_true(possible_directions)) {
        // Yes. Change _receiver_ir_index to ir_all, and add the message
        m.header._receiver_ir_index = ir_all;
        messages.push(m);
    } else {
        // No, create a new received message for each possible directions
        for (uint8_t dir = ir_front; dir < ir_all; dir++) {
            if (!possible_directions[dir])
                continue;
            // Yes. Change _receiver_ir_index, and add the message
            m.header._receiver_ir_index = dir;
            messages.push(m);
        }
    }
}


void PogobotObject::initialize_time() {
    // Identify the level of temporal noise on this object
    if (temporal_noise_stddev > 0) {
        std::uniform_real_distribution<float> dist(0.0f, temporal_noise_stddev);
        temporal_noise = dist(rnd_gen);
    }
}

void PogobotObject::update_time() {
    current_time_microseconds += temporal_noise;
}

void PogobotObject::sleep_µs(uint64_t microseconds) {
    if (microseconds <= 0) return;
    current_time_microseconds += microseconds;
}


/************* Pogobot Objects *************/ // {{{1

PogobjectObject::PogobjectObject(uint16_t _id, float _x, float _y,
       ObjectGeometry& geom, b2WorldId world_id,
       size_t _userdatasize,
       float _communication_radius,
       std::unique_ptr<MsgSuccessRate> _msg_success_rate,
       float _temporal_noise_stddev,
       float _linear_damping, float _angular_damping,
       float _density, float _friction, float _restitution,
       std::string const& _category)
    : PogobotObject::PogobotObject(_id, _x, _y, geom, world_id,
      _userdatasize, _communication_radius, std::move(_msg_success_rate),
      _temporal_noise_stddev, _linear_damping, _angular_damping,
      _density, _friction, _restitution,
      0.0f, 0.0f, 0.0f, 0.0f, _category) {
    for (size_t i = 0; i != motorB; i++)
        set_motor(static_cast<motor_id>(i), 0);
}

PogobjectObject::PogobjectObject(Simulation* simulation, uint16_t _id, float _x, float _y,
       b2WorldId world_id, size_t _userdatasize, Configuration const& config,
       std::string const& _category)
    : PogobotObject::PogobotObject(simulation, _id, _x, _y, world_id, _userdatasize, config, _category) {
    for (size_t i = 0; i != motorB; i++)
        set_motor(static_cast<motor_id>(i), 0);
}

void PogobjectObject::set_motor(motor_id motor, [[maybe_unused]] int speed) {
    // Update motor speeds
    if (motor == motorL) {
        left_motor_speed = 0;
    } else if (motor == motorR) {
        right_motor_speed = 0;
    }

    // Set damping values using those provided during construction.
    b2Body_SetLinearDamping(body_id, linear_damping);
    b2Body_SetAngularDamping(body_id, angular_damping);

    b2Vec2 linear_velocity = {0.0f, 0.0f};
    b2Body_SetLinearVelocity(body_id, linear_velocity);
    b2Body_SetAngularVelocity(body_id, 0.0f);
}

b2Vec2 PogobjectObject::get_IR_emitter_position([[maybe_unused]] ir_direction dir) const {
    return b2Body_GetPosition(body_id);
}

void PogobjectObject::render(SDL_Renderer* renderer, [[maybe_unused]] b2WorldId worldId) const {
    // Get robot's position in the physics world
    b2Vec2 position = b2Body_GetPosition(body_id);

    // Convert to screen coordinates
    float screenX = position.x * VISUALIZATION_SCALE;
    float screenY = position.y * VISUALIZATION_SCALE;
    auto const circle_pos = visualization_position(screenX, screenY);

    // Get the robot's orientation as a rotation (cosine/sine pair)
    b2Rot rotation = b2Body_GetRotation(body_id);
    float cosAngle = rotation.c;
    float sinAngle = rotation.s;

    // Define relative positions for LEDs around the robot based on orientation.
    // For the lateral LEDs we want them exactly on the border, so use full 'radius'.
    std::vector<b2Vec2> ledOffsets = { {0, 0} };

    // Rotate LED offsets based on robot orientation
    std::vector<b2Vec2> rotatedLedOffsets;
    for (const auto& offset : ledOffsets) {
        float rotatedX = cosAngle * offset.x - sinAngle * offset.y;
        float rotatedY = sinAngle * offset.x + cosAngle * offset.y;
        rotatedLedOffsets.push_back({rotatedX, rotatedY});
    }

    // Draw main LED
    color_t const& ledColor = leds[0]; // Get LED color
    uint8_t const r = adjust_color(ledColor.r);
    uint8_t const g = adjust_color(ledColor.g);
    uint8_t const b = adjust_color(ledColor.b);

    // Calculate screen coordinates for the LED.
    // Here, we scale the rotated offset from simulation units to pixels.
    float ledScreenX = screenX + rotatedLedOffsets[0].x  * 0.95;
    float ledScreenY = screenY + rotatedLedOffsets[0].y  * 0.95;

    auto const led_pos = visualization_position(ledScreenX, ledScreenY);
    // Center LED is drawn as a full circle.
    filledCircleRGBA(renderer, led_pos.x, led_pos.y, (radius - 2) * mm_to_pixels, r, g, b, 255);

    // Draw the main robot body (outline)
    circleRGBA(renderer, circle_pos.x, circle_pos.y, radius * mm_to_pixels, 0, 0, 0, 255);
}

/************* Pogowalls *************/ // {{{1

Pogowall::Pogowall(uint16_t _id, float _x, float _y,
       ObjectGeometry& _geom, b2WorldId world_id,
       size_t _userdatasize,
       float _communication_radius,
       std::unique_ptr<MsgSuccessRate> _msg_success_rate,
       float _temporal_noise_stddev,
       float _linear_damping, float _angular_damping,
       float _density, float _friction, float _restitution,
       float _max_linear_speed, float _max_angular_speed,
       float _linear_noise_stddev, float _angular_noise_stddev,
       std::string const& _category)
    : PogobotObject::PogobotObject(_id, _x, _y, _geom, world_id,
      _userdatasize, _communication_radius, std::move(_msg_success_rate),
      _temporal_noise_stddev, _linear_damping, _angular_damping,
      _density, _friction, _restitution,
      _max_linear_speed, _max_angular_speed,
      _linear_noise_stddev, _angular_noise_stddev,
      _category) {
    auto bd = geom->compute_bounding_disk();
    PhysicalObject::move(bd.center_x, bd.center_y);
}

Pogowall::Pogowall(Simulation* simulation, uint16_t _id, float _x, float _y,
       b2WorldId world_id, size_t _userdatasize, Configuration const& config,
       std::string const& _category)
    : PogobotObject::PogobotObject(simulation, _id, _x, _y, world_id, _userdatasize, config, _category) {
    auto bd = geom->compute_bounding_disk();
    PhysicalObject::move(bd.center_x, bd.center_y);
}

b2Vec2 Pogowall::get_IR_emitter_position([[maybe_unused]] ir_direction dir) const {
    return {NAN, NAN};
}

float Pogowall::get_IR_emitter_angle([[maybe_unused]] ir_direction dir) const {
    return NAN;
}


/************* MembraneObject *************/ // {{{1

MembraneObject::MembraneObject(uint16_t _id, float _x, float _y,
       ObjectGeometry& geom, b2WorldId world_id,
       size_t _userdatasize,
       float _communication_radius,
       std::unique_ptr<MsgSuccessRate> _msg_success_rate,
       float _temporal_noise_stddev,
       float _linear_damping, float _angular_damping,
       float _density, float _friction, float _restitution,
       float _max_linear_speed, float _max_angular_speed,
       float _linear_noise_stddev, float _angular_noise_stddev,
       unsigned int _num_dots, float _dot_radius, int _cross_span,
       float _stiffness,
       std::string _colormap,
       std::string const& _category)
    : Pogowall::Pogowall(_id, _x, _y, geom, world_id,
      _userdatasize, _communication_radius, std::move(_msg_success_rate),
      _temporal_noise_stddev, _linear_damping, _angular_damping,
      _density, _friction, _restitution,
      _max_linear_speed, _max_angular_speed,
      _linear_noise_stddev, _angular_noise_stddev, _category),
      num_dots(_num_dots), dot_radius(_dot_radius), cross_span(_cross_span),
      stiffness(_stiffness),
      colormap(_colormap) {
    for (size_t i = 0; i != motorB; i++)
        set_motor(static_cast<motor_id>(i), 0);
    create_robot_body(world_id);
}

MembraneObject::MembraneObject(Simulation* simulation, uint16_t _id, float _x, float _y,
       b2WorldId world_id, size_t _userdatasize, Configuration const& config,
       std::string const& _category)
    : Pogowall::Pogowall(simulation, _id, _x, _y, world_id, _userdatasize, config, _category) {
    for (size_t i = 0; i != motorB; i++)
        set_motor(static_cast<motor_id>(i), 0);
    parse_configuration(config, simulation);
    create_robot_body(world_id);
}

void MembraneObject::parse_configuration(Configuration const& config, Simulation* simulation) {
    Pogowall::parse_configuration(config, simulation);
    num_dots = config["num_dots"].get(100);
    dot_radius = config["dot_radius"].get(10.0f);
    cross_span = config["cross_span"].get(3);
    stiffness = config["stiffness"].get(30.0f);
    colormap = config["colormap"].get(std::string("rainbow"));
}


b2Vec2 MembraneObject::get_position() const {
    if (!dots.size())
        return {NAN, NAN};
    if (b2Body_IsValid(dots[0].body_id)) {
        return b2Body_GetPosition(dots[0].body_id);
    } else {
        return {NAN, NAN};
    }
}

void MembraneObject::render(SDL_Renderer* renderer, [[maybe_unused]] b2WorldId worldId) const {
    // Assign color based on object id
    uint8_t const value = static_cast<uint32_t>(id) % 256;
    uint8_t r, g, b;
    get_cmap_val(colormap, value, &r, &g, &b);
    //glogger->info("# ID={} value={} {} {} {}", id, value, r, g, b);

    // Draw dots
    for (const Dot& d : dots) {
        b2Vec2 p = b2Body_GetPosition(d.body_id);
        auto const pos = visualization_position(p.x * VISUALIZATION_SCALE, p.y * VISUALIZATION_SCALE);
        //auto const pos = visualization_position(p.x, p.y);
        //glogger->info("pos {} {}", pos.x, pos.y);
        filledCircleRGBA(renderer, pos.x, pos.y, dot_radius * mm_to_pixels, r, g, b, 255);
    }

    // Draw joints
    for (const Joint& j : joints) {
        b2Vec2 base_pA = b2Body_GetPosition(b2Joint_GetBodyA(j.joint_id));
        b2Vec2 base_pB = b2Body_GetPosition(b2Joint_GetBodyB(j.joint_id));
        auto const pA = visualization_position(base_pA.x * VISUALIZATION_SCALE, base_pA.y * VISUALIZATION_SCALE);
        auto const pB = visualization_position(base_pB.x * VISUALIZATION_SCALE, base_pB.y * VISUALIZATION_SCALE);
        //glogger->info("joint {} ({},{}) ({},{})", b2Joint_IsValid(j.joint_id), pA.x, pA.y, pB.x, pB.y);

        thickLineRGBA(renderer, pA.x, pA.y, pB.x, pB.y, 4, 255, 0, 0, 150);
    }
}

void MembraneObject::make_distance_joint(b2WorldId world_id,
                         b2BodyId  a,
                         b2BodyId  b,
                         float     stiffness_scale) {
    const b2Vec2 pA = b2Body_GetPosition(a);
    const b2Vec2 pB = b2Body_GetPosition(b);
    const float  len = std::sqrt((pA.x - pB.x) * (pA.x - pB.x) +
                                 (pA.y - pB.y) * (pA.y - pB.y));

    b2DistanceJointDef jd = b2DefaultDistanceJointDef();
    jd.bodyIdA       = a;
    jd.bodyIdB       = b;
    jd.localAnchorA  = {0.0f, 0.0f};
    jd.localAnchorB  = {0.0f, 0.0f};
    jd.length        = len;
    jd.hertz         = stiffness * stiffness_scale;
    jd.dampingRatio  = linear_damping;

    b2JointId j_id = b2CreateDistanceJoint(world_id, &jd);
    joints.push_back({j_id});
}


void MembraneObject::create_robot_body([[maybe_unused]] b2WorldId world_id) {
    b2DestroyBody(body_id);

    /* -------- 1. sample the outlines -------------------------------- */
    arena_polygons_t contours = geom->generate_contours(num_dots);

    // Determine the size of dot_radius proportional to the contours
    float const radius_bd = geom->compute_bounding_disk().radius;
    float prop_dot_radius = dot_radius / radius_bd;
    if (prop_dot_radius > 0.30f) {
        prop_dot_radius = 0.30f;
    }

    for (const auto& contour : contours) {
        const std::size_t first_idx = dots.size();      // index of 1st dot
        const std::size_t n         = contour.size();
        if (n < 2) continue;                            // skip degenerate
        size_contours.push_back(n);

        /* ---- 2. create one Box2D body per vertex ------------------- */
        for (const b2Vec2& v_local : contour) {
            b2BodyDef body_def     = b2DefaultBodyDef();
            body_def.type          = b2_dynamicBody;
            body_def.position      = {(x + v_local.x * (1 - prop_dot_radius)) / VISUALIZATION_SCALE,
                                      (y + v_local.y * (1 - prop_dot_radius)) / VISUALIZATION_SCALE};
            //body_def.position      = {(x + v_local.x),
            //                          (y + v_local.y)};

            b2BodyId dot_body_id = b2CreateBody(world_id, &body_def);

            b2Circle   circle { .center = {0.0f, 0.0f},
                                .radius = dot_radius / VISUALIZATION_SCALE };

            b2ShapeDef shape_def  = b2DefaultShapeDef();
            shape_def.density     = density;
            shape_def.friction    = friction;

            b2CreateCircleShape(dot_body_id, &shape_def, &circle);
            b2Vec2 linear_velocity = {0.0f, 0.0f};
            b2Body_SetLinearVelocity(dot_body_id, linear_velocity);
            b2Body_SetAngularVelocity(dot_body_id, 0.0f);

            dots.push_back({dot_body_id});
        }

        /* ---- 3. joints: edges of the contour ----------------------- */
        for (std::size_t i = 0; i < n; ++i) {
            make_distance_joint(world_id,
                                dots[first_idx + i].body_id,
                                dots[first_idx + (i + 1) % n].body_id);
        }

        /* ---- 4. optional cross‑bracing ----------------------------- */
        if (cross_span > 1) {
            for (std::size_t i = 0; i < n; i++) {
                    make_distance_joint(world_id,
                                        dots[first_idx + i].body_id,
                                        dots[first_idx + (i + cross_span) % n].body_id,
                                        /*slightly softer*/ 0.8f);
            }
        }
    }
}

void MembraneObject::move(float _x, float _y, float _theta) {
    /* Offset expressed in Box2D world units */
    const float dx = (_x - x) / VISUALIZATION_SCALE;
    const float dy = (_y - y) / VISUALIZATION_SCALE;

    for (const Dot& d : dots) {
        b2Vec2 pos = b2Body_GetPosition(d.body_id);
        pos.x += dx;
        pos.y += dy;

        const b2Rot rot = b2Body_GetRotation(d.body_id);   // Keep current angle
        b2Body_SetTransform(d.body_id, pos, rot);
    }

    PogobotObject::move(_x, _y, _theta);
}

static std::vector<b2Vec2>
resample_polygon(const std::vector<b2Vec2>& src, std::size_t out_count) {
    if (out_count == 0 || src.size() < 2) return {};

    /* compute cumulative edge lengths */
    const std::size_t n = src.size();
    std::vector<float> cum(n + 1, 0.0f);          // cum[0] = 0
    for (std::size_t i = 1; i <= n; ++i) {
        const auto& a = src[i - 1];
        const auto& b = src[i % n];
        const float dx = b.x - a.x, dy = b.y - a.y;
        cum[i] = cum[i - 1] + std::sqrt(dx * dx + dy * dy);
    }
    const float perimeter = cum.back();
    const float step      = perimeter / out_count;

    std::vector<b2Vec2> dst;  dst.reserve(out_count);
    /* walk around the polygon and pick points every *step* millimetres */
    std::size_t edge = 0;
    float target = 0.0f;

    for (std::size_t k = 0; k < out_count; ++k, target += step) {
        while (cum[edge + 1] < target) ++edge;

        const auto& a = src[edge];
        const auto& b = src[(edge + 1) % n];
        const float  t = (target - cum[edge]) / (cum[edge + 1] - cum[edge]);

        dst.push_back({a.x + t * (b.x - a.x), a.y + t * (b.y - a.y)});
    }
    return dst;
}

arena_polygons_t MembraneObject::generate_contours(std::size_t points_per_contour) const {
    arena_polygons_t contours;
    contours.reserve(size_contours.size());

    std::size_t dot_index = 0;

    for (std::size_t contour = 0; contour < size_contours.size(); ++contour) {
        const std::size_t n = size_contours[contour];

        std::vector<b2Vec2> poly;   poly.reserve(n);

        for (std::size_t k = 0; k < n; ++k, ++dot_index) {
            b2Vec2 wp = b2Body_GetPosition(dots[dot_index].body_id);   // world units
            /* convert to millimetres, then to local coords around the membrane centre */
            poly.push_back({wp.x * VISUALIZATION_SCALE,
                            wp.y * VISUALIZATION_SCALE});
        }

        /* optional uniform resampling */
        if (points_per_contour != 0 && points_per_contour != poly.size())
            contours.push_back(resample_polygon(poly, points_per_contour));
        else
            contours.push_back(std::move(poly));
    }
    return contours;
}


// MODELINE "{{{1
// vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
// vim:foldmethod=marker
