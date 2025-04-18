
#include <iostream>
#include <thread>
#include <chrono>
#include <cstdarg>
#include <sstream>
#include <string>
#include <cstdint>
#include <cmath>
#include "SDL2_gfxPrimitives.h"

#include "robot.h"
#include "spogobot.h"
#include "pogosim.h"
#include "simulator.h"


/************* GLOBALS *************/ // {{{1

Robot* current_robot;
//std::chrono::time_point<std::chrono::system_clock> sim_starting_time;
uint64_t sim_starting_time_microseconds;


/************* DynamicMsgSuccessRate *************/ // {{{1

DynamicMsgSuccessRate::DynamicMsgSuccessRate(double alpha, double beta, double gamma, double delta)
    : alpha_(alpha), beta_(beta), gamma_(gamma), delta_(delta) {}

double DynamicMsgSuccessRate::operator()(double msg_size, double p_send, double cluster_size) const {
    return 1.0 / (1.0 + (alpha_ * std::pow(msg_size, beta_) *
                                  std::pow(p_send, gamma_) *
                                  std::pow(cluster_size, delta_)));
}

/************* ConstMsgSuccessRate *************/ // {{{1

ConstMsgSuccessRate::ConstMsgSuccessRate(double value)
    : const_value_(value) {}

double ConstMsgSuccessRate::operator()(double /*msg_size*/, double /*p_send*/, double /*cluster_size*/) const {
    return const_value_;
}


/************* SIMULATED ROBOTS *************/ // {{{1

Robot::Robot(uint16_t _id, size_t _userdatasize, float x, float y, float _radius,
             b2WorldId worldId, std::unique_ptr<MsgSuccessRate> _msg_success_rate,
             float _linearDamping, float _angularDamping,
             float _density, float _friction, float _restitution,
             ShapeType _shapeType, float _yRadius,
             const std::vector<b2Vec2>& _polygonVertices,
             float _linear_noise_stddev, float _angular_noise_stddev,
             float _temporal_noise_stddev)
    : id(_id), radius(_radius), msg_success_rate(std::move(_msg_success_rate)), linearDamping(_linearDamping), angularDamping(_angularDamping),
        linear_noise_stddev(_linear_noise_stddev), angular_noise_stddev(_angular_noise_stddev), temporal_noise_stddev(_temporal_noise_stddev) {
    data = malloc(_userdatasize);
    create_body(worldId, x, y,
                _density, _friction, _restitution, _shapeType, _yRadius, _polygonVertices);
}


//Robot::~Robot() {
//    free(data);
//    data = nullptr;
//}


void Robot::create_body(b2WorldId worldId, float x, float y,
                        float density, float friction, float restitution,
                        ShapeType shapeType, float yRadius,
                        const std::vector<b2Vec2>& polygonVertices) {
    // Create the body definition.
    b2BodyDef bodyDef = b2DefaultBodyDef();
    bodyDef.type = b2_dynamicBody;
    bodyDef.position = { x / VISUALIZATION_SCALE, y / VISUALIZATION_SCALE };
    bodyDef.linearDamping = 1000.0f;
    bodyDef.angularDamping = 1000.0f;
    bodyDef.isBullet = false;
    bodyId = b2CreateBody(worldId, &bodyDef);

    // Set up a shape definition with common physical properties.
    b2ShapeDef shapeDef = b2DefaultShapeDef();
    shapeDef.density = density;
    shapeDef.friction = friction;
    shapeDef.restitution = restitution;
    shapeDef.enablePreSolveEvents = true;

    // Create the shape based on the specified shape type.
    switch (shapeType) {
        case ShapeType::Circle: {
            b2Circle circle;
            circle.center = { 0.0f, 0.0f };
            circle.radius = radius / VISUALIZATION_SCALE;
            shapeId = b2CreateCircleShape(bodyId, &shapeDef, &circle);
            break;
        }
        case ShapeType::Ellipse: {
            // For an ellipse, use _yRadius if provided; otherwise default to the circle's radius.
            float effectiveYRadius = (yRadius > 0.0f ? yRadius : radius);
            const int vertexCount = 16; // Number of vertices to approximate the ellipse.
            std::vector<b2Vec2> ellipseVertices;
            ellipseVertices.reserve(vertexCount);
            for (int i = 0; i < vertexCount; i++) {
                float angle = (2.0f * 3.14159265f * i) / vertexCount;
                float vx = (radius * std::cos(angle)) / VISUALIZATION_SCALE;
                float vy = (effectiveYRadius * std::sin(angle)) / VISUALIZATION_SCALE;
                ellipseVertices.push_back({ vx, vy });
            }
            // Create a temporary b2Polygon and fill it with the ellipse vertices.
            b2Polygon poly;
            poly.count = static_cast<int32_t>(ellipseVertices.size());
            for (int i = 0; i < poly.count; ++i) {
                poly.vertices[i] = ellipseVertices[i];
            }
            shapeId = b2CreatePolygonShape(bodyId, &shapeDef, &poly);
            break;
        }
        case ShapeType::Polygon: {
            if (!polygonVertices.empty()) {
                // Create a temporary b2Polygon and fill it with the provided vertices.
                b2Polygon poly;
                poly.count = static_cast<int32_t>(polygonVertices.size());
                for (int i = 0; i < poly.count; ++i) {
                    poly.vertices[i] = polygonVertices[i];
                }
                shapeId = b2CreatePolygonShape(bodyId, &shapeDef, &poly);
            } else {
                // Fallback: if no vertices are provided, default to a circle.
                b2Circle circle;
                circle.center = { 0.0f, 0.0f };
                circle.radius = radius / VISUALIZATION_SCALE;
                shapeId = b2CreateCircleShape(bodyId, &shapeDef, &circle);
            }
            break;
        }
        default: {
            // Fallback to circle for any unrecognized shape type.
            b2Circle circle;
            circle.center = { 0.0f, 0.0f };
            circle.radius = radius / VISUALIZATION_SCALE;
            shapeId = b2CreateCircleShape(bodyId, &shapeDef, &circle);
            break;
        }
    }

    // Assign an initial velocity.
    b2Vec2 velocity = { 1.0f, 1.0f };
    b2Body_SetLinearVelocity(bodyId, velocity);

    // Identify the level of temporal noise on this robot
    if (temporal_noise_stddev > 0) {
        //std::normal_distribution<float> dist(0.0f, temporal_noise_stddev);
        std::uniform_real_distribution<float> dist(0.0f, temporal_noise_stddev);
        temporal_noise = dist(rnd_gen);
    }
}

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

//void Robot::render(SDL_Renderer* renderer, [[maybe_unused]] b2WorldId worldId, bool show_comm, bool show_lateral_leds) const {
//    // Get robot's position in the physics world
//    b2Vec2 position = b2Body_GetPosition(bodyId);
//
//    // Convert to screen coordinates
//    float screenX = position.x * VISUALIZATION_SCALE;
//    float screenY = position.y * VISUALIZATION_SCALE;
//
//    // Draw circle representing the robot's body
////    SDL_SetRenderDrawColor(renderer, 200, 200, 200, 255); // Gray color for robot body
////    SDL_RenderDrawCircle(renderer, static_cast<int>(screenX), static_cast<int>(screenY), radius);
//    auto const circle_pos = visualization_position(screenX, screenY);
//    circleRGBA(renderer, circle_pos.x, circle_pos.y, radius * mm_to_pixels, 0, 0, 0, 200);
//
//    // Get the robot's orientation as a rotation (cosine/sine pair)
//    b2Rot rotation = b2Body_GetRotation(bodyId);
//    float cosAngle = rotation.c;
//    float sinAngle = rotation.s;
//
//    // Draw line indicating robot orientation with increased width
//    float const orientationX = screenX + cosAngle * radius * 2.0; // Increase length of orientation line
//    float const orientationY = screenY + sinAngle * radius * 2.0;
////    SDL_SetRenderDrawColor(renderer, 0, 0, 255, 150); // Blue/transparent color for orientation line
////    for (int offset = -2; offset <= 2; ++offset) {
////        SDL_RenderDrawLine(renderer, screenX + offset, screenY, orientationX + offset, orientationY);
////        SDL_RenderDrawLine(renderer, screenX, screenY + offset, orientationX, orientationY + offset);
////    }
//    auto const orientation_pos = visualization_position(orientationX, orientationY);
//    thickLineRGBA(renderer, circle_pos.x, circle_pos.y, orientation_pos.x, orientation_pos.y, 4, 0, 0, 255, 150);
//
//    // Draw communication channels
//    if (show_comm) {
//        for (Robot* robot : neighbors) {
//            b2Vec2 const r_pos = robot->get_position();
//            auto const r_circle_pos = visualization_position(r_pos.x * VISUALIZATION_SCALE, r_pos.y * VISUALIZATION_SCALE);
//            thickLineRGBA(renderer, circle_pos.x, circle_pos.y, r_circle_pos.x, r_circle_pos.y, 4, 0, 150, 0, 150);
//        }
//    }
//
//    // Define relative positions for LEDs around the robot based on orientation
//    std::vector<b2Vec2> ledOffsets;
//
//    if (show_lateral_leds) {
//        ledOffsets = {
//            {0, 0},                        // Above the robot center
//            {0, -radius / 2},              // Up
//            {radius / 2, 0},               // Right
//            {0, radius / 2},               // Down
//            {-radius / 2, 0}               // Left
//        };
//    } else {
//        ledOffsets = {
//            {0, 0}                         // Above the robot center
//        };
//    }
//
//    // Rotate LED offsets based on robot orientation
//    std::vector<b2Vec2> rotatedLedOffsets;
//    for (const auto& offset : ledOffsets) {
//        float rotatedX = cosAngle * offset.x - sinAngle * offset.y;
//        float rotatedY = sinAngle * offset.x + cosAngle * offset.y;
//        rotatedLedOffsets.push_back({rotatedX, rotatedY});
//    }
//
//    // Draw each LED
//    for (size_t i = 0; i < leds.size() && i < rotatedLedOffsets.size(); ++i) {
//        color_t const& ledColor = leds[i]; // Get LED color
//        uint8_t const r = adjust_color(ledColor.r);
//        uint8_t const g = adjust_color(ledColor.g);
//        uint8_t const b = adjust_color(ledColor.b);
//
//        //// Convert LED color to SDL color
//        ////SDL_SetRenderDrawColor(renderer, ledColor.r / 25.0f * 255.0f, ledColor.g / 25.0f * 255.0f, ledColor.b / 25.0f * 255.0f, 255);
//        ////SDL_SetRenderDrawColor(renderer, ledColor.r, ledColor.g, ledColor.b, 255);
//        //SDL_SetRenderDrawColor(renderer,
//        //        ledColor.r > 25 ? 230 : static_cast<float>(ledColor.r) / 25.0f * 210.0f,
//        //        ledColor.g > 25 ? 230 : static_cast<float>(ledColor.g) / 25.0f * 210.0f,
//        //        ledColor.b > 25 ? 230 : static_cast<float>(ledColor.b) / 25.0f * 210.0f, 255);
//
//        // Calculate screen coordinates for the LED
//        float ledScreenX = screenX + rotatedLedOffsets[i].x * 2.0;
//        float ledScreenY = screenY + rotatedLedOffsets[i].y * 2.0;
//
//        // Draw LED as a small circle
//        auto const led_pos = visualization_position(ledScreenX, ledScreenY);
//        if (i == 0) {
//            //SDL_RenderDrawCircle(renderer, static_cast<int>(ledScreenX), static_cast<int>(ledScreenY), radius - 2);
//            filledCircleRGBA(renderer, led_pos.x, led_pos.y, (radius - 2) * mm_to_pixels, r, g, b, 255);
//        } else {
//            //SDL_RenderDrawCircle(renderer, static_cast<int>(ledScreenX), static_cast<int>(ledScreenY), radius / 2.5);
//            filledCircleRGBA(renderer, led_pos.x, led_pos.y, (radius / 2.5) * mm_to_pixels, r, g, b, 255);
//        }
//    }
//}


void Robot::render(SDL_Renderer* renderer, [[maybe_unused]] b2WorldId worldId, bool show_comm, bool show_lateral_leds) const {
    // Get robot's position in the physics world
    b2Vec2 position = b2Body_GetPosition(bodyId);

    // Convert to screen coordinates
    float screenX = position.x * VISUALIZATION_SCALE;
    float screenY = position.y * VISUALIZATION_SCALE;

    // Get the robot's orientation as a rotation (cosine/sine pair)
    b2Rot rotation = b2Body_GetRotation(bodyId);
    float cosAngle = rotation.c;
    float sinAngle = rotation.s;

    auto const circle_pos = visualization_position(screenX, screenY);

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
        const float angle = M_PI / 4;  // 45 degrees in radians
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
    thickLineRGBA(renderer, circle_pos.x, circle_pos.y, end_pos.x, end_pos.y, 4, arrowR, arrowG, arrowB, 255);

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

    // Draw communication channels if needed
    if (show_comm) {
        for (Robot* robot : neighbors) {
            b2Vec2 const r_pos = robot->get_position();
            auto const r_circle_pos = visualization_position(r_pos.x * VISUALIZATION_SCALE, r_pos.y * VISUALIZATION_SCALE);
            thickLineRGBA(renderer, circle_pos.x, circle_pos.y, r_circle_pos.x, r_circle_pos.y, 4, 0, 150, 0, 150);
        }
    }
}



void Robot::set_motor(motor_id motor, int speed) {
    // Update motor speeds
    if (motor == motorL) {
        left_motor_speed = speed;
    } else if (motor == motorR) {
        right_motor_speed = speed;
    }
    // glogger->debug("set motor: {} {}", left_motor_speed, right_motor_speed);

    // Set damping values using those provided during construction.
    b2Body_SetLinearDamping(bodyId, linearDamping);
    b2Body_SetAngularDamping(bodyId, angularDamping);

    // Compute the desired linear velocity based on motor speeds.
    b2Rot const rot = b2Body_GetRotation(bodyId);
    float const v = 1.0f * (left_motor_speed / static_cast<float>(motorFull) +
                            right_motor_speed / static_cast<float>(motorFull)) / 2.0f;
    b2Vec2 linear_velocity = {rot.c * v, rot.s * v};

    // Add Gaussian noise to linear velocity if the standard deviation is greater than 0.0.
    if (linear_noise_stddev > 0.0f) {
        // Use a static generator so that it persists across calls.
        std::normal_distribution<float> dist(0.0f, linear_noise_stddev);
        linear_velocity.x += dist(rnd_gen);
        linear_velocity.y += dist(rnd_gen);
    }
    b2Body_SetLinearVelocity(bodyId, linear_velocity);

    // Compute the desired angular velocity based on motor speed difference.
    float angular_velocity = 1.0f / (static_cast<float>(motorFull) * 0.5f) *
                             (left_motor_speed - right_motor_speed);

    // Add Gaussian noise to angular velocity if the standard deviation is greater than 0.0.
    if (angular_noise_stddev > 0.0f) {
        std::normal_distribution<float> dist(0.0f, angular_noise_stddev);
        angular_velocity += dist(rnd_gen);
    }
    b2Body_SetAngularVelocity(bodyId, angular_velocity);

    //glogger->debug("velocity: lin={},{}  ang={}  noise={},{}", linear_velocity.x, linear_velocity.y, angular_velocity, linear_noise_stddev, angular_noise_stddev);
}


void Robot::launch_user_step() {
    update_time();
    enable_stop_watches();
    //user_step();
    pogo_main_loop_step(user_step);
    disable_stop_watches();
}

void Robot::update_time() {
    current_time_microseconds += temporal_noise;
}

void Robot::register_stop_watch(time_reference_t* sw) {
    stop_watches.insert(sw);
}

void Robot::enable_stop_watches() {
    for (auto* sw : stop_watches) {
        sw->enable();
    }
}

void Robot::disable_stop_watches() {
    for (auto* sw : stop_watches) {
        sw->disable();
    }
}

b2Vec2 Robot::get_position() const {
    return b2Body_GetPosition(bodyId);
}

float Robot::get_angle() const {
    b2Rot const rotation = b2Body_GetRotation(bodyId);
    return std::atan2(rotation.s, rotation.c);
}


void Robot::send_to_neighbors(short_message_t *const message) {
    // Reconstruct a long message from the short message
    message_t m;
    m.header._packet_type = message->header._packet_type;
    m.header._emitting_power_list = 0; // power all to 0 shouldn't emit something
    m.header._sender_id = 0xFF;
    m.header._sender_ir_index = 0xF; // index not possible
    m.header._receiver_ir_index = 0;
    m.header.payload_length = message->header.payload_length;
    memcpy( m.payload, message->payload, m.header.payload_length);

    send_to_neighbors(&m);
}

void Robot::send_to_neighbors(message_t *const message) {
    // Define a uniform real distribution between 0.0 and 1.0
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    double const payload_size = static_cast<double>(message->header.payload_length); 
    double const msg_size = payload_size + (message->header._packet_type == ir_t_short ? sizeof(message_short_header_t) : sizeof(message_header_t));
    double const p_send = static_cast<double>(percent_msgs_sent_per_ticks) / 100.0;
    double const cluster_size = static_cast<double>(neighbors.size() + 1);

    for (Robot* robot : neighbors) {
        float const prob = dis(rnd_gen);
        //glogger->debug("MESSAGE !! with prob {} / {}: {} -> {}", prob, msg_success_rate, message->header._sender_id, robot->id);
        if (prob <= (*msg_success_rate)(msg_size, p_send, cluster_size) && robot->messages.size() < 100) {
            robot->messages.push(*message);
        }
    }
}

void Robot::sleep_µs(uint64_t microseconds) {
    if (microseconds <= 0) return;
    current_time_microseconds += microseconds;
}


// MODELINE "{{{1
// vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
// vim:foldmethod=marker
