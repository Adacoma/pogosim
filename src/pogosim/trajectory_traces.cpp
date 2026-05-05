#include "trajectory_traces.h"

#include <cmath>

#include "SDL2_gfxPrimitives.h"

#include "robot.h"
#include "render.h"
#include "simulator.h"
#include "utils.h"

namespace {
// drawing style for robot trajectory traces
constexpr uint8_t trace_gray = 20;
constexpr float trace_alpha_range = 210.0f;
constexpr uint8_t trace_thickness = 3;

// keep memory cost bounded: nb_robots * max_trace_points.
constexpr std::size_t max_trace_points = 512;

// store a point only after the tail moved x mm in world
constexpr float min_trace_distance = 2.0f;
}

void TrajectoryTraces::reset(std::vector<std::shared_ptr<PogobotObject>> const& robots) {
    traces.clear();
    traces.resize(robots.size());
    update(robots);
}

b2Vec2 TrajectoryTraces::get_robot_tail_position(PogobotObject const& robot) const {
    b2Vec2 const position = robot.get_position();
    float const angle = robot.get_angle();
    float const tail_offset = robot.radius / VISUALIZATION_SCALE;

    // trace starts behind the robot, not at its center.
    return {
        position.x - std::cos(angle) * tail_offset,
        position.y - std::sin(angle) * tail_offset,
    };
}

void TrajectoryTraces::update(std::vector<std::shared_ptr<PogobotObject>> const& robots) {
    if (traces.size() != robots.size()) {
        traces.clear();
        traces.resize(robots.size());
    }

    float const min_distance_world = min_trace_distance / VISUALIZATION_SCALE;
    float const min_distance_squared = min_distance_world * min_distance_world;

    for (std::size_t i = 0; i < robots.size(); ++i) {
        b2Vec2 const tail = get_robot_tail_position(*robots[i]);
        auto& trace = traces[i];

        // avoid adding nearly identical points every simulation step
        if (!trace.empty()) {
            b2Vec2 const delta = tail - trace.back();
            float const distance_squared = delta.x * delta.x + delta.y * delta.y;
            if (distance_squared < min_distance_squared) {
                continue;
            }
        }

        trace.push_back(tail);
        // drop oldest points first
        while (trace.size() > max_trace_points) {
            trace.pop_front();
        }
    }
}

void TrajectoryTraces::render(SDL_Renderer* renderer,
        boundary_condition_t boundary_condition,
        float domain_w,
        float domain_h) const {
    if (traces.empty()) {
        return;
    }

    SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);

    for (auto const& trace : traces) {
        if (trace.size() < 2) {
            continue;
        }

        std::size_t const nb_segments = trace.size() - 1;
        for (std::size_t i = 1; i < trace.size(); ++i) {
            b2Vec2 const& from = trace[i - 1];
            b2Vec2 const& to = trace[i];
            b2Vec2 const delta = to - from;

            // for periodic boundary conditions, don't draw segments that go across
            if (boundary_condition == boundary_condition_t::periodic && (std::abs(delta.x) > domain_w * 0.5f ||
                                                                         std::abs(delta.y) > domain_h * 0.5f)) {
                continue;
            }

            float const age = i / static_cast<float>(nb_segments);

            // age^10 makes old segments disappear quickly.
            float const fade = std::pow(age, 10.0f);
            uint8_t const alpha = static_cast<uint8_t>(trace_alpha_range * fade);

            b2Vec2 const from_screen = visualization_position(from.x * VISUALIZATION_SCALE, from.y * VISUALIZATION_SCALE);
            b2Vec2 const to_screen = visualization_position(to.x * VISUALIZATION_SCALE, to.y * VISUALIZATION_SCALE);

            thickLineRGBA(renderer,
                    from_screen.x,
                    from_screen.y,
                    to_screen.x,
                    to_screen.y,
                    trace_thickness,
                    trace_gray,
                    trace_gray,
                    trace_gray,
                    alpha);
        }
    }
}
