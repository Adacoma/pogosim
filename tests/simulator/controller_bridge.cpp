#include "pogosim/simulator.h"
#undef main
#include "test_support.h"
#include <type_traits>
#include <vector>

// The C++ core must read the same width that REGISTER_USERDATA defines.
static_assert(std::is_same_v<decltype(UserdataSize), std::size_t>);

namespace { std::vector<PogobotObject*> robots; float moved_angle = 0; }

extern "C" void regression_check(int condition, const char* message) { check(condition, message); }

extern "C" void regression_register_robot(void) {
    check(current_robot && current_robot->is_initialized(), "Controller began before object initialization");
    robots.push_back(current_robot);
}

extern "C" void regression_initialize(uint32_t mode) {
    check(robots.size() == 2 && simulation->get_nb_robots() == 2, "Unexpected fixture robot population");
    if (mode == 3 && current_robot->id == 0) {
        current_robot->move(1005, 500, 0.25f);
        // Wrapping must preserve orientation, including the existing move()
        // conversion between robot and Box2D angle conventions.
        moved_angle = current_robot->get_angle();
    }
}

extern "C" void regression_end(uint32_t mode) {
    check(current_robot->neighbors[ir_all].size() == (mode == 4 ? 0u : 1u), "Neighbor union duplicated or lost a robot");
    if (mode == 3 && current_robot->id == 0) {
        const auto position = current_robot->get_position();
        close_to(position.x * VISUALIZATION_SCALE, 5, 0.02);
        close_to(position.y * VISUALIZATION_SCALE, 500, 0.02);
        close_to(current_robot->get_angle(), moved_angle, 1e-4);
    }
    check(robots[0]->data != robots[1]->data, "Robots share controller storage");
}

extern "C" void regression_schema(uint32_t mode) {
    if (mode == 6) throw std::runtime_error("regression schema failure");
}

extern "C" void regression_export(uint32_t mode) {
    if (mode == 7) throw std::runtime_error("regression export failure");
}
