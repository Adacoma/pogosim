#include "pogosim/simulator.h"
#undef main
#include "test_support.h"
#include <filesystem>
#include <memory>
#include <typeinfo>

namespace {
struct World {
    b2WorldId id;
    World() { auto def = b2DefaultWorldDef(); def.gravity = {0, 0}; id = b2CreateWorld(&def); }
    ~World() { b2DestroyWorld(id); }
};
}

void test_factory(const std::string& type, bool periodic, const std::filesystem::path& source) {
    Configuration config(YAML::Load("GUI: false\nenable_console_logging: false\nenable_data_logging: false\narena_surface: 1000000\n"));
    config.set("arena_file", (source / "arenas/square.csv").generic_string());
    config.set("boundary_condition", periodic ? "periodic" : "solid");
    Simulation context(config);
    context.create_arena();
    World world;
    LightLevelMap map(10, 10, 100, 100);
    Configuration object_config(YAML::Load("geometry: disk\nradius: 60\nx: 500\ny: 500\nnum_dots: 12\ndot_radius: 1\ncross_span: 2\ncommunication_radius: 0\ntemporal_noise_stddev: 0\n"));
    object_config.set("type", type);
    // Specify the communication model explicitly, as production fixtures do.
    object_config.set("msg_success_rate", YAML::Load("{type: static, rate: 1}"));
    if (type == "pogowall") object_config.set("geometry", "arena");
    std::unique_ptr<Object> object(object_factory(&context, 7, 500, 500, world.id, object_config, &map, 16, "fixture"));
    if (periodic && type == "pogowall") {
        check(!object, "Periodic Pogowalls must be skipped");
        return;
    }
    check(object && object->is_initialized() && object->category == "fixture", "Factory failed to initialize/category-tag its object");
    // Exact types catch accidental fall-through to a parent factory branch.
    const std::type_info* expected = nullptr;
    if (type == "pogobot") expected = &typeid(PogobotObject);
    else if (type == "pogobject") expected = &typeid(PogobjectObject);
    else if (type == "pogowall") expected = &typeid(Pogowall);
    else if (type == "membrane") expected = &typeid(MembraneObject);
    else if (type == "rectmembrane") expected = &typeid(RectMembraneObject);
    else if (type == "passive_object") expected = &typeid(PassiveObject);
    else if (type == "active_object") expected = &typeid(ActiveObject);
    else if (type == "static_light") expected = &typeid(StaticLightObject);
    else if (type == "rotating_ray_of_light") expected = &typeid(RotatingRayOfLightObject);
    else if (type == "alternating_rays_of_light") expected = &typeid(AlternatingDualRayOfLightObject);
    check(expected && typeid(*object) == *expected, "Factory returned the wrong concrete object type");
    expect_error([&] { object->init(world.id); }, "called twice");
    b2World_Step(world.id, 0.001f, 4);
    if (auto* physical = dynamic_cast<PhysicalObject*>(object.get())) {
        // Pogowalls inherit the robot API but deliberately report themselves
        // as non-tangible; arena walls handle collisions instead.
        check(object->is_tangible() == (type != "pogowall"), "Factory tangibility changed");
        if (object->is_tangible()) {
            const auto position = physical->get_position();
            check(std::isfinite(position.x) && std::isfinite(position.y), "Factory produced an invalid physics position");
        }
    } else {
        check(!object->is_tangible(), "Light object became tangible");
        object->launch_user_step(0.01f);
        map.update();
        for (unsigned y = 0; y < 10; ++y) for (unsigned x = 0; x < 10; ++x)
            check(std::isfinite(map.get_light_level(x, y)), "Light factory produced non-finite map values");
    }
    // Runtime geometry ownership is currently non-owning. Reclaim the final
    // geometry in this fixture while its Box2D world still exists.
    std::unique_ptr<ObjectGeometry> geometry(object->get_geometry());
    object.reset();
}
