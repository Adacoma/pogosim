#include "pogosim/lights.h"
#include "pogosim/distances.h"
#undef main
#include "test_support.h"
#include <algorithm>
#include <memory>

void test_geometry() {
    DiskGeometry disk(2);
    const auto box = disk.compute_bounding_box();
    close_to(box.x, -2); close_to(box.y, -2);
    close_to(box.width, 4); close_to(box.height, 4);
    const auto circle = disk.generate_contours(16, {3, 4});
    check(circle.size() == 1 && circle[0].size() == 16, "Disk contour vertex count changed");
    for (const auto point : circle[0]) close_to(std::hypot(point.x - 3, point.y - 4), 2);
    check(disk.generate_contours(1)[0].size() == 3, "Disk minimum contour is not a triangle");
    const auto grid = disk.export_geometry_grid(6, 6, 1, 1, 2.5, 2.5);
    check(grid[2][2] && grid[2][4] && !grid[0][0] && !grid[5][5], "Disk grid/translation changed");

    RectangleGeometry rectangle(6, 8);
    close_to(rectangle.compute_bounding_disk().radius, 5);
    const auto corners = rectangle.generate_contours(4, {10, 20});
    check(corners.size() == 1 && corners[0].size() == 4, "Rectangle contour changed");
    for (const auto point : corners[0]) {
        close_to(std::fabs(point.x - 10), 3);
        close_to(std::fabs(point.y - 20), 4);
    }
    const auto rectangle_grid = rectangle.export_geometry_grid(10, 10, 1, 1, 4, 4);
    check(rectangle_grid[4][4] && !rectangle_grid[9][9], "Rectangle grid bounds changed");

    TriangleGeometry triangle(6);
    close_to(triangle.compute_bounding_disk().radius, 6 / std::sqrt(3.0));
    const auto vertices = triangle.generate_contours(3, {5, 5})[0];
    check(vertices.size() == 3, "Triangle contour changed");
    for (std::size_t i = 0; i < vertices.size(); ++i)
        close_to(euclidean_distance(vertices[i], vertices[(i + 1) % 3]), 6);

    arena_polygons_t polygons{{{0, 0}, {10, 0}, {10, 8}, {0, 8}}};
    ArenaGeometry arena(polygons);
    close_to(arena.compute_bounding_box().width, 10);
    close_to(arena.compute_bounding_box().height, 8);
    close_to(arena.get_distance_to({0, 0}, {5, 4}), 4);
    close_to(arena.get_distance_to({0, 0}, {12, 4}), 2);
    GlobalGeometry global;
    const auto everywhere = global.export_geometry_grid(3, 2, 1, 1, 99, 99);
    for (const auto& row : everywhere) for (const bool cell : row) check(cell, "Global geometry missed a bin");
}

void test_lighting() {
    LightLevelMap map(3, 2, 10, 20);
    check(map.get_num_bins_x() == 3 && map.get_num_bins_y() == 2, "Light map dimensions changed");
    close_to(map.get_bin_width(), 10); close_to(map.get_bin_height(), 20);
    map.set_light_level(0, 0, 11);
    map.set_light_level(2, 1, 22);
    close_to(map.get_light_level_at(5, 5), 11);
    // Characterize current clamping, not the outdated 'outside returns zero'
    // header comment: changing that behavior is outside this test-only task.
    close_to(map.get_light_level_at(-1, 5), 11);
    close_to(map.get_light_level_at(30, 40), 22);
    map.set_light_level(1, 0, 32760);
    map.add_light_level(1, 0, 10);
    close_to(map.get_light_level(1, 0), 32767);
    map.clear();
    close_to(map.get_light_level(1, 0), 0);
    unsigned calls = 0;
    map.register_callback([&](LightLevelMap& m) { ++calls; m.add_light_level(0, 0, 7); });
    map.register_callback([](LightLevelMap& m) { m.add_light_level(0, 0, 5); });
    map.update(); map.update();
    check(calls == 2, "Light callbacks not invoked once per update");
    close_to(map.get_light_level(0, 0), 12); // Must clear rather than accumulate across ticks.

    GlobalGeometry global;
    LightLevelMap uniform(3, 1, 1, 1);
    StaticLightObject constant(1.5, 0.5, global, &uniform, 100);
    uniform.update();
    for (unsigned x = 0; x < 3; ++x) close_to(uniform.get_light_level(x, 0), 100);

    LightLevelMap gradient(3, 1, 1, 1);
    StaticLightObject radial(1.5, 0.5, global, &gradient, 100,
                            StaticLightObject::LightMode::GRADIENT, 20, 1);
    gradient.update();
    close_to(gradient.get_light_level(1, 0), 100);
    close_to(gradient.get_light_level(0, 0), 20);
    close_to(gradient.get_light_level(2, 0), 20);

    LightLevelMap plane(3, 1, 1, 1);
    StaticLightObject linear(1.5, 0.5, global, &plane, 100,
                            StaticLightObject::LightMode::PLANE, 20, -1, 0, 1);
    plane.update();
    close_to(plane.get_light_level(0, 0), 100);
    close_to(plane.get_light_level(1, 0), 60);
    close_to(plane.get_light_level(2, 0), 20);

    LightLevelMap pulse_map(1, 1, 1, 1);
    StaticLightObject pulse(0.5, 0.5, global, &pulse_map, 100,
        StaticLightObject::LightMode::STATIC, 0, -1, 0, 1, 1, 0.5, 500);
    pulse_map.update(); close_to(pulse_map.get_light_level(0, 0), 0);
    pulse.launch_user_step(1.0); close_to(pulse_map.get_light_level(0, 0), 500);
    pulse.launch_user_step(1.49); close_to(pulse_map.get_light_level(0, 0), 500);
    pulse.launch_user_step(1.5); close_to(pulse_map.get_light_level(0, 0), 100);
}

void test_neighbors() {
    check(get_grid_cell(-0.1f, -2.1f, 2) == GridCell{-1, -2}, "Negative spatial cells truncate instead of floor");
    close_to(wrap01(-1, 10), 9); close_to(wrap01(21, 10), 1);
    close_to(delta_periodic(9, 10), -1); close_to(delta_periodic(-9, 10), 1);
    close_to(delta_periodic(-5, 10), 5); // Existing half-domain tie convention.
    check(angles::in_fov(-3.13f, 3.13f, 0.1f), "FOV failed across the angle seam");
    std::vector<angles::Interval> intervals;
    angles::add_interval(0, 1, intervals);
    angles::add_interval(2, 3, intervals);
    check(!angles::fully_covered(0.5, 2.5, intervals), "LOS gap incorrectly covered");
    angles::add_interval(0.5, 2.5, intervals);
    check(intervals.size() == 1 && angles::fully_covered(0.1, 2.9, intervals), "LOS interval merge failed");

    // Non-overlapping distances make ordering deterministic across STL versions.
    std::vector<float> xs{0, 2, 4, -2, 20}, ys(5, 0), radii(5, 0.5f), ranges(5, 5), directions(5, 0);
    const auto hash = build_spatial_hash(xs, ys, 10);
    const auto clipped = collect_candidates(0, xs, ys, xs, ys, radii, ranges, directions, hash, 10, true);
    check(clipped.size() == 2 && clipped[0].idx == 1 && clipped[1].idx == 2, "Range/FOV/self filtering failed");
    const auto visible = filter_visible(clipped);
    check(visible == std::vector<std::size_t>{1}, "Collinear far robot was not occluded");
    const auto unfiltered = collect_candidates(0, xs, ys, xs, ys, radii, ranges, directions, hash, 10, false);
    check(unfiltered.size() == 3, "Disabling occlusion no longer includes rear candidates");
    const auto seam = filter_visible({{1, 1, 3.13f, 0.3f}, {2, 4, -3.13f, 0.1f}});
    check(seam == std::vector<std::size_t>{1}, "Occlusion failed across +/-pi");
    std::vector<float> px{0.1f, 9.9f}, py{5, 5};
    const auto periodic = build_spatial_hash_periodic(px, py, 1, {0, 0}, 10, 10);
    check(periodic.find(GridCell{-1, 5}) != periodic.end(), "Periodic neighbor ghost was not inserted");
}

void test_probabilities() {
    ConstMsgSuccessRate constant(0.25);
    close_to(constant(100, 0.5, 20), 0.25);
    DynamicMsgSuccessRate dynamic(1, 1, 1, 1);
    close_to(dynamic(2, 3, 4), 1.0 / 25, 1e-12);
    check(dynamic(4, 3, 4) < dynamic(2, 3, 4), "Message-size reception dependence changed");
    check(dynamic(2, 6, 4) < dynamic(2, 3, 4), "Traffic reception dependence changed");
    check(dynamic(2, 3, 8) < dynamic(2, 3, 4), "Density reception dependence changed");
    std::unique_ptr<MsgSuccessRate> configured(msg_success_rate_factory(Configuration(YAML::Load("{type: STATIC, rate: 0.75}"))));
    close_to((*configured)(100, 1, 100), 0.75);
    expect_error([] { std::unique_ptr<MsgSuccessRate> bad(msg_success_rate_factory(Configuration(YAML::Load("{type: nonexistent}")))); }, "Unknown msg_success_rate type");
}
