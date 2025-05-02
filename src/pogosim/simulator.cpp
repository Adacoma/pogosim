#include <iostream>
#include <string>
#include <chrono>
#include <sstream>
#include <iomanip>

#include <fmt/format.h>
#include <yaml-cpp/yaml.h>
#include <unordered_map>
#include <spdlog/spdlog.h>
#include <filesystem>
#include <algorithm>

#include <cmath>
#include <vector>
#include <SDL2/SDL.h>
#include <box2d/box2d.h>
#include "fpng.h"
#include "SDL2_gfxPrimitives.h"

#include "version.h"
#include "tqdm.hpp"
#include "utils.h"
#include "simulator.h"
#include "render.h"
#include "distances.h"
#include "spogobot.h"
#undef main         // We defined main() as robot_main() in pogobot.h


void set_current_robot(PogobotObject& robot) {
    // Store values of previous robot
    if (current_robot != nullptr) {
        current_robot->callback_export_data          = callback_export_data;
        current_robot->pogobot_ticks                 = pogobot_ticks;
        current_robot->main_loop_hz                  = main_loop_hz;
        current_robot->max_nb_processed_msg_per_tick = max_nb_processed_msg_per_tick;
        current_robot->msg_rx_fn                     = msg_rx_fn;
        current_robot->msg_tx_fn                     = msg_tx_fn;
        current_robot->error_codes_led_idx           = error_codes_led_idx;
        current_robot->_global_timer                 = _global_timer;
        current_robot->timer_main_loop               = timer_main_loop;
        current_robot->_current_time_milliseconds    = _current_time_milliseconds;
        current_robot->_error_code_initial_time      = _error_code_initial_time;
        current_robot->percent_msgs_sent_per_ticks   = percent_msgs_sent_per_ticks;
        current_robot->nb_msgs_sent                  = nb_msgs_sent;
        current_robot->nb_msgs_recv                  = nb_msgs_recv;
    }

    current_robot = &robot;
    mydata = robot.data;

    // Update robot values
    callback_export_data          = robot.callback_export_data;
    pogobot_ticks                 = robot.pogobot_ticks;
    main_loop_hz                  = robot.main_loop_hz;
    max_nb_processed_msg_per_tick = robot.max_nb_processed_msg_per_tick;
    msg_rx_fn                     = robot.msg_rx_fn;
    msg_tx_fn                     = robot.msg_tx_fn;
    error_codes_led_idx           = robot.error_codes_led_idx;
    _global_timer                 = robot._global_timer;
    timer_main_loop               = robot.timer_main_loop;
    _current_time_milliseconds    = robot._current_time_milliseconds;
    _error_code_initial_time      = robot._error_code_initial_time;
    percent_msgs_sent_per_ticks   = robot.percent_msgs_sent_per_ticks;
    nb_msgs_sent                  = robot.nb_msgs_sent;
    nb_msgs_recv                  = robot.nb_msgs_recv;
}


/************* SIMULATION *************/ // {{{1

std::unique_ptr<Simulation> simulation;

Simulation::Simulation(Configuration& _config)
        : config(_config) {
    init_config();
    init_console_logger();
    init_box2d();
    init_SDL();
}

Simulation::~Simulation() {
    FC_FreeFont(font);
    TTF_Quit();
    b2DestroyWorld(worldId);
    if (renderer)
        SDL_DestroyRenderer(renderer);
    if (window)
        SDL_DestroyWindow(window);
    SDL_Quit();
}

void Simulation::init_all() {
    //create_walls();
    create_arena();
    create_objects();
    create_robots();
}

void Simulation::create_objects() {
    uint16_t current_id = 0;
    uint16_t current_other_id = 65535;
    std::vector<std::shared_ptr<Object>> objects_to_move;
    std::vector<float> objects_radii;

    // Create light map
    size_t num_bin_x = 100.0f;
    size_t num_bin_y = 100.0f;
    float bin_width = arena_width / num_bin_x;
    float bin_height = arena_height / num_bin_y;
    light_map.reset(new LightLevelMap(num_bin_x, num_bin_y, bin_width, bin_height));

    // Parse the configuration, and create objects as needed
    for (const auto& [name, obj_config] : config["objects"].children()) {
        // Find number of objects of this category
        size_t nb = obj_config["nb"].get(1);

        // XXX
        // Identify the userspace for this category
        size_t userdatasize = UserdataSize; // XXX

        // Generate all objects of this category
        std::vector<std::shared_ptr<Object>> obj_vec;
        for (size_t i = 0; i < nb; ++i) {
            // Check if this object has an initial coordinate
            float x = obj_config["x"].get(NAN);
            float y = obj_config["y"].get(NAN);

            // Create object from configuration
            if (std::isnan(x) or std::isnan(y)) {
                obj_vec.emplace_back(object_factory(this, current_id, 0.0f, 0.0f, worldId, obj_config, light_map.get(), userdatasize, name));
                if (obj_vec.back()->is_tangible()) {
                    objects_to_move.push_back(obj_vec.back());
                    float radius = obj_vec.back()->get_geometry()->compute_bounding_disk().radius;
                    if (radius < formation_min_space_between_neighbors)
                        radius = formation_min_space_between_neighbors;
                    objects_radii.push_back(radius);
                }
            } else {
                obj_vec.emplace_back(object_factory(this, current_id, x, y, worldId, obj_config, light_map.get(), userdatasize, name));
            }

            // Check if the object is a robot, and store it if this is the case
            if (auto wall = std::dynamic_pointer_cast<Pogowall>(obj_vec.back())) {
                wall->id = current_other_id;
                current_other_id--;
                wall_objects.push_back(wall);
                robots.push_back(wall);
            } else if (auto robot = std::dynamic_pointer_cast<PogobotObject>(obj_vec.back())) {
                robots.push_back(robot);
                current_id++;
                // Update max communication radius
                float const tot_radius = robot->radius + robot->communication_radius;
                if (max_comm_radius < tot_radius)
                    max_comm_radius = tot_radius;
            } else {
                non_robots.push_back(obj_vec.back());
            }
        }
        objects[name] = std::move(obj_vec);
    }

    // Generate random coordinates for all objects of all categories
    std::vector<b2Vec2> points;
    try {
        if (initial_formation == "random") {
            points = generate_random_points_within_polygon_safe(arena_polygons, objects_radii, formation_max_space_between_neighbors, formation_attempts_per_point, formation_max_restarts);
        } else if (initial_formation == "disk") {
            points = generate_regular_disk_points_in_polygon(arena_polygons, objects_radii);
        } else {
            glogger->error("Unknown 'initial_formation' value: '{}'. Assuming random formation...", initial_formation);
            points = generate_random_points_within_polygon_safe(arena_polygons, objects_radii, formation_max_space_between_neighbors, formation_attempts_per_point, formation_max_restarts);
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("Impossible to create robots (number may be too high for the provided arena): " + std::string(e.what()));
    }

    // Move all objects to the new coordinates
    size_t current_point_idx = 0;
    for (const auto& obj : objects_to_move) {
        float const x = points[current_point_idx].x;
        float const y = points[current_point_idx].y;
        obj->move(x, y);
        current_point_idx++;
    }

    // Update the light map
    light_map->update();
}


void Simulation::create_arena() {
    std::string const csv_file = resolve_path(config["arena_file"].get(std::string("test.csv")));

    float const friction = 0.05f;
    float const restitution = 1.8f; // Bounciness
    float const WALL_THICKNESS = 1.0f / VISUALIZATION_SCALE; // Thickness of the wall in SDL units
             // Careful! Values higher than 1.0 / VISUALIZATION_SCALE results in robots outside arena

    // Read multiple polygons from the CSV file
    //arena_polygons = read_poly_from_csv(csv_file, arena_width, arena_height);
    arena_polygons = read_poly_from_csv(csv_file, arena_surface);
    if (arena_polygons.empty()) {
        glogger->error("Error: No polygons found in the arena file");
        throw std::runtime_error("No polygons found in the arena file or unable to open arena file");
    }

    // Compute the bounding box of the main polygon
    std::tie(arena_width, arena_height) = compute_polygon_dimensions(arena_polygons[0]);

    // Process each polygon
    for (auto& polygon : arena_polygons) {
        if (polygon.size() < 2) {
            glogger->error("Error: A polygon must have at least two points to create walls.");
            continue;
        }

        // Remove the duplicate closing vertex, if it is present
        if (!polygon.empty() && polygon.front() == polygon.back()) {
            polygon.pop_back();
        }

        std::vector<b2Vec2> outer_polygon = offset_polygon(polygon, -1.0f * WALL_THICKNESS);

        // Define the static body for each wall segment
        b2BodyDef wallBodyDef = b2DefaultBodyDef();
        wallBodyDef.type = b2_staticBody;

        b2Vec2 p1;
        b2Vec2 p2;
        for (size_t i = 0; i < outer_polygon.size(); ++i) {
            if (i < outer_polygon.size() - 1) {
                p1 = outer_polygon[i];
                p2 = outer_polygon[i + 1];
            } else {
                p1 = outer_polygon[i];
                p2 = outer_polygon[0];
            }

            // Calculate the center of the rectangle
            b2Vec2 center = (p1 + p2) * 0.5f * (1.0f/VISUALIZATION_SCALE);

            // Calculate the angle of the rectangle
            float angle = atan2f(p2.y - p1.y, p2.x - p1.x);

            // Calculate the length of the rectangle
            float length = b2Distance(p1, p2) / VISUALIZATION_SCALE;

            // Create the wall body
            wallBodyDef.position = center;
            wallBodyDef.rotation = b2MakeRot(angle);
            b2BodyId wallBody = b2CreateBody(worldId, &wallBodyDef);

            // Create the rectangular shape
            b2Polygon wallShape = b2MakeBox(length / 2, WALL_THICKNESS / 2);
            b2ShapeDef wallShapeDef = b2DefaultShapeDef();
            wallShapeDef.friction = friction;
            wallShapeDef.restitution = restitution;

            b2CreatePolygonShape(wallBody, &wallShapeDef, &wallShape);
        }
    }

    glogger->info("Arena walls created from CSV file: {}", csv_file);

    // Adjust mm_to_pixels to show the entire arena, by default
    float const ratio_width  = window_width  / arena_width;
    float const ratio_height = window_height / arena_height;
    float const ratio = std::min(ratio_width, ratio_height);
    mm_to_pixels = 0.0f;
    adjust_mm_to_pixels(ratio);
    config.set("mm_to_pixels", std::to_string(mm_to_pixels));
}


void Simulation::create_walls() {
    float const WALL_THICKNESS = 30.0f / VISUALIZATION_SCALE; // Thickness of the wall in Box2D units (30 pixels)
    float const offset = 30.0f / VISUALIZATION_SCALE;        // Offset from the window edge in Box2D units
    float const width = (window_width - 2 * 30) / VISUALIZATION_SCALE; // Width adjusted for 30-pixel offset
    float const height = (window_height - 2 * 30) / VISUALIZATION_SCALE; // Height adjusted for 30-pixel offset
    float const friction = 0.03f;
    float const restitution = 10.8f; // Bounciness

    // Define the static body for each wall
    b2BodyDef wallBodyDef = b2DefaultBodyDef();
    wallBodyDef.type = b2_staticBody;

    // Bottom wall
    wallBodyDef.position = {offset + width / 2, offset - WALL_THICKNESS / 2};
    b2BodyId bottomWall = b2CreateBody(worldId, &wallBodyDef);

    b2Polygon bottomShape = b2MakeBox(width / 2, WALL_THICKNESS / 2);
    b2ShapeDef bottomShapeDef = b2DefaultShapeDef();
    bottomShapeDef.friction = friction;
    bottomShapeDef.restitution = restitution;
    b2CreatePolygonShape(bottomWall, &bottomShapeDef, &bottomShape);

    // Top wall
    wallBodyDef.position = {offset + width / 2, offset + height + WALL_THICKNESS / 2};
    b2BodyId topWall = b2CreateBody(worldId, &wallBodyDef);
    b2Polygon topShape = b2MakeBox(width / 2, WALL_THICKNESS / 2);
    b2ShapeDef topShapeDef = b2DefaultShapeDef();
    topShapeDef.friction = friction;
    topShapeDef.restitution = restitution;
    b2CreatePolygonShape(topWall, &topShapeDef, &topShape);

    // Left wall
    wallBodyDef.position = {offset - WALL_THICKNESS / 2, offset + height / 2};
    b2BodyId leftWall = b2CreateBody(worldId, &wallBodyDef);
    b2Polygon leftShape = b2MakeBox(WALL_THICKNESS / 2, height / 2);
    b2ShapeDef leftShapeDef = b2DefaultShapeDef();
    leftShapeDef.friction = friction;
    leftShapeDef.restitution = restitution;
    b2CreatePolygonShape(leftWall, &leftShapeDef, &leftShape);

    // Right wall
    wallBodyDef.position = {offset + width + WALL_THICKNESS / 2, offset + height / 2};
    b2BodyId rightWall = b2CreateBody(worldId, &wallBodyDef);
    b2Polygon rightShape = b2MakeBox(WALL_THICKNESS / 2, height / 2);
    b2ShapeDef rightShapeDef = b2DefaultShapeDef();
    rightShapeDef.friction = friction;
    rightShapeDef.restitution = restitution;
    b2CreatePolygonShape(rightWall, &rightShapeDef, &rightShape);
}


void Simulation::init_box2d() {
    // Initialize Box2D world
    b2Vec2 gravity = {0.0f, 0.0f}; // No gravity for robots
    b2WorldDef worldDef = b2DefaultWorldDef();
    worldDef.gravity = gravity;
    worldId = b2CreateWorld(&worldDef);
}


void Simulation::init_config() {
    glogger->info("Welcome to the Pogosim simulator, version {}", POGOSIM_VERSION);

    window_width = config["window_width"].get(800);
    window_height = config["window_height"].get(800);

    arena_surface = config["arena_surface"].get(1e6f);

    mm_to_pixels = 0.0f;
    adjust_mm_to_pixels(config["mm_to_pixels"].get(1.0f));
    show_comm = config["show_communication_channels"].get(false);
    show_lateral_leds = config["show_lateral_LEDs"].get(true);
    show_light_levels = config["show_light_levels"].get(false);

    initial_formation = config["initial_formation"].get(std::string("random"));
    formation_min_space_between_neighbors = config["formation_min_space_between_neighbors"].get(0.0f);
    formation_max_space_between_neighbors = config["formation_max_space_between_neighbors"].get(INFINITY);
    formation_attempts_per_point = config["formation_attempts_per_point"].get(100U);
    formation_max_restarts = config["formation_max_restarts"].get(100U);

    enable_gui = config["GUI"].get(true);
    GUI_speed_up = config["GUI_speed_up"].get(1.0f);

    std::srand(std::time(nullptr));
}


void Simulation::init_SDL() {
    if (!enable_gui) {
        /*--------------------------------------------------------------------
          In SDL < 2.0.22 the hint does not exist, but you can still achieve
          the same effect by setting the environment variable instead.
          ------------------------------------------------------------------*/
#if SDL_VERSION_ATLEAST(2, 0, 22)   // compile-time check :contentReference[oaicite:1]{index=1}
        SDL_SetHint(SDL_HINT_VIDEODRIVER, "offscreen");
#else
        /*  SDL_setenv appeared in 2.0.2; fall back for older headers */
        SDL_setenv("SDL_VIDEODRIVER", "offscreen", /*overwrite =*/1);
#endif
    }

    // Initialize SDL
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        SDL_Log("Failed to initialize SDL: %s", SDL_GetError());
        throw std::runtime_error("Error while initializing SDL");
    }

    if (enable_gui) {
        window = SDL_CreateWindow("Swarm Robotics Simulator with Walls",
                SDL_WINDOWPOS_CENTERED,
                SDL_WINDOWPOS_CENTERED,
                window_width, window_height,
                SDL_WINDOW_SHOWN);
    } else {
        window = SDL_CreateWindow("Swarm Robotics Simulator with Walls",
                SDL_WINDOWPOS_CENTERED,
                SDL_WINDOWPOS_CENTERED,
                window_width, window_height,
                SDL_WINDOW_HIDDEN);
    }
    if (!window) {
        SDL_Log("Failed to create window: %s", SDL_GetError());
        SDL_Quit();
        throw std::runtime_error("Error while initializing SDL");
    }

    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (!renderer) {
        SDL_Log("Failed to create renderer: %s", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        throw std::runtime_error("Error while initializing SDL");
    }

    // Init fpng
    fpng::fpng_init();

    // Init fonts
    font = FC_CreateFont();
    //FC_LoadFont(font, renderer, "fonts/helvetica.ttf", 20, FC_MakeColor(0,0,0,255), TTF_STYLE_NORMAL);  
    FC_LoadFont(font, renderer, resolve_path("fonts/helvetica.ttf").c_str(), 20, FC_MakeColor(0,0,0,255), TTF_STYLE_NORMAL);  
}


void Simulation::create_robots() {
    current_robot = robots.front().get();

    glogger->info("Initializing all robots...");
    // Launch main() on all robots
    for (auto robot : robots) {
        set_current_robot(*robot.get());
        robot_main();
    }

    // If there is a global_setup callback, call it
    glogger->info("Global initialization...");
    if(callback_global_setup != nullptr) {
        callback_global_setup();
    }

    // Setup all robots
    for (auto robot : robots) {
        set_current_robot(*robot.get());
        if (current_robot->user_init != nullptr)
            current_robot->user_init();
    }
}


void Simulation::speed_up() {
    if (GUI_speed_up <= 1e05)
        GUI_speed_up *= 1.2;
    glogger->info("Setting GUI speed up to {}", GUI_speed_up);
}

void Simulation::speed_down() {
    if (GUI_speed_up >= 1e-04)
        GUI_speed_up *= 0.8;
    glogger->info("Setting GUI speed up to {}", GUI_speed_up);
}

void Simulation::pause() {
    paused = !paused;
}

void Simulation::help_message() {
    glogger->info("Welcome to the Pogosim's GUI. This is an help message...");
    glogger->info("Here is a list of shortcuts that can be used to control the GUI:");
    glogger->info(" - F1: Help message");
    glogger->info(" - F3: Slow down the simulation");
    glogger->info(" - F4: Speed up the simulation");
    glogger->info(" - F5: Show/Hide the communication channels");
    glogger->info(" - F6: Show/Hide the lateral LEDs");
    glogger->info(" - F7: Show/Hide the light level");
    glogger->info(" - ESC: quit the simulation");
    glogger->info(" - SPACE: pause the simulation");
    glogger->info(" - DOWN, UP, LEFT, RIGHT: move the visualisation coordinates");
    glogger->info(" - Right-Click + Mouse move: move the visualisation coordinates");
    glogger->info(" - PLUS, MINUS or Mouse Wheel: Zoom up or down");
    glogger->info(" - 0: Reset the zoom and visualization coordinates");
}


void Simulation::handle_SDL_events() {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        if (event.type == SDL_QUIT) {
            running = false;

        } else if (event.type == SDL_KEYDOWN) {
            switch (event.key.keysym.sym) {
                case SDLK_F1:
                    help_message();
                    break;
                case SDLK_F3:
                    speed_down();
                    break;
                case SDLK_F4:
                    speed_up();
                    break;
                case SDLK_F5:
                    show_comm = !show_comm;
                    break;
                case SDLK_F6:
                    show_lateral_leds = !show_lateral_leds;
                    break;
                case SDLK_F7:
                    show_light_levels = !show_light_levels;
                    break;
                case SDLK_ESCAPE:
                    running = false;
                    break;
                case SDLK_SPACE:
                    pause();
                    break;
                case SDLK_UP:
                    visualization_y += 10.0f * (1.f / mm_to_pixels);
                    break;
                case SDLK_DOWN:
                    visualization_y -= 10.0f * (1.f / mm_to_pixels);
                    break;
                case SDLK_LEFT:
                    visualization_x += 10.0f * (1.f / mm_to_pixels);
                    break;
                case SDLK_RIGHT:
                    visualization_x -= 10.0f * (1.f / mm_to_pixels);
                    break;
                case SDLK_PLUS:
                    adjust_mm_to_pixels(0.1);
                    break;
                case SDLK_MINUS:
                    adjust_mm_to_pixels(-0.1);
                    break;
                case SDLK_0:
                    visualization_x = 0.0f;
                    visualization_y = 0.0f;
                    mm_to_pixels = 0.0f;
                    adjust_mm_to_pixels(config["mm_to_pixels"].get(1.0f));
                    break;
            }

        } else if (event.type == SDL_MOUSEWHEEL) {
            if (event.wheel.y > 0) {
                adjust_mm_to_pixels(0.1);
            } else if (event.wheel.y < 0) {
                adjust_mm_to_pixels(-0.1);
            }

        } else if (event.type == SDL_MOUSEBUTTONDOWN) {
            if (event.button.button == SDL_BUTTON_RIGHT) {
                dragging_pos_by_mouse = true;
                last_mouse_x = event.button.x;
                last_mouse_y = event.button.y;
            }

        } else if (event.type == SDL_MOUSEBUTTONUP) {
            if (event.button.button == SDL_BUTTON_RIGHT) {
                dragging_pos_by_mouse = false;
            }

        } else if (event.type == SDL_MOUSEMOTION) {
            if (dragging_pos_by_mouse) {
                int dx = event.motion.x - last_mouse_x;
                int dy = event.motion.y - last_mouse_y;
                visualization_x += dx;
                visualization_y += dy;
                last_mouse_x = event.motion.x;
                last_mouse_y = event.motion.y;
                //printf("Visualization moved to: (%d, %d)\n", visualization_x, visualization_y);
            }
        }


    }
}


void Simulation::compute_neighbors() {
    // Find robots that are neighbors
    for (int i = 0; i < IR_RX_COUNT; i++ ) {
        find_neighbors((ir_direction)i, robots, max_comm_radius / VISUALIZATION_SCALE);
        find_neighbors_to_pogowalls(wall_objects, (ir_direction)i, robots);
    }

    // Merge neighbors (without duplicates) from all IR emitters/receivers into the direction ir_all
    for (auto a : robots) {
        for (std::size_t i = 0; i < IR_RX_COUNT; ++i) {
            for (auto* r : a->neighbors[i]) {
                // Check if r is already in neighbors[ir_all]
                if (std::find(a->neighbors[ir_all].begin(), a->neighbors[ir_all].end(), r) == a->neighbors[ir_all].end()) {
                    a->neighbors[ir_all].push_back(r);
                }
            }
        }
    }

    //glogger->debug("Robot 0 has {} neighbors.", robots[0].neighbors.size());
}


void Simulation::init_callbacks() {
    init_data_logger();
}

void Simulation::init_data_logger() {
    enable_data_logging = config["enable_data_logging"].get(true);
    if (!enable_data_logging)
        return;
    std::string data_filename = config["data_filename"].get(std::string("data.feather"));
    if (data_filename.size() == 0) {
        throw std::runtime_error("'enable_data_logging' is set to true, but 'data_filename' is empty.");
    }

    data_logger = std::make_unique<DataLogger>();

    // Init base schema
    data_logger->add_field("time", arrow::float64());
    data_logger->add_field("robot_category", arrow::utf8());
    data_logger->add_field("robot_id", arrow::int32());
    data_logger->add_field("pogobot_ticks", arrow::int64());
    data_logger->add_field("x", arrow::float64());
    data_logger->add_field("y", arrow::float64());
    data_logger->add_field("angle", arrow::float64());

    // Init user-defined schema
    if (robots.size() > 0 && callback_create_data_schema != nullptr) {
        callback_create_data_schema();
    }

    // Open data logger file
    data_logger->open_file(data_filename);
}

void Simulation::init_console_logger() {
    bool const enable_console_logging = config["enable_console_logging"].get(false);
    if (!enable_console_logging)
        return;
    std::string const console_filename = config["console_filename"].get(std::string("console.txt"));
    if (console_filename.size() == 0) {
        throw std::runtime_error("'enable_console_logging' is set to true, but 'console_filename' is empty.");
    }
    loggers_add_file_sink(console_filename);
}


void Simulation::draw_scale_bar() {
    // Get the window size
    int window_width, window_height;
    SDL_GetWindowSize(window, &window_width, &window_height);

    float mm_scale = 100.0f;

    int bar_length = (int)(mm_scale * mm_to_pixels);
    //int bar_thickness = 3; // Thickness of the line

    // Define start and end points of the scale bar
    int x1 = 10;
    int y1 = window_height - 30;
    int x2 = x1 + bar_length;
    int y2 = y1;

    // Draw the scale bar (horizontal line)
    thickLineRGBA(renderer, x1, y1, x2, y2, 4, 0, 0, 0, 255);

    // Render the scale
    std::string formatted_scale = fmt::format("{:.0f} mm", mm_scale);
    FC_Draw(font, renderer, x1, y1 + 5, "%s", formatted_scale.c_str()); 
}


void Simulation::render_all() {
    if (show_light_levels) {
        SDL_RenderClear(renderer);
        light_map->render(renderer);
    } else {
        uint8_t background_level = 200.0f;
        SDL_SetRenderDrawColor(renderer, background_level, background_level, background_level, 255); // Grey background
        SDL_RenderClear(renderer);
    }

    //renderWalls(renderer); // Render the walls
    for(auto const& poly : arena_polygons) {
        draw_polygon(renderer, poly);
    }

    // Render objects
    for (auto robot : robots) {
        robot->show_comm = show_comm;
        robot->show_lateral_leds = show_lateral_leds;
        robot->render(renderer, worldId);
    }
    //SDL_RenderPresent(renderer);

    for (const auto& [cat_name, obj_vec] : objects) {
        for (auto const& obj : obj_vec) {
            obj->render(renderer, worldId);
        }
    }

    // Get the window size
    int windowWidth, windowHeight;
    SDL_GetWindowSize(window, &windowWidth, &windowHeight);

    // Render the current time
    std::string formatted_time  = fmt::format("{:.4f}s", t);
    FC_Draw(font, renderer, windowWidth - 120, 10, "t=%s", formatted_time.c_str()); 

    // Render the scale bar
    draw_scale_bar();
}

void Simulation::export_frames() {
    // If wanted, export to PNG
    float const save_video_period = config["save_video_period"].get(-1.0f);
    std::string const frames_name = config["frames_name"].get(std::string("frames/f{:010.4f}.png"));
    if (save_video_period > 0.0 && frames_name.size()) {
        //float const time_step_duration = config["time_step"].get(0.01667f);
        if (t >= last_frame_saved_t + save_video_period) {
            last_frame_saved_t = t;
            std::string formatted_filename = fmt::format(fmt::runtime(frames_name), t);
            save_window_to_png(renderer, window, formatted_filename);
        }
    }
}

void Simulation::export_data() {
    for (auto robot : robots) {
        data_logger->set_value("time", t);
        data_logger->set_value("robot_category", robot->category);
        data_logger->set_value("robot_id", (int32_t) robot->id);
        data_logger->set_value("pogobot_ticks", (int64_t) robot->pogobot_ticks);
        auto const pos = robot->get_position();
        data_logger->set_value("x", pos.x);
        data_logger->set_value("y", pos.y);
        data_logger->set_value("angle", robot->get_angle());

        // User-defined values
        if (robot->callback_export_data != nullptr) {
            set_current_robot(*robot.get());
            robot->callback_export_data();
        }

        data_logger->save_row();
    }
}


void Simulation::main_loop() {
    // Delete old data, if needed
    delete_old_data();

    bool const progress_bar = config["progress_bar"].get(false);
    double const simulation_time = config["simulation_time"].get(100.0f);
    glogger->info("Launching the main simulation loop.");

    // Print an help message with the GUI keyboard shortcuts
    if (enable_gui)
        help_message();

    double const save_data_period = config["save_data_period"].get(1.0f);
    double const save_video_period = config["save_video_period"].get(-1.0f);
    double time_step_duration = config["time_step"].get(0.01f);
    double GUI_frame_period;

    //sim_starting_time = std::chrono::system_clock::now();
    sim_starting_time_microseconds = get_current_time_microseconds();

    // Prepare main loop
    running = true;
    t = 0.0f;
    last_frame_shown_t = 0.0f - time_step_duration;
    last_frame_saved_t = 0.0f - time_step_duration;
    last_data_saved_t = 0.0f - time_step_duration;
    uint32_t const max_nb_ticks = std::ceil(simulation_time / time_step_duration);
    auto tqdmrange = tq::trange(max_nb_ticks);
    if (progress_bar) {
        tqdmrange.begin();
        tqdmrange.update();
    }
    double gui_delay = time_step_duration;

    // Main loop for all robots
    while (running && t < simulation_time) {
        handle_SDL_events();

        // Check if the simulation is paused
        if (enable_gui && paused) {
            render_all();
            SDL_RenderPresent(renderer);
            // Delay
            SDL_Delay(time_step_duration / GUI_speed_up);
            continue;
        }

        // Adjust simulation speed
        gui_delay = time_step_duration / GUI_speed_up;
        GUI_frame_period = time_step_duration * GUI_speed_up;

        // Launch user code on normal objects
        for (auto obj : non_robots) {
            obj->launch_user_step(t);
        }

        // Launch user code on robots
        for (auto robot : robots) {
            set_current_robot(*robot.get());
            // Check if the robot has waited enough time
            //glogger->debug("Debug main loop. t={}  robot.current_time_microseconds={}", t * 1000000.0f, robot.current_time_microseconds);
            if (t * 1000000.0f >= robot->current_time_microseconds) {
                robot->launch_user_step(t);
            }
            // Check if dt is enough to simulate the main loop frequency of this robot
            double const main_loop_period = 1.0f / main_loop_hz;
            if (time_step_duration > main_loop_period) {
                glogger->warn("Time step duration dt={} is not enough to simulate a main loop frequency of {}. Adjusting to {}", time_step_duration, main_loop_hz, main_loop_period);
                time_step_duration = main_loop_period;
            }
        }
        //glogger->debug("Global: t={}  Robot0: t={}", t, robots[0]._current_time_milliseconds);

        // Step the Box2D world
        b2World_Step(worldId, time_step_duration, sub_step_count);

        // Compute neighbors
        compute_neighbors();

        // Save data, if needed
        if (enable_data_logging && t >= last_data_saved_t + save_data_period) {
            last_data_saved_t = t;
            export_data();
        }

        if (enable_gui) {
            if (    (t >= last_frame_shown_t + GUI_frame_period) ||
                    (save_video_period > 0.0 && t >= last_frame_saved_t + save_video_period) ) {
                // Render
                render_all();
                export_frames();
                SDL_RenderPresent(renderer);
                last_frame_shown_t = t;
                // Delay
                SDL_Delay(gui_delay);
            }
        } else {
            if (save_video_period > 0.0 && t >= last_frame_saved_t + save_video_period) {
                render_all();
                export_frames();
            }
        }

        // Update global time
        t += time_step_duration;

        if (progress_bar) {
            tqdmrange << 1;
            tqdmrange.update();
        }
    }

    // End progress bar, if needed
    if (progress_bar) {
        tqdmrange.end();
    }
}

void Simulation::delete_old_data() {
    bool const delete_old_files = config["delete_old_files"].get(false);
    if (delete_old_files) {
        std::string const frames_name = config["frames_name"].get(std::string("frames/f{:06.4f}.png"));
        std::filesystem::path filePath(frames_name);
        std::filesystem::path directory = filePath.parent_path();
        glogger->info("Deleting old data files in directory: {}", directory.string());
        delete_files_with_extension(directory, ".png", false);
    }
}

DataLogger* Simulation::get_data_logger() {
    return data_logger.get();
}


Configuration& Simulation::get_config() {
    return config;
}

LightLevelMap* Simulation::get_light_map() {
    return light_map.get();
}


bool parse_arguments(int argc, char* argv[], std::string& config_file, bool& verbose, bool& quiet, bool& do_not_show_robot_msg, bool& gui, bool& progress) {
    verbose = false;
    quiet = false;
    do_not_show_robot_msg = false;
    gui = true;
    progress = false;
    config_file.clear();

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-c" || arg == "--config") {
            if (i + 1 < argc) {
                config_file = argv[++i];
            } else {
                std::cerr << "Error: -c requires a configuration file argument." << std::endl;
                return false;
            }
        } else if (arg == "-g" || arg == "--no-GUI") {
            gui = false;
        } else if (arg == "-v" || arg == "--verbose") {
            verbose = true;
        } else if (arg == "-q" || arg == "--quiet") {
            quiet = true;
        } else if (arg == "-nr" || arg == "--do-not-show-robot-msg") {
            do_not_show_robot_msg = true;
        } else if (arg == "-P" || arg == "--progress") {
            progress = true;
        } else if (arg == "-V" || arg == "--version") {
            std::cout << "Pogosim simulator. Version " << POGOSIM_VERSION << "." << std::endl;
            return false;
        } else if (arg == "-h" || arg == "--help") {
            print_help();
            return false;
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            return false;
        }
    }

    return true;
}

void print_help() {
    std::cout << "Usage: pogosim [options]\n"
              << "Options:\n"
              << "  -c, --config <file>             Specify the configuration file.\n"
              << "  -g, --no-GUI                    Disable GUI mode.\n"
              << "  -v, --verbose                   Enable verbose mode.\n"
              << "  -q, --verbose                   Enable quiet mode (ouput only warning and errors on terminal).\n"
              << "  -nr, --do-not-show-robot-msg    Suppress robot messages.\n"
              << "  -P, --progress                  Show progress output.\n"
              << "  -V, --version                   Show version information.\n"
              << "  -h, --help                      Display this help message.\n";
}


int main(int argc, char** argv) {
    std::string config_file;
    bool verbose = false;
    bool quiet = false;
    bool do_not_show_robot_msg = false;
    bool gui = true;
    bool progress = false;

    // Parse command-line arguments
    if (!parse_arguments(argc, argv, config_file, verbose, quiet, do_not_show_robot_msg, gui, progress)) {
        std::cerr << "Usage: " << argv[0] << " -c CONFIG_FILE [-v/-q] [-nr] [-g] [-P] [-V]" << std::endl;
        return 1;
    }

    // Init logging
    init_logger();

    if (quiet) {
        // Quiet mode, only output warnings and error on terminal
        auto glogger_console_sink = glogger->sinks().front();
        glogger_console_sink->set_level(spdlog::level::warn);
        auto robotlogger_console_sink = robotlogger->sinks().front();
        robotlogger_console_sink->set_level(spdlog::level::warn);
    } else if (verbose) {
        // Enable verbose mode if requested
        glogger->info("Verbose mode enabled.");
        glogger->set_level(spdlog::level::debug);
        robotlogger->set_level(spdlog::level::debug);
    }

    if (do_not_show_robot_msg) {
        robotlogger->sinks().clear();
    }

    if (gui) {
        glogger->info("GUI enabled.");
    }

    Configuration config;
    try {
        // Load configuration
        config.load(config_file);

        if (verbose) {
            glogger->info("Loaded configuration from: {}", config_file);
        }

        // Display configuration
        if (verbose)
            glogger->debug(config.summary());

        config.set("GUI", gui ? "true" : "false");
        config.set("progress_bar", progress ? "true" : "false");

        // Create the simulation object
        simulation = std::make_unique<Simulation>(config);
        simulation->init_all();
        simulation->init_callbacks();

        // Launch simulation
        simulation->main_loop();

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

// MODELINE "{{{1
// vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
// vim:foldmethod=marker
