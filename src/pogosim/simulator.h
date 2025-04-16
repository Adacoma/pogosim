#ifndef SIMULATOR_H
#define SIMULATOR_H

#include "pogosim.h"
#include "robot.h"
#include "objects.h"
#include "configuration.h"
#include "data_logger.h"
#include "SDL_FontCache.h"

/**
 * @brief Main entry point for the robot code (C linkage).
 *
 * This function is defined with C linkage so that it can be called from non-C++ code.
 *
 * @return int Exit status code.
 */
extern "C" int robot_main(void);

/**
 * @brief Sets the current active robot.
 *
 * Updates the global current_robot pointer and copies various robot parameters into global variables.
 *
 * @param robot Reference to the Robot to be set as current.
 */
//void set_current_robot(Robot& robot);
void set_current_robot(PogobotObject& robot);

/**
 * @brief Prints the help message to the console.
 *
 * Displays usage instructions and available command-line options.
 */
void print_help();


/**
 * @brief Class representing the simulation environment.
 *
 * The Simulation class encapsulates the configuration, SDL window and renderer,
 * Box2D world, robots, arena geometry, data logging, and GUI event handling for the simulation.
 */
class Simulation {
    Configuration& config;  ///< Reference to the simulation configuration.

    // SDL globals
    SDL_Window* window = nullptr;         ///< SDL window.
    SDL_Renderer* renderer = nullptr;     ///< SDL renderer.
    bool enable_gui = true;               ///< Flag to enable or disable the GUI.
    bool paused = false;                  ///< Flag indicating whether the simulation is paused.
    bool running = true;                  ///< Flag indicating whether the simulation is running.
    bool show_comm = false;               ///< Flag indicating whether to show the communication channels between robots.
    bool show_lateral_leds = false;       ///< Flag indicating whether to show the lateral leds

    double t = 0.0f;                      ///< Simulation time (in seconds).

    uint16_t window_width = 800;          ///< Window width in pixels.
    uint16_t window_height = 600;         ///< Window height in pixels.
    uint16_t sub_step_count = 4;          ///< Number of Box2D sub-steps per simulation step.
    double GUI_speed_up = 1.0;            ///< Factor to speed up the GUI rendering.

    float arena_width = 1000.0;           ///< Arena width in millimeters.
    float arena_height = 1000.0;          ///< Arena height in millimeters.
    float arena_surface = 1e6;            ///< Arena surface area in mm².
    float max_comm_radius = 00.0f;        ///< Max communication radius across all types of objects

    b2WorldId worldId;                    ///< Identifier for the Box2D world.
    //std::vector<Robot> robots;          ///< Vector of robots in the simulation.
    std::vector<std::vector<b2Vec2>> arena_polygons; ///< Arena polygon definitions.

    // Objects
    std::map<std::string, std::vector<std::shared_ptr<Object>>> objects;    ///< Dictionary of simulation objects, by category name.
    std::vector<std::shared_ptr<PogobotObject>> robots;                     ///< Vector of robots in the simulation.
    std::unique_ptr<LightLevelMap> light_map;                               ///< Light map of the arena.
    std::string initial_formation;                                          ///< Type of initial formation of the objects.

    double last_frame_shown_t = -1.0;     ///< Time when the last frame was rendered.
    double last_frame_saved_t = -1.0;     ///< Time when the last frame was saved.
    double last_data_saved_t = -1.0;      ///< Time when the last data export occurred.

    int16_t current_light_value = std::numeric_limits<int16_t>::max(); ///< Current light sensor value.
    double photo_start_at = -1.f;           ///< Time to start a photo capture.
    double photo_start_duration = 1.f;      ///< Duration for the photo capture.

    // Fonts
    FC_Font* font;                          ///< Font used for rendering text.

    // Mouse dragging for visualization
    bool dragging_pos_by_mouse = false;   ///< Flag for mouse dragging.
    int last_mouse_x;                     ///< Last recorded mouse x-coordinate.
    int last_mouse_y;                     ///< Last recorded mouse y-coordinate.

    // Data logger
    bool enable_data_logging;                ///< Flag to enable data logging.
    std::unique_ptr<DataLogger> data_logger; ///< DataLogger instance.

public:
    /**
     * @brief Constructs a Simulation object.
     *
     * Initializes the simulation configuration, console logging, Box2D world, and SDL subsystems.
     *
     * @param _config Reference to a Configuration object.
     */
    Simulation(Configuration& _config);

    /**
     * @brief Destructor.
     *
     * Cleans up fonts, SDL, and Box2D world resources.
     */
    virtual ~Simulation();

    /**
     * @brief Initializes the simulation components.
     *
     * Calls functions to create the arena and robots
     */
    void init_all();

    /**
     * @brief Creates the robot instances.
     *
     * Generates initial positions for robots based on the configuration, creates Robot objects,
     * and initializes their user code.
     */
    void create_robots();

    /**
     * @brief Creates objects in the simulation
     *
     */
    void create_objects();

    /**
     * @brief Creates the arena from a CSV file.
     *
     * Reads the arena polygons from a CSV file, computes the bounding box, creates walls,
     * and adjusts visualization scaling.
     */
    void create_arena();

    /**
     * @brief Creates arena walls.
     *
     * Defines static bodies for arena boundaries using Box2D.
     */
    void create_walls();

    /**
     * @brief Initializes the Box2D world.
     *
     * Creates a Box2D world with zero gravity.
     */
    void init_box2d();

    /**
     * @brief Initializes the simulation configuration.
     *
     * Reads configuration parameters and initializes global variables such as window size,
     * arena dimensions, robot radius, and GUI settings.
     */
    void init_config();

    /**
     * @brief Initializes SDL and related subsystems.
     *
     * Sets up the SDL window, renderer, font, and initializes fpng for image exporting.
     *
     * @throw std::runtime_error if SDL initialization fails.
     */
    void init_SDL();

    /**
     * @brief Speeds up the simulation GUI.
     *
     * Increases the GUI speed-up factor and logs the new value.
     */
    void speed_up();

    /**
     * @brief Slows down the simulation GUI.
     *
     * Decreases the GUI speed-up factor and logs the new value.
     */
    void speed_down();

    /**
     * @brief Toggles the simulation pause state.
     */
    void pause();

    /**
     * @brief Displays a help message with GUI keyboard shortcuts.
     */
    void help_message();

    /**
     * @brief Handles SDL events.
     *
     * Processes SDL events such as keyboard input, mouse movement, and window events,
     * and updates simulation state accordingly.
     */
    void handle_SDL_events();

    /**
     * @brief Computes neighboring robots.
     *
     * Updates each robot's neighbor list based on their positions and communication radius.
     */
    void compute_neighbors();

    /**
     * @brief Initializes simulation callbacks.
     *
     * Currently initializes the data logger.
     */
    void init_callbacks();

    /**
     * @brief Initializes the data logger.
     *
     * Sets up the DataLogger schema based on configuration and opens the data file.
     *
     * @throw std::runtime_error if required configuration values are missing.
     */
    void init_data_logger();

    /**
     * @brief Initializes the console logger.
     *
     * Sets up file-based logging if enabled in the configuration.
     *
     * @throw std::runtime_error if the console filename is empty when logging is enabled.
     */
    void init_console_logger();

    /**
     * @brief Draws a scale bar on the GUI.
     *
     * Renders a horizontal scale bar along with a label indicating the scale in millimeters.
     */
    void draw_scale_bar();

    /**
     * @brief Renders all simulation components.
     *
     * Clears the renderer, draws the arena walls, robots, current time, and scale bar.
     */
    void render_all();

    /**
     * @brief Exports the current frame to a PNG file.
     *
     * Saves the current window content to a PNG file if the frame export period has elapsed.
     */
    void export_frames();

    /**
     * @brief Exports simulation data.
     *
     * Iterates over all robots, logs their state to the DataLogger, and saves the data row.
     */
    void export_data();

    /**
     * @brief Adjusts the simulated light value for photo capture.
     *
     * Sets the current light value to zero during the photo capture period, and resets it otherwise.
     */
    void photo_start();

    /**
     * @brief Runs the main simulation loop.
     *
     * Processes SDL events, updates robot states, steps the Box2D world, computes neighbors,
     * logs data, renders frames, and manages simulation timing until the simulation time is reached or exit is requested.
     */
    void main_loop();

    /**
     * @brief Deletes old data files.
     *
     * Deletes files with a specified extension from the frames directory if configured to do so.
     */
    void delete_old_data();

    /**
     * @brief Retrieves the current light sensor value.
     *
     * @return uint16_t The current light value.
     */
    uint16_t get_current_light_value() const;

    /**
     * @brief Retrieves the current configuration
     *
     * @return Configuration& configuration
     */
    Configuration& get_config();

    /**
     * @brief Retrieves the data logger.
     *
     * @return DataLogger* Pointer to the DataLogger instance.
     */
    DataLogger* get_data_logger();
};

/// Global simulation instance.
extern std::unique_ptr<Simulation> simulation;


#endif // SIMULATOR_H

// MODELINE "{{{1
// vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
// vim:foldmethod=marker
