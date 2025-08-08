#ifndef ROBOT_H
#define ROBOT_H

#include <vector>
#include <cmath>
#include <set>
#include <queue>
#include <chrono>
#include <SDL2/SDL.h>
#include <box2d/box2d.h>

#include "utils.h"
#include "render.h"
#include "colormaps.h"
#include "spogobot.h"
#include "objects.h"

/**
 * @brief Returns a log string for the current robot.
 *
 * This function gathers and returns a string containing relevant log information
 * for the currently active robot.
 *
 * @return std::string A string representing the current robot's log.
 */
std::string log_current_robot();


/**
 * @brief Enumeration to select the shape type of the robot's body.
 */
enum class ShapeType {
    Circle,   /**< Use a circular shape. */
    Ellipse,  /**< Use an elliptical shape (approximated by a polygon). */
    Polygon   /**< Use an arbitrary polygon shape defined by vertices. */
};


/**
 * @brief Abstract base class for message success rate calculations.
 *
 * This class defines the interface for calculating message success rates.
 */
class MsgSuccessRate {
public:
    /**
     * @brief Computes the message success rate.
     *
     * @param msg_size The message size.
     * @param p_send The sending parameter.
     * @param cluster_size The cluster size.
     * @return The computed success rate.
     */
    virtual double operator()(double msg_size, double p_send, double cluster_size) const = 0;

    /**
     * @brief Virtual destructor.
     */
    virtual ~MsgSuccessRate() = default;
};

/**
 * @brief Class for dynamic message success rate calculation using a configurable formula.
 *
 * The success rate is calculated as:
 * \f[
 * \text{success rate} = \frac{1}{1 + (\alpha \cdot \text{msg\_size}^{\beta} \cdot \text{p\_send}^{\gamma} \cdot \text{cluster\_size}^{\delta})}
 * \f]
 */
class DynamicMsgSuccessRate : public MsgSuccessRate {
public:
    /**
     * @brief Constructs a new DynamicMsgSuccessRate object.
     *
     * @param alpha The multiplicative constant (default is 0.000004).
     * @param beta The exponent for msg_size (default is 2.8096).
     * @param gamma The exponent for p_send (default is 2.3807).
     * @param delta The exponent for cluster_size (default is 1.2457).
     */
    DynamicMsgSuccessRate(double alpha = 0.000001, double beta = 3.0708, double gamma = 2.3234, double delta = 1.1897);

    /**
     * @brief Computes the dynamic success rate.
     *
     * @param msg_size The msg size.
     * @param p_send The sending parameter.
     * @param cluster_size The cluster size.
     * @return The computed success rate.
     */
    double operator()(double msg_size, double p_send, double cluster_size) const override;

private:
    double alpha_; ///< The multiplicative constant.
    double beta_;  ///< Exponent for msg size.
    double gamma_; ///< Exponent for sending parameter.
    double delta_; ///< Exponent for cluster size.
};

/**
 * @brief Class for constant message success rate calculation.
 *
 * This class always returns a constant success rate value regardless of the inputs.
 */
class ConstMsgSuccessRate : public MsgSuccessRate {
public:
    /**
     * @brief Constructs a new ConstMsgSuccessRate object.
     *
     * @param value The constant success rate value.
     */
    explicit ConstMsgSuccessRate(double value);

    /**
     * @brief Returns the constant success rate.
     *
     * @param msg_size The msg size (ignored).
     * @param p_send The sending parameter (ignored).
     * @param cluster_size The cluster size (ignored).
     * @return The constant success rate.
     */
    double operator()(double msg_size, double p_send, double cluster_size) const override;

private:
    double const_value_; ///< The constant success rate value.
};

/**
 * @brief Factory of MsgSuccessRate, from a given configuration
 *
 * @param config Configuration entry describing the object properties.
 */
MsgSuccessRate* msg_success_rate_factory(Configuration const& config);



/**
 * @brief Class representing a simulated Pogobot.
 *
 * The PogobotObject class encapsulates the properties and behaviors of a simulated Pogobot robot,
 * including physical properties, timing, LED control, messaging, and rendering.
 */
class PogobotObject : public PhysicalObject {

public:

    /**
     * Initializes a new Pogobot robot with the specified identifier, user data size, initial position,
     * radius, associated Box2D world, and message success rate. It also allows customization
     * of the body's physical properties (linear and angular damping, density, friction, and restitution).
     *
     * @param _id Unique object identifier.
     * @param x Initial x-coordinate in the simulation.
     * @param y Initial y-coordinate in the simulation.
     * @param geom Object's geometry.
     * @param world_id The Box2D world identifier.
     * @param _userdatasize Size of the memory block allocated for user data.
     * @param _communication_radius communication radius of each IR emitter
     * @param _msg_success_rate std::unique_ptr<MsgSuccessRate> describing the probability of successfully sending a message.
     * @param _temporal_noise_stddev Standard deviation of the gaussian noise to apply to time on each object, or 0.0 for deterministic time
     * @param _linear_damping Linear damping value for the physical body (default is 0.0f).
     * @param _angular_damping Angular damping value for the physical body (default is 0.0f).
     * @param _density Density of the body shape (default is 10.0f).
     * @param _friction Friction coefficient of the body shape (default is 0.3f).
     * @param _restitution Restitution (bounciness) of the body shape (default is 0.5f).
     * @param _max_linear_speed Max speed value in mm.s^-1.
     * @param _max_angular_speed Max speed value in rad.s^-1.
     * @param _linear_noise_stddev Standard deviation of the gaussian noise to apply to linear velocity, or 0.0 for deterministic velocity.
     * @param _angular_noise_stddev Standard deviation of the gaussian noise to apply to angular velocity, or 0.0 for deterministic velocity.
     * @param category Name of the category of the object.
     */
    PogobotObject(uint16_t _id, float _x, float _y,
           ObjectGeometry& geom, b2WorldId world_id,
           size_t _userdatasize,
           float _communication_radius = 80.0f,
           std::unique_ptr<MsgSuccessRate> _msg_success_rate = std::make_unique<ConstMsgSuccessRate>(0.5),
           float _temporal_noise_stddev = 0.0f,
           float _linear_damping = 0.0f, float _angular_damping = 0.0f,
           float _density = 10.0f, float _friction = 0.3f, float _restitution = 0.5f,
           float _max_linear_speed = 100.0f, float _max_angular_speed = 1.0f,
           float _linear_noise_stddev = 0.0f, float _angular_noise_stddev = 0.0f,
           std::pair<int16_t, int16_t> photosensors_systematic_bias_domain = {0, 0},
           std::string const& _category = "robots");

    /**
     * @brief Constructs a PogobotObject from a configuration entry.
     *
     * @param simulation Pointer to the underlying simulation.
     * @param _id Unique object identifier.
     * @param x Initial x-coordinate in the simulation.
     * @param y Initial y-coordinate in the simulation.
     * @param world_id The Box2D world identifier.
     * @param _userdatasize Size of the memory block allocated for user data.
     * @param config Configuration entry describing the object properties.
     */
    PogobotObject(Simulation* simulation, uint16_t _id, float _x, float _y,
           b2WorldId world_id, size_t _userdatasize, Configuration const& config,
           std::string const& _category = "robots");


    //virtual ~Robot();

    // Base info
    void* data = nullptr;                ///< Pointer to user data.
    void (*user_init)(void) = nullptr;   ///< Pointer to a user-defined initialization function.
    void (*user_step)(void) = nullptr;   ///< Pointer to a user-defined step function.
    void (*callback_export_data)(void) = nullptr;        ///< Callback to export data.

    /**
     * @brief Launches the user-defined step function.
     *
     * Updates the object's time, enables all registered stop watches, executes the user step
     * function via pogo_main_loop_step, and then disables the stop watches.
     */
    virtual void launch_user_step(float t) override;

    // C-code accessible values
    uint32_t pogobot_ticks = 0;                 ///< Simulation ticks counter.
    uint8_t main_loop_hz = 60;                  ///< Main loop frequency in Hz.
    uint8_t max_nb_processed_msg_per_tick = 3;  ///< Maximum number of messages processed per tick.
    void (*msg_rx_fn)(message_t*) = nullptr;    ///< Function pointer for message reception.
    bool (*msg_tx_fn)(void) = nullptr;          ///< Function pointer for message transmission.
    int8_t error_codes_led_idx = 3;             ///< LED index for error codes.
    time_reference_t _global_timer;             ///< Global timer reference.
    time_reference_t timer_main_loop;           ///< Main loop timer reference.
    uint32_t _current_time_milliseconds = 0;    ///< Current time in milliseconds.
    uint32_t _error_code_initial_time = 0;      ///< Initial time for error code reporting.
    uint8_t percent_msgs_sent_per_ticks = 20;   ///< Percentage of messages sent per tick.
    uint32_t nb_msgs_sent = 0;                  ///< Counter for messages sent.
    uint32_t nb_msgs_recv = 0;                  ///< Counter for messages received.
    std::vector<int16_t> photosensors_biases = {0, 0, 0}; ///< Level of systematic bias to apply to each photosensor

    // Time-related utilities
    std::set<time_reference_t*> stop_watches;   ///< Set of registered stop watches.

    /**
     * @brief Registers a stop watch with the robot.
     *
     * Adds the given stop watch pointer to the set of stop watches.
     *
     * @param sw Pointer to the stop watch to register.
     */
    void register_stop_watch(time_reference_t* sw);

    /**
     * @brief Enables all registered stop watches.
     *
     * Iterates through all registered stop watches and enables them.
     */
    void enable_stop_watches();

    /**
     * @brief Disables all registered stop watches.
     *
     * Iterates through all registered stop watches and disables them.
     */
    void disable_stop_watches();

    /**
     * @brief Updates the object's current time.
     *
     * Computes the current time in microseconds relative to the simulation start.
     */
    virtual void update_time();


    // Physical information
    float radius;                   ///< Radius of this robot.
    float left_motor_speed  = 0;    ///< Current speed of the left motor.
    float right_motor_speed = 0;    ///< Current speed of the right motor.
    bool show_lateral_leds = false; ///< Whether to render lateral LEDs.


    /**
     * @brief Renders the object on the given SDL renderer.
     *
     * @param renderer Pointer to the SDL_Renderer.
     * @param world_id The Box2D world identifier (unused in rendering).
     */
    virtual void render(SDL_Renderer*, b2WorldId) const override;

    /**
     * @brief Renders the communication channels originating from this robot.
     *
     * @param renderer Pointer to the SDL_Renderer.
     * @param world_id The Box2D world identifier (unused in rendering).
     */
    virtual void render_communication_channels(SDL_Renderer*, b2WorldId) const;

    /**
     * @brief Updates the motor speed of the robot and recalculates its velocities.
     *
     * This method sets the motor speed (left or right) and then computes the robot's
     * linear and angular velocities. It applies the damping values
     * that were provided in the constructor. If the noise standard deviations
     * (for linear or angular velocities) are greater than 0.0, a Gaussian noise component is added
     * to the respective velocity.
     *
     * @param motor The identifier of the motor to update.
     * @param speed The new speed value for the selected motor.
     */
    virtual void set_motor(motor_id motor, int speed);

    /**
     * @brief Retrieves the IR emitters current positions
     *
     * Returns the position of one of the robot's IR emitter as a Box2D vector.
     *
     * @return b2Vec2 The current position.
     */
    virtual b2Vec2 get_IR_emitter_position(ir_direction dir) const;

    /**
     * @brief World-frame bearing of the chosen IR emitter.
     *
     * For the four perimeter LEDs the bearing is the angle of the outward
     * normal of the robot’s hull at that LED (i.e. ±90° from the tangent).
     * For dir == MIDDLE we return the robot’s *forward* heading, so the caller
     * still receives a meaningful direction.
     */
    virtual float get_IR_emitter_angle(ir_direction dir) const;

    /**
     * @brief Returns the position of a photosensor.
     * @param sensor_number the id (0-2) of the sensor.
     * @return the position of photosensor 'sensor_number'.
     */
    virtual b2Vec2 get_photosensor_position(uint8_t sensor_number) const;

    // LED control
    std::vector<color_t> leds = std::vector<color_t>(5, {0, 0, 0}); ///< LED colors for the robot.

    // Neighbors and messaging
    std::vector<std::vector<PogobotObject*>> neighbors{ir_all+1};  ///< Pointers to neighboring robots.
    std::queue<message_t> messages; ///< Queue of incoming messages.
    float communication_radius;                       ///< Communication radius of each IR emitter.
    std::unique_ptr<MsgSuccessRate> msg_success_rate; ///< Probability of successfully sending a message.

    /**
     * @brief Sends a short message to neighboring robots.
     *
     * Converts a short message to a full message and forwards it to all neighboring robots.
     *
     * @param dir Direction in which the message is sent (i.e. the number of the IR emitter)
     * @param message Pointer to the short_message_t to send.
     */
    void send_to_neighbors(ir_direction dir, short_message_t *const message);

    /**
     * @brief Sends a message to neighboring robots.
     *
     * Iterates over neighboring robots and, based on a random probability and the message success rate,
     * enqueues the message in each neighbor's message queue.
     *
     * @param dir Direction in which the message is sent (i.e. the number of the IR emitter)
     * @param message Pointer to the message_t to send.
     */
    void send_to_neighbors(ir_direction dir, message_t *const message);

    /**
     * @brief Receive a message from another robot.
     *
     * @param message Pointer to the message_t to receive.
     * @param source Pointer to the originating robot.
     */
    void receive_message(message_t *const message, PogobotObject* source);


    /**
     * @brief Simulate a sleep on a single robot.
     *
     * @param microseconds Number of microseconds to sleep for
     */
    void sleep_µs(uint64_t microseconds);

    uint64_t current_time_microseconds = 0LL;                        ///< Current time in microseconds.

    /// Motors current direction and power ([R, L, B])
    uint8_t motor_dir_mem[3] = {0, 1, 0};
    uint8_t motor_dir[3] = {0, 1, 0};
    uint16_t motor_power_mem[3] = {512, 512, 0};

protected:
    /**
     * @brief Creates the object's physical body in the simulation.
     *
     * @param world_id The Box2D world identifier.
     */
    void create_robot_body(b2WorldId world_id);

    /**
     * @brief Initialize time-related operations
     */
    void initialize_time();

    // Temporal information
    float temporal_noise = 0;
    float temporal_noise_stddev;

    // Physical information
    float max_linear_speed;
    float max_angular_speed;
    float linear_noise_stddev;
    float angular_noise_stddev;

    /**
     * @brief Parse a provided configuration and set associated members values.
     *
     * @param config Configuration entry describing the object properties.
     */
    virtual void parse_configuration(Configuration const& config, Simulation* simulation) override;

    /**
     * @brief Initialize the level of bias each photosensor has, based on provided configuration/parameters
     */
    virtual void initialize_photosensors_bias(std::pair<int16_t, int16_t>& domain);
};


/**
 * @brief Class representing a simulated Pogobject.
 */
class PogobjectObject : public PogobotObject {
public:
    /**
     * Initializes a new Pogobject robot with the specified identifier, user data size, initial position,
     * radius, associated Box2D world, and message success rate. It also allows customization
     * of the body's physical properties (linear and angular damping, density, friction, and restitution).
     *
     * @param _id Unique object identifier.
     * @param x Initial x-coordinate in the simulation.
     * @param y Initial y-coordinate in the simulation.
     * @param geom Object's geometry.
     * @param world_id The Box2D world identifier.
     * @param _userdatasize Size of the memory block allocated for user data.
     * @param _communication_radius communication radius of each IR emitter
     * @param _msg_success_rate std::unique_ptr<MsgSuccessRate> describing the probability of successfully sending a message.
     * @param _temporal_noise_stddev Standard deviation of the gaussian noise to apply to time on each object, or 0.0 for deterministic time
     * @param _linear_damping Linear damping value for the physical body (default is 0.0f).
     * @param _angular_damping Angular damping value for the physical body (default is 0.0f).
     * @param _density Density of the body shape (default is 10.0f).
     * @param _friction Friction coefficient of the body shape (default is 0.3f).
     * @param _restitution Restitution (bounciness) of the body shape (default is 0.5f).
     * @param category Name of the category of the object.
     */
    PogobjectObject(uint16_t _id, float _x, float _y,
           ObjectGeometry& geom, b2WorldId world_id,
           size_t _userdatasize,
           float _communication_radius = 80.0f,
           std::unique_ptr<MsgSuccessRate> _msg_success_rate = std::make_unique<ConstMsgSuccessRate>(0.5),
           float _temporal_noise_stddev = 0.0f,
           float _linear_damping = 0.0f, float _angular_damping = 0.0f,
           float _density = 10.0f, float _friction = 0.3f, float _restitution = 0.5f,
           std::string const& _category = "robots");

    /**
     * @brief Constructs a PogobotObject from a configuration entry.
     *
     * @param simulation Pointer to the underlying simulation.
     * @param _id Unique object identifier.
     * @param x Initial x-coordinate in the simulation.
     * @param y Initial y-coordinate in the simulation.
     * @param world_id The Box2D world identifier.
     * @param _userdatasize Size of the memory block allocated for user data.
     * @param config Configuration entry describing the object properties.
     */
    PogobjectObject(Simulation* simulation, uint16_t _id, float _x, float _y,
           b2WorldId world_id, size_t _userdatasize, Configuration const& config,
           std::string const& _category = "robots");

    /**
     * @brief Updates the motor speed of the robot and recalculates its velocities.
     * Pogobjects do not move, so this method will always set the motors to 0.
     *
     * @param motor The identifier of the motor to update.
     * @param speed (Ignored) speed value for the selected motor.
     */
    virtual void set_motor(motor_id motor, int speed) override;

    /**
     * @brief Retrieves the IR emitters current positions
     *
     * Returns the position of one of the robot's IR emitter as a Box2D vector.
     *
     * @return b2Vec2 The current position.
     */
    virtual b2Vec2 get_IR_emitter_position(ir_direction dir) const override;

    /**
     * @brief Renders the robot on the given SDL renderer.
     *
     * @param renderer Pointer to the SDL_Renderer.
     * @param world_id The Box2D world identifier (unused in rendering).
     */
    virtual void render(SDL_Renderer*, b2WorldId) const override;
};


/**
 * @brief Class representing a simulated Pogowall.
 */
class Pogowall : public PogobotObject {
public:
    /**
     * Initializes a new Pogobject robot with the specified identifier, user data size, initial position,
     * radius, associated Box2D world, and message success rate. It also allows customization
     * of the body's physical properties (linear and angular damping, density, friction, and restitution).
     *
     * @param _id Unique object identifier.
     * @param x Initial x-coordinate in the simulation.
     * @param y Initial y-coordinate in the simulation.
     * @param geom Object's geometry.
     * @param world_id The Box2D world identifier.
     * @param _userdatasize Size of the memory block allocated for user data.
     * @param _communication_radius communication radius of each IR emitter
     * @param _msg_success_rate std::unique_ptr<MsgSuccessRate> describing the probability of successfully sending a message.
     * @param _temporal_noise_stddev Standard deviation of the gaussian noise to apply to time on each object, or 0.0 for deterministic time
     * @param _linear_damping Linear damping value for the physical body (default is 0.0f).
     * @param _angular_damping Angular damping value for the physical body (default is 0.0f).
     * @param _density Density of the body shape (default is 10.0f).
     * @param _friction Friction coefficient of the body shape (default is 0.3f).
     * @param _restitution Restitution (bounciness) of the body shape (default is 0.5f).
     * @param _linear_noise_stddev Standard deviation of the gaussian noise to apply to linear velocity, or 0.0 for deterministic velocity
     * @param _angular_noise_stddev Standard deviation of the gaussian noise to apply to angular velocity, or 0.0 for deterministic velocity
     * @param category Name of the category of the object.
     */
    Pogowall(uint16_t _id, float _x, float _y,
           ObjectGeometry& geom, b2WorldId world_id,
           size_t _userdatasize,
           float _communication_radius = 80.0f,
           std::unique_ptr<MsgSuccessRate> _msg_success_rate = std::make_unique<ConstMsgSuccessRate>(0.5),
           float _temporal_noise_stddev = 0.0f,
           float _linear_damping = 0.0f, float _angular_damping = 0.0f,
           float _density = 10.0f, float _friction = 0.3f, float _restitution = 0.5f,
           float _max_linear_speed = 100.0f, float _max_angular_speed = 1.0f,
           float _linear_noise_stddev = 0.0f, float _angular_noise_stddev = 0.0f,
           std::string const& _category = "robots");

    /**
     * @brief Constructs a PogobotObject from a configuration entry.
     *
     * @param simulation Pointer to the underlying simulation.
     * @param _id Unique object identifier.
     * @param x Initial x-coordinate in the simulation.
     * @param y Initial y-coordinate in the simulation.
     * @param world_id The Box2D world identifier.
     * @param _userdatasize Size of the memory block allocated for user data.
     * @param config Configuration entry describing the object properties.
     */
    Pogowall(Simulation* simulation, uint16_t _id, float _x, float _y,
           b2WorldId world_id, size_t _userdatasize, Configuration const& config,
           std::string const& _category = "robots");

    /**
     * @brief Updates the motor speed of the robot and recalculates its velocities.
     * Pogobjects do not move, so this method will always set the motors to 0.
     *
     * @param motor The identifier of the motor to update.
     * @param speed (Ignored) speed value for the selected motor.
     */
    virtual void set_motor([[maybe_unused]] motor_id motor, [[maybe_unused]] int speed) override { }

    /**
     * @brief Retrieves the IR emitters current positions
     *
     * Returns the position of one of the robot's IR emitter as a Box2D vector.
     *
     * @return b2Vec2 The current position.
     */
    virtual b2Vec2 get_IR_emitter_position(ir_direction dir) const override;

    /**
     * @brief World-frame bearing of the chosen IR emitter.
     *
     * For the four perimeter LEDs the bearing is the angle of the outward
     * normal of the robot’s hull at that LED (i.e. ±90° from the tangent).
     * For dir == MIDDLE we return the robot’s *forward* heading, so the caller
     * still receives a meaningful direction.
     */
    virtual float get_IR_emitter_angle(ir_direction dir) const override;

    /**
     * @brief Renders the robot on the given SDL renderer.
     *
     * @param renderer Pointer to the SDL_Renderer.
     * @param world_id The Box2D world identifier (unused in rendering).
     */
    virtual void render(SDL_Renderer*, b2WorldId) const override { }

    /**
     * @brief Renders the communication channels originating from this robot.
     *
     * @param renderer Pointer to the SDL_Renderer.
     * @param world_id The Box2D world identifier (unused in rendering).
     */
    virtual void render_communication_channels(SDL_Renderer*, b2WorldId) const override { }

    /**
     * @brief Returns whether this object is tangible (e.g. collisions, etc) or not.
     */
    virtual bool is_tangible() const override { return false; }

    /**
     * @brief Move the object to a given coordinate
     *
     * @param x X coordinate.
     * @param y Y coordinate.
     * @param theta Orientation, in rad.
     */
    virtual void move([[maybe_unused]] float x, [[maybe_unused]] float y, [[maybe_unused]] float theta = NAN) override { }
};


/**
 * @brief Class representing a simulated Membrane, with a pogowall on each side.
 */
class MembraneObject : public Pogowall { // PogobotObject {
public:
    struct Dot   { b2BodyId  body_id; };
    struct Joint { b2JointId joint_id; };


    /**
     * Initializes a new Pogobject robot with the specified identifier, user data size, initial position,
     * radius, associated Box2D world, and message success rate. It also allows customization
     * of the body's physical properties (linear and angular damping, density, friction, and restitution).
     *
     * @param _id Unique object identifier.
     * @param x Initial x-coordinate in the simulation.
     * @param y Initial y-coordinate in the simulation.
     * @param geom Object's geometry.
     * @param world_id The Box2D world identifier.
     * @param _userdatasize Size of the memory block allocated for user data.
     * @param _communication_radius communication radius of each IR emitter
     * @param _msg_success_rate std::unique_ptr<MsgSuccessRate> describing the probability of successfully sending a message.
     * @param _temporal_noise_stddev Standard deviation of the gaussian noise to apply to time on each object, or 0.0 for deterministic time
     * @param _linear_damping Linear damping value for the physical body (default is 0.0f).
     * @param _angular_damping Angular damping value for the physical body (default is 0.0f).
     * @param _density Density of the body shape (default is 10.0f).
     * @param _friction Friction coefficient of the body shape (default is 0.3f).
     * @param _restitution Restitution (bounciness) of the body shape (default is 0.5f).
     * @param _linear_noise_stddev Standard deviation of the gaussian noise to apply to linear velocity, or 0.0 for deterministic velocity
     * @param _angular_noise_stddev Standard deviation of the gaussian noise to apply to angular velocity, or 0.0 for deterministic velocity
     * @param _num_dots How many dots (≈ vertices) the membrane should have.
     * @param _dot_radius Physical radius for each dot (Box2D units, not pixels).
     * @param _cross_span Connect every i‑th neighbour to stiffen the sheet (≥ 1).
     * @param _stiffness The stiffness of the joints.
     * @param _colormap Name of the colormap to use to set the color of the object
     * @param _category Name of the category of the object.
     */
    MembraneObject(uint16_t _id, float _x, float _y,
           ObjectGeometry& geom, b2WorldId world_id,
           size_t _userdatasize,
           float _communication_radius = 80.0f,
           std::unique_ptr<MsgSuccessRate> _msg_success_rate = std::make_unique<ConstMsgSuccessRate>(0.5),
           float _temporal_noise_stddev = 0.0f,
           float _linear_damping = 0.0f, float _angular_damping = 0.0f,
           float _density = 10.0f, float _friction = 0.3f, float _restitution = 0.5f,
           float _max_linear_speed = 100.0f, float _max_angular_speed = 1.0f,
           float _linear_noise_stddev = 0.0f, float _angular_noise_stddev = 0.0f,
           unsigned int _num_dots = 100, float _dot_radius = 10.0f, int _cross_span = 3,
           float _stiffness = 30.f,
           std::string _colormap = "rainbow",
           std::string const& _category = "robots");

    /**
     * @brief Constructs a PogobotObject from a configuration entry.
     *
     * @param simulation Pointer to the underlying simulation.
     * @param _id Unique object identifier.
     * @param x Initial x-coordinate in the simulation.
     * @param y Initial y-coordinate in the simulation.
     * @param world_id The Box2D world identifier.
     * @param _userdatasize Size of the memory block allocated for user data.
     * @param config Configuration entry describing the object properties.
     */
    MembraneObject(Simulation* simulation, uint16_t _id, float _x, float _y,
           b2WorldId world_id, size_t _userdatasize, Configuration const& config,
           std::string const& _category = "robots");

    /**
     * @brief Updates the motor speed of the robot and recalculates its velocities.
     * Pogobjects do not move, so this method will always set the motors to 0.
     *
     * @param motor The identifier of the motor to update.
     * @param speed (Ignored) speed value for the selected motor.
     */
    virtual void set_motor([[maybe_unused]] motor_id motor, [[maybe_unused]] int speed) override { }

    /**
     * @brief Retrieves the object's current position.
     *
     * Returns the position of the object's physical body as a Box2D vector.
     *
     * @return b2Vec2 The current position.
     */
    virtual b2Vec2 get_position() const override;

    /**
     * @brief Renders the robot on the given SDL renderer.
     *
     * @param renderer Pointer to the SDL_Renderer.
     * @param world_id The Box2D world identifier (unused in rendering).
     */
    virtual void render(SDL_Renderer*, b2WorldId) const override;

    /**
     * @brief Returns whether this object is tangible (e.g. collisions, etc) or not.
     */
    virtual bool is_tangible() const override { return true; }

    /**
     * @brief Move the object to a given coordinate
     *
     * @param x X coordinate.
     * @param y Y coordinate.
     * @param theta Orientation, in rad.
     */
    virtual void move([[maybe_unused]] float x, [[maybe_unused]] float y, [[maybe_unused]] float theta = NAN) override;

    /**
     * @brief Return one or more polygonal contours that represent the current geometry of the object.
     *
     * @param points_per_contour  Desired number of vertices for each contour
     *                            (a rectangle has one contour, a disk has one,
     *                             an arena may have many – one per wall).
     *                            If this value is 0, the function will determine
     *                            automatically the best number of points to represent
     *                            the shape.
     *
     * @return arena_polygons_t   A vector of closed polygons (counter‑clockwise,
     *                            last vertex different from the first – the caller
     *                            may close the loop if needed).
     */
    virtual arena_polygons_t generate_contours(std::size_t points_per_contour = 0) const override;


protected:
    /**
     * @brief Creates the object's physical body in the simulation.
     *
     * @param world_id The Box2D world identifier.
     */
    void create_robot_body(b2WorldId world_id);

    /**
     * @brief Parse a provided configuration and set associated members values.
     *
     * @param config Configuration entry describing the object properties.
     */
    virtual void parse_configuration(Configuration const& config, Simulation* simulation) override;

    /* helper – create one distance joint with the usual parameters */
    void make_distance_joint(b2WorldId world_id,
                             b2BodyId  a,
                             b2BodyId  b,
                             float     stiffness_scale = 1.0f);

    // Physical information
    int num_dots;
    std::vector<size_t> size_contours;
    float dot_radius;
    int cross_span;
    float stiffness;
    std::string colormap;
    std::vector<Dot> dots;
    std::vector<Joint> joints;
};




//extern Robot* current_robot;
extern PogobotObject* current_robot;
extern int UserdataSize;
extern void* mydata;
extern uint64_t sim_starting_time_microseconds;


/**
 * @brief Retrieves the current time in microseconds.
 *
 * Computes and returns the current simulation time in microseconds.
 *
 * @return uint64_t Current time in microseconds.
 */
uint64_t get_current_time_microseconds();

#endif


// MODELINE "{{{1
// vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
// vim:foldmethod=marker
