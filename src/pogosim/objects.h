#ifndef OBJECTS_H
#define OBJECTS_H

#include <functional>

#include "utils.h"
#include "configuration.h"
#include "render.h"
#include "colormaps.h"
#include "geometry.h"
#include "objects_geometry.h"
#include "data_logger.h"


class Simulation;
class LightLevelMap;


/**
 * @brief Base class of any object contained within the simulation.
 *
 */
class Object {
public:
    /**
     * @brief Constructs an Object.
     *
     * @param x Initial x-coordinate in the simulation.
     * @param y Initial y-coordinate in the simulation.
     * @param geom Object's geometry.
     * @param category Name of the category of the object.
     */
    Object(float _x, float _y, ObjectGeometry& _geom, std::string const& _category = "objects");

    /**
     * @brief Constructs an Object from a configuration entry.
     *
     * @param simulation Pointer to the underlying simulation.
     * @param x Initial x-coordinate in the simulation.
     * @param y Initial y-coordinate in the simulation.
     * @param config Configuration entry describing the object properties.
     * @param category Name of the category of the object.
     */
    Object(Simulation* simulation, float _x, float _y, Configuration const& config, std::string const& _category = "objects");

    /**
     * @brief Destructor
     */
    virtual ~Object();


    /**
     * @brief Launch virtual function 'do_init' that will perform base initialization (e.g. create Box2D objects)
     */
    void init(b2WorldId world_id) {
        if (initialized) {
            throw std::runtime_error("Object::init() called twice.");
        }
        do_init(world_id);
        initialized = true;
    }

    /**
     * @brief Check if the object was correctly initialized yet
     */
    bool is_initialized() const noexcept {
        return initialized;
    }


    /**
     * @brief Renders the object on the given SDL renderer.
     *
     * @param renderer Pointer to the SDL_Renderer.
     * @param world_id The Box2D world identifier (unused in rendering).
     */
    virtual void render(SDL_Renderer* renderer, b2WorldId world_id) const = 0;

    /**
     * @brief Launches the user-defined step function.
     *
     * @param t current simulation time
     */
    virtual void launch_user_step(float f);

    /**
     * @brief Return the object's geometry.
     */
    ObjectGeometry* get_geometry() { return geom;} ;

    /**
     * @brief Move the object to a given coordinate
     *
     * @param x X coordinate.
     * @param y Y coordinate.
     * @param theta Orientation, in rad.
     */
    virtual void move(float x, float y, float theta = NAN);

    /**
     * @brief Returns whether this object is tangible (e.g. collisions, etc) or not.
     */
    virtual bool is_tangible() const { return false; };

    /**
     * @brief Save base values of the object into a data logger row.
     *
     * @param data_logger Pointer to a DataLogger used to serialize base values
     */
    virtual void serialize_base_values(DataLogger* data_logger);

    /**
     * @brief Create serialization fields of the data logger
     *
     * @param data_logger Pointer to a DataLogger used for serialization
     */
    virtual void create_serialization_fields(DataLogger* data_logger);

    /**
     * @brief Return one or more polygonal contours that represent the geometry of the object.
     *
     * @param points_per_contour  Desired number of vertices for each contour
     *                            (a rectangle has one contour, a disk has one,
     *                             an arena may have many – one per wall).
     *
     * @return arena_polygons_t   A vector of closed polygons (counter‑clockwise,
     *                            last vertex different from the first – the caller
     *                            may close the loop if needed).
     */
    virtual arena_polygons_t generate_contours(std::size_t points_per_contour = 0) const;

    // Physical information
    float x;                            ///< X position
    float y;                            ///< Y position
    float theta;                        ///< Orientation (in rad)

    // Base information
    std::string category;               ///< Category of the object

protected:
    /**
     * @brief Perform the base initialization (e.g. create Box2D objects).
     *  Called once by `init(world_id)`.
     *
     * @param world_id The Box2D world identifier.
     */
    virtual void do_init([[maybe_unused]] b2WorldId world_id) { }

    /**
     * @brief Parse a provided configuration and set associated members values.
     *
     * @param config Configuration entry describing the object properties.
     */
    virtual void parse_configuration(Configuration const& config, Simulation* simulation);

    ObjectGeometry* geom;                ///< Geometry of the object.

private:
    bool initialized = false;
};


/**
 * @brief A physical object, i.e. with physics properties (e.g. collisions) modelled by Box2D
 *
 */
class PhysicalObject : public Object {
public:

    /**
     * @brief Constructs a PhysicalObject.
     *
     * @param _id Unique object identifier.
     * @param x Initial x-coordinate in the simulation.
     * @param y Initial y-coordinate in the simulation.
     * @param geom Object's geometry.
     * @param _linear_damping Linear damping value for the physical body (default is 0.0f).
     * @param _angular_damping Angular damping value for the physical body (default is 0.0f).
     * @param _density Density of the body shape (default is 10.0f).
     * @param _friction Friction coefficient of the body shape (default is 0.3f).
     * @param _restitution Restitution (bounciness) of the body shape (default is 0.5f).
     * @param category Name of the category of the object.
     */
    PhysicalObject(uint16_t _id, float _x, float _y,
           ObjectGeometry& geom,
           float _linear_damping = 0.0f, float _angular_damping = 0.0f,
           float _density = 10.0f, float _friction = 0.3f, float _restitution = 0.5f,
           std::string const& _category = "objects");

    /**
     * @brief Constructs a PhysicalObject from a configuration entry.
     *
     * @param simulation Pointer to the underlying simulation.
     * @param _id Unique object identifier.
     * @param x Initial x-coordinate in the simulation.
     * @param y Initial y-coordinate in the simulation.
     * @param config Configuration entry describing the object properties.
     * @param category Name of the category of the object.
     */
    PhysicalObject(Simulation* simulation, uint16_t _id, float _x, float _y,
           Configuration const& config,
           std::string const& _category = "objects");

    /**
     * @brief Launches the user-defined step function.
     *          For PhysicalObject, it is also used to compute acceleration statistics.
     */
    virtual void launch_user_step(float t) override;

    /**
     * @brief Retrieves the object's current position.
     *
     * Returns the position of the object's physical body as a Box2D vector.
     *
     * @return b2Vec2 The current position.
     */
    virtual b2Vec2 get_position() const;

    /**
     * @brief Retrieves the object's current orientation angle.
     *
     * Computes and returns the orientation angle (in radians) of the object's body.
     *
     * @return float The orientation angle.
     */
    float get_angle() const;

    /**
     * @brief Retrieves the object's current angular velocity
     *
     * @return float The angular velocity
     */
    float get_angular_velocity() const;

    /**
     * @brief Retrieves the object's current angular velocity
     *
     * @return float The angular velocity
     */
    b2Vec2 get_linear_acceleration() const;

    /**
     * @brief Renders the object on the given SDL renderer.
     *
     * @param renderer Pointer to the SDL_Renderer.
     * @param world_id The Box2D world identifier (unused in rendering).
     */
    virtual void render(SDL_Renderer* renderer, b2WorldId world_id) const override = 0 ;

    /**
     * @brief Move the object to a given coordinate
     *
     * @param x X coordinate.
     * @param y Y coordinate.
     * @param theta Orientation, in rad.
     */
    virtual void move(float x, float y, float theta = NAN) override;

    /**
     * @brief Returns whether this object is tangible (e.g. collisions, etc) or not.
     */
    virtual bool is_tangible() const override { return true; };

    /**
     * @brief Create serialization fields of the data logger
     *
     * @param data_logger Pointer to a DataLogger used for serialization
     */
    virtual void create_serialization_fields(DataLogger* data_logger) override;

    /**
     * @brief Save base values of the object into a data logger row.
     *
     * @param data_logger Pointer to a DataLogger used to serialize base values
     */
    virtual void serialize_base_values(DataLogger* data_logger) override;

    /**
     * @brief Return one or more polygonal contours that represent the geometry of the object.
     *
     * @param points_per_contour  Desired number of vertices for each contour
     *                            (a rectangle has one contour, a disk has one,
     *                             an arena may have many – one per wall).
     *
     * @return arena_polygons_t   A vector of closed polygons (counter‑clockwise,
     *                            last vertex different from the first – the caller
     *                            may close the loop if needed).
     */
    virtual arena_polygons_t generate_contours(std::size_t points_per_contour = 0) const override;

    // Base info
    uint16_t id;                         ///< Object identifier.


protected:
    /**
     * @brief Perform the base initialization (e.g. create Box2D objects).
     *  Called once by `init(world_id)`.
     *
     * @param world_id The Box2D world identifier.
     */
    virtual void do_init([[maybe_unused]] b2WorldId world_id) override;

    /**
     * @brief Parse a provided configuration and set associated members values.
     *
     * @param config Configuration entry describing the object properties.
     */
    virtual void parse_configuration(Configuration const& config, Simulation* simulation) override;

    /**
     * @brief Creates the object's physical body in the simulation.
     *
     * Constructs a dynamic body in the Box2D world at the specified position, defines its shape
     * based on the provided geometry.
     *
     * @param world_id The Box2D world identifier.
     */
    virtual void create_body(b2WorldId world_id);

    // Physical information
    float linear_damping;
    float angular_damping;
    float density;
    float friction;
    float restitution;
    b2BodyId body_id;      ///< Box2D body identifier.

    // Useful to compute acceleration
    float _estimated_dt = 0.0f;
    float _last_time = 0.0f;
    b2Vec2 _prev_v = {NAN, NAN};
    b2Vec2 _lin_acc = {NAN, NAN};
};


/**
 * @brief Physical object without user code (i.e. passive). Can still interact with other objects (e.g. collisions, etc).
 *
 */
class PassiveObject : public PhysicalObject {
public:

    /**
     * @brief Constructs a PassiveObject.
     *
     * @param _id Unique object identifier.
     * @param x Initial x-coordinate in the simulation.
     * @param y Initial y-coordinate in the simulation.
     * @param geom Object's geometry.
     * @param _linear_damping Linear damping value for the physical body (default is 0.0f).
     * @param _angular_damping Angular damping value for the physical body (default is 0.0f).
     * @param _density Density of the body shape (default is 10.0f).
     * @param _friction Friction coefficient of the body shape (default is 0.3f).
     * @param _restitution Restitution (bounciness) of the body shape (default is 0.5f).
     * @param _colormap Name of the colormap to use to set the color of the object
     * @param category Name of the category of the object.
     */
    PassiveObject(uint16_t _id, float _x, float _y,
           ObjectGeometry& geom,
           float _linear_damping = 0.0f, float _angular_damping = 0.0f,
           float _density = 10.0f, float _friction = 0.3f, float _restitution = 0.5f,
           std::string _colormap = "rainbow",
           std::string const& _category = "objects");

    /**
     * @brief Constructs a PassiveObject from a configuration entry.
     *
     * @param simulation Pointer to the underlying simulation.
     * @param _id Unique object identifier.
     * @param x Initial x-coordinate in the simulation.
     * @param y Initial y-coordinate in the simulation.
     * @param config Configuration entry describing the object properties.
     * @param category Name of the category of the object.
     */
    PassiveObject(Simulation* simulation, uint16_t _id, float _x, float _y,
           Configuration const& config,
           std::string const& _category = "objects");

    /**
     * @brief Renders the object on the given SDL renderer.
     *
     * @param renderer Pointer to the SDL_Renderer.
     * @param world_id The Box2D world identifier (unused in rendering).
     */
    void render(SDL_Renderer* renderer, b2WorldId world_id) const override;

protected:
    std::string colormap;

    /**
     * @brief Parse a provided configuration and set associated members values.
     *
     * @param config Configuration entry describing the object properties.
     */
    virtual void parse_configuration(Configuration const& config, Simulation* simulation) override;
};


/**
 * @brief Factory of simulation Objects. Return a constructed object from configuration.
 *
 * @param simulation Pointer to the underlying simulation.
 * @param world_id The Box2D world identifier (unused in rendering).
 * @param config Configuration entry describing the object properties.
 * @param light_map Pointer to the global light level map.
 * @param userdatasize Size of the memory block allocated for user data.
 * @param category Name of the category of the object.
 */
Object* object_factory(Simulation* simulation, uint16_t id, float x, float y, b2WorldId world_id, Configuration const& config, LightLevelMap* light_map, size_t userdatasize = 0, std::string const& category = "objects");

/**
 * @brief Interface to colormaps
 *
 * @param name Name of the colormap.
 * @param value Value to determine the color in this colormap
 * @param r Red color component
 * @param g Green color component
 * @param b Blue color component
 */
void get_cmap_val(std::string const name, uint8_t const value, uint8_t* r, uint8_t* g, uint8_t* b);

#endif


// MODELINE "{{{1
// vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
// vim:foldmethod=marker
