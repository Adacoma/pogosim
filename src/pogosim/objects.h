#ifndef OBJECTS_H
#define OBJECTS_H

#include <functional>

#include "utils.h"
#include "configuration.h"
#include "render.h"
#include "colormaps.h"


class Simulation;

/**
 * @brief Represents a disk with center (x, y) and radius.
 */
struct BoundingDisk {
    float center_x;
    float center_y;
    float radius;
};

/**
 * @brief Represents an axis-aligned bounding box with top-left corner (x, y) and dimensions width and height.
 */
struct BoundingBox {
    float x;
    float y;
    float width;
    float height;
};



/**
 * @brief Geometry of an object.
 *
 */
class ObjectGeometry {
public:

    /**
     * @brief Construct an ObjectGeometry.
     */
    ObjectGeometry() {}

    /**
     * @brief Destructor
     */
    virtual ~ObjectGeometry();

    /**
     * @brief Create Box2D shape based on this geometry
     */
    virtual void create_box2d_shape(b2BodyId body_id, b2ShapeDef& shape_def) = 0;

    /**
     * @brief Return Box2D shape_id
     */
    b2ShapeId get_shape_id() const { return shape_id; }

    /**
     * @brief Exports a boolean 2D grid showing which bins are covered by the geometry.
     *
     * @param num_bins_x Number of bins along the X-axis.
     * @param num_bins_y Number of bins along the Y-axis.
     * @param bin_width Width (size) of each bin.
     * @param bin_height Height (size) of each bin.
     * @param obj_x The x-coordinate of the object (geometry center).
     * @param obj_y The y-coordinate of the object (geometry center).
     * @return A 2D vector of booleans. True in a given cell indicates that the geometry covers that bin.
     */
    virtual std::vector<std::vector<bool>> export_geometry_grid(size_t num_bins_x,
                                                                  size_t num_bins_y,
                                                                  float bin_width,
                                                                  float bin_height,
                                                                  float obj_x,
                                                                  float obj_y) const = 0;

    /**
     * @brief Renders the object on the given SDL renderer.
     *
     * @param renderer Pointer to the SDL_Renderer.
     * @param world_id The Box2D world identifier (unused in rendering).
     * @param x X coordinate
     * @param y Y coordinate
     * @param r Red color component
     * @param g Green color component
     * @param b Blue color component
     * @param alpha Alpha color component
     */
    virtual void render(SDL_Renderer* renderer, b2WorldId world_id, float x, float y, uint8_t r, uint8_t g, uint8_t b, uint8_t alpha = 255) const = 0;

    /**
     * @brief Computes the bounding disk that completely encloses the geometry.
     *
     * @return A BoundingDisk with center (x,y) and radius.
     */
    virtual BoundingDisk compute_bounding_disk() const = 0;

    /**
     * @brief Computes the axis-aligned bounding box that completely encloses the geometry.
     *
     * @return A BoundingBox with top-left corner (x,y) and width and height.
     */
    virtual BoundingBox compute_bounding_box() const = 0;

    /**
     * @brief Compute the distance from a given point to the geometry.
     */
    virtual float get_distance_to(b2Vec2 orig, b2Vec2 point) const;

    /**
     * @brief Return one or more polygonal contours that approximate / represent
     *        this geometry.
     *
     * @param points_per_contour  Desired number of vertices for each contour
     *                            (a rectangle has one contour, a disk has one,
     *                             an arena may have many – one per wall).
     *
     * @return arena_polygons_t   A vector of closed polygons (counter‑clockwise,
     *                            last vertex different from the first – the caller
     *                            may close the loop if needed).
     */
    virtual arena_polygons_t generate_contours(std::size_t points_per_contour = 0, b2Vec2 position = {0.0f, 0.0f}) const = 0;

protected:
    bool shape_created = false;
    b2ShapeId shape_id;     ///< Box2D shape identifier.
};

/**
 * @brief Disk-shaped geometry
 *
 */
class DiskGeometry : public ObjectGeometry {
public:

    /**
     * @brief Construct an ObjectGeometry.
     */
    DiskGeometry(float _radius) : radius(_radius) {}

    /**
     * @brief Create Box2D shape based on this geometry.
     */
    virtual void create_box2d_shape(b2BodyId body_id, b2ShapeDef& shape_def) override;

    /**
     * @brief Return radius of the disk.
     */
    float get_radius() const { return radius; }

    /**
     * @brief Exports a boolean 2D grid showing which bins are covered by the geometry.
     *
     * @param num_bins_x Number of bins along the X-axis.
     * @param num_bins_y Number of bins along the Y-axis.
     * @param bin_width Width (size) of each bin.
     * @param bin_height Height (size) of each bin.
     * @param obj_x The x-coordinate of the object (geometry center).
     * @param obj_y The y-coordinate of the object (geometry center).
     * @return A 2D vector of booleans. True in a given cell indicates that the geometry covers that bin.
     */
    virtual std::vector<std::vector<bool>> export_geometry_grid(size_t num_bins_x,
                                                                  size_t num_bins_y,
                                                                  float bin_width,
                                                                  float bin_height,
                                                                  float obj_x,
                                                                  float obj_y) const override;

    /**
     * @brief Renders the object on the given SDL renderer.
     *
     * @param renderer Pointer to the SDL_Renderer.
     * @param world_id The Box2D world identifier (unused in rendering).
     * @param x X coordinate
     * @param y Y coordinate
     * @param r Red color component
     * @param g Green color component
     * @param b Blue color component
     * @param alpha Alpha color component
     */
    virtual void render(SDL_Renderer* renderer, b2WorldId world_id, float x, float y, uint8_t r, uint8_t g, uint8_t b, uint8_t alpha = 255) const override;

    /**
     * @brief Computes the bounding disk that completely encloses the geometry.
     *
     * @return A BoundingDisk with center (x,y) and radius.
     */
    virtual BoundingDisk compute_bounding_disk() const override;

    /**
     * @brief Computes the axis-aligned bounding box that completely encloses the geometry.
     *
     * @return A BoundingBox with top-left corner (x,y) and width and height.
     */
    virtual BoundingBox compute_bounding_box() const override;

    /**
     * @brief Return one or more polygonal contours that approximate / represent
     *        this geometry.
     *
     * @param points_per_contour  Desired number of vertices for each contour
     *                            (a rectangle has one contour, a disk has one,
     *                             an arena may have many – one per wall).
     *
     * @return arena_polygons_t   A vector of closed polygons (counter‑clockwise,
     *                            last vertex different from the first – the caller
     *                            may close the loop if needed).
     */
    virtual arena_polygons_t generate_contours(std::size_t points_per_contour = 0, b2Vec2 position  = {0.0f, 0.0f}) const override;

protected:
    float radius;           ///< Radius of the disk
};

class RectangleGeometry : public ObjectGeometry {
public:
    /**
     * @brief Construct a RectangleGeometry.
     * @param _width The width of the rectangle.
     * @param _height The height of the rectangle.
     */
    RectangleGeometry(float _width, float _height) : width(_width), height(_height) {}

    /**
     * @brief Create a Box2D shape based on this geometry.
     */
    virtual void create_box2d_shape(b2BodyId body_id, b2ShapeDef& shape_def) override;

    /**
     * @brief Exports a boolean 2D grid showing which bins are covered by the rectangle.
     *
     * @param num_bins_x Number of bins along the X-axis.
     * @param num_bins_y Number of bins along the Y-axis.
     * @param bin_width Width (size) of each bin.
     * @param bin_height Height (size) of each bin.
     * @param obj_x The x-coordinate of the object (geometry center).
     * @param obj_y The y-coordinate of the object (geometry center).
     * @return A 2D vector of booleans. True in a given cell indicates that the geometry covers that bin.
     */
    virtual std::vector<std::vector<bool>> export_geometry_grid(size_t num_bins_x,
                                                                  size_t num_bins_y,
                                                                  float bin_width,
                                                                  float bin_height,
                                                                  float obj_x,
                                                                  float obj_y) const override;

    /**
     * @brief Renders the rectangle on the given SDL renderer.
     *
     * @param renderer Pointer to the SDL_Renderer.
     * @param world_id The Box2D world identifier (unused in rendering).
     * @param x X coordinate of the rectangle center.
     * @param y Y coordinate of the rectangle center.
     * @param r Red color component.
     * @param g Green color component.
     * @param b Blue color component.
     * @param alpha Alpha color component.
     */
    virtual void render(SDL_Renderer* renderer, b2WorldId world_id, float x, float y,
                        uint8_t r, uint8_t g, uint8_t b, uint8_t alpha = 255) const override;

    /**
     * @brief Returns the width of the rectangle.
     */
    float get_width() const { return width; }

    /**
     * @brief Returns the height of the rectangle.
     */
    float get_height() const { return height; }

    /**
     * @brief Computes the bounding disk that completely encloses the geometry.
     *
     * @return A BoundingDisk with center (x,y) and radius.
     */
    virtual BoundingDisk compute_bounding_disk() const override;

    /**
     * @brief Computes the axis-aligned bounding box that completely encloses the geometry.
     *
     * @return A BoundingBox with top-left corner (x,y) and width and height.
     */
    virtual BoundingBox compute_bounding_box() const override;

    /**
     * @brief Return one or more polygonal contours that approximate / represent
     *        this geometry.
     *
     * @param points_per_contour  Desired number of vertices for each contour
     *                            (a rectangle has one contour, a disk has one,
     *                             an arena may have many – one per wall).
     *
     * @return arena_polygons_t   A vector of closed polygons (counter‑clockwise,
     *                            last vertex different from the first – the caller
     *                            may close the loop if needed).
     */
    virtual arena_polygons_t generate_contours(std::size_t points_per_contour = 0, b2Vec2 position  = {0.0f, 0.0f}) const override;

protected:
    float width;   ///< Width of the rectangle.
    float height;  ///< Height of the rectangle.
};


/**
 * @brief Geometry representing the entire simulation.
 *
 */
class GlobalGeometry final : public ObjectGeometry {
public:
    /**
     * @brief Construct an ObjectGeometry.
     */
    GlobalGeometry() {}

    /**
     * @brief Create Box2D shape based on this geometry.
     */
    virtual void create_box2d_shape(b2BodyId, b2ShapeDef&) override {};

    /**
     * @brief Exports a boolean 2D grid showing which bins are covered by the geometry.
     *
     * @param num_bins_x Number of bins along the X-axis.
     * @param num_bins_y Number of bins along the Y-axis.
     * @param bin_width Width (size) of each bin.
     * @param bin_height Height (size) of each bin.
     * @param obj_x The x-coordinate of the object (geometry center).
     * @param obj_y The y-coordinate of the object (geometry center).
     * @return A 2D vector of booleans. True in a given cell indicates that the geometry covers that bin.
     */
    virtual std::vector<std::vector<bool>> export_geometry_grid(size_t num_bins_x,
                                                                  size_t num_bins_y,
                                                                  float bin_width,
                                                                  float bin_height,
                                                                  float obj_x,
                                                                  float obj_y) const override;

    /**
     * @brief Renders the object on the given SDL renderer.
     *
     * @param renderer Pointer to the SDL_Renderer.
     * @param world_id The Box2D world identifier (unused in rendering).
     * @param x X coordinate
     * @param y Y coordinate
     * @param r Red color component
     * @param g Green color component
     * @param b Blue color component
     * @param alpha Alpha color component
     */
    virtual void render(SDL_Renderer*, b2WorldId, float, float, uint8_t, uint8_t, uint8_t, uint8_t = 255) const override {}

    /**
     * @brief Computes the bounding disk that completely encloses the geometry.
     *
     * @return A BoundingDisk with center (x,y) and radius.
     */
    virtual BoundingDisk compute_bounding_disk() const override;

    /**
     * @brief Computes the axis-aligned bounding box that completely encloses the geometry.
     *
     * @return A BoundingBox with top-left corner (x,y) and width and height.
     */
    virtual BoundingBox compute_bounding_box() const override;

    /**
     * @brief Compute the distance from a given point to the geometry.
     */
    virtual float get_distance_to([[maybe_unused]] b2Vec2 orig, [[maybe_unused]] b2Vec2 point) const override { return 0.0f; }

    /**
     * @brief Return one or more polygonal contours that approximate / represent
     *        this geometry.
     *
     * @param points_per_contour  Desired number of vertices for each contour
     *                            (a rectangle has one contour, a disk has one,
     *                             an arena may have many – one per wall).
     *
     * @return arena_polygons_t   A vector of closed polygons (counter‑clockwise,
     *                            last vertex different from the first – the caller
     *                            may close the loop if needed).
     */
    virtual arena_polygons_t generate_contours([[maybe_unused]] std::size_t points_per_contour = 0, [[maybe_unused]] b2Vec2 position  = {0.0f, 0.0f}) const override { return {}; }
};


/**
 * @brief Geometry representing the entire arena (collection of wall polygons).
 *
 * Each element of @p arena_polygons describes one closed wall loop.
 */
class ArenaGeometry final : public ObjectGeometry {
public:
    /// Ctor – keeps a ref to the polygon container that lives elsewhere.
    explicit ArenaGeometry(arena_polygons_t const& arena_polygons) noexcept
        : arena_polygons_{arena_polygons} {}

    /* ---------- ObjectGeometry interface ---------- */

    void create_box2d_shape([[maybe_unused]] b2BodyId body_id, [[maybe_unused]] b2ShapeDef& shape_def) override { }

    std::vector<std::vector<bool>>
    export_geometry_grid(std::size_t num_bins_x,
                         std::size_t num_bins_y,
                         float      bin_width,
                         float      bin_height,
                         float      obj_x,
                         float      obj_y) const override;

    void render(SDL_Renderer*, b2WorldId, float, float,
                uint8_t, uint8_t, uint8_t, uint8_t = 255) const override { }

    BoundingDisk  compute_bounding_disk() const override;
    BoundingBox   compute_bounding_box()  const override;

    /**
     * @brief Return the *shortest* Euclidean distance between @p point and all arena walls.
     *
     * The @p orig parameter is ignored (it is only useful for geometries that need a reference
     * point such as Catmull‑Rom splines).
     */
    float get_distance_to(b2Vec2 /*orig*/, b2Vec2 point) const override;

    arena_polygons_t const& get_arena_polygons() { return arena_polygons_; }

    /**
     * @brief Return one or more polygonal contours that approximate / represent
     *        this geometry.
     *
     * @param points_per_contour  Desired number of vertices for each contour
     *                            (a rectangle has one contour, a disk has one,
     *                             an arena may have many – one per wall).
     *
     * @return arena_polygons_t   A vector of closed polygons (counter‑clockwise,
     *                            last vertex different from the first – the caller
     *                            may close the loop if needed).
     */
    virtual arena_polygons_t generate_contours(std::size_t points_per_contour = 0, b2Vec2 position  = {0.0f, 0.0f}) const override;

private:
    static float distance_point_segment(b2Vec2 p, b2Vec2 a, b2Vec2 b) noexcept;
    static bool  point_inside_polygon(b2Vec2 p, const std::vector<b2Vec2>& poly) noexcept;

    const arena_polygons_t& arena_polygons_;
};


/**
 * @brief A discretized 2D grid representing light intensities over a simulation area.
 */
class LightLevelMap {
public:
    /**
     * @brief Construct a LightLevelMap.
     * @param num_bins_x Number of bins along the x-axis.
     * @param num_bins_y Number of bins along the y-axis.
     * @param bin_width Physical width of each bin.
     * @param bin_height Physical height of each bin.
     */
    LightLevelMap(size_t num_bins_x, size_t num_bins_y, float bin_width, float bin_height);

    /// Destructor.
    ~LightLevelMap();

    /**
     * @brief Get the light level at a physical coordinate (world‐space).
     * @param x  X coordinate in the same units as bin_width_.
     * @param y  Y coordinate in the same units as bin_height_.
     * @return   The light level at the bin containing (x,y), or 0 if outside.
     */
    float get_light_level_at(float x, float y) const;

    /// Returns the light level stored at the given bin (bin_x, bin_y).
    float get_light_level(size_t bin_x, size_t bin_y) const;

    /// Sets the light level at the given bin (bin_x, bin_y).
    void set_light_level(size_t bin_x, size_t bin_y, int16_t value);

    /// Adds a given value to the light level at the given bin (bin_x, bin_y).
    void add_light_level(size_t bin_x, size_t bin_y, int16_t value);

    /// Resets all bins to 0.
    void clear();

    /// Accessors for the grid properties.
    size_t get_num_bins_x() const;
    size_t get_num_bins_y() const;
    float get_bin_width() const;
    float get_bin_height() const;

    /**
     * @brief Renders the light level map to the given SDL_Renderer.
     *
     * This method scales each bin's light level into a brightness value. For each bin, it:
     * - Normalizes the light level from the int16_t range [-32768, 32767] to [0, 1].
     * - Maps that normalized value to a brightness in the range [100, 200].
     * - Renders a filled rectangle with that brightness.
     *
     * @param renderer A pointer to the SDL_Renderer used for drawing.
     */
    void render(SDL_Renderer* renderer) const;

    /// Register a callback which will be called with the map
    /// whenever update() is run.
    void register_callback(std::function<void(LightLevelMap&)> cb);

    /// Clears the map and invokes all registered callbacks.
    void update();

private:
    size_t num_bins_x_;
    size_t num_bins_y_;
    float bin_width_;
    float bin_height_;
    std::vector<std::vector<int16_t>> levels_;
    std::vector<std::function<void(LightLevelMap&)>> callbacks_;
};


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
     * @brief Parse a provided configuration and set associated members values.
     *
     * @param config Configuration entry describing the object properties.
     */
    virtual void parse_configuration(Configuration const& config, Simulation* simulation);

    ObjectGeometry* geom;                ///< Geometry of the object.
};



/**
 * @brief StaticLightObject that contributes light to the simulation's light level map.
 */
class StaticLightObject : public Object {
public:

    /**
     * @brief Constructs a StaticLightObject object.
     *
     * @param x Initial x-coordinate in the simulation.
     * @param y Initial y-coordinate in the simulation.
     * @param geom Object's geometry.
     * @param light_map Pointer to the global light level map.
     * @param value Light value between 0 (min) and 32767 (max).
     * @param photo_start_at Change the light value at the specified time to trigger the synchronised photo start of the robots.
     * @param photo_start_duration Amount of time to stay in the photo start stage.
     * @param photo_start_value Light value during the photo start stage.
     * @param category Name of the category of the object.
     */
    StaticLightObject(float _x, float _y,
           ObjectGeometry& _geom, LightLevelMap* light_map,
           int16_t _value, float _photo_start_at = -1.0f, float _photo_start_duration = 1.0f, int16_t _photo_start_value = 32767,
           std::string const& _category = "objects");

    /**
     * @brief Constructs a StaticLightObject object from a configuration entry.
     *
     * @param simulation Pointer to the underlying simulation.
     * @param x Initial x-coordinate in the simulation.
     * @param y Initial y-coordinate in the simulation.
     * @param light_map Pointer to the global light level map.
     * @param config Configuration entry describing the object properties.
     * @param category Name of the category of the object.
     */
    StaticLightObject(Simulation* simulation, float _x, float _y,
            LightLevelMap* light_map, Configuration const& config,
            std::string const& _category = "objects");

    /**
     * @brief Renders the object on the given SDL renderer.
     *
     * @param renderer Pointer to the SDL_Renderer.
     * @param world_id The Box2D world identifier (unused in rendering).
     */
    virtual void render(SDL_Renderer*, b2WorldId) const override {}

    /// Updates the object's contribution to the light level map.
    virtual void update_light_map(LightLevelMap& l);

    /**
     * @brief Launches the user-defined step function.
     *
     * Updates the object's time, enables all registered stop watches, executes the user step
     * function via pogo_main_loop_step, and then disables the stop watches.
     */
    virtual void launch_user_step(float t) override;

protected:
    /**
     * @brief Parse a provided configuration and set associated members values.
     *
     * @param config Configuration entry describing the object properties.
     */
    virtual void parse_configuration(Configuration const& config, Simulation* simulation) override;

    int16_t value;
    int16_t orig_value;
    LightLevelMap* light_map;  ///< Pointer to the global light level map.
    float photo_start_at;
    float photo_start_duration;
    int16_t photo_start_value;
    bool performing_photo_start = false;
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
     * @param world_id The Box2D world identifier.
     * @param _linear_damping Linear damping value for the physical body (default is 0.0f).
     * @param _angular_damping Angular damping value for the physical body (default is 0.0f).
     * @param _density Density of the body shape (default is 10.0f).
     * @param _friction Friction coefficient of the body shape (default is 0.3f).
     * @param _restitution Restitution (bounciness) of the body shape (default is 0.5f).
     * @param category Name of the category of the object.
     */
    PhysicalObject(uint16_t _id, float _x, float _y,
           ObjectGeometry& geom, b2WorldId world_id,
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
     * @param world_id The Box2D world identifier.
     * @param config Configuration entry describing the object properties.
     * @param category Name of the category of the object.
     */
    PhysicalObject(Simulation* simulation, uint16_t _id, float _x, float _y,
           b2WorldId world_id, Configuration const& config,
           std::string const& _category = "objects");

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
     * @param world_id The Box2D world identifier.
     * @param _linear_damping Linear damping value for the physical body (default is 0.0f).
     * @param _angular_damping Angular damping value for the physical body (default is 0.0f).
     * @param _density Density of the body shape (default is 10.0f).
     * @param _friction Friction coefficient of the body shape (default is 0.3f).
     * @param _restitution Restitution (bounciness) of the body shape (default is 0.5f).
     * @param _colormap Name of the colormap to use to set the color of the object
     * @param category Name of the category of the object.
     */
    PassiveObject(uint16_t _id, float _x, float _y,
           ObjectGeometry& geom, b2WorldId world_id,
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
     * @param world_id The Box2D world identifier.
     * @param config Configuration entry describing the object properties.
     * @param category Name of the category of the object.
     */
    PassiveObject(Simulation* simulation, uint16_t _id, float _x, float _y,
           b2WorldId world_id, Configuration const& config,
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
 * @brief Factory of ObjectGeometries
 *
 * @param config Configuration entry describing the object properties.
 * @param simulation Pointer to the underlying simulation.
 */
ObjectGeometry* object_geometry_factory(Configuration const& config, Simulation* simulation);


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
