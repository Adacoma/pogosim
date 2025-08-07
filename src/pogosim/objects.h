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
 * @class StaticLightObject
 * @brief Light-emitting object with optional radial gradient.
 *
 * @ingroup objects
 *
 * @details
 *  A #StaticLightObject registers a callback on the global
 *  #LightLevelMap given at construction time.  
 *  Once invoked, the callback iterates on every covered bin and calls
 *  #LightLevelMap::add_light_level() with the intensity calculated
 *  according to the current @ref LightMode.
 *
 *  The class is **pod-friendly** (all data members are trivially
 *  copyable / reset-able), yet provides high-level behaviour such as:
 *  - automatic update of the light map when switching in/out of
 *    a photo-start pulse;
 *  - automatic gradient-radius computation if the user passes
 *    `gradient_radius ≤ 0`.
 */
class StaticLightObject : public Object {
public:

    /**
     * @enum LightMode
     * @brief Selects how intensity is distributed over the object geometry.
     *
     * @var LightMode::STATIC
     *  All bins receive an identical intensity = #value.
     *
     * @var LightMode::GRADIENT
     *  Bins receive an intensity that decays **linearly** with the distance
     *  from the geometrical centre (`x`,`y`).  
     *  The intensity at the outer edge of the gradient is #edge_value.
     */
    enum class LightMode { STATIC, GRADIENT, PLANE };

    /**
     * @brief Construct a light object programmatically.
     *
     * @param _x                  Initial *x* coordinate of the object centre
     *                            (in world units, mm).
     * @param _y                  Initial *y* coordinate of the object centre.
     * @param _geom               Reference to the geometry that delimits
     *                            where the object exists in space.
     * @param light_map           Pointer to the global #LightLevelMap
     *                            (must remain valid for the lifetime
     *                            of the object).
     * @param _value              **Centre** intensity in the range \[0, 32767\]
     *                            (for `STATIC` this is also the only intensity).
     * @param _mode               Intensity distribution strategy
     *                            (default =`STATIC`).
     * @param _edge_value         Intensity at @p gradient_radius
     *                            (only meaningful in `GRADIENT` mode).
     * @param _gradient_radius    Radius, in mm, at which the intensity
     *                            reaches @p _edge_value.  
     *                            If ≤ 0 the radius is computed automatically
     *                            as the farthest covered bin centre.
     * @param _photo_start_at     Start time (seconds, simulation clock) of the
     *                            optional photo-start pulse.  
     *                            Set to a negative value to disable.
     * @param _photo_start_duration
     *                            Duration (seconds) of the pulse.
     * @param _photo_start_value  Intensity during the pulse.  After the pulse
     *                            the object reverts to @p _value.
     * @param _category           Category name for profiling / filtering.
     *
     * @note The object automatically registers a callback on
     *       @p light_map; **do not** call #update_light_map() manually.
     */
    StaticLightObject(float _x, float _y,
                      ObjectGeometry& _geom, LightLevelMap* light_map,
                      int16_t _value,
                      LightMode _mode               = LightMode::STATIC,
                      int16_t _edge_value           = 0,
                      float   _gradient_radius      = -1.0f,
                      float   _plane_angle          = 0.0f,
                      float   _plane_half_span      = 1000.0f,
                      float   _photo_start_at       = -1.0f,
                      float   _photo_start_duration = 1.0f,
                      int16_t _photo_start_value    = 32767,
                      std::string const& _category  = "objects");


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


    /** Centre intensity (also the uniform level for `STATIC` mode). */
    int16_t value = 0;

    /** Saved intensity for restoring after a photo-start pulse. */
    int16_t orig_value = 0;

    /** Intensity at #gradient_radius (GRADIENT and PLANE modes only). */
    int16_t edge_value = 0;

    /**
     * Radius (mm) used for linear fall-off in GRADIENT mode.
     * A non-positive value means “compute automatically”.
     */
    float gradient_radius = -1.0f;

    // Direction of the plane’s normal, *in radians* (0 → +x, π/2 → +y)
    float plane_angle = 0.0f;

    // Half-width (mm) of the transition zone.  
    // The intensity goes from `value` to `edge_value` over a span of 2·plane_half_span.
    float plane_half_span = 1000.0f;

    /** Selected intensity distribution strategy. */
    LightMode mode = LightMode::STATIC;

    /** Pointer to the light map this object contributes to. */
    LightLevelMap* light_map = nullptr;

    // ---------- Photo-start pulse parameters ---------------------------- //
    float   photo_start_at        = -1.0f;  ///< start time (s), negative ⇒ off
    float   photo_start_duration  = 1.0f;   ///< pulse width  (s)
    int16_t photo_start_value     = 32767;  ///< intensity during pulse
    bool    performing_photo_start = false; ///< internal state flag
};

/**
 * @class RotatingRayOfLightObject
 * @brief Single ray of light that sweeps around its centre,
 *  and that becomes visible only after an optional photo-start delay.
 *
 * The object overwrites each covered light-map bin with either
 * `value` (inside the ray) or `0` (outside the ray) via
 * LightLevelMap::set_light_level().
 *
 * @param angular_speed  Sweep speed in rad · s⁻¹ (default = 3).
 * @param ray_half_width Half the angular aperture of the ray in
 *                       radians (default ≈ 5.7 ° = 0.1 rad).
 */
class RotatingRayOfLightObject : public Object {
public:
    RotatingRayOfLightObject(float x, float y, ObjectGeometry& geom,
                             LightLevelMap* light_map, int16_t value,
                             float ray_half_width  = 0.1f,
                             float angular_speed   = 3.0f,
                             float photo_start_at  = -1.0f,
                             float photo_start_dur = 1.0f,
                             float _white_frame_dur = 1.0f,
                             int16_t _white_frame_val = 32767,
                             std::string const& category = "objects");

    RotatingRayOfLightObject(Simulation* simulation, float x, float y,
                             LightLevelMap* light_map,
                             Configuration const& config,
                             std::string const& category = "objects");

    void render(SDL_Renderer*, b2WorldId) const override {}
    void update_light_map(LightLevelMap& l);
    void launch_user_step(float t) override;

protected:
    void parse_configuration(Configuration const& config,
                             Simulation* simulation) override;

    void start_white_frame(float now_s);

private:
    static float normalise_angle(float a);

    /* --- parameters --------------------------------------------------- */
    LightLevelMap* light_map = nullptr;
    int16_t value            = 0;
    float ray_half_width     = 0.1f;   ///< radians
    float angular_speed      = 3.0f;   ///< rad·s⁻¹
    float photo_start_at     = -1.0f;  ///< s, <0 ⇒ disabled
    float photo_start_dur    = 1.0f;   ///< s
    float white_frame_dur    = 0.03f;  ///< s, 0 ⇒ disabled
    int16_t white_frame_val  = 32767;  ///< level inside the white frame

    /* --- evolving state ----------------------------------------------- */
    float  current_angle         = 0.0f;  ///< radians
    bool   ray_is_active         = true;  ///< false until photo-start is over
    float  previous_angle        = 0.f;   ///< rad, for wrap-around test
    bool   white_frame_active    = false;
    float  white_frame_end_time  = 0.f;   ///< s, when to stop white frame
    float  sim_prev_t            = 0.f;
    float  ray_current_t         = 0.f;
};


class AlternatingDualRayOfLightObject : public Object {
public:
    AlternatingDualRayOfLightObject(float x, float y, ObjectGeometry& geom,
                                    LightLevelMap* light_map, int16_t value,
                                    float ray_half_width          = 0.1f,
                                    float angular_speed           = 3.0f,
                                    float long_white_frame_dur    = 1.0f,
                                    float short_white_frame_dur   = 0.03f,
                                    int16_t white_frame_val       = 32767,
                                    std::string const& category   = "objects");

    AlternatingDualRayOfLightObject(Simulation* simulation, float x, float y,
                                    LightLevelMap* light_map,
                                    Configuration const& config,
                                    std::string const& category = "objects");

    void render(SDL_Renderer*, b2WorldId) const override {}
    void launch_user_step(float t) override;
    void update_light_map(LightLevelMap& l);

protected:
    void parse_configuration(Configuration const& config,
                             Simulation* simulation) override;

private:
    /* helpers ---------------------------------------------------------- */
    static float normalise_angle(float a);
    void         recompute_geometry();
    void         request_map_refresh() { light_map->update(); }

    /* phase machine ---------------------------------------------------- */
    enum class phase_t { LONG_WHITE, LEFT_RAY, SHORT_WHITE, RIGHT_RAY };

    void enter_phase(phase_t p, float now_s);

    /* parameters ------------------------------------------------------- */
    LightLevelMap* light_map       = nullptr;
    int16_t         value           = 0;
    float           ray_half_width  = 0.1f;     /* rad  */
    float           angular_speed   = 3.0f;     /* rad·s⁻¹ */
    float           long_white_dur  = 1.0f;     /* s    */
    float           short_white_dur = 0.03f;    /* s    */
    int16_t         white_val       = 32767;

    /* geometry --------------------------------------------------------- */
    BoundingBox bbox {};
    float       cx = 0.f, cy = 0.f;          /* centre of geometry         */
    float       ax_l = 0.f, ay_l = 0.f;      /* top-left  apex             */
    float       ax_r = 0.f, ay_r = 0.f;      /* top-right apex             */
    float       left_a0  = 0.f, left_a1  = 0.f;
    float       right_a0 = 0.f, right_a1 = 0.f;

    /* evolving state --------------------------------------------------- */
    phase_t phase            = phase_t::LONG_WHITE;
    float   phase_start_t    = 0.f;          /* s                           */
    float   prev_t           = 0.f;          /* s                           */
    float   current_angle    = 0.f;          /* rad – of active ray         */
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
