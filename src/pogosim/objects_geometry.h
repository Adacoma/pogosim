#ifndef OBJECTS_GEOMETRY_H
#define OBJECTS_GEOMETRY_H

#include <functional>
#include <array>

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
    virtual void render(SDL_Renderer* renderer, b2WorldId world_id, float x, float y, float theta,
            uint8_t r, uint8_t g, uint8_t b, uint8_t alpha = 255) const = 0;

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
    virtual void render(SDL_Renderer* renderer, b2WorldId world_id, float x, float y, float theta,
            uint8_t r, uint8_t g, uint8_t b, uint8_t alpha = 255) const override;

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
    virtual void render(SDL_Renderer* renderer, b2WorldId world_id, float x, float y, float theta,
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


class TriangleGeometry : public ObjectGeometry {
public:
    /**
     * @brief Construct an equilateral TriangleGeometry.
     *
     * The triangle is centered at (0,0) in local coordinates (its centroid).
     *
     * @param _side_length The side length of the equilateral triangle.
     */
    explicit TriangleGeometry(float _side_length) : side_length(_side_length) {}

    /**
     * @brief Create a Box2D shape based on this geometry.
     */
    void create_box2d_shape(b2BodyId body_id, b2ShapeDef& shape_def) override;

    /**
     * @brief Exports a boolean 2D grid showing which bins are covered by the triangle.
     */
    std::vector<std::vector<bool>> export_geometry_grid(size_t num_bins_x,
                                                        size_t num_bins_y,
                                                        float bin_width,
                                                        float bin_height,
                                                        float obj_x,
                                                        float obj_y) const override;

    /**
     * @brief Renders the triangle on the given SDL renderer.
     */
    void render(SDL_Renderer* renderer, b2WorldId world_id, float x, float y, float theta,
                uint8_t r, uint8_t g, uint8_t b, uint8_t alpha = 255) const override;

    /**
     * @brief Returns the side length of the equilateral triangle.
     */
    float get_side_length() const { return side_length; }

    /**
     * @brief Computes the bounding disk that completely encloses the geometry.
     */
    BoundingDisk compute_bounding_disk() const override;

    /**
     * @brief Computes the axis-aligned bounding box that completely encloses the geometry.
     */
    BoundingBox compute_bounding_box() const override;

    /**
     * @brief Return one or more polygonal contours that approximate / represent
     *        this geometry.
     */
    arena_polygons_t generate_contours(std::size_t points_per_contour = 0, b2Vec2 position  = {0.0f, 0.0f}) const override;

private:
    /**
     * @brief Returns the local-space vertices of the triangle (centroid at origin),
     *        counter-clockwise (CCW).
     */
    std::array<b2Vec2, 3> get_local_vertices() const;

    float side_length;  ///< Side length of the equilateral triangle.
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
    virtual void render(SDL_Renderer*, b2WorldId, float, float, float, uint8_t, uint8_t, uint8_t, uint8_t = 255) const override {}

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

    void render(SDL_Renderer*, b2WorldId, float, float, float,
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
 * @brief Factory of ObjectGeometries
 *
 * @param config Configuration entry describing the object properties.
 * @param simulation Pointer to the underlying simulation.
 */
ObjectGeometry* object_geometry_factory(Configuration const& config, Simulation* simulation);


#endif

// MODELINE "{{{1
// vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
// vim:foldmethod=marker
