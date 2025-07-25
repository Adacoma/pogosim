#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <box2d/box2d.h>
#include <vector>
#include <string>
#include <tuple>
#include <utility>
#include <cstdint>

typedef std::vector<std::vector<b2Vec2>> arena_polygons_t;

float const VISUALIZATION_SCALE = 100.0f; // 1 Box2D unit = 100 pixels

/// Global scaling factor from millimeters to pixels.
extern float mm_to_pixels;
/// Global visualization offset for the x-coordinate.
extern float visualization_x;
/// Global visualization offset for the y-coordinate.
extern float visualization_y;

/**
 * @brief Adjusts the global mm_to_pixels scaling factor.
 *
 * This function modifies the conversion factor from millimeters to pixels by adding
 * the provided delta. The resulting value is clamped between 0.10 and 10.0.
 *
 * @param delta The value to add to mm_to_pixels.
 */
void adjust_mm_to_pixels(float delta);

/**
 * @brief Calculates the visualization position for given x and y coordinates.
 *
 * This function converts physical coordinates to visualization coordinates using the global
 * mm_to_pixels scaling factor and visualization offsets.
 *
 * @param x The x-coordinate in the original space.
 * @param y The y-coordinate in the original space.
 * @return b2Vec2 The computed visualization position.
 */
b2Vec2 visualization_position(float x, float y);

/**
 * @brief Calculates the visualization position for a given point.
 *
 * This overloaded function converts a b2Vec2 point to visualization coordinates using the global
 * mm_to_pixels scaling factor and visualization offsets.
 *
 * @param pos The original position as a b2Vec2.
 * @return b2Vec2 The computed visualization position.
 */
b2Vec2 visualization_position(b2Vec2 pos);

/**
 * @brief Reads polygons from a CSV file and scales them to match a specified surface area.
 *
 * This function loads polygons from a CSV file where each line contains the x and y
 * coordinates separated by a comma. An empty line indicates the end of a polygon.
 * Once the polygons are loaded, the function normalizes all points into the [0,1] range
 * using the overall bounding box of the data. The effective area is then computed as the
 * area of the first (main) polygon minus the sum of the areas of any subsequent polygons
 * (considered holes). A uniform scaling factor is determined so that when applied to the
 * normalized polygons, the effective area becomes equal to the specified total_surface.
 *
 * @param filename The path to the CSV file containing the polygon vertices.
 * @param total_surface The desired surface area for the main polygon after subtracting the holes.
 * @return std::vector<std::vector<b2Vec2>> A vector containing the scaled polygons, where each
 *         polygon is represented as a vector of b2Vec2 points.
 *
 * @throw std::runtime_error If the file cannot be opened, no polygons are loaded, or the
 *         effective area (main polygon area minus holes area) is non-positive.
 */
std::vector<std::vector<b2Vec2>> read_poly_from_csv(const std::string& filename, float total_surface);

/**
 * @brief Generates a random point within the specified polygon.
 *
 * This function calculates the bounding box of the provided polygon and repeatedly generates random points
 * within that box until one is found that lies inside the polygon.
 *
 * @param polygon A vector of b2Vec2 points defining the polygon.
 * @return b2Vec2 A randomly generated point within the polygon.
 *
 * @throws std::runtime_error If the polygon has fewer than 3 points.
 */

b2Vec2 generate_random_point_within_polygon(const std::vector<b2Vec2>& polygon);

/**
 * @brief Determines whether a point is within a polygon.
 *
 * This function uses a ray-casting algorithm to test whether the point (x, y) lies inside the given polygon.
 *
 * @param polygon A vector of b2Vec2 points defining the polygon.
 * @param x The x-coordinate of the point.
 * @param y The y-coordinate of the point.
 * @return true If the point is inside the polygon.
 * @return false Otherwise.
 */
bool is_point_within_polygon(const std::vector<b2Vec2>& polygon, float x, float y);

/**
 * @brief Computes an offset polygon.
 *
 * This function generates a new polygon by offsetting the original polygon inward or outward by a specified distance.
 * It calculates normals at each vertex and computes new offset points accordingly.
 *
 * @param polygon A vector of b2Vec2 points defining the original polygon.
 * @param offset The offset distance to apply.
 * @return std::vector<b2Vec2> The resulting offset polygon.
 *
 * @throws std::runtime_error If the polygon has fewer than 3 points.
 */
std::vector<b2Vec2> offset_polygon(const std::vector<b2Vec2>& polygon, float offset);


/**
 * Generate random points inside a (possibly holed) polygonal domain while
 * respecting a per‑point exclusion radius and a global connectivity limit.
 *
 * A candidate is accepted only if it is:
 *   1. Inside the outer polygon and outside every “hole” polygon.
 *   2. At a distance ≥ r_i + r_j from every previously accepted point j.
 *   3. Within `max_neighbor_distance` of at least one previously accepted
 *      point (unless it is the very first point or `max_neighbor_distance`
 *      is +∞).
 *
 * If radius is NaN for a point, it will be set to {NaN, NaN} in the result.
 *
 * If it fails to build the whole set after `attempts_per_point` rejected
 * candidates, the algorithm discards all progress and restarts.  It will
 * attempt the whole sampling process up to `max_restarts` times before
 * throwing.
 */
std::vector<b2Vec2> generate_random_points_within_polygon_safe(
        const std::vector<std::vector<b2Vec2>> &polygons,
        const std::vector<float> &reserve_radii,
        float max_neighbor_distance = std::numeric_limits<float>::infinity(),
        std::uint32_t attempts_per_point = 100U,
        std::uint32_t max_restarts = 100U);

/**
 * Generate approximately equi-spaced random points inside a (possibly holed)
 * polygonal domain by running a few iterations of Lloyd’s relaxation
 * (a.k.a. “K-means” on a dense uniform sample).
 *
 *  1. Draw `n_samples` uniform random points inside the domain.
 *  2. Initialise `k` cluster centres by picking `k` of those samples at random.
 *  3. Repeat `kmeans_iterations` times (or until convergence):
 *     • Assign every sample to its closest centre.
 *     • Replace each centre by the arithmetic mean of the samples in its
 *       cluster.  If the mean falls outside the domain, snap it to the
 *       in-domain sample that is nearest to the mean.
 *  4. Return the final centres.
 *
 * In practice after ~15–25 iterations you already get a centroidal Voronoi
 * tessellation good enough for swarm-robot initialisation.
 *
 * @param polygons              outer polygon first, holes afterwards
 * @param k                     number of points to generate
 * @param n_samples             size of the uniform background sample
 * @param kmeans_iterations     maximum number of Lloyd relaxations
 * @param max_restarts          how many times we may restart if a step fails
 * @throws std::runtime_error   if the domain is invalid or no solution is found
 */
std::vector<b2Vec2> generate_points_voronoi_lloyd(const std::vector<std::vector<b2Vec2>> &polygons,
                                                  std::size_t k,
                                                  std::size_t n_samples      = 20'000,
                                                  std::size_t kmeans_iterations = 20,
                                                  std::uint32_t max_restarts = 3);


/**
 * @brief Uniformly sample approximately equi-spaced points in a (possibly holed)
 *        polygonal domain, while respecting *per-point* exclusion radii.
 *
 * The routine performs **Lloyd relaxation in power-distance space**:
 *   1. Draw `n_samples` uniform random points inside the domain.
 *   2. Initialise `k = reserve_radii.size()` centres with random samples.
 *   3. Iterate *kmeans_iterations* times (or until all moves < *convergence_eps*):
 *        • Assign every sample to the centre that minimises the
 *          power distance ‖p − c‖² − rᵢ².
 *        • Replace each centre by the arithmetic mean of its samples.
 *          If that mean falls outside the domain, snap it back to the
 *          closest *in-domain* sample in the same cluster.
 *   4. Run a lightweight “push-apart” pass so that no two centres end up
 *      closer than rᵢ + rⱼ (handles the rare residual overlaps).
 *
 * The power metric naturally enlarges Voronoi cells for large radii, giving
 * a blue-noise layout where big robots get more space.
 *
 * @param polygons            0-th entry = outer boundary, 1…N = holes
 * @param reserve_radii       desired exclusion radius per point (size k)
 * @param n_samples           size of the background uniform sample cloud
 * @param kmeans_iterations   maximum Lloyd iterations (≈15–30 is plenty)
 * @param convergence_eps     stop early if every shift < this (world units)
 * @param max_restarts        number of times we may restart on hard failure
 *
 * @return std::vector<b2Vec2>  the *k* centre positions
 *
 * @throws std::runtime_error  if the domain is invalid or the algorithm
 *                             fails after `max_restarts` attempts.
 */
std::vector<b2Vec2> generate_random_points_power_lloyd(
        const std::vector<std::vector<b2Vec2>> &polygons,
        const std::vector<float>               &reserve_radii,
        std::size_t        n_samples          = 25'000,
        std::size_t        kmeans_iterations  = 25,
        float              convergence_eps    = 1e-3f,
        std::uint32_t      max_restarts       = 3);


/**
 * Priority-wall layered sampler.
 *
 * Stage 0  : fill the boundary (outer wall or hole walls) greedily;
 * Stage 1+ : build concentric layers around the already-accepted points.
 *
 * A point i always keeps `reserve_radii[i]` clearance from walls *and*
 * from every other point.  NaN radii → return {NaN,NaN}.
 *
 * Throws std::runtime_error after `max_restarts` failed global attempts.
 */
std::vector<b2Vec2> generate_random_points_layered(
        const std::vector<std::vector<b2Vec2>> &polygons,
        const std::vector<float> &reserve_radii,
        std::uint32_t attempts_per_point = 1'000U,
        std::uint32_t max_restarts       = 25U);


/**
 * @brief Generate a square-grid ("checkerboard") layout with **exactly
 *        @p n_points** nodes inside a (possibly holed) polygonal arena.
 *
 * The grid is axis-aligned with spacing approximately @p pitch. The function
 * will try multiple strategies to achieve exactly the requested number of points:
 *   1. Try exact grid with the given pitch
 *   2. Try grids with slightly adjusted pitch values
 *   3. Use grid points as base and add/remove points strategically
 *   4. For small numbers, use optimized placement
 *
 * @param[in] polygons      0 = outer boundary (≥ 3 vertices), 1…N = holes
 * @param[in] n_points      Exact number of nodes requested (≥ 1)
 * @param[in] pitch         Target grid spacing in metres (> 0)
 * @param[in] cluster_center If true, creates compact formation centered in arena
 *                          without holes. If false, uses distributed placement.
 *
 * @return std::vector<b2Vec2>  Exactly @p n_points valid nodes
 *
 * @throws std::runtime_error
 *         • if @p pitch ≤ 0 or @p n_points == 0  
 *         • if the outer polygon is missing / degenerate  
 *         • if it is impossible to place @p n_points nodes in the arena
 *
 * @note Requires `is_point_within_polygon(poly,x,y)` treating boundary points
 *       as inside.  Uses Box2D's `b2Vec2` for coordinates.
 */
std::vector<b2Vec2> generate_chessboard_points(
        const std::vector<std::vector<b2Vec2>> &polygons,
        std::size_t                             n_points,
        float                                   pitch,
        bool                                    cluster_center = false);

/**
 * @brief Computes the width and height of a polygon.
 *
 * This function calculates the dimensions of a polygon by determining the minimum
 * and maximum x and y coordinates of its vertices. The width is computed as the
 * difference between the maximum and minimum x values, and the height is the difference
 * between the maximum and minimum y values.
 *
 * @param polygon A vector of b2Vec2 points representing the vertices of the polygon.
 * @return std::pair<float, float> A pair where the first element is the width and the second element is the height of the polygon.
 */
std::pair<float, float> compute_polygon_dimensions(const std::vector<b2Vec2>& polygon);

/**
 * @brief Computes the area of a polygon using the shoelace formula.
 *
 * This function calculates the area of a polygon defined by a sequence of
 * points (b2Vec2) using the shoelace algorithm. The formula sums the cross
 * products of consecutive vertices and returns half of the absolute value.
 *
 * @param poly A vector of b2Vec2 points representing the vertices of the polygon.
 * @return float The computed area of the polygon.
 */
float compute_polygon_area(const std::vector<b2Vec2>& poly);

/**
 * @brief Computes the centroid of a polygon.
 *
 * This function calculates the geometric center of the polygon using the shoelace formula.
 *
 * @param polygon A vector of b2Vec2 points defining the polygon.
 * @return b2Vec2 The centroid of the polygon.
 */
b2Vec2 polygon_centroid(const std::vector<b2Vec2>& polygon);

/**
 * @brief Calculates the distance from a point to a line segment.
 *
 * This function computes the shortest distance from point p to the line segment defined by endpoints a and b.
 *
 * @param p The point from which the distance is measured.
 * @param a The first endpoint of the line segment.
 * @param b The second endpoint of the line segment.
 * @return float The distance from point p to the line segment.
 */
float point_to_line_segment_distance(const b2Vec2& p, const b2Vec2& a, const b2Vec2& b);

/**
 * @brief  Place points on (approximate) concentric rings inside polygons[0]
 *         so that:
 *           • point i stays ≥ reserve_radii[i] from every polygon edge,
 *           • point i stays ≥ reserve_radii[i] + reserve_radii[j]
 *             from every previously accepted point j,
 *           • no point falls inside a hole (polygons[1…]).
 *
 * @param  polygons        polygons[0] is the main area; polygons[1…] are holes.
 * @param  reserve_radii   exclusion radii for each requested point (size N).
 * @return vector<b2Vec2>  the generated points (size == reserve_radii.size()).
 */
std::vector<b2Vec2> generate_regular_disk_points_in_polygon( const std::vector<std::vector<b2Vec2>>& polygons, const std::vector<float>& reserve_radii);


/**
 * @brief Import and rescale robot formation points.
 *
 * This routine loads a formation description from @p formation_filename,
 * rescales every (x, y) so that the entire formation fits inside the bounding
 * box of @p scaled_arena_polygons, and preserves θ exactly as provided.  The
 * caller must supply the minimum and maximum (x, y) that were present in the
 * file (these can be pre‑computed with a simple pass over the input).
 *
 * @param scaled_arena_polygons  Destination geometry (already scaled).
 * @param nb_objects             Number of objects.
 * @param formation_filename     Path to a *.csv* or *.feather* file with three
 *                               numeric columns: *x*, *y*, *theta*.
 * @param imported_formation_min_coords  Minimum (x, y) encountered in the
 *                                       source file.
 * @param imported_formation_max_coords  Maximum (x, y) encountered in the
 *                                       source file.
 *
 * @return std::tuple containing:
 *   - **std::vector<b2Vec2>** – positions mapped to arena space.
 *   - **std::vector<float>**  – corresponding headings (in radians).
 *
 * @throws std::runtime_error if the file cannot be read or the extension is
 *                            unsupported.
 *
 * @note No attempt is made to normalise θ; it is copied verbatim.
 */
std::tuple<std::vector<b2Vec2>, std::vector<float>>
import_points_from_file(const arena_polygons_t& scaled_arena_polygons,
                        const size_t nb_objects,
                        const std::string&      formation_filename,
                        const std::pair<float, float>& imported_formation_min_coords,
                        const std::pair<float, float>& imported_formation_max_coords);


#endif // GEOMETRY_H

// MODELINE "{{{1
// vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
// vim:foldmethod=marker
