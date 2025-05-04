#ifndef DISTANCES_H
#define DISTANCES_H

#include <box2d/box2d.h>
#include <unordered_map>
#include <vector>
#include <cmath>

#include "utils.h"
#include "spogobot.h"
#include "robot.h"

/**
 * @brief Computes the Euclidean distance between two 2D points.
 *
 * This function calculates the Euclidean distance between the points represented by
 * the Box2D vectors @a a and @a b.
 *
 * @param a The first point as a b2Vec2.
 * @param b The second point as a b2Vec2.
 * @return float The Euclidean distance between @a a and @a b.
 */
float euclidean_distance(const b2Vec2& a, const b2Vec2& b);

/**
 * @brief Represents a cell in a spatial grid.
 *
 * A GridCell is defined by its integer coordinates (x, y) and is used in spatial
 * hashing to partition a 2D space into discrete cells.
 */
struct GridCell {
    int x, y; ///< The x and y coordinates of the grid cell.

    /**
     * @brief Compares two GridCell objects for equality.
     *
     * @param other The other GridCell to compare against.
     * @return true if both the x and y coordinates are equal.
     * @return false otherwise.
     */
    bool operator==(const GridCell& other) const {
        return x == other.x && y == other.y;
    }
};

/**
 * @brief Hash functor for GridCell.
 *
 * This structure provides a hash function for GridCell objects, allowing them to be used
 * as keys in unordered associative containers.
 */
struct GridCellHash {
    /**
     * @brief Computes a hash value for a GridCell.
     *
     * Combines the hash of the x and y coordinates.
     *
     * @param cell The GridCell to hash.
     * @return std::size_t The computed hash value.
     */
    std::size_t operator()(const GridCell& cell) const {
        // Simple hash combining for x and y.
        return std::hash<int>()(cell.x) ^ (std::hash<int>()(cell.y) << 1);
    }
};

/**
 * @brief Precomputed offsets for neighbor grid cells.
 *
 * This constant array contains the relative offsets for a cell's neighbors in a 3x3 grid,
 * including the cell itself. It is used to quickly access adjacent cells during spatial queries.
 */
constexpr std::array<GridCell,9> precomputed_neighbor_cells{
    GridCell{-1,-1}, GridCell{-1,0}, GridCell{-1,1},
    GridCell{ 0,-1}, GridCell{ 0,0}, GridCell{ 0,1},
    GridCell{ 1,-1}, GridCell{ 1,0}, GridCell{ 1,1}
};

/**
 * @brief Converts a 2D position to a grid cell index.
 *
 * This inline function maps the provided (x, y) coordinates into a GridCell based on
 * the specified cell size. It uses std::floor to determine the appropriate cell index.
 *
 * @param x The x-coordinate of the position.
 * @param y The y-coordinate of the position.
 * @param cellSize The size of a single grid cell.
 * @return GridCell The corresponding grid cell for the given position.
 */
inline GridCell get_grid_cell(float x, float y, float cell_size) {
    return {static_cast<int>(std::floor(x / cell_size)),
            static_cast<int>(std::floor(y / cell_size))};
}




struct Candidate {
    std::size_t idx;   /* index inside robots[]                               */
    float       dist_sq;
    float       angle; /* atan2(dy,dx)                                        */
    float       half_ap; /* asin(r_body / dist)                               */
};

/* ------------------------------------------------------------------------ */
/* Angular-interval utilities (LOS test)                                    */
/* ------------------------------------------------------------------------ */
namespace angles {
    inline float wrap(float a) {
        while (a <= -M_PI) a += 2 * M_PI;
        while (a >   M_PI) a -= 2 * M_PI;
        return a;
    }

    struct Interval { float a, b; };  /* half-open, assume a < b in (-π, π]  */

    /* insert [a,b) into sorted union without overlaps                       */
    void add_interval(float a, float b, std::vector<Interval>& ivs);

    /* true iff [a,b) completely covered by ivs                              */
    bool fully_covered(float a, float b, const std::vector<Interval>& ivs);

    inline bool in_fov(float bearing,   /* atan2(dy,dx)                 */
            float center,    /* LED direction                 */
            float half_ap)   /* half-FOV (e.g. π/3)           */
    {
        return std::fabs(wrap(bearing - center)) <= half_ap;
    }
} // namespace angles

/* ------------------------------------------------------------------------ */
/* Building blocks for find_neighbors                                       */
/* ------------------------------------------------------------------------ */

/* Build a spatial hash of LED positions. */
std::unordered_map<GridCell,std::vector<std::size_t>,GridCellHash>
build_spatial_hash(span_t<float> xs,
                   span_t<float> ys,
                   float cell_size);

/* Collect robots that lie                                           *
 *   – in the 3×3 grid block around emitter i                        *
 *   – inside emitter i’s communication radius                       */
std::vector<Candidate>
collect_candidates(std::size_t                  i,
                   span_t<float>       xs,
                   span_t<float>       ys,
                   span_t<float>       cx,
                   span_t<float>       cy,
                   span_t<float>       body_rad,
                   span_t<float>       comm_rad,
                   span_t<float>       led_dir,
                   const std::unordered_map<GridCell,
                                            std::vector<std::size_t>,
                                            GridCellHash>& hash,
                   float cell_size,
                   bool  clip_fov);

/* Given distance-sorted candidates, keep only those not shadowed. */
std::vector<std::size_t>
filter_visible(const std::vector<Candidate>& cand);


/**
 * @brief Finds neighboring robots within a specified maximum distance.
 *
 * This function uses spatial hashing to efficiently determine the neighbors for each robot.
 * It partitions the 2D space into grid cells of size @a maxDistance and assigns each robot
 * to a cell. For every robot, the function then checks the same cell and all adjacent cells
 * for other robots, and updates the robot's neighbor list if the Euclidean distance (squared)
 * is within the allowed maximum distance.
 *
 * @param dir Direction in which messages are sent (i.e. the ID number of the IR emitter)
 * @param robots A vector of Robot objects. Each Robot must implement a method get_position()
 *               returning a b2Vec2, and contain a public member "neighbors" (e.g., a vector)
 *               that can store pointers to neighboring Robot objects.
 * @param maxDistance The maximum distance within which two robots are considered neighbors.
 * @param enable_occlusion  When true, use the angular-shadow-map LOS filter;
 *                          when false, accept every range-culled candidate.
 */
void find_neighbors(ir_direction dir, std::vector<std::shared_ptr<PogobotObject>>& robots,
        float maxDistance,
        bool  enable_occlusion = true);


/**
 * @brief Compute, for every robot, the distance from its IR–emitter @p dir to the
 *        nearest arena wall.
 *
 * The function returns a vector<double> whose i‑th element is the distance for
 * robots[i].  The caller decides what to do with the result (store it in the robot,
 * feed a sensor model, etc.).
 *
 * @param dir             Which IR emitter of the robot is queried.
 * @param robots          Robots for which the distance is required.
 * @param arena_polygons  Arena walls described as polygons (see arena_polygons_t).
 * @return std::vector<float> A vector of distances (same ordering as @p robots).
 */
std::vector<float>
compute_wall_distances(ir_direction                           dir,
                       const std::vector<std::shared_ptr<PogobotObject>>& robots,
                       const arena_polygons_t&                arena_polygons);


/**
 * @brief Finds robots that are close to given pogowalls.
 *
 * @param pogowalls A vector of Pogowall objects.
 * @param dir Direction in which messages are sent (i.e. the ID number of the IR emitter)
 * @param robots A vector of Robot objects.
 * @param max_distance The maximum distance within which a robot and a pogowalls are considered neighbors.
 */
void find_neighbors_to_pogowalls(std::vector<std::shared_ptr<Pogowall>>& pogowalls, ir_direction dir, std::vector<std::shared_ptr<PogobotObject>>& robots);

#endif // DISTANCES_H

// MODELINE "{{{1
// vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
// vim:foldmethod=marker
