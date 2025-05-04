#include "distances.h"

#include <box2d/box2d.h>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <set>
#include <iostream>


float euclidean_distance(const b2Vec2& a, const b2Vec2& b) {
    float const dx = a.x - b.x;
    float const dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

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
constexpr std::array<GridCell, 9> precomputedNeighborCells{
    GridCell{-1, -1}, GridCell{-1,  0}, GridCell{-1,  1},
    GridCell{ 0, -1}, GridCell{ 0,  0}, GridCell{ 0,  1},
    GridCell{ 1, -1}, GridCell{ 1,  0}, GridCell{ 1,  1},
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
inline GridCell getGridCell(float x, float y, float cellSize) {
    return {static_cast<int>(std::floor(x / cellSize)),
            static_cast<int>(std::floor(y / cellSize))};
}


//inline float length_squared(const b2Vec2& v) {
//    return b2Dot(v, v);
//}
//
//// XXX Take into account robot communication radius !!!
//bool has_line_of_sight(const b2Vec2& emitter_pos,
//                       const b2Vec2& target_pos,
//                       std::span<const b2Vec2> all_centres,
//                       float robot_radius) {
//    const b2Vec2 seg = target_pos - emitter_pos;
//    const float  seg_len_sq = length_squared(seg);
//    if (seg_len_sq == 0.0f) {          // identical points ⇒ undefined LOS
//        return false;
//    }
//
//    const float radius_sq = robot_radius * robot_radius;
//
//    for (const b2Vec2& c : all_centres) {
//        /* Skip the target itself and anything farther than the target
//           along the segment direction. */
//        const float t = b2Dot(c - emitter_pos, seg) / seg_len_sq;
//        if (t <= 0.0f || t >= 1.0f) {
//            continue;
//        }
//
//        const b2Vec2 proj = emitter_pos + t * seg;
//        if (length_squared(c - proj) < radius_sq) {
//            return false;              // segment hits another robot ⇒ blocked
//        }
//    }
//    return true;                       // no blocker found
//}

// XXX Take into account robot communication radius !!!
void find_neighbors(ir_direction dir, std::vector<std::shared_ptr<PogobotObject>>& robots, float maxDistance, bool detect_line_of_sight) {
    const float cellSize = maxDistance;
    const float maxDistSq = maxDistance * maxDistance;
    const size_t N = robots.size();
    const float radius = robots[0]->radius; // XXX assume all robots have the same radius

    // XXX
//    // Cache robot centres once – we need them for LOS
//    std::vector<b2Vec2> centres(N);
//    for (size_t i = 0; i < N; ++i) {
//        centres[i] = robots[i]->get_position();
//    }
//    std::span<const b2Vec2> centres_view{centres};

    // 1) Build separate x and y arrays (SoA layout).
    //    This avoids calling get_position() repeatedly and also helps cache locality.
    std::vector<float> xs(N), ys(N);
    for (size_t i = 0; i < N; i++) {
        b2Vec2 pos = robots[i]->get_IR_emitter_position(dir);
        xs[i] = pos.x;
        ys[i] = pos.y;
    }

    // 2) Build a spatial hash from GridCell -> list of robot indices.
    std::unordered_map<GridCell, std::vector<size_t>, GridCellHash> spatialHash;
    spatialHash.reserve(N); // optional, to reduce rehashes if you know approximate N

    for (size_t i = 0; i < N; i++) {
        GridCell cell = getGridCell(xs[i], ys[i], cellSize);
        spatialHash[cell].push_back(i);
    }

    // 3) For each robot, find neighbors in the same or adjacent cells.
    for (size_t i = 0; i < N; i++) {
        // Clear the old neighbor list
        robots[i]->neighbors[dir].clear();

        GridCell cell = getGridCell(xs[i], ys[i], cellSize);
        b2Vec2 emitter_pos{xs[i], ys[i]};

        for (const auto& offset : precomputedNeighborCells) {
            GridCell neighborCell{ cell.x + offset.x, cell.y + offset.y };
            auto it = spatialHash.find(neighborCell);
            if (it != spatialHash.end()) {
                // For each candidate robot in this neighbor cell
                for (size_t otherIdx : it->second) {
                    if (otherIdx == i) {
                        continue; // skip self
                    }
                    // 4) Compare squared distances to avoid sqrt.
                    float dx = xs[i] - xs[otherIdx];
                    float dy = ys[i] - ys[otherIdx];
                    float distSq = dx*dx + dy*dy;
                    if (distSq <= maxDistSq) {
                        // Robot i and otherIdx are neighbors
                        robots[i]->neighbors[dir].push_back(robots[otherIdx].get());
                        // XXX
                        //if (!detect_line_of_sight ||
                        //        has_line_of_sight(emitter_pos,
                        //            centres[otherIdx],
                        //            centres_view,
                        //            radius)) {
                        //    // Robot i and otherIdx are neighbors
                        //    robots[i]->neighbors[dir].push_back(robots[otherIdx].get());
                        //}
                    }
                } // for
            } // if
        } // for
    } // for
}


namespace {

/* squared length of a vector */
[[nodiscard]] inline float len_sq(b2Vec2 v) noexcept { return v.x * v.x + v.y * v.y; }

/**
 * @brief Shortest distance P — segment AB.
 *
 * Identical math to what we used in ArenaGeometry; kept here in an unnamed
 * namespace so the compiler can inline it.
 */
[[nodiscard]] float distance_point_segment(b2Vec2 p, b2Vec2 a, b2Vec2 b) noexcept {
    const b2Vec2 ab{b.x - a.x, b.y - a.y};
    const float  ab_len2 = len_sq(ab);
    if (ab_len2 == 0.0f) {          // degenerate edge
        return std::sqrt(len_sq({p.x - a.x, p.y - a.y}));
    }

    const b2Vec2 ap{p.x - a.x, p.y - a.y};
    float t = (ap.x * ab.x + ap.y * ab.y) / ab_len2;   // scalar projection
    t = std::clamp(t, 0.0f, 1.0f);

    const b2Vec2 closest{a.x + t * ab.x, a.y + t * ab.y};
    return std::sqrt(len_sq({p.x - closest.x, p.y - closest.y}));
}

} // namespace

std::vector<float>
compute_wall_distances(ir_direction                           dir,
                       const std::vector<std::shared_ptr<PogobotObject>>& robots,
                       const arena_polygons_t&                arena_polygons) {
    const std::size_t N = robots.size();
    std::vector<float> distances;
    distances.reserve(N);

    /* 1)  Structure‑of‑arrays cache‑friendly layout for robot positions          */
    std::vector<float> xs(N), ys(N);
    for (std::size_t i = 0; i < N; ++i) {
        const b2Vec2 pos = robots[i]->get_IR_emitter_position(dir);
        xs[i] = pos.x;
        ys[i] = pos.y;
    }

    /* 2)  Walk every robot × every arena edge  (usually still very small)        */
    for (std::size_t i = 0; i < N; ++i) {
        const b2Vec2 p{xs[i] * VISUALIZATION_SCALE, ys[i] * VISUALIZATION_SCALE};
        float best = std::numeric_limits<float>::infinity();

        for (const auto& poly : arena_polygons) {
            const std::size_t m = poly.size();
            if (m < 2) continue;

            for (std::size_t k = 0, j = m - 1; k < m; j = k++) {
                best = std::min(best, distance_point_segment(p, poly[j], poly[k]));
                //glogger->info("compute_wall_distances p=({},{}) poly[i]=({},{}), poly[k]=({},{})", p.x, p.y, poly[i].x, poly[i].y, poly[k].x, poly[k].y);
            }
        }
        distances.push_back(best);
    }
    return distances;
}

void find_neighbors_to_pogowalls(std::vector<std::shared_ptr<Pogowall>>& pogowalls, ir_direction dir, std::vector<std::shared_ptr<PogobotObject>>& robots) {
    size_t N = robots.size();

    for (auto wall : pogowalls) {
        arena_polygons_t contours = wall->generate_contours();

        auto dists = compute_wall_distances(dir, robots, contours);
        for (size_t i = 0; i < N; i++) {
            //glogger->info("Not close to wall {}: {} ({})", i, dists[i], robots[i]->communication_radius);
            if (dists[i] <= robots[i]->communication_radius) {
                //glogger->info("Close to wall {}: {} ({})", i, dists[i], robots[i]->communication_radius);
                robots[i]->neighbors[dir].push_back(wall.get());
            }
            if (dists[i] <= wall->communication_radius) {
                wall->neighbors[0].push_back(robots[i].get());
            }
        }
    }
}


// MODELINE "{{{1
// vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
// vim:foldmethod=marker
