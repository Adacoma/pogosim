#ifndef DISTANCES_H
#define DISTANCES_H

#include <box2d/box2d.h>
#include <unordered_map>
#include <vector>

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
 * @param detect_line_of_sight Whether to detect line of sight (i.e. simple way of handling occlusions).
 */
void find_neighbors(ir_direction dir, std::vector<std::shared_ptr<PogobotObject>>& robots,
        float maxDistance,
        bool detect_line_of_sight = true);

bool has_line_of_sight(const b2Vec2& emitter,
                       const b2Vec2& target,
                       std::span<const b2Vec2> centres,
                       float robot_radius);


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
