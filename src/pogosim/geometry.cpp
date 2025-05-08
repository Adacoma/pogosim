#include <SDL2/SDL.h>
#include <box2d/box2d.h>
#include <random>
#include <iostream>
#include <fstream>
#include <sstream>

#include <algorithm>
#include <arrow/api.h>
#include <arrow/io/file.h>
#include <arrow/ipc/feather.h>
#include <box2d/box2d.h>
#include <filesystem>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include "utils.h"
#include "distances.h"
#include "render.h"
#include "spogobot.h"
#include "fpng.h"


float mm_to_pixels = 1.0f;
float visualization_x = 0.0f;
float visualization_y = 0.0f;


void adjust_mm_to_pixels(float delta) {
    mm_to_pixels += delta;
    if (mm_to_pixels <= 0.10) {
        mm_to_pixels = 0.10;
    } else if (mm_to_pixels >= 10.0) {
        mm_to_pixels = 10.0;
    }
}

b2Vec2 visualization_position(float x, float y) {
    b2Vec2 res = {.x = (x + visualization_x) * mm_to_pixels, .y = (y + visualization_y) * mm_to_pixels};
    return res;
}

b2Vec2 visualization_position(b2Vec2 pos) {
    b2Vec2 res = {.x = (pos.x + visualization_x) * mm_to_pixels, .y = (pos.y + visualization_y) * mm_to_pixels};
    return res;
}


std::vector<std::vector<b2Vec2>> read_poly_from_csv(const std::string& filename, float total_surface) {
    std::vector<std::vector<b2Vec2>> polygons;
    std::ifstream file(filename);
    if (!file.is_open()) {
        glogger->error("Error: Unable to open file {}", filename);
        throw std::runtime_error("Unable to open arena file");
    }

    std::vector<b2Vec2> currentPolygon;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) {
            if (!currentPolygon.empty()) {
                polygons.push_back(currentPolygon);
                currentPolygon.clear();
            }
            continue;
        }
        std::istringstream ss(line);
        std::string xStr, yStr;
        if (std::getline(ss, xStr, ',') && std::getline(ss, yStr)) {
            float x = std::stof(xStr);
            float y = std::stof(yStr);
            currentPolygon.push_back({x, y});
        }
    }
    file.close();
    if (!currentPolygon.empty()) {
        polygons.push_back(currentPolygon);
    }

    // Compute the overall bounding box.
    float minX = std::numeric_limits<float>::max();
    float maxX = std::numeric_limits<float>::lowest();
    float minY = std::numeric_limits<float>::max();
    float maxY = std::numeric_limits<float>::lowest();
    for (const auto& poly : polygons) {
        for (const auto& point : poly) {
            minX = std::min(minX, point.x);
            maxX = std::max(maxX, point.x);
            minY = std::min(minY, point.y);
            maxY = std::max(maxY, point.y);
        }
    }

    // Normalize all polygons into a [0,1] range.
    std::vector<std::vector<b2Vec2>> normalized_polygons;
    for (const auto& poly : polygons) {
        std::vector<b2Vec2> normPoly;
        for (const auto& point : poly) {
            float normX = (point.x - minX) / (maxX - minX);
            float normY = (point.y - minY) / (maxY - minY);
            normPoly.push_back({normX, normY});
        }
        normalized_polygons.push_back(normPoly);
    }

    if (normalized_polygons.empty()) {
        throw std::runtime_error("No polygons loaded from file.");
    }

    // Compute effective area in normalized space:
    // effective_area = (area of main polygon) - (sum of areas of holes)
    float mainArea = compute_polygon_area(normalized_polygons[0]);
    float holesArea = 0.0f;
    for (size_t i = 1; i < normalized_polygons.size(); i++) {
        holesArea += compute_polygon_area(normalized_polygons[i]);
    }
    float effectiveArea = mainArea - holesArea;
    if (effectiveArea <= 0) {
        throw std::runtime_error("Effective area of polygons is non-positive.");
    }

    // Determine the scale factor s so that:
    // (normalized effective area) * s^2 = total_surface
    float scale = std::sqrt(total_surface / effectiveArea);

    // Apply uniform scaling to all normalized polygons.
    std::vector<std::vector<b2Vec2>> scaled_polygons;
    for (const auto& poly : normalized_polygons) {
        std::vector<b2Vec2> scaledPoly;
        for (const auto& point : poly) {
            scaledPoly.push_back({point.x * scale, point.y * scale});
        }
        scaled_polygons.push_back(scaledPoly);
    }

    return scaled_polygons;
}


b2Vec2 generate_random_point_within_polygon(const std::vector<b2Vec2>& polygon) {
    if (polygon.size() < 3) {
        throw std::runtime_error("Polygon must have at least 3 points to define a valid area.");
    }

    // Calculate the bounding box of the polygon
    float minX = std::numeric_limits<float>::max();
    float maxX = std::numeric_limits<float>::lowest();
    float minY = std::numeric_limits<float>::max();
    float maxY = std::numeric_limits<float>::lowest();

    for (const auto& point : polygon) {
        minX = std::min(minX, point.x);
        maxX = std::max(maxX, point.x);
        minY = std::min(minY, point.y);
        maxY = std::max(maxY, point.y);
    }

    // Random number generator
    std::uniform_real_distribution<float> disX(minX, maxX);
    std::uniform_real_distribution<float> disY(minY, maxY);

    // Generate random points until one is inside the polygon
    while (true) {
        float x = disX(rnd_gen);
        float y = disY(rnd_gen);

        if (is_point_within_polygon(polygon, x, y)) {
            return b2Vec2{x, y};
        }
    }
}

bool is_point_within_polygon(const std::vector<b2Vec2>& polygon, float x, float y) {
    int n = polygon.size();
    if (n < 3) {
        std::cerr << "Error: Polygon must have at least 3 points." << std::endl;
        return false;
    }

    bool isInside = false;
    for (int i = 0, j = n - 1; i < n; j = i++) {
        float xi = polygon[i].x, yi = polygon[i].y;
        float xj = polygon[j].x, yj = polygon[j].y;

        // Check if point is within edge bounds
        bool intersect = ((yi > y) != (yj > y)) &&
                         (x < (xj - xi) * (y - yi) / (yj - yi) + xi);

        if (intersect) {
            isInside = !isInside;
        }
    }
    return isInside;
}

std::vector<b2Vec2> offset_polygon(const std::vector<b2Vec2>& polygon, float offset) {
    std::vector<b2Vec2> offsetPolygon;

    int n = polygon.size();
    if (n < 3) {
        throw std::runtime_error("Polygon must have at least 3 points to offset.");
    }

    for (int i = 0; i < n; ++i) {
        // Get the previous, current, and next points
        b2Vec2 prev = polygon[(i - 1 + n) % n];
        b2Vec2 curr = polygon[i];
        b2Vec2 next = polygon[(i + 1) % n];

        // Calculate vectors for the current edge and the previous edge
        b2Vec2 edge1 = curr - prev;
        b2Vec2 edge2 = next - curr;

        // Normalize and find perpendiculars
        b2Vec2 norm1 = b2Vec2{-edge1.y, edge1.x};
        b2Vec2 norm2 = b2Vec2{-edge2.y, edge2.x};

        norm1 *= (offset / b2Distance(b2Vec2{0, 0}, edge1));
        norm2 *= (offset / b2Distance(b2Vec2{0, 0}, edge2));

        // Compute the inward offset point using normals
        b2Vec2 offsetPoint = curr + 0.5f * (norm1 + norm2);
        offsetPolygon.push_back(offsetPoint);
    }

    return offsetPolygon;
}


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
 * If it fails to build the whole set after `attempts_per_point` rejected
 * candidates, the algorithm discards all progress and restarts.  It will
 * attempt the whole sampling process up to `max_restarts` times before
 * throwing.
 */
std::vector<b2Vec2> generate_random_points_within_polygon_safe(
        const std::vector<std::vector<b2Vec2>> &polygons,
        const std::vector<float> &reserve_radii,
        float max_neighbor_distance,
        std::uint32_t attempts_per_point,
        std::uint32_t max_restarts) {
    // ─── basic sanity checks ──────────────────────────────────────────────
    if (polygons.empty()) {
        throw std::runtime_error("At least one polygon must be supplied.");
    }
    for (const auto &p : polygons) {
        if (p.size() < 3) {
            throw std::runtime_error("Every polygon needs ≥ 3 vertices.");
        }
    }

    const std::size_t n_points = reserve_radii.size();
    if (n_points == 0U) { return {}; }

    // ─── conservative bounding‑box (contracted by largest radius) ─────────
    const float bb_margin = *std::max_element(reserve_radii.begin(), reserve_radii.end());

    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float max_y = std::numeric_limits<float>::lowest();

    const auto &outer_poly = polygons.front();
    for (const auto &v : outer_poly) {
        min_x = std::min(min_x, v.x);
        min_y = std::min(min_y, v.y);
        max_x = std::max(max_x, v.x);
        max_y = std::max(max_y, v.y);
    }
    min_x += bb_margin;  min_y += bb_margin;
    max_x -= bb_margin;  max_y -= bb_margin;

    if (min_x >= max_x || min_y >= max_y) {
        throw std::runtime_error("Reserve radii are too large for the given polygon.");
    }

    // ─── random‑number engine ─────────────────────────────────────────────
    std::uniform_real_distribution<float> dis_x(min_x, max_x);
    std::uniform_real_distribution<float> dis_y(min_y, max_y);

    // ─── outer restart loop ───────────────────────────────────────────────
    for (std::uint32_t restart = 0U; restart < max_restarts; ++restart) {
        std::vector<b2Vec2> points; points.reserve(n_points);
        std::uint32_t attempts = 0U;

        // ─── rejection‑sampling loop ──────────────────────────────────────
        while (points.size() < n_points) {
            const float x = dis_x(rnd_gen);
            const float y = dis_y(rnd_gen);

            const float r_curr = reserve_radii[points.size()];

            // 1️⃣ inside outer polygon?
            if (!is_point_within_polygon(outer_poly, x, y)) { continue; }

            // 2️⃣ outside every hole polygon?
            bool ok = true;
            for (std::size_t i = 1; i < polygons.size() && ok; ++i) {
                if (is_point_within_polygon(polygons[i], x, y)) { ok = false; }
            }

            // 3️⃣ exclusion radius + connectivity checks
            if (ok && !points.empty()) {
                float min_dist = std::numeric_limits<float>::infinity();
                for (std::size_t i = 0; i < points.size(); ++i) {
                    const float min_sep = reserve_radii[i] + r_curr;
                    const float d = euclidean_distance(points[i], {x, y});
                    if (d < min_sep) { ok = false; break; } // too close
                    min_dist = std::min(min_dist, d);
                }
                if (ok && min_dist > max_neighbor_distance) { ok = false; }
            }

            // 4️⃣ accept or reject
            if (ok) {
                points.push_back({x, y});
                attempts = 0U;                  // reset attempt counter
            } else if (++attempts >= attempts_per_point) {
                // Give up on this run and start over.
                break; // triggers outer restart loop
            }
        }

        if (points.size() == n_points) {
            return points; // success
        }
    }

    // If we fall through the loop, all restarts failed.
    throw std::runtime_error("Impossible to create random points within polygon: too many points or radii too large, even after multiple restarts.");
}

inline float sqr(float v) { return v * v; }

float euclidean_distance_sq(const b2Vec2 &a, const b2Vec2 &b) {
    return sqr(a.x - b.x) + sqr(a.y - b.y);
}

std::vector<b2Vec2> generate_points_voronoi_lloyd(const std::vector<std::vector<b2Vec2>> &polygons,
                                                  std::size_t k,
                                                  std::size_t n_samples,
                                                  std::size_t kmeans_iterations,
                                                  std::uint32_t max_restarts) {
    // ─── sanity checks ───────────────────────────────────────────────────
    if (polygons.empty()) {
        throw std::runtime_error("At least one polygon must be supplied.");
    }
    if (k == 0U) { return {}; }

    for (const auto &p : polygons) {
        if (p.size() < 3) {
            throw std::runtime_error("Every polygon needs ≥ 3 vertices.");
        }
    }

    // ─── compute bounding box of outer polygon ───────────────────────────
    const auto &outer = polygons.front();
    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float max_y = std::numeric_limits<float>::lowest();
    for (const auto &v : outer) {
        min_x = std::min(min_x, v.x);  min_y = std::min(min_y, v.y);
        max_x = std::max(max_x, v.x);  max_y = std::max(max_y, v.y);
    }
    std::uniform_real_distribution<float> dis_x(min_x, max_x);
    std::uniform_real_distribution<float> dis_y(min_y, max_y);

    // ─── outer restart loop (rarely needed) ──────────────────────────────
    for (std::uint32_t restart = 0; restart < max_restarts; ++restart) {

        // 1️⃣  Dense uniform sampling inside the domain
        std::vector<b2Vec2> samples;  samples.reserve(n_samples);
        while (samples.size() < n_samples) {
            const float x = dis_x(rnd_gen);
            const float y = dis_y(rnd_gen);
            if (!is_point_within_polygon(outer, x, y)) { continue; }

            bool in_hole = false;
            for (std::size_t h = 1; h < polygons.size() && !in_hole; ++h) {
                if (is_point_within_polygon(polygons[h], x, y)) { in_hole = true; }
            }
            if (!in_hole) { samples.push_back({x, y}); }
        }

        // 2️⃣  Random-init cluster centres (K-means++ gives nicer convergence,
        //     but plain random is simpler and good enough here)
        if (k > samples.size()) { throw std::runtime_error("Too few domain samples."); }

        std::vector<std::size_t> indices(samples.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rnd_gen);

        std::vector<b2Vec2> centres;
        centres.reserve(k);
        for (std::size_t i = 0; i < k; ++i) { centres.push_back(samples[indices[i]]); }

        // 3️⃣  Lloyd relaxations
        std::vector<std::size_t> membership(samples.size());
        const float converge_eps_sq = sqr(1e-3f);   // stop if every move < 1 mm
        bool converged = false;

        for (std::size_t it = 0; it < kmeans_iterations && !converged; ++it) {
            // Assignment step --------------------------------------------------
            for (std::size_t s = 0; s < samples.size(); ++s) {
                float best_d = std::numeric_limits<float>::max();
                std::size_t best_c = 0;
                for (std::size_t c = 0; c < k; ++c) {
                    float d = euclidean_distance_sq(samples[s], centres[c]);
                    if (d < best_d) { best_d = d; best_c = c; }
                }
                membership[s] = best_c;
            }

            // Update step ------------------------------------------------------
            std::vector<b2Vec2> new_centres(k, {0.f, 0.f});
            std::vector<std::size_t> counts(k, 0);
            for (std::size_t s = 0; s < samples.size(); ++s) {
                const auto &pt = samples[s];
                std::size_t c  = membership[s];
                new_centres[c].x += pt.x;
                new_centres[c].y += pt.y;
                ++counts[c];
            }
            // Handle empty clusters by re-seeding them with a random sample
            for (std::size_t c = 0; c < k; ++c) {
                if (counts[c] == 0) {
                    std::uniform_int_distribution<std::size_t> dis(0, samples.size() - 1);
                    new_centres[c] = samples[dis(rnd_gen)];
                    counts[c] = 1;
                } else {
                    new_centres[c].x /= static_cast<float>(counts[c]);
                    new_centres[c].y /= static_cast<float>(counts[c]);
                    // If centroid left the domain (possible for highly concave
                    // shapes), snap to nearest in-domain sample of the cluster.
                    if (!is_point_within_polygon(outer, new_centres[c].x, new_centres[c].y)) {
                        float best_d = std::numeric_limits<float>::max();
                        std::size_t best_s = 0;
                        for (std::size_t s = 0; s < samples.size(); ++s) {
                            if (membership[s] != c) { continue; }
                            float d = euclidean_distance_sq(samples[s], new_centres[c]);
                            if (d < best_d) { best_d = d; best_s = s; }
                        }
                        new_centres[c] = samples[best_s];
                    }
                }
            }

            // Convergence test -------------------------------------------------
            converged = true;
            for (std::size_t c = 0; c < k; ++c) {
                if (euclidean_distance_sq(centres[c], new_centres[c]) > converge_eps_sq) {
                    converged = false; break;
                }
            }
            centres.swap(new_centres);
        }

        // 4️⃣  All good → return
        return centres;
    }

    throw std::runtime_error("Failed to create centroidal Voronoi points after several restarts.");
}


std::vector<b2Vec2> generate_random_points_power_lloyd(
        const std::vector<std::vector<b2Vec2>> &polygons,
        const std::vector<float>               &reserve_radii,
        std::size_t        n_samples,
        std::size_t        kmeans_iterations,
        float              convergence_eps,
        std::uint32_t      max_restarts) {
    const std::size_t k = reserve_radii.size();
    if (k == 0U) { return {}; }

    // ─── basic input sanity checks ───────────────────────────────────────
    if (polygons.empty()) {
        throw std::runtime_error("At least one polygon must be supplied.");
    }
    for (const auto &poly : polygons) {
        if (poly.size() < 3) {
            throw std::runtime_error("Every polygon needs ≥ 3 vertices.");
        }
    }

    // ─── axis-aligned bounding box of the *outer* polygon ───────────────
    const auto &outer = polygons.front();
    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float max_y = std::numeric_limits<float>::lowest();

    for (const auto &v : outer) {
        min_x = std::min(min_x, v.x);  min_y = std::min(min_y, v.y);
        max_x = std::max(max_x, v.x);  max_y = std::max(max_y, v.y);
    }
    if (min_x >= max_x || min_y >= max_y) {
        throw std::runtime_error("Degenerate outer polygon.");
    }

    std::uniform_real_distribution<float> dis_x(min_x, max_x);
    std::uniform_real_distribution<float> dis_y(min_y, max_y);

    // ─── outer restart loop (rarely taken) ───────────────────────────────
    for (std::uint32_t restart = 0U; restart < max_restarts; ++restart) {

        // 1️⃣  Dense uniform sampling inside the domain -------------------
        std::vector<b2Vec2> samples;
        samples.reserve(n_samples);
        while (samples.size() < n_samples) {
            const float x = dis_x(rnd_gen);
            const float y = dis_y(rnd_gen);
            if (!is_point_within_polygon(outer, x, y)) { continue; }

            bool inside_hole = false;
            for (std::size_t h = 1; h < polygons.size() && !inside_hole; ++h) {
                if (is_point_within_polygon(polygons[h], x, y)) { inside_hole = true; }
            }
            if (!inside_hole) { samples.push_back({x, y}); }
        }

        // 2️⃣  Random-initialise the centres ------------------------------
        if (k > samples.size()) {
            throw std::runtime_error("Not enough domain samples for the requested k.");
        }
        std::vector<std::size_t> perm(samples.size());
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rnd_gen);

        std::vector<b2Vec2> centres(k);
        for (std::size_t i = 0; i < k; ++i) { centres[i] = samples[perm[i]]; }

        // radii must follow the centre order (copy once so we may shuffle)
        std::vector<float> radii = reserve_radii;

        // 3️⃣  Lloyd iterations in POWER space ----------------------------
        std::vector<std::size_t> membership(samples.size());
        const float eps_sq = sqr(convergence_eps);
        bool converged = false;

        for (std::size_t it = 0; it < kmeans_iterations && !converged; ++it) {

            // — assignment step (power distance) —
            for (std::size_t s = 0; s < samples.size(); ++s) {
                float best_val = std::numeric_limits<float>::max();
                std::size_t best_c = 0;
                const auto &p = samples[s];
                for (std::size_t c = 0; c < k; ++c) {
                    float val = euclidean_distance_sq(p, centres[c]) - sqr(radii[c]);
                    if (val < best_val) { best_val = val; best_c = c; }
                }
                membership[s] = best_c;
            }

            // — update step: arithmetic mean of each cluster —
            std::vector<b2Vec2> new_centres(k, {0.f, 0.f});
            std::vector<std::size_t> counts(k, 0);

            for (std::size_t s = 0; s < samples.size(); ++s) {
                std::size_t c = membership[s];
                new_centres[c].x += samples[s].x;
                new_centres[c].y += samples[s].y;
                ++counts[c];
            }
            // handle empty clusters by reseeding them
            std::uniform_int_distribution<std::size_t> dis_sample(0, samples.size() - 1);
            for (std::size_t c = 0; c < k; ++c) {
                if (counts[c] == 0) {
                    new_centres[c] = samples[dis_sample(rnd_gen)];
                    counts[c] = 1;
                } else {
                    new_centres[c].x /= static_cast<float>(counts[c]);
                    new_centres[c].y /= static_cast<float>(counts[c]);
                }

                // snap back into domain if centroid drifted outside
                if (!is_point_within_polygon(outer, new_centres[c].x, new_centres[c].y)) {
                    float best_d = std::numeric_limits<float>::max();
                    std::size_t best_s = 0;
                    for (std::size_t s = 0; s < samples.size(); ++s) {
                        if (membership[s] != c) { continue; }
                        float d = euclidean_distance_sq(samples[s], new_centres[c]);
                        if (d < best_d) { best_d = d; best_s = s; }
                    }
                    new_centres[c] = samples[best_s]; // guaranteed to exist
                }
            }

            // — convergence test —
            converged = true;
            for (std::size_t c = 0; c < k && converged; ++c) {
                if (euclidean_distance_sq(centres[c], new_centres[c]) > eps_sq) {
                    converged = false;
                }
            }
            centres.swap(new_centres);
        }

        // 4️⃣  quick push-apart pass to eliminate residual overlaps --------
        constexpr std::size_t overlap_relax_iters = 4;
        for (std::size_t pass = 0; pass < overlap_relax_iters; ++pass) {
            bool overlap_found = false;
            for (std::size_t i = 0; i < k; ++i) {
                for (std::size_t j = i + 1; j < k; ++j) {
                    const float min_sep = radii[i] + radii[j];
                    b2Vec2 d = centres[j] - centres[i];
                    const float dist_sq = sqr(d.x) + sqr(d.y);
                    if (dist_sq < sqr(min_sep) && dist_sq > 1e-12f) {
                        overlap_found = true;
                        const float dist = std::sqrt(dist_sq);
                        const float push = 0.5f * (min_sep - dist);
                        d *= (push / dist);  // normalised * push
                        centres[i] -= d;
                        centres[j] += d;
                    }
                }
            }
            if (!overlap_found) { break; }
        }

        return centres;                 // success on this restart
    }

    // — all restarts failed —
    throw std::runtime_error("Power-Lloyd sampler failed after multiple restarts; "
                             "check radii or decrease k.");
}


std::pair<float, float> compute_polygon_dimensions(const std::vector<b2Vec2>& polygon) {
    float minX = std::numeric_limits<float>::max();
    float maxX = std::numeric_limits<float>::lowest();
    float minY = std::numeric_limits<float>::max();
    float maxY = std::numeric_limits<float>::lowest();

    for (const auto& point : polygon) {
        minX = std::min(minX, point.x);
        maxX = std::max(maxX, point.x);
        minY = std::min(minY, point.y);
        maxY = std::max(maxY, point.y);
    }

    float width = maxX - minX;
    float height = maxY - minY;
    return std::make_pair(width, height);
}


float compute_polygon_area(const std::vector<b2Vec2>& poly) {
    float area = 0.0f;
    int n = poly.size();
    for (int i = 0; i < n; i++) {
        const b2Vec2& p1 = poly[i];
        const b2Vec2& p2 = poly[(i + 1) % n];
        area += p1.x * p2.y - p2.x * p1.y;
    }
    return 0.5f * std::abs(area);
}


// Simple polygon centroid function (using the "shoelace" formula):
b2Vec2 polygon_centroid(const std::vector<b2Vec2>& polygon) {
    // Expecting polygon to be non-empty and closed (first == last) is optional.
    double signedArea = 0.0;
    double cx = 0.0, cy = 0.0;
    for (size_t i = 0; i < polygon.size(); i++) {
        const b2Vec2& p0 = polygon[i];
        const b2Vec2& p1 = polygon[(i+1) % polygon.size()];
        double cross = (p0.x * p1.y) - (p1.x * p0.y);
        signedArea += cross;
        cx += (p0.x + p1.x) * cross;
        cy += (p0.y + p1.y) * cross;
    }
    signedArea *= 0.5;
    cx /= (6.0 * signedArea);
    cy /= (6.0 * signedArea);
    return b2Vec2{(float)cx, (float)cy};
}

// Distance from point to line segment (used for computing min-dist from centroid to edges):
float point_to_line_segment_distance(const b2Vec2& p,
                                     const b2Vec2& a,
                                     const b2Vec2& b) {
    // Vector AP and AB
    float vx = p.x - a.x;
    float vy = p.y - a.y;
    float ux = b.x - a.x;
    float uy = b.y - a.y;
    // Compute the dot product AP·AB
    float dot = vx * ux + vy * uy;
    // Compute squared length of AB
    float len2 = ux * ux + uy * uy;
    // Parameter t along AB to project point P
    float t = (len2 == 0.0f ? 0.0f : dot / len2);
    // Clamp t to [0,1] so we stay in segment
    t = std::max(0.0f, std::min(1.0f, t));
    // Projection point on AB
    float projx = a.x + t * ux;
    float projy = a.y + t * uy;
    // Distance from P to projection
    float dx = p.x - projx;
    float dy = p.y - projy;
    return std::sqrt(dx*dx + dy*dy);
}


std::vector<b2Vec2> generate_regular_disk_points_in_polygon(
        const std::vector<std::vector<b2Vec2>>& polygons,
        const std::vector<float>& reserve_radii) {
    /* ---------- 1. sanity checks ---------------------------------------- */
    if (polygons.empty()) {
        throw std::runtime_error("No polygons provided.");
    }
    const auto& main_polygon = polygons.front();
    if (main_polygon.size() < 3) {
        throw std::runtime_error("Polygon must have at least 3 vertices.");
    }

    const std::size_t n_points = reserve_radii.size();
    if (n_points == 0U) { return {}; }

    const float r_max = *std::max_element(reserve_radii.begin(), reserve_radii.end());
    const float r_min = *std::min_element(reserve_radii.begin(), reserve_radii.end());

    /* ---------- 2. centroid & “inscribed circle” radius ----------------- */
    b2Vec2 center = polygon_centroid(main_polygon);

    float max_edge_dist = std::numeric_limits<float>::lowest();
    for (std::size_t i = 0; i < main_polygon.size(); ++i) {
        float d = point_to_line_segment_distance(center,
                                                 main_polygon[i],
                                                 main_polygon[(i + 1) % main_polygon.size()]);
        max_edge_dist = std::max(max_edge_dist, d);
    }
    const float allowed_radius = max_edge_dist - r_max;  // keep every circle inside
    if (allowed_radius <= 0.0f) {
        throw std::runtime_error("Polygon too small or some reserve radius too large.");
    }

    /* ---------- 3. helper: does a candidate fit? ------------------------ */
    const auto fits = [&](const b2Vec2& c, float r_curr,
                          const std::vector<b2Vec2>& accepted) -> bool
    {
        /* outside holes & inside outer polygon */
        if (!is_point_within_polygon(main_polygon, c.x, c.y)) return false;
        for (std::size_t h = 1; h < polygons.size(); ++h)
            if (is_point_within_polygon(polygons[h], c.x, c.y)) return false;

        /* far enough from outer edges (quick test using inscribed circle) */
        if (euclidean_distance(center, c) + r_curr > allowed_radius + 1e-5f) return false;

        /* far enough from previously accepted points */
        for (std::size_t i = 0; i < accepted.size(); ++i)
            if (euclidean_distance(accepted[i], c) < reserve_radii[i] + r_curr) return false;

        return true;
    };

    /* ---------- 4. place points ---------------------------------------- */
    std::vector<b2Vec2> result;  result.reserve(n_points);

    /* try the centroid for the first point -------------------------------- */
    if (fits(center, reserve_radii[0], result)) {
        result.push_back(center);
    }

    /* concentric‑ring search --------------------------------------------- */
    float ring_radius = 0.0f;
    const float radial_step = r_min;           // move outwards by the *smallest* radius

    while (result.size() < n_points && ring_radius <= allowed_radius) {
        ring_radius += radial_step;
        if (ring_radius > allowed_radius) break;

        /* angular step so that neighbouring *small* points are ~2 r_min apart */
        const float arc_len = 2.0f * r_min;
        const float d_theta = (ring_radius < 1e-6f) ? 2.0f * float(M_PI)
                                                   : arc_len / ring_radius;

        for (float theta = 0.0f; theta < 2.0f * float(M_PI) && result.size() < n_points;
             theta += d_theta)
        {
            const std::size_t idx   = result.size();        // next point’s index
            const float       r_cur = reserve_radii[idx];

            const float px = center.x + ring_radius * std::cos(theta);
            const float py = center.y + ring_radius * std::sin(theta);
            const b2Vec2 cand{px, py};

            if (fits(cand, r_cur, result)) {
                result.push_back(cand);
            }
        }
    }

    if (result.size() != n_points) {
        throw std::runtime_error(
            "Impossible to place all requested points with the given reserve_radii.");
    }
    return result;
}



//////////////////////////////////// IMPORT FROM FILE //////////////////////////////////// {{{1

using arrow::Status;
using arrow::Result;
using arrow::DoubleArray;
using arrow::FloatArray;
using arrow::Int32Array;
using arrow::Int64Array;

namespace {

/*----------------------------------------------------------------------
 * Utility helpers
 *--------------------------------------------------------------------*/

inline float random_angle() {
    static thread_local std::uniform_real_distribution<float> d(
        0.0f, 2.0f * static_cast<float>(M_PI));
    return d(rnd_gen);
}

struct bbox_t {
    float min_x{ std::numeric_limits<float>::max() };
    float max_x{ std::numeric_limits<float>::lowest() };
    float min_y{ std::numeric_limits<float>::max() };
    float max_y{ std::numeric_limits<float>::lowest() };
};

bbox_t accumulate_bbox(const arena_polygons_t& polys) {
    bbox_t bb;
    for (auto& poly : polys)
        for (auto& p : poly) {
            bb.min_x = std::min(bb.min_x, p.x);
            bb.max_x = std::max(bb.max_x, p.x);
            bb.min_y = std::min(bb.min_y, p.y);
            bb.max_y = std::max(bb.max_y, p.y);
        }
    return bb;
}

inline float map_lin(float v, float smin, float smax,
                     float dmin, float dmax) {
    if (std::fabs(smax - smin) < std::numeric_limits<float>::epsilon())
        return 0.5f * (dmin + dmax);
    return dmin + (v - smin) * (dmax - dmin) / (smax - smin);
}


/* robust numeric → double (allows surrounding blanks, “nan”, “NaN”) */
double parse_num(const std::string& s) {
    if (s.empty()) return std::numeric_limits<double>::quiet_NaN();

    const char* c = s.c_str();
    char* end     = nullptr;
    double v      = std::strtod(c, &end);

    /* skip any trailing spaces / tabs */
    while (end && std::isspace(static_cast<unsigned char>(*end))) ++end;

    if (*end == '\0') return v;                 // fully consumed → ok

    std::string lc;
    lc.reserve(s.size());
    for (char ch : s) lc.push_back(std::tolower(static_cast<unsigned char>(ch)));
    return (lc == "nan") ? std::numeric_limits<double>::quiet_NaN()
                         : std::numeric_limits<double>::quiet_NaN();
}


void read_csv(const std::filesystem::path& file,
              std::vector<b2Vec2>& pts,
              std::vector<float>&  ths) {
    std::ifstream in(file);
    if (!in) throw std::runtime_error("Cannot open CSV " + file.string());

    auto split = [](const std::string& l) {
        std::vector<std::string> v;
        std::stringstream ss(l);
        std::string tok;
        while (std::getline(ss, tok, ',')) v.push_back(tok);
        return v;
    };

    /* ---------- detect header ---------- */
    std::string line;
    while (std::getline(in, line) &&
           line.find_first_not_of(" \t\r\n") == std::string::npos) {}
    if (in.eof()) return;                            // empty file

    std::vector<std::string> first = split(line);
    bool has_header = std::any_of(first.begin(), first.end(), [](const std::string& t) {
        return t.find_first_not_of("0123456789+-.eE") != std::string::npos;
    });

    int ix_x = 0, ix_y = 1, ix_theta = -1, ix_time = -1;
    if (has_header) {
        auto find = [&](std::initializer_list<std::string> names) -> int {
            for (size_t i = 0; i < first.size(); ++i) {
                std::string n = first[i];
                std::transform(n.begin(), n.end(), n.begin(), ::tolower);
                if (std::find(names.begin(), names.end(), n) != names.end())
                    return static_cast<int>(i);
            }
            return -1;
        };
        ix_x     = find({ "x" });
        ix_y     = find({ "y" });
        ix_theta = find({ "theta", "angle" });
        ix_time  = find({ "time" });

        if (ix_x < 0 || ix_y < 0)
            throw std::runtime_error("CSV header missing x or y column");
        /* data starts on next getline() */
    } else {
        in.seekg(0);                                 // first line is data
    }

    const int max_optional =
        std::max({ ix_theta, ix_time, ix_x, ix_y }); // highest column we may access

    struct row_t { float x, y, theta; double time; bool has_time; };
    std::vector<row_t> rows;
    double t_min = std::numeric_limits<double>::infinity();

    while (std::getline(in, line)) {
        auto f = split(line);
        if (static_cast<int>(f.size()) <= std::max(ix_x, ix_y))
            continue;                                // x or y missing → skip

        /* pad with empty strings so indexing is always safe */
        if (static_cast<int>(f.size()) <= max_optional)
            f.resize(max_optional + 1);

        double x = parse_num(f[ix_x]);
        double y = parse_num(f[ix_y]);
        if (std::isnan(x) || std::isnan(y)) continue;

        double theta_raw = (ix_theta >= 0) ? parse_num(f[ix_theta])
                                           : std::numeric_limits<double>::quiet_NaN();

        bool   has_time = ix_time >= 0;
        double time_raw = has_time ? parse_num(f[ix_time])
                                   : std::numeric_limits<double>::quiet_NaN();

        if (has_time && !std::isnan(time_raw) && time_raw < t_min) t_min = time_raw;

        rows.push_back({ static_cast<float>(x),
                         static_cast<float>(y),
                         static_cast<float>(theta_raw),
                         time_raw,
                         has_time });
    }

    /* ---------- keep rows with min-time (if any) ---------- */
    const double eps = 1e-9;
    for (const auto& r : rows) {
        if (r.has_time && std::fabs(r.time - t_min) > eps) continue;

        float th = (!std::isnan(r.theta)) ? r.theta : random_angle();
        pts.push_back({r.x, r.y});
        ths.emplace_back(th);
    }
}


double scalar_at(const std::shared_ptr<arrow::Array>& arr, int64_t i) {
    if (!arr->IsValid(i)) return std::numeric_limits<double>::quiet_NaN();

    switch (arr->type_id()) {
        case arrow::Type::DOUBLE:
            return static_cast<const arrow::DoubleArray&>(*arr).Value(i);
        case arrow::Type::FLOAT:
            return static_cast<const arrow::FloatArray&>(*arr).Value(i);
        case arrow::Type::INT64:
            return static_cast<const arrow::Int64Array&>(*arr).Value(i);
        case arrow::Type::INT32:
            return static_cast<const arrow::Int32Array&>(*arr).Value(i);
        default:
            return std::numeric_limits<double>::quiet_NaN();
    }
}

void read_feather(const std::filesystem::path& file,
                  std::vector<b2Vec2>& pts,
                  std::vector<float>&  ths) {
    /* open -------------------------------------------------------------- */
    auto rf_res = arrow::io::ReadableFile::Open(file.string());
    if (!rf_res.ok()) throw std::runtime_error(rf_res.status().ToString());
    std::shared_ptr<arrow::io::ReadableFile> rf = *rf_res;

    auto r_res = arrow::ipc::feather::Reader::Open(rf);
    if (!r_res.ok()) throw std::runtime_error(r_res.status().ToString());
    std::shared_ptr<arrow::ipc::feather::Reader> rdr = *r_res;

    std::shared_ptr<arrow::Table> tbl;
    if (!rdr->Read(&tbl).ok())
        throw std::runtime_error("Feather: cannot read table");

    /* make each column a single contiguous array */
    auto comb_res = tbl->CombineChunks(arrow::default_memory_pool());
    if (!comb_res.ok()) throw std::runtime_error(comb_res.status().ToString());
    tbl = *comb_res;

    /* locate columns ---------------------------------------------------- */
    auto find = [&](std::initializer_list<std::string> names)->int{
        for (int i=0;i<tbl->num_columns();++i){
            std::string n = tbl->field(i)->name();
            std::transform(n.begin(), n.end(), n.begin(), ::tolower);
            if (std::find(names.begin(), names.end(), n)!=names.end()) return i;
        }
        return -1;
    };

    int ix_x     = find({ "x" });
    int ix_y     = find({ "y" });
    int ix_theta = find({ "theta", "angle" });
    int ix_time  = find({ "time" });

    if (ix_x < 0 || ix_y < 0)
        throw std::runtime_error("Feather missing x or y column");

    /* choose rows (min-time) ------------------------------------------- */
    std::vector<int64_t> rows;
    if (ix_time >= 0) {
        auto time_arr = tbl->column(ix_time)->chunk(0);
        double t_min = std::numeric_limits<double>::infinity();
        for (int64_t r=0;r<time_arr->length();++r){
            double v = scalar_at(time_arr,r);
            if (!std::isnan(v) && v < t_min) t_min = v;
        }
        for (int64_t r=0;r<time_arr->length();++r)
            if (scalar_at(time_arr,r)==t_min) rows.push_back(r);
    } else {
        rows.resize(tbl->num_rows());
        std::iota(rows.begin(), rows.end(), 0);
    }

    /* extract ----------------------------------------------------------- */
    auto col_x = tbl->column(ix_x)->chunk(0);
    auto col_y = tbl->column(ix_y)->chunk(0);
    std::shared_ptr<arrow::Array> col_th =
        ix_theta >= 0 ? tbl->column(ix_theta)->chunk(0) : nullptr;

    pts.reserve(rows.size());
    ths.reserve(rows.size());

    for (auto r : rows) {
        float x = static_cast<float>(scalar_at(col_x, r));
        float y = static_cast<float>(scalar_at(col_y, r));

        float th = random_angle();
        if (col_th) {
            double raw = scalar_at(col_th, r);
            if (!std::isnan(raw)) th = static_cast<float>(raw);
        }
        pts.push_back({x, y});
        ths.emplace_back(th);
    }
}


} // anonymous namespace

std::tuple<std::vector<b2Vec2>, std::vector<float>>
import_points_from_file(const arena_polygons_t &scaled_arena_polygons,
                        const size_t nb_objects,
                        const std::string &formation_filename,
                        const std::pair<float, float> &imported_formation_min_coords,
                        const std::pair<float, float> &imported_formation_max_coords) {
    std::vector<b2Vec2> points;
    std::vector<float> thetas;

    // Load formation file ---------------------------------------------------
    std::filesystem::path path { formation_filename };
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    if (ext == ".csv") {
        read_csv(path, points, thetas);
    } else if (ext == ".feather") {
        read_feather(path, points, thetas);
    } else {
        throw std::runtime_error("Unsupported formation file extension: " + ext);
    }
    if (points.empty()) {
        throw std::runtime_error("Formation file contains no valid rows");
    }

    // Decide whether rescaling is requested ---------------------------------
    bool do_rescale = std::isfinite(imported_formation_min_coords.first) &&
                      std::isfinite(imported_formation_max_coords.first) &&
                      std::isfinite(imported_formation_min_coords.second) &&
                      std::isfinite(imported_formation_max_coords.second);

    if (do_rescale) {
        float src_min_x = imported_formation_min_coords.first;
        float src_max_x = imported_formation_max_coords.first;
        float src_min_y = imported_formation_min_coords.second;
        float src_max_y = imported_formation_max_coords.second;

        const bbox_t arena_bbox = accumulate_bbox(scaled_arena_polygons);
        for (auto &p : points) {
            p.x = map_lin(p.x, src_min_x, src_max_x, arena_bbox.min_x, arena_bbox.max_x);
            p.y = map_lin(p.y, src_min_y, src_max_y, arena_bbox.min_y, arena_bbox.max_y);
        }
    } else {
        for (auto &p : points) {
            p.x *= VISUALIZATION_SCALE;
            p.y *= VISUALIZATION_SCALE;
        }
    }

    if (points.size() < nb_objects || thetas.size() < nb_objects) {
        throw std::runtime_error("Not enough points in imported data file: " + std::to_string(points.size()) + " but at least " + std::to_string(nb_objects) + " are needed.");
    }

    return { std::move(points), std::move(thetas) };
}


// MODELINE "{{{1
// vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
// vim:foldmethod=marker
