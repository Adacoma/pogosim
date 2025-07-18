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

    std::vector<b2Vec2> current_polygon;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) {
            if (!current_polygon.empty()) {
                polygons.push_back(current_polygon);
                current_polygon.clear();
            }
            continue;
        }
        std::istringstream ss(line);
        std::string x_str, y_str;
        if (std::getline(ss, x_str, ',') && std::getline(ss, y_str)) {
            float x = std::stof(x_str);
            float y = std::stof(y_str);
            current_polygon.push_back({x, y});
        }
    }
    file.close();
    if (!current_polygon.empty()) {
        polygons.push_back(current_polygon);
    }

    // Compute the overall bounding box.
    float min_x = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float min_y = std::numeric_limits<float>::max();
    float max_y = std::numeric_limits<float>::lowest();
    for (const auto& poly : polygons) {
        for (const auto& point : poly) {
            min_x = std::min(min_x, point.x);
            max_x = std::max(max_x, point.x);
            min_y = std::min(min_y, point.y);
            max_y = std::max(max_y, point.y);
        }
    }

    // Normalize all polygons into a [0,1] range and flip Y.
    std::vector<std::vector<b2Vec2>> normalized_polygons;
    for (const auto& poly : polygons) {
        std::vector<b2Vec2> norm_poly;
        for (const auto& point : poly) {
            float norm_x = (point.x - min_x) / (max_x - min_x);
            float norm_y = (point.y - min_y) / (max_y - min_y);
            norm_poly.push_back({norm_x, norm_y});
        }
        normalized_polygons.push_back(norm_poly);
    }

    if (normalized_polygons.empty()) {
        throw std::runtime_error("No polygons loaded from file.");
    }

    // Compute effective area in normalized space.
    float main_area = compute_polygon_area(normalized_polygons[0]);
    float holes_area = 0.0f;
    for (size_t i = 1; i < normalized_polygons.size(); ++i) {
        holes_area += compute_polygon_area(normalized_polygons[i]);
    }
    float effective_area = main_area - holes_area;
    if (effective_area <= 0) {
        throw std::runtime_error("Effective area of polygons is non-positive.");
    }

    // Determine the scale factor s.
    float scale = std::sqrt(total_surface / effective_area);

    // Apply uniform scaling to all normalized polygons.
    std::vector<std::vector<b2Vec2>> scaled_polygons;
    for (const auto& poly : normalized_polygons) {
        std::vector<b2Vec2> scaled_poly;
        for (const auto& point : poly) {
            scaled_poly.push_back({point.x * scale, point.y * scale});
        }
        scaled_polygons.push_back(scaled_poly);
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


// ───── distance from point P to segment AB ────────────────────────────
static float distance_point_to_segment(const b2Vec2 &p,
                                       const b2Vec2 &a,
                                       const b2Vec2 &b) {
    const b2Vec2 ab = b - a, ap = p - a;
    const float   ab2 = ab.x * ab.x + ab.y * ab.y;
    if (ab2 == 0.f) {                           // degenerate segment
        return std::hypot(ap.x, ap.y);
    }
    float t = (ap.x * ab.x + ap.y * ab.y) / ab2;
    t = std::clamp(t, 0.f, 1.f);
    const b2Vec2 q = a + t * ab;
    return std::hypot(p.x - q.x, p.y - q.y);
}

// ───── minimum distance from P to *any* polygon edge ──────────────────
static float min_distance_to_polygons_edges(
        const b2Vec2 &p,
        const std::vector<std::vector<b2Vec2>> &polys) {
    float d_min = std::numeric_limits<float>::infinity();
    for (const auto &poly : polys) {
        for (std::size_t i = 0, n = poly.size(); i < n; ++i) {
            d_min = std::min(d_min,
                distance_point_to_segment(p, poly[i], poly[(i + 1) % n]));
        }
    }
    return d_min;
}


/**
 * Generate random points inside a (possibly holed) polygonal domain while
 * respecting a per‑point exclusion radius and a global connectivity limit.
 *
 * A candidate is accepted only if it is:
 *   1. Inside the outer polygon and outside every "hole" polygon.
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

    // Count non-NaN points that need to be placed
    std::size_t n_points_to_place = 0;
    for (const auto &radius : reserve_radii) {
        if (!std::isnan(radius)) {
            n_points_to_place++;
        }
    }
    
    // If all points are NaN, return a vector of NaN points
    if (n_points_to_place == 0) {
        std::vector<b2Vec2> nan_points(n_points, b2Vec2{NAN, NAN});
        return nan_points;
    }

    // ─── conservative bounding‑box (contracted by largest radius) ─────────
    float bb_margin = 0.0f;
    for (const auto &radius : reserve_radii) {
        if (!std::isnan(radius)) {
            bb_margin = std::max(bb_margin, radius * 1.5f);
        }
    }

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
        // Pre-fill result vector with placeholders
        std::vector<b2Vec2> result(n_points);
        // Track which indices have been filled with real points
        std::vector<bool> filled(n_points, false);
        // Create a separate vector to hold the actually placed points for distance checking
        std::vector<b2Vec2> placed_points;
        // Map from placed_points index to result index
        std::vector<std::size_t> placed_to_result_index;
        
        std::uint32_t attempts = 0U;
        std::size_t next_idx_to_try = 0;

        // ─── rejection‑sampling loop ──────────────────────────────────────
        while (placed_points.size() < n_points_to_place) {
            // Find next unfilled non-NaN position
            while (next_idx_to_try < n_points && 
                  (filled[next_idx_to_try] || std::isnan(reserve_radii[next_idx_to_try]))) {
                // If this is a NaN radius and not yet filled, mark it as NaN point
                if (!filled[next_idx_to_try] && std::isnan(reserve_radii[next_idx_to_try])) {
                    result[next_idx_to_try] = b2Vec2{NAN, NAN};
                    filled[next_idx_to_try] = true;
                }
                next_idx_to_try++;
            }
            
            // If all indices have been processed, we're done
            if (next_idx_to_try >= n_points) {
                break;
            }
            
            const float x = dis_x(rnd_gen);
            const float y = dis_y(rnd_gen);

            const float r_curr = reserve_radii[next_idx_to_try];

            // 1️⃣ inside outer polygon?
            if (!is_point_within_polygon(outer_poly, x, y)) { 
                if (++attempts >= attempts_per_point) break;
                continue; 
            }

            // 2️⃣ outside every hole polygon?
            bool ok = true;
            for (std::size_t i = 1; i < polygons.size() && ok; ++i) {
                if (is_point_within_polygon(polygons[i], x, y)) { ok = false; }
            }
            // At least 1.5 × r_i away from every wall/vertex
            if (ok) {
                const float d_wall = min_distance_to_polygons_edges({x, y}, polygons);
                if (d_wall < 1.5f * r_curr) { ok = false; }
            }

            // 3️⃣ exclusion radius + connectivity checks
            if (ok && !placed_points.empty()) {
                float min_dist = std::numeric_limits<float>::infinity();
                for (std::size_t i = 0; i < placed_points.size(); ++i) {
                    const std::size_t result_idx = placed_to_result_index[i];
                    const float min_sep = reserve_radii[result_idx] + r_curr;
                    const float d = euclidean_distance(placed_points[i], {x, y});
                    if (d < min_sep) { ok = false; break; } // too close
                    min_dist = std::min(min_dist, d);
                }
                if (ok && min_dist > max_neighbor_distance) { ok = false; }
            }

            // 4️⃣ accept or reject
            if (ok) {
                result[next_idx_to_try] = {x, y};
                filled[next_idx_to_try] = true;
                placed_points.push_back({x, y});
                placed_to_result_index.push_back(next_idx_to_try);
                next_idx_to_try++;
                attempts = 0U;                  // reset attempt counter
            } else if (++attempts >= attempts_per_point) {
                // Give up on this run and start over.
                break; // triggers outer restart loop
            }
        }

        // Check if we successfully placed all non-NaN points
        if (placed_points.size() == n_points_to_place) {
            // Fill any remaining NaN points before returning
            for (std::size_t i = 0; i < n_points; ++i) {
                if (!filled[i]) {
                    result[i] = b2Vec2{NAN, NAN};
                }
            }
            return result; // success
        }
    }

    // If we fall through the loop, all restarts failed.
    throw std::runtime_error("Impossible to create random points within polygon: too many points or radii too large, even after multiple restarts.");
}



std::vector<b2Vec2> generate_random_points_layered(
        const std::vector<std::vector<b2Vec2>> &polygons,
        const std::vector<float> &reserve_radii,
        std::uint32_t attempts_per_point,
        std::uint32_t max_restarts) {
    // ─── parameters ────────────────────────────────────────────────────
    constexpr float wall_clearance_factor = 1.50f;   // ← requested 1.50×
    constexpr float eps = 1e-4f;                     // tiny extra spacing

    // ─── sanity checks ────────────────────────────────────────────────
    if (polygons.empty()) throw std::runtime_error("Need at least one polygon.");
    for (auto const &poly : polygons)
        if (poly.size() < 3) throw std::runtime_error("Polygon needs ≥3 vertices.");

    const std::size_t n_points = reserve_radii.size();
    if (!n_points) return {};

    const std::size_t required_pts = static_cast<std::size_t>(std::count_if(
        reserve_radii.begin(), reserve_radii.end(),
        [](float r){ return !std::isnan(r); }));

    // ─── build boundary segment roulette wheel ────────────────────────
    struct seg_t { b2Vec2 a, b; float cumulative; };
    std::vector<seg_t> segs; segs.reserve(polygons.size() * 4);
    float total_len = 0.f;
    auto add_poly = [&](const std::vector<b2Vec2> &poly){
        for (std::size_t i = 0, n = poly.size(); i < n; ++i){
            const b2Vec2 a = poly[i], b = poly[(i+1)%n];
            total_len += euclidean_distance(a,b);
            segs.push_back({a,b,total_len});
        }
    };
    add_poly(polygons[0]);                           // outer wall
    for (std::size_t h=1; h<polygons.size(); ++h) add_poly(polygons[h]);

    // ─── RNG ───────────────────────────────────────────────────────────
    static std::mt19937 gen{std::random_device{}()};
    std::uniform_real_distribution<float> pick_len(0.f, total_len);
    std::uniform_real_distribution<float> pick_t(0.f, 1.f);
    std::uniform_real_distribution<float> pick_ang(0.f, 2.f*static_cast<float>(M_PI));

    // ─── helpers ───────────────────────────────────────────────────────
    auto inward_normal = [&](const seg_t &s)->b2Vec2{
        b2Vec2 n{-(s.b-s.a).y,(s.b-s.a).x};
        float len = std::hypot(n.x,n.y); n.x/=len; n.y/=len;
        b2Vec2 probe = s.a + eps*n;
        bool inside = is_point_within_polygon(polygons[0],probe.x,probe.y);
        for (std::size_t h=1; inside && h<polygons.size(); ++h)
            if (is_point_within_polygon(polygons[h],probe.x,probe.y)) inside=false;
        if (!inside){ n.x=-n.x; n.y=-n.y; }
        return n;
    };

    auto dist_pt_seg = [](const b2Vec2&p,const b2Vec2&a,const b2Vec2&b){
        b2Vec2 ab=b-a, ap=p-a;
        float ab2=ab.x*ab.x+ab.y*ab.y;
        if (ab2==0.f) return std::hypot(ap.x,ap.y);
        float t=((ap.x*ab.x)+(ap.y*ab.y))/ab2;
        t=std::clamp(t,0.f,1.f);
        b2Vec2 q{a.x+t*ab.x, a.y+t*ab.y};
        return std::hypot(p.x-q.x,p.y-q.y);
    };
    auto min_dist_edges = [&](const b2Vec2&p){
        float d=std::numeric_limits<float>::infinity();
        for (auto const &poly:polygons)
            for (std::size_t i=0,n=poly.size(); i<n; ++i)
                d = std::min(d, dist_pt_seg(p, poly[i], poly[(i+1)%n]));
        return d;
    };

    // sort indices by descending radius so big robots claim walls first
    std::vector<std::size_t> sorted_idx;
    for (std::size_t i=0;i<n_points;++i) if (!std::isnan(reserve_radii[i])) sorted_idx.push_back(i);
    std::ranges::sort(sorted_idx, [&](auto a,auto b){ return reserve_radii[a]>reserve_radii[b]; });

    // ─── restart loop ──────────────────────────────────────────────────
    for (std::uint32_t restart=0; restart<max_restarts; ++restart){
        struct placed_t{ b2Vec2 p; float r; };
        std::vector<placed_t> placed; placed.reserve(required_pts);
        std::vector<b2Vec2> result(n_points,{NAN,NAN});

        // ---------- STAGE 0 : wall pass --------------------------------
        std::vector<std::size_t> deferred;
        for (std::size_t idx : sorted_idx){
            float r_i = reserve_radii[idx];
            float clearance = r_i * wall_clearance_factor;

            bool ok=false;
            for (std::uint32_t a=0; a<attempts_per_point && !ok; ++a){
                float s = pick_len(gen);
                auto it = std::lower_bound(segs.begin(),segs.end(),s,
                     [](const seg_t&sg,float v){ return sg.cumulative<v; });
                const seg_t &seg=*it;
                float t=pick_t(gen);
                b2Vec2 wall_pt{seg.a.x+t*(seg.b.x-seg.a.x),
                               seg.a.y+t*(seg.b.y-seg.a.y)};
                b2Vec2 cand = wall_pt + (clearance+eps)*inward_normal(seg);

                // domain & clearance to every wall
                if (!is_point_within_polygon(polygons[0],cand.x,cand.y)) continue;
                bool inside=true;
                for (std::size_t h=1; inside && h<polygons.size(); ++h)
                    if (is_point_within_polygon(polygons[h],cand.x,cand.y)) inside=false;
                if (!inside) continue;
                if (min_dist_edges(cand) < clearance) continue;

                // overlap with placed points
                for (auto const &pl:placed)
                    if (euclidean_distance(pl.p,cand) < pl.r + r_i) { inside=false; break; }
                if (!inside) continue;

                // accept
                placed.push_back({cand,r_i});
                result[idx]=cand;
                ok=true;
            }
            if (!ok) deferred.push_back(idx);
        }

        // ---------- STAGE 1+ : inner layers ----------------------------
        if (placed.empty()) goto failed_this_restart;

        while (!deferred.empty()){
            std::uniform_int_distribution<std::size_t> pick_neigh(0,placed.size()-1);
            std::vector<std::size_t> next;
            for (std::size_t idx: deferred){
                float r_i=reserve_radii[idx];
                float clearance = r_i * wall_clearance_factor;
                bool ok=false;
                for (std::uint32_t a=0; a<attempts_per_point && !ok; ++a){
                    const placed_t &seed = placed[pick_neigh(gen)];
                    float θ=pick_ang(gen);
                    b2Vec2 dir{std::cos(θ),std::sin(θ)};
                    b2Vec2 cand = seed.p + (r_i+seed.r+eps)*dir;

                    if (!is_point_within_polygon(polygons[0],cand.x,cand.y)) continue;
                    bool inside=true;
                    for (std::size_t h=1; inside && h<polygons.size(); ++h)
                        if (is_point_within_polygon(polygons[h],cand.x,cand.y)) inside=false;
                    if (!inside) continue;
                    if (min_dist_edges(cand) < clearance) continue;

                    for (auto const &pl:placed)
                        if (euclidean_distance(pl.p,cand) < pl.r + r_i) { inside=false; break; }
                    if (!inside) continue;

                    placed.push_back({cand,r_i});
                    result[idx]=cand;
                    ok=true;
                }
                if (!ok) next.push_back(idx);
            }
            if (next.size()==deferred.size()) goto failed_this_restart; // no progress
            deferred.swap(next);
        }

        if (placed.size()==required_pts) return result;
failed_this_restart: ;
    }

    throw std::runtime_error("Could not place every point 1.5×r away from walls.");
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


std::vector<b2Vec2> generate_chessboard_points(
        const std::vector<std::vector<b2Vec2>> &polygons,
        std::size_t                             n_points,
        float                                   pitch,
        bool                                    cluster_center) {

    // ──────────────────────── guards ────────────────────────────────
    if (pitch <= 0.0f)            { throw std::runtime_error("pitch must be > 0"); }
    if (n_points == 0)            { throw std::runtime_error("n_points must be > 0"); }
    if (polygons.empty())         { throw std::runtime_error("polygons vector is empty"); }
    if (polygons.front().size()<3){ throw std::runtime_error("outer polygon degenerate"); }

    const auto &outer = polygons.front();

    // ─────────────────── outer AABB of arena ───────────────────────
    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float max_y = std::numeric_limits<float>::lowest();

    for (const auto &v: outer) {
        min_x = std::min(min_x, v.x);   min_y = std::min(min_y, v.y);
        max_x = std::max(max_x, v.x);   max_y = std::max(max_y, v.y);
    }
    if (min_x >= max_x || min_y >= max_y) {
        throw std::runtime_error("degenerate outer polygon");
    }

    // ──────────────── helpers for point validity ───────────────────
    const auto is_valid_point = [&](float x, float y)->bool {
        if (!is_point_within_polygon(outer, x, y)) { return false; }
        for (std::size_t h = 1; h < polygons.size(); ++h) {
            if (is_point_within_polygon(polygons[h], x, y)) { return false; }
        }
        return true;
    };

    const auto collect_grid_nodes =
        [&](float grid_pitch, float x_origin, float y_origin)->std::vector<b2Vec2> {

        std::vector<b2Vec2> nodes;
        const float margin = grid_pitch * 0.01f;

        for (float x = x_origin; x <= max_x + margin; x += grid_pitch) {
            for (float y = y_origin; y <= max_y + margin; y += grid_pitch) {
                if (is_valid_point(x, y)) { nodes.push_back({x, y}); }
            }
        }
        return nodes;
    };

    // ────────────── CLUSTER-CENTER (compact) mode ──────────────────
    if (cluster_center) {
        // 1. compute signed‐area centroid of the outer polygon
        double A = 0.0, Cx = 0.0, Cy = 0.0;
        for (std::size_t i = 0, j = outer.size() - 1; i < outer.size(); j = i++) {
            double cross = outer[j].x * outer[i].y - outer[i].x * outer[j].y;
            A  += cross;
            Cx += (outer[j].x + outer[i].x) * cross;
            Cy += (outer[j].y + outer[i].y) * cross;
        }
        A *= 0.5;
        float center_x, center_y;
        if (std::fabs(A) > 1e-12) {
            center_x = static_cast<float>(Cx / (6.0 * A));
            center_y = static_cast<float>(Cy / (6.0 * A));
        } else {
            center_x = 0.5f * (min_x + max_x);
            center_y = 0.5f * (min_y + max_y);
        }

        // 2. ensure the centre itself is valid
        if (!is_valid_point(center_x, center_y)) {
            float best_x = center_x, best_y = center_y;
            float best_d2 = std::numeric_limits<float>::max();
            float step = std::min(pitch * 0.1f,
                                  std::min(max_x - min_x, max_y - min_y) * 0.05f);
            for (float x = min_x; x <= max_x; x += step) {
                for (float y = min_y; y <= max_y; y += step) {
                    if (!is_valid_point(x, y)) continue;
                    float d2 = (x - center_x)*(x - center_x)
                             + (y - center_y)*(y - center_y);
                    if (d2 < best_d2) {
                        best_d2 = d2;
                        best_x = x;
                        best_y = y;
                    }
                }
            }
            center_x = best_x;
            center_y = best_y;
        }

        // 3. build expanding square rings until we have exactly n_points
        std::vector<b2Vec2> points{{center_x, center_y}};
        if (n_points == 1) {
            return points;
        }
        for (int radius = 1; points.size() < n_points; ++radius) {
            std::vector<b2Vec2> ring;
            for (int dx = -radius; dx <= radius; ++dx) {
                for (int dy = -radius; dy <= radius; ++dy) {
                    if (std::abs(dx) != radius && std::abs(dy) != radius) continue;
                    float x = center_x + dx * pitch;
                    float y = center_y + dy * pitch;
                    if (is_valid_point(x, y)) {
                        ring.push_back({x, y});
                    }
                }
            }
            // if we can’t grow any further, give up with an exception
            if (ring.empty()) {
                throw std::runtime_error("cannot place requested points: pitch too large for arena");
            }
            std::sort(ring.begin(), ring.end(),
                      [cx = center_x, cy = center_y](auto &a, auto &b){
                float da = (a.x - cx)*(a.x - cx) + (a.y - cy)*(a.y - cy);
                float db = (b.x - cx)*(b.x - cx) + (b.y - cy)*(b.y - cy);
                return da < db;
            });
            points.insert(points.end(), ring.begin(), ring.end());
        }

        // 4. now we must have ≥ n_points – if somehow we still didn’t, that’s an error
        if (points.size() < n_points) {
            throw std::runtime_error("cannot place requested points with cluster_center");
        }

        // 5. trim any surplus (never dropping the centre)
        if (points.size() > n_points) {
            b2Vec2 centre = points.front();
            std::vector<b2Vec2> others(points.begin() + 1, points.end());
            std::sort(others.begin(), others.end(),
                      [centre](auto &a, auto &b){
                float da = (a.x - centre.x)*(a.x - centre.x)
                         + (a.y - centre.y)*(a.y - centre.y);
                float db = (b.x - centre.x)*(b.x - centre.x)
                         + (b.y - centre.y)*(b.y - centre.y);
                return da < db;
            });
            others.resize(n_points - 1);
            others.insert(others.begin(), centre);
            return others;
        }

        return points; // exactly n_points
    }


    // ──────────── DISTRIBUTED (original) mode below ────────────────────
    const auto try_exact_grid = [&](float test_pitch)
            ->std::optional<std::vector<b2Vec2>> {

        const float margin = test_pitch * 0.1f;
        const float start_x = min_x - margin;
        const float start_y = min_y - margin;
        const float step    = test_pitch * 0.05f;

        for (float dx = 0.0f; dx < test_pitch; dx += step) {
            for (float dy = 0.0f; dy < test_pitch; dy += step) {
                auto nodes = collect_grid_nodes(test_pitch,
                                                start_x + dx, start_y + dy);
                if (nodes.size() == n_points) { return nodes; }
            }
        }
        return std::nullopt;
    };

    /* ---- Strategy 1: exact grid with given pitch --------------------- */
    if (auto r = try_exact_grid(pitch)) { return *r; }

    /* ---- Strategy 2: slight pitch tweaks (±5 %) ---------------------- */
    for (float k : {0.95f, 1.05f, 0.98f, 1.02f, 0.92f, 1.08f}) {
        if (auto r = try_exact_grid(pitch * k)) { return *r; }
    }

    /* ---- Strategy 3: use raw grid then subsample / complain ---------- */
    auto grid_pts = collect_grid_nodes(pitch, min_x, min_y);
    if (!grid_pts.empty()) {
        if (grid_pts.size() == n_points) { return grid_pts; }

        if (grid_pts.size() > n_points) {
            std::vector<b2Vec2> subset;
            if (n_points == 1) {
                subset.push_back(grid_pts[grid_pts.size()/2]);
            } else {
                float step = static_cast<float>(grid_pts.size()-1)
                           / static_cast<float>(n_points-1);
                for (std::size_t i = 0; i < n_points; ++i) {
                    subset.push_back(grid_pts[static_cast<std::size_t>(i*step)]);
                }
            }
            return subset;
        }
        throw std::runtime_error("pitch too large – not enough grid points");
    }

    /* ---- Strategy 4: modest pitch shrink for very small n ----------- */
    if (n_points <= 5) {
        for (float k: {0.8f, 0.7f, 0.6f}) {
            auto small = collect_grid_nodes(pitch * k, min_x, min_y);
            if (small.size() >= n_points) {
                std::vector<b2Vec2> subset;
                float step = static_cast<float>(small.size()-1)
                           / static_cast<float>(n_points-1);
                for (std::size_t i = 0; i < n_points; ++i) {
                    subset.push_back(small[static_cast<std::size_t>(i*step)]);
                }
                return subset;
            }
        }
    }

    throw std::runtime_error("cannot place requested points with given pitch");
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
    /* ---------- 1. sanity checks -------------------------------------- */
    if (polygons.empty()) {
        throw std::runtime_error("No polygons provided.");
    }
    const auto& outer_poly = polygons.front();
    if (outer_poly.size() < 3) {
        throw std::runtime_error("Polygon must have at least 3 vertices.");
    }
    const std::size_t n_points = reserve_radii.size();
    if (n_points == 0U) { return {}; }

    // Count how many valid (non-NaN) radii we have
    std::size_t valid_points_count = 0;
    for (const float& r : reserve_radii) {
        if (!std::isnan(r)) {
            valid_points_count++;
        }
    }

    // Find min/max of valid radii only
    float r_max = 0.0f;
    float r_min = std::numeric_limits<float>::max();
    bool has_valid_radius = false;

    for (const float& r : reserve_radii) {
        if (!std::isnan(r)) {
            r_max = std::max(r_max, r);
            r_min = std::min(r_min, r);
            has_valid_radius = true;
        }
    }

    // If all radii are NaN, return a vector of NaN points
    if (!has_valid_radius) {
        std::vector<b2Vec2> nan_points(n_points, b2Vec2{std::numeric_limits<float>::quiet_NaN(),
                                                      std::numeric_limits<float>::quiet_NaN()});
        return nan_points;
    }

    if (r_min <= 0.0f) {
        throw std::runtime_error("Reserve radii must be strictly positive.");
    }

    /* ---------- 2. centroid & admissible radius ----------------------- */
    const b2Vec2 centroid = polygon_centroid(outer_poly);
    float max_edge_dist = std::numeric_limits<float>::lowest();
    for (std::size_t i = 0; i < outer_poly.size(); ++i) {
        max_edge_dist = std::max(
            max_edge_dist,
            point_to_line_segment_distance(centroid,
                                           outer_poly[i],
                                           outer_poly[(i + 1) % outer_poly.size()]));
    }
    const float allowed_radius = max_edge_dist - r_max;
    if (allowed_radius <= 0.0f) {
        throw std::runtime_error("Polygon too small or some reserve radius too large.");
    }

    /* ---------- 3. helper lambda -------------------------------------- */
    const auto fits = [&](const b2Vec2& c, float r_curr,
                          const std::vector<std::pair<b2Vec2, size_t>>& accepted) -> bool
    {
        if (!is_point_within_polygon(outer_poly, c.x, c.y)) { return false; }
        for (std::size_t h = 1; h < polygons.size(); ++h) {
            if (is_point_within_polygon(polygons[h], c.x, c.y)) { return false; }
        }
        if (euclidean_distance(centroid, c) + r_curr > allowed_radius + 1e-5f) {
            return false;
        }
        for (const auto& [point, idx] : accepted) {
            if (euclidean_distance(point, c) < reserve_radii[idx] + r_curr) {
                return false;
            }
        }
        return true;
    };

    /* ---------- 4. place points --------------------------------------- */
    // Track placed points with their original indices
    std::vector<std::pair<b2Vec2, size_t>> placed_with_indices;
    placed_with_indices.reserve(valid_points_count);

    // Final result vector (will have NaN for unplaced points)
    std::vector<b2Vec2> result(n_points, b2Vec2{std::numeric_limits<float>::quiet_NaN(),
                                              std::numeric_limits<float>::quiet_NaN()});

    // First try placing at centroid for the first valid radius
    for (size_t i = 0; i < n_points; ++i) {
        if (!std::isnan(reserve_radii[i])) {
            if (fits(centroid, reserve_radii[i], placed_with_indices)) {
                placed_with_indices.push_back({centroid, i});
                result[i] = centroid;
            }
            break;
        }
    }

    float ring_radius = r_min;                  // start one full step out
    const float radial_step = r_min;

    // Add a safety counter to prevent infinite loops
    const size_t max_iterations = 1000; // Adjust based on expected complexity
    size_t iteration_count = 0;

    while (placed_with_indices.size() < valid_points_count &&
           ring_radius <= allowed_radius &&
           iteration_count < max_iterations) {

        /* angular step so that neighbouring *small* points are ≈2 r_min apart */
        const float d_theta = 2.0f * r_min / ring_radius;
        bool placed_any_this_ring = false;

        for (float theta = 0.0f;
             theta < 2.0f * float(M_PI) && placed_with_indices.size() < valid_points_count;
             theta += d_theta)
        {
            // Find the next point with valid radius that needs placement
            for (size_t i = 0; i < n_points && placed_with_indices.size() < valid_points_count; ++i) {
                // Skip if already placed or if radius is NaN
                if (!std::isnan(result[i].x) || std::isnan(reserve_radii[i])) {
                    continue;
                }

                const float r_cur = reserve_radii[i];
                const b2Vec2 cand{
                    centroid.x + ring_radius * std::cos(theta),
                    centroid.y + ring_radius * std::sin(theta)
                };

                if (fits(cand, r_cur, placed_with_indices)) {
                    placed_with_indices.push_back({cand, i});
                    result[i] = cand;
                    placed_any_this_ring = true;
                    break; // Move to next angle after successful placement
                }
            }
        }

        // If we completed a full ring and couldn't place any points, we may be stuck
        if (!placed_any_this_ring) {
            iteration_count++;
        }

        ring_radius += radial_step;
    }

    // Check if we managed to place all valid points
    if (placed_with_indices.size() < valid_points_count) {
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
    }

//    for (auto &p : points) {
//        glogger->info("DEBUG imported: {}, {}", p.x, p.y);
//    }

    if (points.size() < nb_objects || thetas.size() < nb_objects) {
        throw std::runtime_error("Not enough points in imported data file: " + std::to_string(points.size()) + " but at least " + std::to_string(nb_objects) + " are needed.");
    }

    return { std::move(points), std::move(thetas) };
}


// MODELINE "{{{1
// vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
// vim:foldmethod=marker
