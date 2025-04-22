#include <SDL2/SDL.h>
#include <box2d/box2d.h>
#include <vector>
#include <random>
#include <iostream>
#include <fstream>
#include <sstream>

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
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> disX(minX, maxX);
    std::uniform_real_distribution<float> disY(minY, maxY);

    // Generate random points until one is inside the polygon
    while (true) {
        float x = disX(gen);
        float y = disY(gen);

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


std::vector<b2Vec2> generate_random_points_within_polygon_safe(
        const std::vector<std::vector<b2Vec2>>& polygons,
        const std::vector<float>& reserve_radii) {
    if (polygons.empty()) {
        throw std::runtime_error("At least one polygon must be supplied.");
    }
    for (const auto& p : polygons) {
        if (p.size() < 3) {
            throw std::runtime_error("Every polygon needs ≥ 3 vertices.");
        }
    }

    const std::size_t n_points = reserve_radii.size();
    if (n_points == 0U) { return {}; }

    /* --- build a conservative bounding box: contract by the                *
     *     largest exclusion radius so that every random candidate           *
     *     automatically satisfies the “stay away from the outer edge” rule. */
    const float bb_margin = *std::max_element(reserve_radii.begin(), reserve_radii.end());

    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float max_y = std::numeric_limits<float>::lowest();

    const auto& outer_poly = polygons.front();
    for (const auto& v : outer_poly) {
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

    /* --- RNG ------------------------------------------------------------- */
    std::random_device rd;
    std::mt19937                 gen(rd());
    std::uniform_real_distribution<float> dis_x(min_x, max_x);
    std::uniform_real_distribution<float> dis_y(min_y, max_y);

    std::vector<b2Vec2> points;   points.reserve(n_points);
    std::uint32_t attempts = 0U;

    /* --- rejection sampling --------------------------------------------- */
    while (points.size() < n_points) {
        const float x = dis_x(gen);
        const float y = dis_y(gen);

        /* Candidate’s personal exclusion radius. */
        const float r_curr = reserve_radii[points.size()];

        /* 1️⃣  inside outer polygon? (ignore if outside) */
        if (!is_point_within_polygon(outer_poly, x, y)) { continue; }

        /* 2️⃣  outside every “hole” polygon? */
        bool ok = true;
        for (std::size_t i = 1; i < polygons.size() && ok; ++i) {
            if (is_point_within_polygon(polygons[i], x, y)) { ok = false; }
        }

        /* 3️⃣  far enough from all previously accepted points? */
        if (ok) {
            for (std::size_t i = 0; i < points.size(); ++i) {
                const float min_sep = reserve_radii[i] + r_curr;
                if (euclidean_distance(points[i], {x, y}) < min_sep) {
                    ok = false;
                    break;
                }
            }
        }

        if (ok) {                       // accept candidate
            points.emplace_back(x, y);
            attempts = 0U;
        } else if (++attempts >= 10'000'000U) {
            throw std::runtime_error(
                "Impossible to create random points within polygon: "
                "too many points or radii too large.");
        }
    }
    return points;
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


void draw_polygon(SDL_Renderer* renderer, const std::vector<b2Vec2>& polygon) {
    if (polygon.size() < 3) {
        std::cerr << "Error: Polygon must have at least 3 points to be drawable." << std::endl;
        return;
    }

    // Set the drawing color for the polygon
    SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255); // Red color with full opacity

    // Draw lines between consecutive points
    for (size_t i = 0; i < polygon.size() - 1; ++i) {
        auto const orig_pos = visualization_position(polygon[i].x, polygon[i].y);
        auto const dest_pos = visualization_position(polygon[i+1].x, polygon[i+1].y);
        SDL_RenderDrawLine(renderer,
                           static_cast<int>(orig_pos.x),
                           static_cast<int>(orig_pos.y),
                           static_cast<int>(dest_pos.x),
                           static_cast<int>(dest_pos.y));
    }

    // Connect the last point to the first to close the polygon
    auto const back_pos  = visualization_position(polygon.back().x, polygon.back().y);
    auto const front_pos = visualization_position(polygon.front().x, polygon.front().y);
    SDL_RenderDrawLine(renderer,
                       static_cast<int>(back_pos.x),
                       static_cast<int>(back_pos.y),
                       static_cast<int>(front_pos.x),
                       static_cast<int>(front_pos.y));
}


void save_window_to_png(SDL_Renderer* renderer, SDL_Window* window, const std::string& filename) {
    int width, height;
    SDL_GetWindowSize(window, &width, &height);

    ensure_directories_exist(filename);

    std::vector<Uint8> pixels(width * height * 4); // RGBA format
    if (SDL_RenderReadPixels(renderer, NULL, SDL_PIXELFORMAT_RGBA32, pixels.data(), width * 4) != 0) {
        std::cerr << "Error reading pixels: " << SDL_GetError() << std::endl;
        return;
    }

//    // Flip the image vertically (SDL's origin is top-left, PNG's is bottom-left)
//    std::vector<Uint8> flippedPixels(width * height * 4);
//    for (int row = 0; row < height; ++row) {
//        std::copy_n(
//            &pixels[(height - row - 1) * width * 4], // Source row (flipped)
//            width * 4,                               // Row size
//            &flippedPixels[row * width * 4]          // Destination row
//        );
//    }

    // Save to PNG
    if (!fpng::fpng_encode_image_to_file(filename.c_str(), pixels.data(), width, height, 4)) {
        glogger->warn("Error writing PNG file '{}'", filename);
    } else {
        glogger->debug("Saved window content to '{}'", filename);
    }
}


// MODELINE "{{{1
// vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
// vim:foldmethod=marker
