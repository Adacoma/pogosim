
#include "utils.h"
#include "objects_geometry.h"
#include "robot.h"
#include "distances.h"
#include "simulator.h"

#include <cmath>
#include "SDL2_gfxPrimitives.h"


static b2Vec2 rotate_local(b2Vec2 v, float a) {
    float const c = std::cos(a);
    float const s = std::sin(a);
    return {c * v.x - s * v.y, s * v.x + c * v.y};
}


/************* ObjectGeometry *************/ // {{{1

ObjectGeometry::~ObjectGeometry() {
    if (shape_created && b2Shape_IsValid(shape_id))
        b2DestroyShape(shape_id, false);
}

float ObjectGeometry::get_distance_to(b2Vec2 orig, b2Vec2 point) const {
    return euclidean_distance(orig, point);
}

/************* DiskGeometry *************/ // {{{1

void DiskGeometry::create_box2d_shape(b2BodyId body_id, b2ShapeDef& shape_def) {
    b2Circle circle;
    circle.center = { 0.0f, 0.0f };
    circle.radius = radius / VISUALIZATION_SCALE;
    shape_id = b2CreateCircleShape(body_id, &shape_def, &circle);
    shape_created = true;
}

std::vector<std::vector<bool>> DiskGeometry::export_geometry_grid(size_t num_bins_x,
                                                                  size_t num_bins_y,
                                                                  float bin_width,
                                                                  float bin_height,
                                                                  float obj_x,
                                                                  float obj_y) const {
    std::vector<std::vector<bool>> grid(num_bins_y, std::vector<bool>(num_bins_x, false));

    for (size_t j = 0; j < num_bins_y; ++j) {
        for (size_t i = 0; i < num_bins_x; ++i) {
            // Determine the center of this bin.
            float center_x = (i + 0.5f) * bin_width;
            float center_y = (j + 0.5f) * bin_height;
            // Calculate squared distance from the bin center to the object center.
            float dx = center_x - obj_x;
            float dy = center_y - obj_y;
            if ((dx * dx + dy * dy) <= (radius * radius)) {
                grid[j][i] = true;
            }
        }
    }
    return grid;
}


void DiskGeometry::render(SDL_Renderer* renderer, [[maybe_unused]] b2WorldId world_id, float x, float y, [[maybe_unused]] float theta, uint8_t r, uint8_t g, uint8_t b, uint8_t alpha) const {
    filledCircleRGBA(renderer, x, y, radius * mm_to_pixels, r, g, b, alpha);
}

BoundingDisk DiskGeometry::compute_bounding_disk() const {
    // The disk is defined by its own radius and is centered at (0,0) in local coordinates.
    return { 0.0f, 0.0f, radius };
}

BoundingBox DiskGeometry::compute_bounding_box() const {
    // The bounding box of a disk centered at (0,0) is a square from (-radius,-radius)
    // with width and height equal to 2*radius.
    return { -radius, -radius, 2.0f * radius, 2.0f * radius };
}

arena_polygons_t DiskGeometry::generate_contours(std::size_t n, b2Vec2 position) const {
    if (n == 0) { // Identify best number
        n = 100;
    }

    if (n < 3) n = 3;                                // A disk needs ≥ 3 points
    arena_polygons_t contours(1);
    auto& poly = contours.front();
    poly.reserve(n);

    const float step = 2.0f * M_PI / static_cast<float>(n);
    for (std::size_t i = 0; i < n; ++i) {
        const float a = i * step;
        poly.push_back({position.x + radius * std::cos(a), position.y + radius * std::sin(a)});
        //glogger->info("generate_contours ({},{}) ({},{})", position.x, position.y, radius * std::cos(a), radius * std::sin(a));
    }
    return contours;
}


/************* RectangleGeometry *************/ // {{{1

void RectangleGeometry::create_box2d_shape(b2BodyId body_id, b2ShapeDef& shape_def) {
    float half_width = width / (2.0f * VISUALIZATION_SCALE);
    float half_height = height / (2.0f * VISUALIZATION_SCALE);
    b2Polygon polygon = b2MakeBox(half_width, half_height);

    // Create the polygon shape similarly to how the circle was created.
    shape_id = b2CreatePolygonShape(body_id, &shape_def, &polygon);
    shape_created = true;
}

// Export a boolean grid where each cell is marked true if its center lies within the rectangle.
std::vector<std::vector<bool>> RectangleGeometry::export_geometry_grid(size_t num_bins_x,
                                                                       size_t num_bins_y,
                                                                       float bin_width,
                                                                       float bin_height,
                                                                       float obj_x,
                                                                       float obj_y) const {
    // Initialize a grid with false values.
    std::vector<std::vector<bool>> grid(num_bins_y, std::vector<bool>(num_bins_x, false));

    // Calculate the rectangle boundaries (assumed centered on (obj_x, obj_y)).
    float left   = obj_x - width / 2.0f;
    float right  = obj_x + width / 2.0f;
    float top    = obj_y - height / 2.0f;
    float bottom = obj_y + height / 2.0f;

    // Loop over each bin in the grid.
    for (size_t j = 0; j < num_bins_y; ++j) {
        for (size_t i = 0; i < num_bins_x; ++i) {
            // Determine the center of the current bin.
            float center_x = (i + 0.5f) * bin_width;
            float center_y = (j + 0.5f) * bin_height;
            // Check if the bin center is inside the rectangle boundaries.
            if (center_x >= left && center_x <= right &&
                center_y >= top  && center_y <= bottom) {
                grid[j][i] = true;
            }
        }
    }

    return grid;
}


// The rectangle is drawn as a filled rotated box centered at (x, y).
void RectangleGeometry::render(SDL_Renderer* renderer, [[maybe_unused]] b2WorldId world_id,
                               float x, float y, float theta,
                               uint8_t r, uint8_t g, uint8_t b, uint8_t alpha) const {
    float const hw = 0.5f * width;
    float const hh = 0.5f * height;

    auto rotate_local = [](float lx, float ly, float a) {
        float const c = std::cos(a);
        float const s = std::sin(a);
        return b2Vec2{c * lx - s * ly, s * lx + c * ly};
    };

    // Local corners in mm, CCW
    b2Vec2 const p0 = rotate_local(-hw, -hh, theta);
    b2Vec2 const p1 = rotate_local( hw, -hh, theta);
    b2Vec2 const p2 = rotate_local( hw,  hh, theta);
    b2Vec2 const p3 = rotate_local(-hw,  hh, theta);

    // Convert to pixels and translate to (x, y)
    Sint16 const vx[4] = {
        static_cast<Sint16>(x + p0.x * mm_to_pixels),
        static_cast<Sint16>(x + p1.x * mm_to_pixels),
        static_cast<Sint16>(x + p2.x * mm_to_pixels),
        static_cast<Sint16>(x + p3.x * mm_to_pixels),
    };

    Sint16 const vy[4] = {
        static_cast<Sint16>(y + p0.y * mm_to_pixels),
        static_cast<Sint16>(y + p1.y * mm_to_pixels),
        static_cast<Sint16>(y + p2.y * mm_to_pixels),
        static_cast<Sint16>(y + p3.y * mm_to_pixels),
    };

    // Filled rotated polygon (requires SDL2_gfx)
    filledPolygonRGBA(renderer, vx, vy, 4, r, g, b, alpha);
}


BoundingDisk RectangleGeometry::compute_bounding_disk() const {
    // The bounding disk must cover the entire rectangle.
    // Its radius is half the diagonal of the rectangle.
    float half_width = width / 2.0f;
    float half_height = height / 2.0f;
    float radius = std::sqrt(half_width * half_width + half_height * half_height);
    return { 0.0f, 0.0f, radius };
}

BoundingBox RectangleGeometry::compute_bounding_box() const {
    // The rectangle is its own bounding box (centered at (0,0)).
    return { -width / 2.0f, -height / 2.0f, width, height };
}

arena_polygons_t RectangleGeometry::generate_contours(std::size_t n, b2Vec2 position) const {
    if (n == 0) { // Identify best number
        n = 100;
    }

    /* At least the four corners – distribute the rest along the edges. */
    if (n < 4) n = 4;
    const std::size_t per_edge = n / 4;
    const std::size_t extras   = n % 4;

    const float hw = width  * 0.5f;
    const float hh = height * 0.5f;

    auto edge_points = [per_edge,position](b2Vec2 a, b2Vec2 b) {
        std::vector<b2Vec2> pts;
        pts.reserve(per_edge + 1);
        for (std::size_t i = 0; i < per_edge; ++i) {
            const float t = static_cast<float>(i) / per_edge;
            pts.push_back({position.x + a.x + t * (b.x - a.x), position.y + a.y + t * (b.y - a.y)});
        }
        return pts;
    };

    arena_polygons_t contours(1);
    auto& poly = contours.front();

    /* Counter‑clockwise: left‑top‑right‑bottom edges. */
    const b2Vec2 lt{-hw, -hh}, rt{ hw, -hh}, rb{ hw,  hh}, lb{-hw,  hh};
    auto append = [&](auto&& vec){ poly.insert(poly.end(), vec.begin(), vec.end()); };

    append(edge_points(lt, rt));
    append(edge_points(rt, rb));
    append(edge_points(rb, lb));
    append(edge_points(lb, lt));

    /* Distribute any extra vertices on the first edges */
    for (std::size_t i = 0; i < extras; ++i)
        poly.push_back(poly[i]);

    return contours;
}

/************* TriangleGeometry *************/ // {{{1

std::array<b2Vec2, 3> TriangleGeometry::get_local_vertices() const {
    // Equilateral triangle, centroid at origin.
    // Let h be the height. For an equilateral triangle:
    //   h = sqrt(3)/2 * s
    // The centroid lies at a distance h/3 from the base and 2h/3 from the apex.
    const float s = side_length;
    const float h = std::sqrt(3.0f) * 0.5f * s;

    // CCW order.
    const b2Vec2 v0{0.0f, -2.0f * h / 3.0f};
    const b2Vec2 v1{ s * 0.5f,  h / 3.0f};
    const b2Vec2 v2{-s * 0.5f,  h / 3.0f};
    return {v0, v1, v2};
}

void TriangleGeometry::create_box2d_shape(b2BodyId body_id, b2ShapeDef& shape_def) {
    const auto verts_mm = get_local_vertices();

    // Convert to Box2D world units.
    b2Vec2 verts[3];
    for (int i = 0; i < 3; ++i) {
        verts[i] = {verts_mm[i].x / VISUALIZATION_SCALE, verts_mm[i].y / VISUALIZATION_SCALE};
    }

    b2Polygon polygon{};
    polygon.count = 3;
    polygon.radius = 0.0f;

    // Vertices must be CCW.
    for (int i = 0; i < 3; ++i) {
        polygon.vertices[i] = verts[i];
    }

    auto normalize = [](b2Vec2 v) {
        const float len2 = v.x * v.x + v.y * v.y;
        if (len2 <= 0.0f) return b2Vec2{0.0f, 0.0f};
        const float inv_len = 1.0f / std::sqrt(len2);
        return b2Vec2{v.x * inv_len, v.y * inv_len};
    };

    for (int i = 0; i < 3; ++i) {
        const int j = (i + 1) % 3;
        const b2Vec2 e{polygon.vertices[j].x - polygon.vertices[i].x,
                       polygon.vertices[j].y - polygon.vertices[i].y};
        // Outward normal for CCW polygon.
        polygon.normals[i] = normalize({e.y, -e.x});
    }

    shape_id = b2CreatePolygonShape(body_id, &shape_def, &polygon);
    shape_created = true;
}

static bool point_in_triangle(b2Vec2 p, b2Vec2 a, b2Vec2 b, b2Vec2 c) {
    // Barycentric technique, robust for triangles.
    const b2Vec2 v0{c.x - a.x, c.y - a.y};
    const b2Vec2 v1{b.x - a.x, b.y - a.y};
    const b2Vec2 v2{p.x - a.x, p.y - a.y};

    const float dot00 = v0.x * v0.x + v0.y * v0.y;
    const float dot01 = v0.x * v1.x + v0.y * v1.y;
    const float dot02 = v0.x * v2.x + v0.y * v2.y;
    const float dot11 = v1.x * v1.x + v1.y * v1.y;
    const float dot12 = v1.x * v2.x + v1.y * v2.y;

    const float denom = dot00 * dot11 - dot01 * dot01;
    if (std::abs(denom) < 1e-12f) return false;

    const float inv_denom = 1.0f / denom;
    const float u = (dot11 * dot02 - dot01 * dot12) * inv_denom;
    const float v = (dot00 * dot12 - dot01 * dot02) * inv_denom;

    return (u >= 0.0f) && (v >= 0.0f) && (u + v <= 1.0f);
}

std::vector<std::vector<bool>> TriangleGeometry::export_geometry_grid(size_t num_bins_x,
                                                                     size_t num_bins_y,
                                                                     float bin_width,
                                                                     float bin_height,
                                                                     float obj_x,
                                                                     float obj_y) const {
    std::vector<std::vector<bool>> grid(num_bins_y, std::vector<bool>(num_bins_x, false));

    const auto v = get_local_vertices();
    const b2Vec2 a{obj_x + v[0].x, obj_y + v[0].y};
    const b2Vec2 b{obj_x + v[1].x, obj_y + v[1].y};
    const b2Vec2 c{obj_x + v[2].x, obj_y + v[2].y};

    for (size_t j = 0; j < num_bins_y; ++j) {
        for (size_t i = 0; i < num_bins_x; ++i) {
            const float center_x = (i + 0.5f) * bin_width;
            const float center_y = (j + 0.5f) * bin_height;
            const b2Vec2 p{center_x, center_y};

            if (point_in_triangle(p, a, b, c)) {
                grid[j][i] = true;
            }
        }
    }

    return grid;
}

void TriangleGeometry::render(SDL_Renderer* renderer, b2WorldId world_id, float x, float y, float theta,
                             uint8_t r, uint8_t g, uint8_t b, uint8_t alpha) const {
    (void)world_id;

    auto v = get_local_vertices();
    b2Vec2 v0 = rotate_local(v[0], theta);
    b2Vec2 v1 = rotate_local(v[1], theta);
    b2Vec2 v2 = rotate_local(v[2], theta);

    Sint16 const x1 = static_cast<Sint16>(x + v0.x * mm_to_pixels);
    Sint16 const y1 = static_cast<Sint16>(y + v0.y * mm_to_pixels);
    Sint16 const x2 = static_cast<Sint16>(x + v1.x * mm_to_pixels);
    Sint16 const y2 = static_cast<Sint16>(y + v1.y * mm_to_pixels);
    Sint16 const x3 = static_cast<Sint16>(x + v2.x * mm_to_pixels);
    Sint16 const y3 = static_cast<Sint16>(y + v2.y * mm_to_pixels);

    filledTrigonRGBA(renderer, x1, y1, x2, y2, x3, y3, r, g, b, alpha);
}

BoundingDisk TriangleGeometry::compute_bounding_disk() const {
    // Circumradius for an equilateral triangle: R = s / sqrt(3).
    return {0.0f, 0.0f, side_length / std::sqrt(3.0f)};
}

BoundingBox TriangleGeometry::compute_bounding_box() const {
    const float s = side_length;
    const float h = std::sqrt(3.0f) * 0.5f * s;

    // Local vertices are:
    //  (0, -2h/3), (-s/2, h/3), (s/2, h/3)
    const float min_x = -s * 0.5f;
    const float max_x =  s * 0.5f;
    const float min_y = -2.0f * h / 3.0f;
    const float max_y =  h / 3.0f;

    return {min_x, min_y, max_x - min_x, max_y - min_y};
}

arena_polygons_t TriangleGeometry::generate_contours(std::size_t n, b2Vec2 position) const {
    if (n == 0) {
        n = 3;
    }

    if (n < 3) n = 3;

    const auto v = get_local_vertices();
    const std::size_t per_edge = n / 3;
    const std::size_t extras   = n % 3;

    auto edge_points = [per_edge, position](b2Vec2 a, b2Vec2 b) {
        std::vector<b2Vec2> pts;
        pts.reserve(per_edge + 1);
        for (std::size_t i = 0; i < per_edge; ++i) {
            const float t = static_cast<float>(i) / per_edge;
            pts.push_back({position.x + a.x + t * (b.x - a.x), position.y + a.y + t * (b.y - a.y)});
        }
        return pts;
    };

    arena_polygons_t contours(1);
    auto& poly = contours.front();

    const b2Vec2 a{v[0].x, v[0].y};
    const b2Vec2 b{v[1].x, v[1].y};
    const b2Vec2 c{v[2].x, v[2].y};

    auto append = [&](auto&& vec) { poly.insert(poly.end(), vec.begin(), vec.end()); };

    // CCW.
    append(edge_points(a, b));
    append(edge_points(b, c));
    append(edge_points(c, a));

    for (std::size_t i = 0; i < extras; ++i) {
        poly.push_back(poly[i]);
    }

    return contours;
}



/************* GlobalGeometry *************/ // {{{1

std::vector<std::vector<bool>> GlobalGeometry::export_geometry_grid(size_t num_bins_x,
                                                                    size_t num_bins_y,
                                                                    float /*bin_width*/,
                                                                    float /*bin_height*/,
                                                                    float /*obj_x*/,
                                                                    float /*obj_y*/) const {
    return std::vector<std::vector<bool>>(num_bins_y, std::vector<bool>(num_bins_x, true));
}

BoundingDisk GlobalGeometry::compute_bounding_disk() const {
    return { 0.0f, 0.0f, 0.0f };
}

BoundingBox GlobalGeometry::compute_bounding_box() const {
    return { 0.0f, 0.0f, 0.0f, 0.0f };
}

/************* ArenaGeometry *************/ // {{{1


float ArenaGeometry::distance_point_segment(b2Vec2 p, b2Vec2 a, b2Vec2 b) noexcept {
    const b2Vec2 ab {b.x - a.x, b.y - a.y};
    const float  ab_len2 = ab.x * ab.x + ab.y * ab.y;
    if (ab_len2 == 0.0f) {                    // degenerate segment
        const float dx = p.x - a.x;
        const float dy = p.y - a.y;
        return std::sqrt(dx * dx + dy * dy);
    }

    const b2Vec2 ap {p.x - a.x, p.y - a.y};
    float t = (ap.x * ab.x + ap.y * ab.y) / ab_len2;   // projection factor
    t = std::clamp(t, 0.0f, 1.0f);

    const b2Vec2 closest {a.x + t * ab.x, a.y + t * ab.y};
    const float  dx = p.x - closest.x;
    const float  dy = p.y - closest.y;
    return std::sqrt(dx * dx + dy * dy);
}

/* Ray‑casting, even‑odd rule */
bool ArenaGeometry::point_inside_polygon(b2Vec2 p,
        const std::vector<b2Vec2>& poly) noexcept {
    bool inside = false;
    const std::size_t n = poly.size();
    for (std::size_t i = 0, j = n - 1; i < n; j = i++) {
        const auto& vi = poly[i];
        const auto& vj = poly[j];

        const bool intersect = ((vi.y > p.y) != (vj.y > p.y)) &&
                               (p.x < (vj.x - vi.x) * (p.y - vi.y) / (vj.y - vi.y + 1e-9f) + vi.x);
        if (intersect) inside = !inside;
    }
    return inside;
}

std::vector<std::vector<bool>>
ArenaGeometry::export_geometry_grid(std::size_t num_bins_x,
                                    std::size_t num_bins_y,
                                    float       bin_width,
                                    float       bin_height,
                                    float       obj_x,
                                    float       obj_y) const {
    std::vector<std::vector<bool>> grid(num_bins_y, std::vector<bool>(num_bins_x, false));

    for (std::size_t j = 0; j < num_bins_y; ++j) {
        for (std::size_t i = 0; i < num_bins_x; ++i) {
            const float cx = (i + 0.5f) * bin_width  - obj_x;
            const float cy = (j + 0.5f) * bin_height - obj_y;
            const b2Vec2 p{cx, cy};

            /* Mark the cell if the point is *inside* any polygon */
            for (const auto& poly : arena_polygons_) {
                if (point_inside_polygon(p, poly)) {
                    grid[j][i] = true;
                    break;
                }
            }
        }
    }
    return grid;
}

BoundingBox ArenaGeometry::compute_bounding_box() const {
    float min_x =  std::numeric_limits<float>::max();
    float min_y =  std::numeric_limits<float>::max();
    float max_x = -std::numeric_limits<float>::max();
    float max_y = -std::numeric_limits<float>::max();

    for (const auto& poly : arena_polygons_) {
        for (const auto& v : poly) {
            min_x = std::min(min_x, v.x);
            min_y = std::min(min_y, v.y);
            max_x = std::max(max_x, v.x);
            max_y = std::max(max_y, v.y);
        }
    }
    return {min_x, min_y, max_x - min_x, max_y - min_y};
}

BoundingDisk ArenaGeometry::compute_bounding_disk() const {
    const auto bb   = compute_bounding_box();
    const float cx  = bb.x + bb.width  * 0.5f;
    const float cy  = bb.y + bb.height * 0.5f;
    const float rad = std::sqrt(bb.width * bb.width + bb.height * bb.height) * 0.5f;
    return {cx, cy, rad};
}

float ArenaGeometry::get_distance_to([[maybe_unused]] b2Vec2 orig, b2Vec2 point) const {
    float best = std::numeric_limits<float>::infinity();

    for (const auto& poly : arena_polygons_) {
        const std::size_t n = poly.size();
        if (n < 2) continue;

        /* Walk every segment of the polygon loop */
        for (std::size_t i = 0, j = n - 1; i < n; j = i++) {
            const float d = distance_point_segment(point, poly[j], poly[i]);
            best = std::min(best, d);
        }
    }
    return best;
}

arena_polygons_t ArenaGeometry::generate_contours(std::size_t points, b2Vec2 position) const {
    /* If points ≥ current vertex count we can just return the wall polygons
       unchanged – the caller can decimate if necessary.                     */
    if (arena_polygons_.empty())
        return {};

    arena_polygons_t result;
    result.reserve(arena_polygons_.size());

    for (const auto& src : arena_polygons_) {
        if (src.size() <= points || points == 0) {          // Keep original
            //result.push_back(src);
            std::vector<b2Vec2> dst;
            dst.reserve(src.size());
            for (size_t i = 0; i < src.size(); i++) {
                dst.push_back({position.x + src[i].x, position.y + src[i].y});
            }
            result.push_back(dst);
            continue;
        }

        /* Uniform resampling so every polygon gets exactly @points vertices */
        std::vector<float> edge_len(src.size());
        float perimeter = 0.0f;

        for (std::size_t i = 0, j = src.size() - 1; i < src.size(); j = i++) {
            const auto& a = src[j];
            const auto& b = src[i];
            const float dx = b.x - a.x, dy = b.y - a.y;
            perimeter += edge_len[i] = std::sqrt(dx * dx + dy * dy);
        }

        const float step = perimeter / points;
        std::vector<b2Vec2> dst;
        dst.reserve(points);

        std::size_t  i  = 0,   j = src.size() - 1;
        float        d = 0.0f, next_d = step;

        while (dst.size() < points) {
            const auto& a = src[j];
            const auto& b = src[i];
            const float seg = edge_len[i];

            while (d + seg >= next_d && dst.size() < points) {
                const float t = (next_d - d) / seg;
                dst.push_back({position.x + a.x + t * (b.x - a.x), position.y + a.y + t * (b.y - a.y)});
                next_d += step;
            }
            d += seg;
            j = i++;
        }
        result.emplace_back(std::move(dst));
    }
    return result;
}



ObjectGeometry* object_geometry_factory(Configuration const& config, Simulation* simulation) {
    std::string const geometry_str = to_lowercase(config["geometry"].get(std::string("unknown")));
    if (geometry_str == "global") {
        return new GlobalGeometry();
    } else if (geometry_str == "disk") {
        float const radius = config["radius"].get(10.0);
        return new DiskGeometry(radius);
    } else if (geometry_str == "rectangle") {
        float const body_width = config["body_width"].get(10.0);
        float const body_height = config["body_height"].get(10.0);
        return new RectangleGeometry(body_width, body_height);
    } else if (geometry_str == "triangle") {
        float const side_length = config["side_length"].get(10.0);
        return new TriangleGeometry(side_length);
    } else if (geometry_str == "arena") {
        return new ArenaGeometry(simulation->get_arena_geometry());
    } else {
        throw std::runtime_error("Unknown geometry type '" + geometry_str + "'.");
    }
}

// MODELINE "{{{1
// vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
// vim:foldmethod=marker
