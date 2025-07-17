
#include "utils.h"
#include "objects.h"
#include "robot.h"
#include "distances.h"
#include "simulator.h"

#include <cmath>
#include "SDL2_gfxPrimitives.h"


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


void DiskGeometry::render(SDL_Renderer* renderer, [[maybe_unused]] b2WorldId world_id, float x, float y, uint8_t r, uint8_t g, uint8_t b, uint8_t alpha) const {
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

// The rectangle is drawn as a filled box centered at (x, y) converted to pixels.
void RectangleGeometry::render(SDL_Renderer* renderer, [[maybe_unused]] b2WorldId world_id,
                               float x, float y, uint8_t r, uint8_t g, uint8_t b, uint8_t alpha) const {
    // Calculate the pixel size and position.
    SDL_Rect rect;
    rect.w = static_cast<int>(width * mm_to_pixels);
    rect.h = static_cast<int>(height * mm_to_pixels);
    // Center the rectangle at (x, y) by offsetting by half its width and height.
    rect.x = static_cast<int>(x - rect.w / 2);
    rect.y = static_cast<int>(y - rect.h / 2);

    // Set the drawing color and render the filled rectangle.
    SDL_SetRenderDrawColor(renderer, r, g, b, alpha);
    SDL_RenderFillRect(renderer, &rect);
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



/************* LightLevelMap *************/ // {{{1

// Constructor: Initialize the grid with all light levels set to 0.
LightLevelMap::LightLevelMap(size_t num_bins_x, size_t num_bins_y, float bin_width, float bin_height)
        : num_bins_x_(num_bins_x), num_bins_y_(num_bins_y), bin_width_(bin_width), bin_height_(bin_height) {
    levels_.resize(num_bins_y_);
    for (auto &row : levels_) {
        row.resize(num_bins_x_, 0.0f);
    }
}

// Destructor.
LightLevelMap::~LightLevelMap() {
    // No special cleanup is necessary here.
}

float LightLevelMap::get_light_level_at(float x, float y) const {
    // 1) Early‐out if outside the overall map
    if (x < 0.0f || y < 0.0f)
        return 0.0f;

    // 2) Compute which bin this falls into
    size_t bin_x = static_cast<size_t>(std::floor(x / bin_width_));
    size_t bin_y = static_cast<size_t>(std::floor(y / bin_height_));

    // 3) Check bounds
    if (bin_x >= num_bins_x_ || bin_y >= num_bins_y_)
        return 0.0f;

    // 4) Delegate to your existing getter
    return get_light_level(bin_x, bin_y);
}

// Returns the light level at a specified bin.
float LightLevelMap::get_light_level(size_t bin_x, size_t bin_y) const {
    assert(bin_x < num_bins_x_ && bin_y < num_bins_y_);
    return levels_[bin_y][bin_x];
}

// Sets the light level at a specified bin.
void LightLevelMap::set_light_level(size_t bin_x, size_t bin_y, int16_t value) {
    assert(bin_x < num_bins_x_ && bin_y < num_bins_y_);
    levels_[bin_y][bin_x] = value;
}

// Adds a value to the light level at a specified bin.
void LightLevelMap::add_light_level(size_t bin_x, size_t bin_y, int16_t value) {
    assert(bin_x < num_bins_x_ && bin_y < num_bins_y_);
    if (levels_[bin_y][bin_x] + value < 32767)
        levels_[bin_y][bin_x] += value;
    else
        levels_[bin_y][bin_x] = 32767;
}

// Resets all bins to 0.
void LightLevelMap::clear() {
    for (auto &row : levels_) {
        std::fill(row.begin(), row.end(), 0);
    }
}

// Accessor for the number of bins along the x-axis.
size_t LightLevelMap::get_num_bins_x() const {
    return num_bins_x_;
}

// Accessor for the number of bins along the y-axis.
size_t LightLevelMap::get_num_bins_y() const {
    return num_bins_y_;
}

// Returns the physical width of a bin.
float LightLevelMap::get_bin_width() const {
    return bin_width_;
}

// Returns the physical height of a bin.
float LightLevelMap::get_bin_height() const {
    return bin_height_;
}

void LightLevelMap::render(SDL_Renderer* renderer) const {
    // Iterate over each bin in the grid.
    for (size_t y = 0; y < num_bins_y_; ++y) {
        for (size_t x = 0; x < num_bins_x_; ++x) {
            // Retrieve the current light level value.
            // The value is in the range [-32768, 32767].
            int16_t current_value = levels_[y][x];
            if (current_value < 0)
                current_value = 0;

            // Normalize the current value to a [0, 1] range.
            float normalized = (static_cast<float>(current_value)) / 32768.0f;

            // Map the normalized value into a brightness range [100, 200].
            // You can adjust these constants to get a different brightness range.
            //float scaled = 100.0f + (200.0f - 100.0f) * normalized;
            float scaled = 100.f + (200.0f - 100.0f) * normalized;
            uint8_t brightness = static_cast<uint8_t>(std::round(scaled));

            // Identify object X and Y coordinates in visualization instance
            float screen_x = x * bin_width_;
            float screen_y = y * bin_height_;
            auto const pos = visualization_position(screen_x, screen_y);
            float screen_w = bin_width_ + 1;
            float screen_h = bin_height_ + 1;
            auto const wh = visualization_position(screen_w, screen_h);

            // Create the rectangle representing the bin's position and size.
            SDL_Rect rect;
            rect.x = static_cast<int>(pos.x);
            rect.y = static_cast<int>(pos.y);
            rect.w = static_cast<int>(wh.x + 1);
            rect.h = static_cast<int>(wh.y + 1);

            // Set the drawing color to the computed brightness.
            // Using the same value for red, green, and blue creates a gray color.
            SDL_SetRenderDrawColor(renderer, brightness, brightness, brightness, 255);
            SDL_RenderFillRect(renderer, &rect);
        }
    }
}

void LightLevelMap::register_callback(std::function<void(LightLevelMap&)> cb) {
    callbacks_.emplace_back(std::move(cb));
}

void LightLevelMap::update() {
    // 1) zero out the entire grid
    clear();

    // 2) let every callback “paint” its contribution
    for (auto& cb : callbacks_) {
        cb(*this);
    }
}


/************* OBJECT *************/ // {{{1

Object::Object(float _x, float _y, ObjectGeometry& _geom, std::string const& _category)
        : x(_x), y(_y), category(_category), geom(&_geom) {
    // ...
}

Object::Object(Simulation* simulation, float _x, float _y, Configuration const& config, std::string const& _category)
        : x(_x), y(_y), category(_category) {
    parse_configuration(config, simulation);
}

// XXX : destroy geom ??
Object::~Object() { }

void Object::launch_user_step([[maybe_unused]] float t) {
    // ...
}

void Object::parse_configuration(Configuration const& config, Simulation* simulation) {
    x = config["x"].get(x);
    y = config["y"].get(y);

    // Initialize geometry
    geom = object_geometry_factory(config, simulation); // XXX never destroyed
}

void Object::move(float _x, float _y, float _theta) {
    x = _x;
    y = _y;
    if (!std::isnan(_theta))
        theta = _theta;
}

arena_polygons_t Object::generate_contours(std::size_t points_per_contour) const {
    return geom->generate_contours(points_per_contour, {x, y});
}


/************* StaticLightObject *************/ // {{{1

StaticLightObject::StaticLightObject(float x, float y,
                                     ObjectGeometry& geom, LightLevelMap* lmap,
                                     int16_t _value,
                                     LightMode _mode,
                                     int16_t _edge_value,
                                     float   _gradient_radius,
                                     float   _photo_start_at,
                                     float   _photo_start_duration,
                                     int16_t _photo_start_value,
                                     std::string const& _category)
    : Object(x, y, geom, _category),
      value(_value),
      orig_value(_value),
      edge_value(_edge_value),
      gradient_radius(_gradient_radius),
      mode(_mode),
      light_map(lmap),
      photo_start_at(_photo_start_at),
      photo_start_duration(_photo_start_duration),
      photo_start_value(_photo_start_value) {
    light_map->register_callback([this](LightLevelMap& m){ this->update_light_map(m); });
    //update_light_map();
}

StaticLightObject::StaticLightObject(Simulation* simulation, float _x, float _y,
        LightLevelMap* light_map, Configuration const& config,
        std::string const& _category)
    : Object(simulation, _x, _y, config, _category),
      light_map(light_map) {
    parse_configuration(config, simulation);
    light_map->register_callback([this](LightLevelMap& m){ this->update_light_map(m); });
    //update_light_map();
}


//void StaticLightObject::update_light_map(LightLevelMap& l) {
//    // Retrieve grid parameters from the light map.
//    size_t num_bins_x = l.get_num_bins_x();
//    size_t num_bins_y = l.get_num_bins_y();
//    float bin_width = l.get_bin_width();
//    float bin_height = l.get_bin_height();
//
//    // Use the geometry's export method to get a grid indicating where the geometry exists.
//    std::vector<std::vector<bool>> geometry_grid =
//        geom->export_geometry_grid(num_bins_x, num_bins_y, bin_width, bin_height, x, y);
//
//    // Update each bin in the light map that is covered by this object's geometry.
//    for (size_t j = 0; j < num_bins_y; ++j) {
//        for (size_t i = 0; i < num_bins_x; ++i) {
//            if (geometry_grid[j][i]) {
//                l.add_light_level(i, j, value);
//            }
//        }
//    }
//}

void StaticLightObject::update_light_map(LightLevelMap& l) {
    const size_t nx = l.get_num_bins_x();
    const size_t ny = l.get_num_bins_y();
    const float  bw = l.get_bin_width();
    const float  bh = l.get_bin_height();

    auto geometry_grid = geom->export_geometry_grid(nx, ny, bw, bh, x, y);

    /* ----- optional automatic radius (max distance inside geometry) ------ */
    float effective_radius = gradient_radius;
    if (!performing_photo_start && mode == LightMode::GRADIENT && effective_radius <= 0.0f) {
        for (size_t j = 0; j < ny; ++j) {
            for (size_t i = 0; i < nx; ++i) {
                if (!geometry_grid[j][i]) { continue; }
                const float cx = (i + 0.5f) * bw;
                const float cy = (j + 0.5f) * bh;
                const float dist = std::hypot(cx - x, cy - y);
                effective_radius = std::max(effective_radius, dist);
            }
        }
        /* Degenerate case: single-bin objects */
        if (effective_radius <= 0.0f) { effective_radius = std::max(bw, bh) * 0.5f; }
    }

    /* ----------------------- write contribution -------------------------- */
    for (size_t j = 0; j < ny; ++j) {
        for (size_t i = 0; i < nx; ++i) {
            if (!geometry_grid[j][i]) { continue; }

            int16_t level = value;   // default: STATIC

            if (!performing_photo_start && mode == LightMode::GRADIENT) {
                const float cx     = (i + 0.5f) * bw;
                const float cy     = (j + 0.5f) * bh;
                const float dist   = std::hypot(cx - x, cy - y);
                const float ratio  = std::clamp(dist / effective_radius, 0.f, 1.f);
                level = static_cast<int16_t>(
                        std::lround(value + (edge_value - value) * ratio));
            }
            l.add_light_level(i, j, level);
        }
    }
}

void StaticLightObject::parse_configuration(Configuration const& config, Simulation* simulation) {
    Object::parse_configuration(config, simulation);
    mode = config["light_mode"].get(std::string("static")) == "gradient" ? LightMode::GRADIENT : LightMode::STATIC;
    value = config["value"].get(10);
    orig_value = value;
    edge_value = config["edge_value"].get(0);              // Only used in GRADIENT
    gradient_radius = config["gradient_radius"].get(-1.f); // Only used in GRADIENT
    photo_start_at = config["photo_start_at"].get(-1.0f);
    photo_start_duration = config["photo_start_duration"].get(1.0f);
    photo_start_value = config["photo_start_value"].get(32767);
}

void StaticLightObject::launch_user_step(float t) {
    Object::launch_user_step(t);

    // Check if we should launch photo_start
    if (photo_start_at >= 0 && t >= photo_start_at && t < photo_start_at + photo_start_duration) {
        // Check if we just started photo_start
        if (!performing_photo_start) {
            performing_photo_start = true;
            orig_value = value;
            value = photo_start_value;
            light_map->update();
        }
    } else {
        // Check if we just finished photo_start
        if (performing_photo_start) {
            value = orig_value;
            performing_photo_start = false;
            light_map->update();
        }
    }
}


/************* PhysicalObject *************/ // {{{1

PhysicalObject::PhysicalObject(uint16_t _id, float _x, float _y,
       ObjectGeometry& geom, b2WorldId world_id,
       float _linear_damping, float _angular_damping,
       float _density, float _friction, float _restitution,
       std::string const& _category)
    : Object(_x, _y, geom, _category),
      id(_id),
      linear_damping(_linear_damping),
      angular_damping(_angular_damping),
      density(_density),
      friction(_friction),
      restitution(_restitution) {
    create_body(world_id);
}

PhysicalObject::PhysicalObject(Simulation* simulation, uint16_t _id, float _x, float _y,
       b2WorldId world_id, Configuration const& config,
       std::string const& _category)
    : Object(simulation, _x, _y, config, _category), id(_id) {
    parse_configuration(config, simulation);
    create_body(world_id);
}

void PhysicalObject::launch_user_step(float t) {
    Object::launch_user_step(t);

    // Compute acceleration statistics
    _estimated_dt = t - _last_time;
    _last_time = t;
    // Translational acceleration in world frame
    b2Vec2 now_v = b2Body_GetLinearVelocity(body_id);
    b2Vec2 a_world = (now_v - _prev_v) * (1.0f / _estimated_dt);
    _prev_v = now_v;
    // Specific force (proper accel) → subtract gravity
    b2Vec2 gravity(0.0f, 0.0f); // Same as Box2D world. TODO update
    b2Vec2 f_world = a_world - gravity;
    // Rotate into body frame
    _lin_acc = b2Body_GetLocalVector(body_id, f_world);

}

b2Vec2 PhysicalObject::get_position() const {
    if (b2Body_IsValid(body_id)) {
        return b2Body_GetPosition(body_id);
    } else {
        return {NAN, NAN};
    }
}

float PhysicalObject::get_angle() const {
    if (b2Body_IsValid(body_id)) {
        b2Rot const rotation = b2Body_GetRotation(body_id);
        return std::atan2(rotation.s, rotation.c);
    } else {
        return NAN;
    }
}

float PhysicalObject::get_angular_velocity() const {
    if (b2Body_IsValid(body_id)) {
        return b2Body_GetAngularVelocity(body_id);
    } else {
        return NAN;
    }
}

b2Vec2 PhysicalObject::get_linear_acceleration() const {
    return _lin_acc;
}

void PhysicalObject::parse_configuration(Configuration const& config, Simulation* simulation) {
    Object::parse_configuration(config, simulation);
    linear_damping = config["body_linear_damping"].get(0.0f);
    angular_damping = config["body_angular_damping"].get(0.0f);
    density = config["body_density"].get(10.0f);
    friction = config["body_friction"].get(0.3f);
    restitution = config["body_restitution"].get(0.5f);
}

void PhysicalObject::create_body(b2WorldId world_id) {
    // Create the body definition.
    b2BodyDef bodyDef = b2DefaultBodyDef();
    bodyDef.type = b2_dynamicBody;
    bodyDef.position = { x / VISUALIZATION_SCALE, y / VISUALIZATION_SCALE };
    bodyDef.linearDamping = linear_damping;
    bodyDef.angularDamping = angular_damping;
    bodyDef.isBullet = false;
    body_id = b2CreateBody(world_id, &bodyDef);

    // Set up a shape definition with common physical properties.
    b2ShapeDef shapeDef = b2DefaultShapeDef();
    shapeDef.density = density;
    shapeDef.friction = friction;
    shapeDef.restitution = restitution;
    shapeDef.enablePreSolveEvents = true;

    // Create shape
    geom->create_box2d_shape(body_id, shapeDef);

    // Assign an initial velocity.
    b2Vec2 velocity = { 0.0f, 0.0f };
    b2Body_SetLinearVelocity(body_id, velocity);
    b2Body_SetAngularVelocity(body_id, 0.0f);
    //b2Body_SetAwake(body_id, false);
}

void PhysicalObject::move(float _x, float _y, float _theta) {
    Object::move(_x, _y, _theta);
    if (b2Body_IsValid(body_id)) {
        b2Vec2 position = {_x / VISUALIZATION_SCALE, _y / VISUALIZATION_SCALE};
        b2Rot rotation = {sinf(_theta), cosf(_theta)};
        if (std::isnan(_theta)) {
            rotation = b2Body_GetRotation(body_id);
        }
        b2Body_SetTransform(body_id, position, rotation);
    }
}

arena_polygons_t PhysicalObject::generate_contours(std::size_t points_per_contour) const {
    return geom->generate_contours(points_per_contour, get_position());
}



/************* PassiveObject *************/ // {{{1

PassiveObject::PassiveObject(uint16_t _id, float _x, float _y,
       ObjectGeometry& geom, b2WorldId world_id,
       float _linear_damping, float _angular_damping,
       float _density, float _friction, float _restitution,
       std::string _colormap,
       std::string const& _category)
    : PhysicalObject(_id, _x, _y, geom, world_id,
      _linear_damping, _angular_damping,
      _density, _friction, _restitution, _category),
      colormap(_colormap) {
    // ...
}

PassiveObject::PassiveObject(Simulation* simulation, uint16_t _id, float _x, float _y,
       b2WorldId world_id, Configuration const& config,
       std::string const& _category)
    : PhysicalObject(simulation, _id, _x, _y, world_id, config, _category) {
    parse_configuration(config, simulation);
    // ...
}

void PassiveObject::render(SDL_Renderer* renderer, b2WorldId world_id) const {
    // Get object's position in the physics world
    b2Vec2 body_position = b2Body_GetPosition(body_id);

    // Identify object X and Y coordinates in visualization instance
    float screen_x = body_position.x * VISUALIZATION_SCALE;
    float screen_y = body_position.y * VISUALIZATION_SCALE;
    auto const pos = visualization_position(screen_x, screen_y);

    // Assign color based on object initial position
    uint8_t const value = (static_cast<int32_t>(x) + static_cast<int32_t>(y)) % 256;
    //uint8_t const value = (reinterpret_cast<intptr_t>(this)) % 256;
    uint8_t r, g, b;
    get_cmap_val(colormap, value, &r, &g, &b);

    // Draw the object main body
    geom->render(renderer, world_id, pos.x, pos.y, r, g, b, 255);
}

void PassiveObject::parse_configuration(Configuration const& config, Simulation* simulation) {
    PhysicalObject::parse_configuration(config, simulation);
    colormap = config["colormap"].get(std::string("rainbow"));
}



/************* Factories *************/ // {{{1


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
    } else if (geometry_str == "arena") {
        return new ArenaGeometry(simulation->get_arena_geometry());
    } else {
        throw std::runtime_error("Unknown geometry type '" + geometry_str + "'.");
    }
}

Object* object_factory(Simulation* simulation, uint16_t id, float x, float y, b2WorldId world_id, Configuration const& config, LightLevelMap* light_map, size_t userdatasize, std::string const& category) {
    std::string const type = to_lowercase(config["type"].get(std::string("unknown")));
    Object* res = nullptr;

    if (type == "static_light") {
        res = new StaticLightObject(simulation, x, y, light_map, config, category);

    } else if (type == "passive_object") {
        res = new PassiveObject(simulation, id, x, y, world_id, config, category);

    } else if (type == "pogobot") {
        res = new PogobotObject(simulation, id, x, y, world_id, userdatasize, config, category);

    } else if (type == "pogobject") {
        res = new PogobjectObject(simulation, id, x, y, world_id, userdatasize, config, category);

    } else if (type == "pogowall") {
        res = new Pogowall(simulation, id, x, y, world_id, userdatasize, config, category);

    } else if (type == "membrane") {
        res = new MembraneObject(simulation, id, x, y, world_id, userdatasize, config, category);

    } else {
        throw std::runtime_error("Unknown object type '" + type + "'.");
    }

    return res;
}


void get_cmap_val(std::string const name, uint8_t const value, uint8_t* r, uint8_t* g, uint8_t* b) {
    if (name == "rainbow") {
        rainbow_colormap(value, r, g, b);
    } else if (name == "qualitative") {
        qualitative_colormap(value, r, g, b);
    } else {
        throw std::runtime_error("Unknown colormap '" + name + "'.");
    }
}

// MODELINE "{{{1
// vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
// vim:foldmethod=marker
