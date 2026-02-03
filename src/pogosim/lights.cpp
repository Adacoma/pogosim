
#include "utils.h"
#include "robot.h"
#include "distances.h"
#include "simulator.h"
#include "lights.h"
#include "objects.h"

#include <cmath>
#include "SDL2_gfxPrimitives.h"




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
        return get_light_level(0, 0);

    // 2) Compute which bin this falls into
    size_t bin_x = static_cast<size_t>(std::floor(x / bin_width_));
    size_t bin_y = static_cast<size_t>(std::floor(y / bin_height_));

    // 3) Check bounds
    if (bin_x >= num_bins_x_ || bin_y >= num_bins_y_)
        return get_light_level(num_bins_x_-1, num_bins_y_-1);

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


/************* StaticLightObject *************/ // {{{1

StaticLightObject::StaticLightObject(float x, float y,
                                     ObjectGeometry& geom, LightLevelMap* lmap,
                                     int16_t _value,
                                     LightMode _mode,
                                     int16_t _edge_value,
                                     float   _gradient_radius,
                                     float   _plane_angle,
                                     float   _plane_half_span,
                                     float   _photo_start_at,
                                     float   _photo_start_duration,
                                     int16_t _photo_start_value,
                                     std::string const& _category)
    : Object(x, y, geom, _category),
      value(_value),
      orig_value(_value),
      edge_value(_edge_value),
      gradient_radius(_gradient_radius),
      plane_angle(_plane_angle),
      plane_half_span(_plane_half_span),
      mode(_mode),
      light_map(lmap),
      photo_start_at(_photo_start_at),
      photo_start_duration(_photo_start_duration),
      photo_start_value(_photo_start_value) {
    if (photo_start_at >= 0) {
        value = 0.0f;
    }
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

    // Automatic radius (max distance inside geometry)
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
        // Degenerate case: single-bin objects
        if (effective_radius <= 0.0f) { effective_radius = std::max(bw, bh) * 0.5f; }
    }

    if (!performing_photo_start && mode == LightMode::PLANE) {
        // Pre–compute the unit normal once
        const float nx_plane = std::cos(plane_angle);
        const float ny_plane = std::sin(plane_angle);
        int16_t level = value;

        for (size_t j = 0; j < ny; ++j) {
            for (size_t i = 0; i < nx; ++i) {
                if (!geometry_grid[j][i]) { continue; }

                const float cx   = (i + 0.5f) * bw;
                const float cy   = (j + 0.5f) * bh;

                // Signed distance of the bin centre to the plane origin (x,y)
                const float proj = (cx - x) * nx_plane + (cy - y) * ny_plane;

                // Normalised position in [0,1] inside the transition zone
                const float ratio = std::clamp(
                    (proj / plane_half_span + 1.f) * 0.5f, 0.f, 1.f);

                level = static_cast<int16_t>(
                    std::lround(value + (edge_value - value) * ratio));
                l.add_light_level(i, j, level);
            }
        }
        return;
    }

    // Write contribution
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
    auto mode_str = config["light_mode"].get(std::string("static"));
    if (mode_str == "static") {
        mode = LightMode::STATIC;
    } else if (mode_str == "gradient") {
        mode = LightMode::GRADIENT;
    } else if (mode_str == "plane") {
        mode = LightMode::PLANE;
    } else {
        glogger->warn("Unknown light_mode: '{}'. Use either 'static', 'gradient' or 'plane'. Using 'static' by default.", mode_str);
        mode = LightMode::STATIC;
    }
    value = config["value"].get(10);
    orig_value = value;
    edge_value = config["edge_value"].get(0);               // Only used in GRADIENT and PLANE
    gradient_radius = config["gradient_radius"].get(-1.f);  // Only used in GRADIENT
    plane_angle     = config["plane_angle"].get(0.f);       // Only used in PLANE
    plane_half_span = config["plane_half_span"].get(1000.f);// Only used in PLANE
    photo_start_at  = config["photo_start_at"].get(-1.0f);
    photo_start_duration = config["photo_start_duration"].get(1.0f);
    photo_start_value = config["photo_start_value"].get(32767);

    if (photo_start_at >= 0) {
        value = 0.0f;
    }
}

void StaticLightObject::launch_user_step(float t) {
    Object::launch_user_step(t);

    // Check if we should launch photo_start
    if (photo_start_at >= 0 && t >= photo_start_at && t < photo_start_at + photo_start_duration) {
        // Check if we just started photo_start
        if (!performing_photo_start) {
            performing_photo_start = true;
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


/************* RotatingRayOfLightObject *************/ // {{{1

RotatingRayOfLightObject::RotatingRayOfLightObject(float x, float y,
        ObjectGeometry& geom, LightLevelMap* lmap, int16_t _value,
        float _ray_half_width, float _angular_speed,
        float _photo_start_at, float _photo_start_dur,
        float _white_frame_dur, int16_t _white_frame_val,
        std::string const& category)
    : Object(x, y, geom, category),
      light_map(lmap),
      value(_value),
      ray_half_width(_ray_half_width),
      angular_speed(_angular_speed),
      photo_start_at(_photo_start_at),
      photo_start_dur(_photo_start_dur),
      white_frame_dur(_white_frame_dur),
      white_frame_val(_white_frame_val),
      ray_is_active(_photo_start_at < 0.f),
      white_frame_active(_white_frame_dur > 0.f) {
    light_map->register_callback([this](LightLevelMap& m){ this->update_light_map(m); });
    if (ray_is_active) {
        previous_angle = 0.f;          // Makes wrap test work later
        start_white_frame(0.f);        // First flash at t = 0 s
    }
}

RotatingRayOfLightObject::RotatingRayOfLightObject(Simulation* simulation,
        float x, float y, LightLevelMap* lmap,
        Configuration const& config, std::string const& category)
    : Object(simulation, x, y, config, category),
      light_map(lmap) {
    parse_configuration(config, simulation);
    ray_is_active = (photo_start_at < 0.f);
    white_frame_active = (white_frame_dur > 0.f);
    light_map->register_callback([this](LightLevelMap& m){ this->update_light_map(m); });
    if (ray_is_active) {
        previous_angle = 0.f;          // Makes wrap test work later
        start_white_frame(0.f);        // First flash at t = 0 s
    }
}

void RotatingRayOfLightObject::parse_configuration(Configuration const& config,
                                                   Simulation* simulation) {
    Object::parse_configuration(config, simulation);
    value            = config["value"].get(10);
    ray_half_width   = config["ray_half_width"].get(0.1f);
    angular_speed    = config["angular_speed"].get(3.0f);
    photo_start_at   = config["photo_start_at"].get(-1.0f);
    photo_start_dur  = config["photo_start_duration"].get(1.0f);
    white_frame_dur  = config["white_frame_duration"].get(0.03f);
    white_frame_val  = config["white_frame_value"].get(32767);
}

float RotatingRayOfLightObject::normalise_angle(float a) {
    while (a <= -M_PI) a += 2.f * M_PI;
    while (a >   M_PI) a -= 2.f * M_PI;
    return a;
}

void RotatingRayOfLightObject::start_white_frame(float now_s) {
    if (white_frame_dur <= 0.f) return;
    white_frame_active   = true;
    white_frame_end_time = now_s + white_frame_dur;
    light_map->update();
}

void RotatingRayOfLightObject::update_light_map(LightLevelMap& l) {
    const size_t nx = l.get_num_bins_x();
    const size_t ny = l.get_num_bins_y();
    const float  bw = l.get_bin_width();
    const float  bh = l.get_bin_height();

    auto bdisk      = geom->compute_bounding_disk();
    auto geom_grid  = geom->export_geometry_grid(nx, ny, bw, bh, x, y);

    for (size_t j = 0; j < ny; ++j) {
        for (size_t i = 0; i < nx; ++i) {
            if (!geom_grid[j][i]) continue;

            int16_t level = 0;

            if (white_frame_active) {
                level = white_frame_val;                 // flood-fill
            } else if (ray_is_active) {
                const float cx   = (i + 0.5f) * bw - bdisk.center_x;
                const float cy   = (j + 0.5f) * bh - bdisk.center_y;
                const float ang  = std::atan2(cy - y, cx - x);
                const float diff = std::fabs(normalise_angle(ang - current_angle));
                level = (diff <= ray_half_width) ? value : 0;
            }

            if (level > 0)
                l.set_light_level(i, j, level);
        }
    }
}

void RotatingRayOfLightObject::launch_user_step(float t) {
    Object::launch_user_step(t);

    // Manage photo-start window
    bool new_state = ray_is_active;
    if (photo_start_at >= 0.f) {
        new_state = (t >= photo_start_at + photo_start_dur);
    }
    if (new_state != ray_is_active) {
        ray_is_active = new_state;
        if (ray_is_active) {           // Just became active → flash
            previous_angle = 0.f;
            start_white_frame(t);
        } else {
            white_frame_active = false;
            light_map->update();
        }
    }

    if (ray_is_active && !white_frame_active) {
        float const delta_t = t - sim_prev_t;
        ray_current_t += delta_t;
    }
    sim_prev_t = t;

    // Update angle only when active
    float new_angle = std::fmod(angular_speed * ray_current_t, 2.f * M_PI);
    if (ray_is_active && white_frame_dur > 0.f) {
        bool wrapped = (new_angle < previous_angle);     // 2π → 0 crossing
        if (wrapped) {
            white_frame_active   = true;
            white_frame_end_time = t + white_frame_dur;
            light_map->update();
        }
        if (white_frame_active && t >= white_frame_end_time) {
            white_frame_active = false;
            light_map->update();
        }
    }
    previous_angle = new_angle;

    // Advance the ray only when visible
    if (ray_is_active && !white_frame_active) {
        current_angle = normalise_angle(new_angle);
        light_map->update();
    }
}


/************* AlternatingDualRayOfLightObject *************/ // {{{1

float AlternatingDualRayOfLightObject::normalise_angle(float a) {
    while (a <= -M_PI) a += 2.f * M_PI;
    while (a >   M_PI) a -= 2.f * M_PI;
    return a;
}


void AlternatingDualRayOfLightObject::recompute_geometry() {
    bbox = geom->compute_bounding_box();

    /* centre of bounding box */
    cx = bbox.x + 0.5f * bbox.width;
    cy = bbox.y + 0.5f * bbox.height;

    /* apexes: top-left & top-right                                                         *
     * NB: if the engine Y-axis points downwards, swap +height ↔ 0 as needed.               */
    ax_l = bbox.x;
    ay_l = bbox.y + bbox.height;
    ax_r = bbox.x + bbox.width;
    ay_r = ay_l;

    auto baseline_l = std::atan2(cy - ay_l, cx - ax_l);
    auto baseline_r = std::atan2(cy - ay_r, cx - ax_r);

    /* each ray sweeps its π/2 quadrant centred on the baseline direction */
    left_a0   = baseline_l - M_PI_4;      /* start angle                   */
    left_a1   = baseline_l + M_PI_4;      /* end   angle                   */
    right_a0  = baseline_r - M_PI_4;
    right_a1  = baseline_r + M_PI_4;
}


AlternatingDualRayOfLightObject::AlternatingDualRayOfLightObject(
        float x, float y, ObjectGeometry& g, LightLevelMap* lm, int16_t _val,
        float _ray_hw, float _ang_speed, float _long_dur, float _short_dur,
        int16_t _white_val, std::string const& category)
    : Object(x, y, g, category),
      light_map(lm),
      value(_val),
      ray_half_width(_ray_hw),
      angular_speed(_ang_speed),
      long_white_dur(_long_dur),
      short_white_dur(_short_dur),
      white_val(_white_val) {

    recompute_geometry();
    light_map->register_callback(
        [this](LightLevelMap& m){ update_light_map(m); });

    /* begin with the long white frame */
    phase_start_t = 0.f;
    request_map_refresh();
}

AlternatingDualRayOfLightObject::AlternatingDualRayOfLightObject(
        Simulation* simulation, float x, float y, LightLevelMap* lm,
        Configuration const& cfg, std::string const& category)
    : Object(simulation, x, y, cfg, category),
      light_map(lm) {

    parse_configuration(cfg, simulation);
    recompute_geometry();
    light_map->register_callback(
        [this](LightLevelMap& m){ update_light_map(m); });

    phase_start_t = 0.f;
    request_map_refresh();
}

void AlternatingDualRayOfLightObject::parse_configuration(
        Configuration const& cfg, Simulation* simulation) {
    Object::parse_configuration(cfg, simulation);

    value            = cfg["value"].get(10);
    ray_half_width   = cfg["ray_half_width"].get(0.1f);
    angular_speed    = cfg["angular_speed"].get(3.f);
    long_white_dur   = cfg["long_white_frame_duration"].get(1.f);
    short_white_dur  = cfg["short_white_frame_duration"].get(0.03f);
    white_val        = cfg["white_frame_value"].get(32767);
}


void AlternatingDualRayOfLightObject::enter_phase(phase_t p, float now_s) {
    phase          = p;
    phase_start_t  = now_s;

    switch (phase) {
    case phase_t::LEFT_RAY:  current_angle = left_a0;   break;
    case phase_t::RIGHT_RAY: current_angle = right_a0;  break;
    default: break;
    }
    request_map_refresh();
}


void AlternatingDualRayOfLightObject::launch_user_step(float t) {
    Object::launch_user_step(t);
    float const dt = t - prev_t;
    prev_t = t;

    switch (phase) {
    /* ─────── long white flash ─────── */
    case phase_t::LONG_WHITE:
        if (t - phase_start_t >= long_white_dur)
            enter_phase(phase_t::LEFT_RAY, t);
        break;

    /* ─────── left ray sweeping ────── */
    case phase_t::LEFT_RAY: {
        current_angle += angular_speed * dt;
        if (current_angle >= left_a1)
            enter_phase(phase_t::SHORT_WHITE, t);
        else
            request_map_refresh();
        break;
    }

    /* ─────── short white flash ────── */
    case phase_t::SHORT_WHITE:
        if (t - phase_start_t >= short_white_dur)
            enter_phase(phase_t::RIGHT_RAY, t);
        break;

    /* ─────── right ray sweeping ───── */
    case phase_t::RIGHT_RAY: {
        current_angle += angular_speed * dt;
        if (current_angle >= right_a1)
            enter_phase(phase_t::LONG_WHITE, t);
        else
            request_map_refresh();
        break;
    }
    }
}


void AlternatingDualRayOfLightObject::update_light_map(LightLevelMap& l) {
    const size_t nx = l.get_num_bins_x();
    const size_t ny = l.get_num_bins_y();
    const float  bw = l.get_bin_width();
    const float  bh = l.get_bin_height();

    auto geom_grid = geom->export_geometry_grid(nx, ny, bw, bh, x, y);

    /* pre-compute which apex / mode is active */
    bool  white_mode   = (phase == phase_t::LONG_WHITE ||
                          phase == phase_t::SHORT_WHITE);
    bool  use_left_ray = (phase == phase_t::LEFT_RAY);

    float ax = use_left_ray ? ax_l : ax_r;
    float ay = use_left_ray ? ay_l : ay_r;

    for (size_t j = 0; j < ny; ++j) {
        for (size_t i = 0; i < nx; ++i) {
            if (!geom_grid[j][i]) continue;

            int16_t level = 0;

            if (white_mode) {
                level = white_val;                                   /* flood */
            } else {
                /* cell centre in world coordinates */
                float px = (i + 0.5f) * bw;
                float py = (j + 0.5f) * bh;

                float ang = std::atan2(py - ay, px - ax);
                float diff = std::fabs(normalise_angle(ang - current_angle));
                if (diff <= ray_half_width) level = value;
            }

            if (level > 0) l.set_light_level(i, j, level);
        }
    }
}


// MODELINE "{{{1
// vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
// vim:foldmethod=marker
