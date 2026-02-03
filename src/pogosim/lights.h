#ifndef LIGHTS_H
#define LIGHTS_H

#include <functional>

#include "utils.h"
#include "configuration.h"
#include "render.h"
#include "colormaps.h"
#include "objects.h"
#include "objects_geometry.h"


class Simulation;


/**
 * @brief A discretized 2D grid representing light intensities over a simulation area.
 */
class LightLevelMap {
public:
    /**
     * @brief Construct a LightLevelMap.
     * @param num_bins_x Number of bins along the x-axis.
     * @param num_bins_y Number of bins along the y-axis.
     * @param bin_width Physical width of each bin.
     * @param bin_height Physical height of each bin.
     */
    LightLevelMap(size_t num_bins_x, size_t num_bins_y, float bin_width, float bin_height);

    /// Destructor.
    ~LightLevelMap();

    /**
     * @brief Get the light level at a physical coordinate (world‐space).
     * @param x  X coordinate in the same units as bin_width_.
     * @param y  Y coordinate in the same units as bin_height_.
     * @return   The light level at the bin containing (x,y), or 0 if outside.
     */
    float get_light_level_at(float x, float y) const;

    /// Returns the light level stored at the given bin (bin_x, bin_y).
    float get_light_level(size_t bin_x, size_t bin_y) const;

    /// Sets the light level at the given bin (bin_x, bin_y).
    void set_light_level(size_t bin_x, size_t bin_y, int16_t value);

    /// Adds a given value to the light level at the given bin (bin_x, bin_y).
    void add_light_level(size_t bin_x, size_t bin_y, int16_t value);

    /// Resets all bins to 0.
    void clear();

    /// Accessors for the grid properties.
    size_t get_num_bins_x() const;
    size_t get_num_bins_y() const;
    float get_bin_width() const;
    float get_bin_height() const;

    /**
     * @brief Renders the light level map to the given SDL_Renderer.
     *
     * This method scales each bin's light level into a brightness value. For each bin, it:
     * - Normalizes the light level from the int16_t range [-32768, 32767] to [0, 1].
     * - Maps that normalized value to a brightness in the range [100, 200].
     * - Renders a filled rectangle with that brightness.
     *
     * @param renderer A pointer to the SDL_Renderer used for drawing.
     */
    void render(SDL_Renderer* renderer) const;

    /// Register a callback which will be called with the map
    /// whenever update() is run.
    void register_callback(std::function<void(LightLevelMap&)> cb);

    /// Clears the map and invokes all registered callbacks.
    void update();

private:
    size_t num_bins_x_;
    size_t num_bins_y_;
    float bin_width_;
    float bin_height_;
    std::vector<std::vector<int16_t>> levels_;
    std::vector<std::function<void(LightLevelMap&)>> callbacks_;
};



/**
 * @class StaticLightObject
 * @brief Light-emitting object with optional radial gradient.
 *
 * @ingroup objects
 *
 * @details
 *  A #StaticLightObject registers a callback on the global
 *  #LightLevelMap given at construction time.  
 *  Once invoked, the callback iterates on every covered bin and calls
 *  #LightLevelMap::add_light_level() with the intensity calculated
 *  according to the current @ref LightMode.
 *
 *  The class is **pod-friendly** (all data members are trivially
 *  copyable / reset-able), yet provides high-level behaviour such as:
 *  - automatic update of the light map when switching in/out of
 *    a photo-start pulse;
 *  - automatic gradient-radius computation if the user passes
 *    `gradient_radius ≤ 0`.
 */
class StaticLightObject : public Object {
public:

    /**
     * @enum LightMode
     * @brief Selects how intensity is distributed over the object geometry.
     *
     * @var LightMode::STATIC
     *  All bins receive an identical intensity = #value.
     *
     * @var LightMode::GRADIENT
     *  Bins receive an intensity that decays **linearly** with the distance
     *  from the geometrical centre (`x`,`y`).  
     *  The intensity at the outer edge of the gradient is #edge_value.
     */
    enum class LightMode { STATIC, GRADIENT, PLANE };

    /**
     * @brief Construct a light object programmatically.
     *
     * @param _x                  Initial *x* coordinate of the object centre
     *                            (in world units, mm).
     * @param _y                  Initial *y* coordinate of the object centre.
     * @param _geom               Reference to the geometry that delimits
     *                            where the object exists in space.
     * @param light_map           Pointer to the global #LightLevelMap
     *                            (must remain valid for the lifetime
     *                            of the object).
     * @param _value              **Centre** intensity in the range \[0, 32767\]
     *                            (for `STATIC` this is also the only intensity).
     * @param _mode               Intensity distribution strategy
     *                            (default =`STATIC`).
     * @param _edge_value         Intensity at @p gradient_radius
     *                            (only meaningful in `GRADIENT` mode).
     * @param _gradient_radius    Radius, in mm, at which the intensity
     *                            reaches @p _edge_value.  
     *                            If ≤ 0 the radius is computed automatically
     *                            as the farthest covered bin centre.
     * @param _photo_start_at     Start time (seconds, simulation clock) of the
     *                            optional photo-start pulse.  
     *                            Set to a negative value to disable.
     * @param _photo_start_duration
     *                            Duration (seconds) of the pulse.
     * @param _photo_start_value  Intensity during the pulse.  After the pulse
     *                            the object reverts to @p _value.
     * @param _category           Category name for profiling / filtering.
     *
     * @note The object automatically registers a callback on
     *       @p light_map; **do not** call #update_light_map() manually.
     */
    StaticLightObject(float _x, float _y,
                      ObjectGeometry& _geom, LightLevelMap* light_map,
                      int16_t _value,
                      LightMode _mode               = LightMode::STATIC,
                      int16_t _edge_value           = 0,
                      float   _gradient_radius      = -1.0f,
                      float   _plane_angle          = 0.0f,
                      float   _plane_half_span      = 1000.0f,
                      float   _photo_start_at       = -1.0f,
                      float   _photo_start_duration = 1.0f,
                      int16_t _photo_start_value    = 32767,
                      std::string const& _category  = "objects");


    /**
     * @brief Constructs a StaticLightObject object from a configuration entry.
     *
     * @param simulation Pointer to the underlying simulation.
     * @param x Initial x-coordinate in the simulation.
     * @param y Initial y-coordinate in the simulation.
     * @param light_map Pointer to the global light level map.
     * @param config Configuration entry describing the object properties.
     * @param category Name of the category of the object.
     */
    StaticLightObject(Simulation* simulation, float _x, float _y,
            LightLevelMap* light_map, Configuration const& config,
            std::string const& _category = "objects");

    /**
     * @brief Renders the object on the given SDL renderer.
     *
     * @param renderer Pointer to the SDL_Renderer.
     * @param world_id The Box2D world identifier (unused in rendering).
     */
    virtual void render(SDL_Renderer*, b2WorldId) const override {}

    /// Updates the object's contribution to the light level map.
    virtual void update_light_map(LightLevelMap& l);

    /**
     * @brief Launches the user-defined step function.
     *
     * Updates the object's time, enables all registered stop watches, executes the user step
     * function via pogo_main_loop_step, and then disables the stop watches.
     */
    virtual void launch_user_step(float t) override;

protected:
    /**
     * @brief Parse a provided configuration and set associated members values.
     *
     * @param config Configuration entry describing the object properties.
     */
    virtual void parse_configuration(Configuration const& config, Simulation* simulation) override;


    /** Centre intensity (also the uniform level for `STATIC` mode). */
    int16_t value = 0;

    /** Saved intensity for restoring after a photo-start pulse. */
    int16_t orig_value = 0;

    /** Intensity at #gradient_radius (GRADIENT and PLANE modes only). */
    int16_t edge_value = 0;

    /**
     * Radius (mm) used for linear fall-off in GRADIENT mode.
     * A non-positive value means “compute automatically”.
     */
    float gradient_radius = -1.0f;

    // Direction of the plane’s normal, *in radians* (0 → +x, π/2 → +y)
    float plane_angle = 0.0f;

    // Half-width (mm) of the transition zone.  
    // The intensity goes from `value` to `edge_value` over a span of 2·plane_half_span.
    float plane_half_span = 1000.0f;

    /** Selected intensity distribution strategy. */
    LightMode mode = LightMode::STATIC;

    /** Pointer to the light map this object contributes to. */
    LightLevelMap* light_map = nullptr;

    // ---------- Photo-start pulse parameters ---------------------------- //
    float   photo_start_at        = -1.0f;  ///< start time (s), negative ⇒ off
    float   photo_start_duration  = 1.0f;   ///< pulse width  (s)
    int16_t photo_start_value     = 32767;  ///< intensity during pulse
    bool    performing_photo_start = false; ///< internal state flag
};

/**
 * @class RotatingRayOfLightObject
 * @brief Single ray of light that sweeps around its centre,
 *  and that becomes visible only after an optional photo-start delay.
 *
 * The object overwrites each covered light-map bin with either
 * `value` (inside the ray) or `0` (outside the ray) via
 * LightLevelMap::set_light_level().
 *
 * @param angular_speed  Sweep speed in rad · s⁻¹ (default = 3).
 * @param ray_half_width Half the angular aperture of the ray in
 *                       radians (default ≈ 5.7 ° = 0.1 rad).
 */
class RotatingRayOfLightObject : public Object {
public:
    RotatingRayOfLightObject(float x, float y, ObjectGeometry& geom,
                             LightLevelMap* light_map, int16_t value,
                             float ray_half_width  = 0.1f,
                             float angular_speed   = 3.0f,
                             float photo_start_at  = -1.0f,
                             float photo_start_dur = 1.0f,
                             float _white_frame_dur = 1.0f,
                             int16_t _white_frame_val = 32767,
                             std::string const& category = "objects");

    RotatingRayOfLightObject(Simulation* simulation, float x, float y,
                             LightLevelMap* light_map,
                             Configuration const& config,
                             std::string const& category = "objects");

    void render(SDL_Renderer*, b2WorldId) const override {}
    void update_light_map(LightLevelMap& l);
    void launch_user_step(float t) override;

protected:
    void parse_configuration(Configuration const& config,
                             Simulation* simulation) override;

    void start_white_frame(float now_s);

private:
    static float normalise_angle(float a);

    /* --- parameters --------------------------------------------------- */
    LightLevelMap* light_map = nullptr;
    int16_t value            = 0;
    float ray_half_width     = 0.1f;   ///< radians
    float angular_speed      = 3.0f;   ///< rad·s⁻¹
    float photo_start_at     = -1.0f;  ///< s, <0 ⇒ disabled
    float photo_start_dur    = 1.0f;   ///< s
    float white_frame_dur    = 0.03f;  ///< s, 0 ⇒ disabled
    int16_t white_frame_val  = 32767;  ///< level inside the white frame

    /* --- evolving state ----------------------------------------------- */
    float  current_angle         = 0.0f;  ///< radians
    bool   ray_is_active         = true;  ///< false until photo-start is over
    float  previous_angle        = 0.f;   ///< rad, for wrap-around test
    bool   white_frame_active    = false;
    float  white_frame_end_time  = 0.f;   ///< s, when to stop white frame
    float  sim_prev_t            = 0.f;
    float  ray_current_t         = 0.f;
};


class AlternatingDualRayOfLightObject : public Object {
public:
    AlternatingDualRayOfLightObject(float x, float y, ObjectGeometry& geom,
                                    LightLevelMap* light_map, int16_t value,
                                    float ray_half_width          = 0.1f,
                                    float angular_speed           = 3.0f,
                                    float long_white_frame_dur    = 1.0f,
                                    float short_white_frame_dur   = 0.03f,
                                    int16_t white_frame_val       = 32767,
                                    std::string const& category   = "objects");

    AlternatingDualRayOfLightObject(Simulation* simulation, float x, float y,
                                    LightLevelMap* light_map,
                                    Configuration const& config,
                                    std::string const& category = "objects");

    void render(SDL_Renderer*, b2WorldId) const override {}
    void launch_user_step(float t) override;
    void update_light_map(LightLevelMap& l);

protected:
    void parse_configuration(Configuration const& config,
                             Simulation* simulation) override;

private:
    /* helpers ---------------------------------------------------------- */
    static float normalise_angle(float a);
    void         recompute_geometry();
    void         request_map_refresh() { light_map->update(); }

    /* phase machine ---------------------------------------------------- */
    enum class phase_t { LONG_WHITE, LEFT_RAY, SHORT_WHITE, RIGHT_RAY };

    void enter_phase(phase_t p, float now_s);

    /* parameters ------------------------------------------------------- */
    LightLevelMap* light_map       = nullptr;
    int16_t         value           = 0;
    float           ray_half_width  = 0.1f;     /* rad  */
    float           angular_speed   = 3.0f;     /* rad·s⁻¹ */
    float           long_white_dur  = 1.0f;     /* s    */
    float           short_white_dur = 0.03f;    /* s    */
    int16_t         white_val       = 32767;

    /* geometry --------------------------------------------------------- */
    BoundingBox bbox {};
    float       cx = 0.f, cy = 0.f;          /* centre of geometry         */
    float       ax_l = 0.f, ay_l = 0.f;      /* top-left  apex             */
    float       ax_r = 0.f, ay_r = 0.f;      /* top-right apex             */
    float       left_a0  = 0.f, left_a1  = 0.f;
    float       right_a0 = 0.f, right_a1 = 0.f;

    /* evolving state --------------------------------------------------- */
    phase_t phase            = phase_t::LONG_WHITE;
    float   phase_start_t    = 0.f;          /* s                           */
    float   prev_t           = 0.f;          /* s                           */
    float   current_angle    = 0.f;          /* rad – of active ray         */
};


#endif

// MODELINE "{{{1
// vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
// vim:foldmethod=marker
