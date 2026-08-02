#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <string>
#include <vector>

namespace pogosim::magnetometer {

struct vector3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

struct scalar_range {
    double minimum = 0.0;
    double maximum = 0.0;
};

struct vector3_range {
    scalar_range x;
    scalar_range y;
    scalar_range z;
};

struct domain_interval {
    bool enabled = false;
    double minimum = 0.0;
    double maximum = 0.0;
};

enum class out_of_domain_policy {
    allow,
    clamp,
    throw_error
};

enum class angle_unit {
    radians,
    degrees
};

struct simulation_domain {
    domain_interval x;
    domain_interval y;
    domain_interval theta;
    bool wrap_theta = true;
    out_of_domain_policy policy = out_of_domain_policy::allow;
};

// The field at position (x, y) is:
// base_field + field_gradient_x * (x - origin_x)
//            + field_gradient_y * (y - origin_y).
struct world_field_config {
    vector3 base_field{1.0, 0.0, 0.0};
    vector3 field_gradient_x{};
    vector3 field_gradient_y{};
    double origin_x = 0.0;
    double origin_y = 0.0;
};

struct output_config {
    // A non-positive value disables quantization.
    double quantization_step = 1.0;

    bool clamp_output = false;
    vector3_range output_limits{
        scalar_range{250.0, 600.0},
        scalar_range{250.0, 600.0},
        scalar_range{250.0, 600.0}
    };
};

// One harmonic contributes:
// cos_coefficient * cos(order * relative_angle)
// + sin_coefficient * sin(order * relative_angle).
struct harmonic_parameter_ranges {
    std::size_t order = 2;
    vector3_range cos_coefficient;
    vector3_range sin_coefficient;
};

// The first harmonic describes a rotated ellipse in the raw X/Y plane.
// Z may also vary sinusoidally with heading.
struct ellipse_parameter_ranges {
    vector3_range center{
        scalar_range{425.0, 425.0},
        scalar_range{425.0, 425.0},
        scalar_range{425.0, 425.0}
    };

    scalar_range major_radius{160.0, 160.0};
    scalar_range minor_radius{140.0, 140.0};
    scalar_range ellipse_rotation{0.0, 0.0};
    scalar_range sensor_phase_offset{0.0, 0.0};

    scalar_range z_cos_amplitude{15.0, 15.0};
    scalar_range z_sin_amplitude{-10.0, -10.0};
};

struct robot_variation_config {
    // Added to the fitted/procedural center independently for each robot.
    vector3_range additional_bias{
        scalar_range{0.0, 0.0},
        scalar_range{0.0, 0.0},
        scalar_range{0.0, 0.0}
    };

    // Multiplies the angle-dependent signal, not the center/bias.
    scalar_range signal_gain{1.0, 1.0};

    // Fixed heading offset for an individual sensor installation.
    scalar_range angle_offset{0.0, 0.0};

    // Fixed per-robot noise standard deviation.
    vector3_range noise_stddev{
        scalar_range{3.0, 3.0},
        scalar_range{3.0, 3.0},
        scalar_range{3.0, 3.0}
    };
};

struct procedural_magnetometer_config {
    procedural_magnetometer_config();

    simulation_domain domain;
    world_field_config world_field;
    ellipse_parameter_ranges ellipse;
    std::vector<harmonic_parameter_ranges> distortion_harmonics;
    robot_variation_config robot_variation;
    output_config output;

    // Used to generate stable, robot-specific parameters.
    std::uint64_t random_seed = 0;

    // If true, angle-dependent coefficients scale with the local horizontal
    // field magnitude relative to the field magnitude at the configured origin.
    bool scale_signal_with_field_magnitude = true;
};

struct csv_column_mapping {
    // Used when has_header is true.
    std::string angle_column = "angle";
    std::string magnetometer_x_column = "mag_x";
    std::string magnetometer_y_column = "mag_y";
    std::string magnetometer_z_column = "mag_z";

    // Used when has_header is false.
    std::size_t angle_index = 0;
    std::size_t magnetometer_x_index = 1;
    std::size_t magnetometer_y_index = 2;
    std::size_t magnetometer_z_index = 3;
};

struct csv_magnetometer_config {
    std::string file_path;
    csv_column_mapping columns;
    bool has_header = true;
    char delimiter = ',';
    char comment_prefix = '#';
    angle_unit input_angle_unit = angle_unit::degrees;

    // Order 1 fits an ellipse. Order 2 additionally captures symmetric
    // head/tail distortion. Higher orders can model more complex artifacts.
    std::size_t harmonic_order = 2;
    double ridge_regularization = 1.0e-8;
    bool skip_invalid_rows = true;

    simulation_domain domain;
    world_field_config world_field;
    robot_variation_config robot_variation;
    output_config output;
    std::uint64_t random_seed = 0;
    bool scale_signal_with_field_magnitude = true;

    // The field angle present during calibration. When unset, it is inferred
    // from world_field.base_field.
    std::optional<double> calibration_field_angle;

    // If true, the fit residual standard deviation is added in quadrature to
    // the configured robot noise standard deviation.
    bool add_fit_residual_as_noise = true;
};

struct magnetometer_robot_state {
    double x = 0.0;
    double y = 0.0;
    double theta = 0.0;

    // Runtime effects, such as temperature drift or temporary interference.
    vector3 additional_bias{};
    vector3 additional_noise_stddev{};
    double signal_gain = 1.0;
};

struct magnetometer_robot_profile {
    std::uint64_t robot_id = 0;
    vector3 center{};

    // Index 0 stores order 1, index 1 stores order 2, etc.
    std::vector<vector3> cos_coefficients;
    std::vector<vector3> sin_coefficients;

    double angle_offset = 0.0;
    double signal_gain = 1.0;
    vector3 noise_stddev{};
};

struct csv_fit_statistics {
    std::size_t accepted_rows = 0;
    std::size_t rejected_rows = 0;
    std::size_t harmonic_order = 0;
    vector3 root_mean_square_error{};
};

enum class magnetometer_model_source {
    procedural,
    csv_fit
};

class raw_magnetometer_model {
public:
    static raw_magnetometer_model from_procedural_config(
        const procedural_magnetometer_config& config
    );

    static raw_magnetometer_model from_csv(
        const csv_magnetometer_config& config
    );

    [[nodiscard]] magnetometer_robot_profile create_robot_profile(
        std::uint64_t robot_id
    ) const;

    [[nodiscard]] vector3 sample(
        const magnetometer_robot_state& state,
        const magnetometer_robot_profile& profile,
        std::mt19937_64& random_engine
    ) const;

    // Reproducible and thread-friendly convenience overload. A given
    // (robot_id, sample_index) pair always generates the same noise sequence.
    [[nodiscard]] vector3 sample_deterministic(
        const magnetometer_robot_state& state,
        const magnetometer_robot_profile& profile,
        std::uint64_t sample_index
    ) const;

    [[nodiscard]] vector3 world_field_at(double x, double y) const;
    [[nodiscard]] magnetometer_model_source source() const noexcept;
    [[nodiscard]] const csv_fit_statistics& fit_statistics() const noexcept;
    [[nodiscard]] std::size_t harmonic_order() const noexcept;

private:
    raw_magnetometer_model() = default;

    magnetometer_model_source source_ = magnetometer_model_source::procedural;
    simulation_domain domain_;
    world_field_config world_field_;
    robot_variation_config robot_variation_;
    output_config output_;
    std::uint64_t random_seed_ = 0;
    bool scale_signal_with_field_magnitude_ = true;
    double reference_horizontal_field_magnitude_ = 1.0;

    // Procedural mode stores ranges; CSV mode stores one fitted base profile.
    std::optional<ellipse_parameter_ranges> ellipse_ranges_;
    std::vector<harmonic_parameter_ranges> harmonic_ranges_;
    std::optional<magnetometer_robot_profile> fitted_base_profile_;
    csv_fit_statistics fit_statistics_;
};

[[nodiscard]] world_field_config make_uniform_world_field(
    double magnetic_north_angle,
    double horizontal_magnitude = 1.0,
    double vertical_component = 0.0
);

[[nodiscard]] double degrees_to_radians(double angle_degrees) noexcept;

} // namespace pogosim::magnetometer
