#include "raw_magnetometer_model.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <numbers>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace pogosim::magnetometer {
namespace {

constexpr double minimum_field_magnitude = 1.0e-12;
constexpr double minimum_pivot = 1.0e-14;

[[nodiscard]] double square(double value) {
    return value * value;
}

[[nodiscard]] bool is_finite(double value) {
    return std::isfinite(value);
}

void validate_range(const scalar_range& range, const std::string& name) {
    if (!is_finite(range.minimum) || !is_finite(range.maximum)) {
        throw std::invalid_argument(name + " must contain finite values");
    }
    if (range.minimum > range.maximum) {
        throw std::invalid_argument(name + " minimum is greater than maximum");
    }
}

void validate_vector_range(const vector3_range& range, const std::string& name) {
    validate_range(range.x, name + ".x");
    validate_range(range.y, name + ".y");
    validate_range(range.z, name + ".z");
}

void validate_domain_interval(const domain_interval& interval, const std::string& name) {
    if (!interval.enabled) {
        return;
    }
    if (!is_finite(interval.minimum) || !is_finite(interval.maximum)) {
        throw std::invalid_argument(name + " must contain finite values");
    }
    if (interval.minimum >= interval.maximum) {
        throw std::invalid_argument(name + " minimum must be smaller than maximum");
    }
}

void validate_domain(const simulation_domain& domain) {
    validate_domain_interval(domain.x, "domain.x");
    validate_domain_interval(domain.y, "domain.y");
    validate_domain_interval(domain.theta, "domain.theta");
}

void validate_world_field(const world_field_config& config) {
    const std::array<double, 9> values{
        config.base_field.x,
        config.base_field.y,
        config.base_field.z,
        config.field_gradient_x.x,
        config.field_gradient_x.y,
        config.field_gradient_x.z,
        config.field_gradient_y.x,
        config.field_gradient_y.y,
        config.field_gradient_y.z
    };
    for (double value : values) {
        if (!is_finite(value)) {
            throw std::invalid_argument("world field values must be finite");
        }
    }
    if (!is_finite(config.origin_x) || !is_finite(config.origin_y)) {
        throw std::invalid_argument("world field origin must be finite");
    }
}

void validate_output_config(const output_config& config) {
    if (!is_finite(config.quantization_step)) {
        throw std::invalid_argument("output.quantization_step must be finite");
    }
    validate_vector_range(config.output_limits, "output.output_limits");
}

void validate_robot_variation(const robot_variation_config& config) {
    validate_vector_range(config.additional_bias, "robot_variation.additional_bias");
    validate_range(config.signal_gain, "robot_variation.signal_gain");
    validate_range(config.angle_offset, "robot_variation.angle_offset");
    validate_vector_range(config.noise_stddev, "robot_variation.noise_stddev");

    if (config.noise_stddev.x.minimum < 0.0 ||
        config.noise_stddev.y.minimum < 0.0 ||
        config.noise_stddev.z.minimum < 0.0) {
        throw std::invalid_argument("noise standard deviations cannot be negative");
    }
}

void validate_ellipse(const ellipse_parameter_ranges& ellipse) {
    validate_vector_range(ellipse.center, "ellipse.center");
    validate_range(ellipse.major_radius, "ellipse.major_radius");
    validate_range(ellipse.minor_radius, "ellipse.minor_radius");
    validate_range(ellipse.ellipse_rotation, "ellipse.ellipse_rotation");
    validate_range(ellipse.sensor_phase_offset, "ellipse.sensor_phase_offset");
    validate_range(ellipse.z_cos_amplitude, "ellipse.z_cos_amplitude");
    validate_range(ellipse.z_sin_amplitude, "ellipse.z_sin_amplitude");

    if (ellipse.major_radius.minimum < 0.0 || ellipse.minor_radius.minimum < 0.0) {
        throw std::invalid_argument("ellipse radii cannot be negative");
    }
}

void validate_harmonics(const std::vector<harmonic_parameter_ranges>& harmonics) {
    for (std::size_t index = 0; index < harmonics.size(); ++index) {
        const auto& harmonic = harmonics[index];
        if (harmonic.order < 2) {
            throw std::invalid_argument(
                "distortion harmonic order must be at least 2 at index " +
                std::to_string(index)
            );
        }
        validate_vector_range(
            harmonic.cos_coefficient,
            "distortion_harmonics[" + std::to_string(index) + "].cos_coefficient"
        );
        validate_vector_range(
            harmonic.sin_coefficient,
            "distortion_harmonics[" + std::to_string(index) + "].sin_coefficient"
        );
    }
}

[[nodiscard]] std::uint64_t splitmix64(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

[[nodiscard]] double sample_uniform(const scalar_range& range, std::mt19937_64& engine) {
    if (range.minimum == range.maximum) {
        return range.minimum;
    }
    std::uniform_real_distribution<double> distribution(range.minimum, range.maximum);
    return distribution(engine);
}

[[nodiscard]] vector3 sample_uniform(
    const vector3_range& range,
    std::mt19937_64& engine
) {
    return {
        sample_uniform(range.x, engine),
        sample_uniform(range.y, engine),
        sample_uniform(range.z, engine)
    };
}

[[nodiscard]] vector3 add(const vector3& left, const vector3& right) {
    return {left.x + right.x, left.y + right.y, left.z + right.z};
}


[[nodiscard]] double horizontal_magnitude(const vector3& value) {
    return std::hypot(value.x, value.y);
}

[[nodiscard]] double clamp_value(double value, const scalar_range& range) {
    return std::clamp(value, range.minimum, range.maximum);
}

[[nodiscard]] double apply_interval(
    double value,
    const domain_interval& interval,
    out_of_domain_policy policy,
    const std::string& name
) {
    if (!interval.enabled || policy == out_of_domain_policy::allow) {
        return value;
    }
    if (value >= interval.minimum && value <= interval.maximum) {
        return value;
    }
    if (policy == out_of_domain_policy::clamp) {
        return std::clamp(value, interval.minimum, interval.maximum);
    }
    throw std::out_of_range(name + " is outside the configured simulation domain");
}

[[nodiscard]] double wrap_angle(double angle, const domain_interval& interval) {
    const double width = interval.maximum - interval.minimum;
    double wrapped = std::fmod(angle - interval.minimum, width);
    if (wrapped < 0.0) {
        wrapped += width;
    }
    return interval.minimum + wrapped;
}

[[nodiscard]] magnetometer_robot_state apply_domain(
    const magnetometer_robot_state& state,
    const simulation_domain& domain
) {
    magnetometer_robot_state result = state;
    result.x = apply_interval(result.x, domain.x, domain.policy, "robot x");
    result.y = apply_interval(result.y, domain.y, domain.policy, "robot y");

    if (domain.theta.enabled && domain.wrap_theta) {
        result.theta = wrap_angle(result.theta, domain.theta);
    } else {
        result.theta = apply_interval(
            result.theta,
            domain.theta,
            domain.policy,
            "robot theta"
        );
    }
    return result;
}

[[nodiscard]] double quantize(double value, double step) {
    if (step <= 0.0) {
        return value;
    }
    return std::round(value / step) * step;
}

[[nodiscard]] double gaussian_noise(double standard_deviation, std::mt19937_64& engine) {
    if (standard_deviation <= 0.0) {
        return 0.0;
    }
    std::normal_distribution<double> distribution(0.0, standard_deviation);
    return distribution(engine);
}

[[nodiscard]] std::vector<std::string> parse_csv_line(
    const std::string& line,
    char delimiter
) {
    std::vector<std::string> fields;
    std::string current;
    bool inside_quotes = false;

    for (std::size_t index = 0; index < line.size(); ++index) {
        const char character = line[index];
        if (character == '"') {
            if (inside_quotes && index + 1 < line.size() && line[index + 1] == '"') {
                current.push_back('"');
                ++index;
            } else {
                inside_quotes = !inside_quotes;
            }
        } else if (character == delimiter && !inside_quotes) {
            fields.push_back(current);
            current.clear();
        } else {
            current.push_back(character);
        }
    }

    if (inside_quotes) {
        throw std::runtime_error("unterminated quoted CSV field");
    }

    fields.push_back(current);
    return fields;
}

[[nodiscard]] std::string trim_copy(const std::string& value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

[[nodiscard]] bool parse_double(const std::string& text, double& result) {
    try {
        const std::string trimmed = trim_copy(text);
        if (trimmed.empty()) {
            return false;
        }
        std::size_t consumed = 0;
        result = std::stod(trimmed, &consumed);
        return consumed == trimmed.size() && is_finite(result);
    } catch (const std::exception&) {
        return false;
    }
}

[[nodiscard]] std::size_t find_header_index(
    const std::unordered_map<std::string, std::size_t>& header_indices,
    const std::string& name
) {
    const auto iterator = header_indices.find(name);
    if (iterator == header_indices.end()) {
        throw std::runtime_error("CSV column not found: " + name);
    }
    return iterator->second;
}

[[nodiscard]] std::vector<double> basis_for_angle(
    double relative_angle,
    std::size_t harmonic_order
) {
    std::vector<double> basis(1 + 2 * harmonic_order, 0.0);
    basis[0] = 1.0;
    for (std::size_t order = 1; order <= harmonic_order; ++order) {
        const double phase = static_cast<double>(order) * relative_angle;
        basis[2 * order - 1] = std::cos(phase);
        basis[2 * order] = std::sin(phase);
    }
    return basis;
}

[[nodiscard]] std::vector<double> solve_linear_system(
    std::vector<std::vector<double>> matrix,
    std::vector<double> right_hand_side
) {
    const std::size_t size = right_hand_side.size();
    if (matrix.size() != size) {
        throw std::invalid_argument("linear system has inconsistent dimensions");
    }

    for (std::size_t pivot_index = 0; pivot_index < size; ++pivot_index) {
        std::size_t best_row = pivot_index;
        double best_value = std::abs(matrix[pivot_index][pivot_index]);
        for (std::size_t row = pivot_index + 1; row < size; ++row) {
            const double candidate = std::abs(matrix[row][pivot_index]);
            if (candidate > best_value) {
                best_value = candidate;
                best_row = row;
            }
        }

        if (best_value < minimum_pivot) {
            throw std::runtime_error(
                "CSV fit is singular; provide more angle coverage or increase ridge_regularization"
            );
        }

        if (best_row != pivot_index) {
            std::swap(matrix[best_row], matrix[pivot_index]);
            std::swap(right_hand_side[best_row], right_hand_side[pivot_index]);
        }

        const double pivot = matrix[pivot_index][pivot_index];
        for (std::size_t column = pivot_index; column < size; ++column) {
            matrix[pivot_index][column] /= pivot;
        }
        right_hand_side[pivot_index] /= pivot;

        for (std::size_t row = 0; row < size; ++row) {
            if (row == pivot_index) {
                continue;
            }
            const double factor = matrix[row][pivot_index];
            if (factor == 0.0) {
                continue;
            }
            for (std::size_t column = pivot_index; column < size; ++column) {
                matrix[row][column] -= factor * matrix[pivot_index][column];
            }
            right_hand_side[row] -= factor * right_hand_side[pivot_index];
        }
    }

    return right_hand_side;
}

struct csv_row {
    double relative_angle = 0.0;
    vector3 measurement{};
};

struct fitted_data {
    magnetometer_robot_profile profile;
    csv_fit_statistics statistics;
};

[[nodiscard]] fitted_data fit_csv_data(const csv_magnetometer_config& config) {
    std::ifstream input(config.file_path);
    if (!input) {
        throw std::runtime_error("cannot open magnetometer CSV file: " + config.file_path);
    }

    std::size_t angle_index = config.columns.angle_index;
    std::size_t x_index = config.columns.magnetometer_x_index;
    std::size_t y_index = config.columns.magnetometer_y_index;
    std::size_t z_index = config.columns.magnetometer_z_index;

    std::string line;
    std::size_t line_number = 0;
    if (config.has_header) {
        while (std::getline(input, line)) {
            ++line_number;
            const std::string trimmed = trim_copy(line);
            if (trimmed.empty() || trimmed.front() == config.comment_prefix) {
                continue;
            }

            auto header = parse_csv_line(line, config.delimiter);
            if (!header.empty() && header.front().size() >= 3 &&
                static_cast<unsigned char>(header.front()[0]) == 0xefU &&
                static_cast<unsigned char>(header.front()[1]) == 0xbbU &&
                static_cast<unsigned char>(header.front()[2]) == 0xbfU) {
                header.front().erase(0, 3);
            }

            std::unordered_map<std::string, std::size_t> header_indices;
            for (std::size_t index = 0; index < header.size(); ++index) {
                header_indices.emplace(trim_copy(header[index]), index);
            }

            angle_index = find_header_index(header_indices, config.columns.angle_column);
            x_index = find_header_index(header_indices, config.columns.magnetometer_x_column);
            y_index = find_header_index(header_indices, config.columns.magnetometer_y_column);
            z_index = find_header_index(header_indices, config.columns.magnetometer_z_column);
            break;
        }
    }

    const double calibration_field_angle = config.calibration_field_angle.value_or(
        std::atan2(config.world_field.base_field.y, config.world_field.base_field.x)
    );

    std::vector<csv_row> rows;
    std::size_t rejected_rows = 0;
    const std::size_t largest_index = std::max({angle_index, x_index, y_index, z_index});

    while (std::getline(input, line)) {
        ++line_number;
        const std::string trimmed = trim_copy(line);
        if (trimmed.empty() || trimmed.front() == config.comment_prefix) {
            continue;
        }

        try {
            const auto fields = parse_csv_line(line, config.delimiter);
            if (fields.size() <= largest_index) {
                throw std::runtime_error("not enough columns");
            }

            double angle = 0.0;
            vector3 measurement;
            const bool valid =
                parse_double(fields[angle_index], angle) &&
                parse_double(fields[x_index], measurement.x) &&
                parse_double(fields[y_index], measurement.y) &&
                parse_double(fields[z_index], measurement.z);

            if (!valid) {
                throw std::runtime_error("invalid numeric value");
            }

            if (config.input_angle_unit == angle_unit::degrees) {
                angle = degrees_to_radians(angle);
            }

            rows.push_back({calibration_field_angle - angle, measurement});
        } catch (const std::exception& exception) {
            ++rejected_rows;
            if (!config.skip_invalid_rows) {
                throw std::runtime_error(
                    "invalid CSV row " + std::to_string(line_number) + ": " + exception.what()
                );
            }
        }
    }

    const std::size_t parameter_count = 1 + 2 * config.harmonic_order;
    if (rows.size() < parameter_count) {
        throw std::runtime_error(
            "not enough valid CSV rows for harmonic order " +
            std::to_string(config.harmonic_order) +
            "; need at least " + std::to_string(parameter_count)
        );
    }

    std::vector<std::vector<double>> normal_matrix(
        parameter_count,
        std::vector<double>(parameter_count, 0.0)
    );
    std::array<std::vector<double>, 3> normal_rhs{
        std::vector<double>(parameter_count, 0.0),
        std::vector<double>(parameter_count, 0.0),
        std::vector<double>(parameter_count, 0.0)
    };

    for (const auto& row : rows) {
        const auto basis = basis_for_angle(row.relative_angle, config.harmonic_order);
        const std::array<double, 3> measurements{
            row.measurement.x,
            row.measurement.y,
            row.measurement.z
        };

        for (std::size_t row_index = 0; row_index < parameter_count; ++row_index) {
            for (std::size_t column_index = 0; column_index < parameter_count; ++column_index) {
                normal_matrix[row_index][column_index] +=
                    basis[row_index] * basis[column_index];
            }
            for (std::size_t axis = 0; axis < 3; ++axis) {
                normal_rhs[axis][row_index] += basis[row_index] * measurements[axis];
            }
        }
    }

    for (std::size_t index = 1; index < parameter_count; ++index) {
        normal_matrix[index][index] += config.ridge_regularization;
    }

    const auto beta_x = solve_linear_system(normal_matrix, normal_rhs[0]);
    const auto beta_y = solve_linear_system(normal_matrix, normal_rhs[1]);
    const auto beta_z = solve_linear_system(normal_matrix, normal_rhs[2]);

    magnetometer_robot_profile profile;
    profile.center = {beta_x[0], beta_y[0], beta_z[0]};
    profile.cos_coefficients.resize(config.harmonic_order);
    profile.sin_coefficients.resize(config.harmonic_order);

    for (std::size_t order = 1; order <= config.harmonic_order; ++order) {
        profile.cos_coefficients[order - 1] = {
            beta_x[2 * order - 1],
            beta_y[2 * order - 1],
            beta_z[2 * order - 1]
        };
        profile.sin_coefficients[order - 1] = {
            beta_x[2 * order],
            beta_y[2 * order],
            beta_z[2 * order]
        };
    }

    vector3 sum_squared_error{};
    for (const auto& row : rows) {
        const auto basis = basis_for_angle(row.relative_angle, config.harmonic_order);
        vector3 prediction{beta_x[0], beta_y[0], beta_z[0]};
        for (std::size_t order = 1; order <= config.harmonic_order; ++order) {
            prediction.x += beta_x[2 * order - 1] * basis[2 * order - 1] +
                beta_x[2 * order] * basis[2 * order];
            prediction.y += beta_y[2 * order - 1] * basis[2 * order - 1] +
                beta_y[2 * order] * basis[2 * order];
            prediction.z += beta_z[2 * order - 1] * basis[2 * order - 1] +
                beta_z[2 * order] * basis[2 * order];
        }
        sum_squared_error.x += square(prediction.x - row.measurement.x);
        sum_squared_error.y += square(prediction.y - row.measurement.y);
        sum_squared_error.z += square(prediction.z - row.measurement.z);
    }

    csv_fit_statistics statistics;
    statistics.accepted_rows = rows.size();
    statistics.rejected_rows = rejected_rows;
    statistics.harmonic_order = config.harmonic_order;
    statistics.root_mean_square_error = {
        std::sqrt(sum_squared_error.x / static_cast<double>(rows.size())),
        std::sqrt(sum_squared_error.y / static_cast<double>(rows.size())),
        std::sqrt(sum_squared_error.z / static_cast<double>(rows.size()))
    };

    return {std::move(profile), statistics};
}

[[nodiscard]] vector3 evaluate_signal(
    const magnetometer_robot_profile& profile,
    double relative_angle
) {
    vector3 signal{};
    const std::size_t order_count = std::min(
        profile.cos_coefficients.size(),
        profile.sin_coefficients.size()
    );

    for (std::size_t index = 0; index < order_count; ++index) {
        const double phase = static_cast<double>(index + 1) * relative_angle;
        const double cosine = std::cos(phase);
        const double sine = std::sin(phase);
        signal.x += profile.cos_coefficients[index].x * cosine +
            profile.sin_coefficients[index].x * sine;
        signal.y += profile.cos_coefficients[index].y * cosine +
            profile.sin_coefficients[index].y * sine;
        signal.z += profile.cos_coefficients[index].z * cosine +
            profile.sin_coefficients[index].z * sine;
    }

    return signal;
}

} // namespace

procedural_magnetometer_config::procedural_magnetometer_config() {
    harmonic_parameter_ranges second_harmonic;
    second_harmonic.order = 2;
    second_harmonic.cos_coefficient = {
        scalar_range{12.0, 12.0},
        scalar_range{-6.0, -6.0},
        scalar_range{0.0, 0.0}
    };
    second_harmonic.sin_coefficient = {
        scalar_range{0.0, 0.0},
        scalar_range{4.0, 4.0},
        scalar_range{0.0, 0.0}
    };
    distortion_harmonics.push_back(second_harmonic);
}

raw_magnetometer_model raw_magnetometer_model::from_procedural_config(
    const procedural_magnetometer_config& config
) {
    validate_domain(config.domain);
    validate_world_field(config.world_field);
    validate_ellipse(config.ellipse);
    validate_harmonics(config.distortion_harmonics);
    validate_robot_variation(config.robot_variation);
    validate_output_config(config.output);

    raw_magnetometer_model model;
    model.source_ = magnetometer_model_source::procedural;
    model.domain_ = config.domain;
    model.world_field_ = config.world_field;
    model.robot_variation_ = config.robot_variation;
    model.output_ = config.output;
    model.random_seed_ = config.random_seed;
    model.scale_signal_with_field_magnitude_ = config.scale_signal_with_field_magnitude;
    model.reference_horizontal_field_magnitude_ = std::max(
        horizontal_magnitude(config.world_field.base_field),
        minimum_field_magnitude
    );
    model.ellipse_ranges_ = config.ellipse;
    model.harmonic_ranges_ = config.distortion_harmonics;
    return model;
}

raw_magnetometer_model raw_magnetometer_model::from_csv(
    const csv_magnetometer_config& config
) {
    if (config.file_path.empty()) {
        throw std::invalid_argument("csv file_path cannot be empty");
    }
    if (config.harmonic_order == 0) {
        throw std::invalid_argument("csv harmonic_order must be at least 1");
    }
    if (!is_finite(config.ridge_regularization) || config.ridge_regularization < 0.0) {
        throw std::invalid_argument("ridge_regularization must be finite and non-negative");
    }

    validate_domain(config.domain);
    validate_world_field(config.world_field);
    validate_robot_variation(config.robot_variation);
    validate_output_config(config.output);

    fitted_data fitted = fit_csv_data(config);

    raw_magnetometer_model model;
    model.source_ = magnetometer_model_source::csv_fit;
    model.domain_ = config.domain;
    model.world_field_ = config.world_field;
    model.robot_variation_ = config.robot_variation;
    model.output_ = config.output;
    model.random_seed_ = config.random_seed;
    model.scale_signal_with_field_magnitude_ = config.scale_signal_with_field_magnitude;
    model.reference_horizontal_field_magnitude_ = std::max(
        horizontal_magnitude(config.world_field.base_field),
        minimum_field_magnitude
    );
    model.fitted_base_profile_ = std::move(fitted.profile);
    model.fit_statistics_ = fitted.statistics;

    if (config.add_fit_residual_as_noise) {
        const vector3 rmse = model.fit_statistics_.root_mean_square_error;
        auto add_rmse_to_range = [](scalar_range& range, double residual) {
            range.minimum = std::hypot(range.minimum, residual);
            range.maximum = std::hypot(range.maximum, residual);
        };
        add_rmse_to_range(model.robot_variation_.noise_stddev.x, rmse.x);
        add_rmse_to_range(model.robot_variation_.noise_stddev.y, rmse.y);
        add_rmse_to_range(model.robot_variation_.noise_stddev.z, rmse.z);
    }

    return model;
}

magnetometer_robot_profile raw_magnetometer_model::create_robot_profile(
    std::uint64_t robot_id
) const {
    std::mt19937_64 engine(splitmix64(random_seed_ ^ splitmix64(robot_id)));

    magnetometer_robot_profile profile;
    profile.robot_id = robot_id;

    if (source_ == magnetometer_model_source::procedural) {
        if (!ellipse_ranges_) {
            throw std::logic_error("procedural model has no ellipse parameters");
        }

        const auto& ellipse = *ellipse_ranges_;
        profile.center = sample_uniform(ellipse.center, engine);

        const double major_radius = sample_uniform(ellipse.major_radius, engine);
        const double minor_radius = sample_uniform(ellipse.minor_radius, engine);
        const double ellipse_rotation = sample_uniform(ellipse.ellipse_rotation, engine);
        profile.angle_offset = sample_uniform(ellipse.sensor_phase_offset, engine);

        profile.cos_coefficients.resize(1);
        profile.sin_coefficients.resize(1);

        profile.cos_coefficients[0] = {
            major_radius * std::cos(ellipse_rotation),
            major_radius * std::sin(ellipse_rotation),
            sample_uniform(ellipse.z_cos_amplitude, engine)
        };
        profile.sin_coefficients[0] = {
            -minor_radius * std::sin(ellipse_rotation),
            minor_radius * std::cos(ellipse_rotation),
            sample_uniform(ellipse.z_sin_amplitude, engine)
        };

        std::size_t maximum_order = 1;
        for (const auto& harmonic : harmonic_ranges_) {
            maximum_order = std::max(maximum_order, harmonic.order);
        }
        profile.cos_coefficients.resize(maximum_order);
        profile.sin_coefficients.resize(maximum_order);

        for (const auto& harmonic : harmonic_ranges_) {
            profile.cos_coefficients[harmonic.order - 1] =
                sample_uniform(harmonic.cos_coefficient, engine);
            profile.sin_coefficients[harmonic.order - 1] =
                sample_uniform(harmonic.sin_coefficient, engine);
        }
    } else {
        if (!fitted_base_profile_) {
            throw std::logic_error("CSV model has no fitted base profile");
        }
        profile = *fitted_base_profile_;
        profile.robot_id = robot_id;
        profile.angle_offset = 0.0;
    }

    profile.center = add(profile.center, sample_uniform(robot_variation_.additional_bias, engine));
    profile.signal_gain = sample_uniform(robot_variation_.signal_gain, engine);
    profile.angle_offset += sample_uniform(robot_variation_.angle_offset, engine);
    profile.noise_stddev = sample_uniform(robot_variation_.noise_stddev, engine);
    return profile;
}

vector3 raw_magnetometer_model::sample(
    const magnetometer_robot_state& state,
    const magnetometer_robot_profile& profile,
    std::mt19937_64& random_engine
) const {
    const magnetometer_robot_state bounded_state = apply_domain(state, domain_);
    const vector3 field = world_field_at(bounded_state.x, bounded_state.y);
    const double field_angle = std::atan2(field.y, field.x);
    const double field_magnitude = horizontal_magnitude(field);
    const double relative_angle = field_angle - bounded_state.theta + profile.angle_offset;

    double field_scale = 1.0;
    if (scale_signal_with_field_magnitude_) {
        field_scale = field_magnitude / reference_horizontal_field_magnitude_;
    }

    const vector3 signal = evaluate_signal(profile, relative_angle);
    const double combined_gain = profile.signal_gain * bounded_state.signal_gain * field_scale;

    vector3 result{
        profile.center.x + bounded_state.additional_bias.x + combined_gain * signal.x,
        profile.center.y + bounded_state.additional_bias.y + combined_gain * signal.y,
        profile.center.z + bounded_state.additional_bias.z + combined_gain * signal.z
    };

    const vector3 total_noise_stddev{
        std::hypot(profile.noise_stddev.x, bounded_state.additional_noise_stddev.x),
        std::hypot(profile.noise_stddev.y, bounded_state.additional_noise_stddev.y),
        std::hypot(profile.noise_stddev.z, bounded_state.additional_noise_stddev.z)
    };

    result.x += gaussian_noise(total_noise_stddev.x, random_engine);
    result.y += gaussian_noise(total_noise_stddev.y, random_engine);
    result.z += gaussian_noise(total_noise_stddev.z, random_engine);

    result.x = quantize(result.x, output_.quantization_step);
    result.y = quantize(result.y, output_.quantization_step);
    result.z = quantize(result.z, output_.quantization_step);

    if (output_.clamp_output) {
        result.x = clamp_value(result.x, output_.output_limits.x);
        result.y = clamp_value(result.y, output_.output_limits.y);
        result.z = clamp_value(result.z, output_.output_limits.z);
    }

    return result;
}

vector3 raw_magnetometer_model::sample_deterministic(
    const magnetometer_robot_state& state,
    const magnetometer_robot_profile& profile,
    std::uint64_t sample_index
) const {
    const std::uint64_t seed = splitmix64(
        random_seed_ ^ splitmix64(profile.robot_id) ^ splitmix64(sample_index)
    );
    std::mt19937_64 engine(seed);
    return sample(state, profile, engine);
}

vector3 raw_magnetometer_model::world_field_at(double x, double y) const {
    const double dx = x - world_field_.origin_x;
    const double dy = y - world_field_.origin_y;
    return {
        world_field_.base_field.x + world_field_.field_gradient_x.x * dx +
            world_field_.field_gradient_y.x * dy,
        world_field_.base_field.y + world_field_.field_gradient_x.y * dx +
            world_field_.field_gradient_y.y * dy,
        world_field_.base_field.z + world_field_.field_gradient_x.z * dx +
            world_field_.field_gradient_y.z * dy
    };
}

magnetometer_model_source raw_magnetometer_model::source() const noexcept {
    return source_;
}

const csv_fit_statistics& raw_magnetometer_model::fit_statistics() const noexcept {
    return fit_statistics_;
}

std::size_t raw_magnetometer_model::harmonic_order() const noexcept {
    if (source_ == magnetometer_model_source::csv_fit && fitted_base_profile_) {
        return fitted_base_profile_->cos_coefficients.size();
    }

    std::size_t result = ellipse_ranges_ ? 1 : 0;
    for (const auto& harmonic : harmonic_ranges_) {
        result = std::max(result, harmonic.order);
    }
    return result;
}

world_field_config make_uniform_world_field(
    double magnetic_north_angle,
    double horizontal_magnitude_value,
    double vertical_component
) {
    if (!is_finite(magnetic_north_angle) ||
        !is_finite(horizontal_magnitude_value) ||
        !is_finite(vertical_component)) {
        throw std::invalid_argument("uniform world field parameters must be finite");
    }
    if (horizontal_magnitude_value < 0.0) {
        throw std::invalid_argument("horizontal field magnitude cannot be negative");
    }

    world_field_config result;
    result.base_field = {
        horizontal_magnitude_value * std::cos(magnetic_north_angle),
        horizontal_magnitude_value * std::sin(magnetic_north_angle),
        vertical_component
    };
    return result;
}

double degrees_to_radians(double angle_degrees) noexcept {
    return angle_degrees * std::numbers::pi / 180.0;
}

} // namespace pogosim::magnetometer
