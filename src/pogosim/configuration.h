#ifndef CONFIGURATION_H
#define CONFIGURATION_H

#include "pogosim.h"
#include "geometry.h"

#include <yaml-cpp/yaml.h>
#include <string>
#include <stdexcept>
#include <sstream>
#include <vector>
#include <utility>
#include <type_traits>
#include <cmath>
#include <limits>

/**
 * @brief Convert an \c arena_polygons_t structure into a compact YAML string.
 * @param polygons   Polygons to serialise (outer vector = polygons, inner
 *                   vector = ordered vertices).
 * @return A UTF-8 YAML document (no trailing newline) suitable for Arrow
 *         metadata.
 *
 * @note The generated YAML uses a two-level nested sequence:
 * @code{yaml}
 * - - x: 0.0
 *     y: 0.0
 *   - x: 1.0
 *     y: 0.0
 *   - x: 1.0
 *     y: 1.0
 *   - x: 0.0
 *     y: 1.0
 * - - x: 2.0
 *     y: 2.0
 *   ...
 * @endcode
 */
inline std::string polygons_to_yaml(const arena_polygons_t &polygons) {
    YAML::Node root(YAML::NodeType::Sequence);

    for (const auto &poly : polygons) {
        YAML::Node poly_node(YAML::NodeType::Sequence);
        for (const auto &v : poly) {
            YAML::Node vtx;
            vtx["x"] = v.x;
            vtx["y"] = v.y;
            poly_node.push_back(vtx);
        }
        root.push_back(poly_node);
    }

    YAML::Emitter emitter;
    emitter << root;              // Default = block style, indented
    return {emitter.c_str(), static_cast<std::size_t>(emitter.size())};
}


/**
 * @brief Class for managing hierarchical configuration parameters.
 *
 * The Configuration class wraps a YAML::Node, preserving the nested structure of the configuration.
 * It provides direct access to sub-parts of the configuration via the [] operator and allows iteration
 * over sub-entries.
 */
class Configuration {
public:
    /// Default constructor creates an empty configuration.
    Configuration();

    /// Construct Configuration from an existing YAML::Node.
    explicit Configuration(const YAML::Node &node);

    /// Opt-in type checking follows child lookups; unknown keys remain allowed.
    void enable_validation(bool enabled = true) { validate_ = enabled; }

    /// Check core safety constraints without initializing the simulation or files.
    void validate_simulator() const;

    /// Optional domain constraint for a parameter; inactive in legacy mode.
    void require(bool condition, const std::string& expectation) const;

    /**
     * @brief Loads configuration parameters from a YAML file.
     *
     * @param file_name The path to the YAML configuration file.
     * @throws std::runtime_error if the YAML file cannot be read or parsed.
     */
    void load(const std::string& file_name);

    /**
     * @brief Access a sub-configuration.
     *
     * Returns a Configuration object wrapping the sub-node corresponding to the provided key.
     *
     * @param key The key for the sub-configuration.
     * @return Configuration The sub-configuration.
     */
    Configuration operator[](const std::string& key) const;

    /// Access a nested sub-configuration via a dotted path (supports '\.' to escape dots).
    Configuration at_path(const std::string& dotted_key) const;

    /// Read a value at a dotted path with a default (convenience wrapper around at_path(...).get(...)).
    template<typename T>
    T get_path(const std::string& dotted_key, const T& default_value = T()) const;

    /**
     * @brief Retrieves the configuration value cast to type T.
     *
     * If the current node is defined, attempts to cast it to type T; otherwise returns default_value.
     *
     * @param T The expected type.
     * @param default_value The default value for missing nodes or legacy conversion failures.
     * @throws std::invalid_argument on conversion failures when validation is enabled.
     * @return T The value of the node cast to type T.
     */
    template<typename T>
    T get(const T& default_value = T()) const;

    /**
     * @brief Sets the configuration entry for the given key.
     *
     * If the current node is not a map, it is converted to one.
     *
     * @tparam T The type of the value.
     * @param key The key where the value should be set.
     * @param value The value to set.
     */
    template<typename T>
    void set(const std::string& key, const T& value);

    /**
     * @brief Checks if the current node is defined.
     *
     * @return true if the node exists and is valid; false otherwise.
     */
    bool exists() const;

    /**
     * @brief Provides a summary of the configuration.
     *
     * @return A string representation of the configuration.
     */
    std::string summary() const;

    /**
     * @brief Returns the children (sub-entries) of the current node.
     *
     * If the current node is a map or sequence, returns a vector of pairs where each pair consists of
     * the key (or index as a string) and the corresponding Configuration.
     * If the node is not a container, returns an empty vector.
     *
     * @return std::vector<std::pair<std::string, Configuration>> Vector of key/Configuration pairs.
     */
    std::vector<std::pair<std::string, Configuration>> children() const;

private:
    YAML::Node node_;
    mutable YAML::Node resolved_cache_;  // Cache for resolved hierarchical default
    mutable bool cache_valid_;           // Whether the cache is valid
    bool validate_ = false;
    std::string path_; // Construct diagnostic paths only when validation is enabled.

    Configuration child(const YAML::Node& node, const std::string& key) const;
    [[noreturn]] void invalid(const std::string& expectation) const;

    /**
     * @brief If @p n is a map containing a 'batch_hierarchical_options' map with a
     *        'default' map, return a new node equal to @p n but with that default
     *        merged at the same level and the key removed. Otherwise return @p n.
     */
    static YAML::Node resolve_hierarchical_default(const YAML::Node& n);
    
    /**
     * @brief Get the resolved node for this configuration, using cache if available.
     */
    const YAML::Node& get_resolved() const;

};


namespace detail {

/// compile-time branch for arithmetic targets
template<typename T>
typename std::enable_if<std::is_arithmetic<T>::value, T>::type
numeric_fallback(const YAML::Node& n, const T& default_value) {
    try {
        // integral → go through double, floating-point → long double
        typedef typename std::conditional<
                std::is_integral<T>::value, double, long double>::type tmp_t;

        tmp_t tmp = n.as<tmp_t>();            // may still throw
        return static_cast<T>(tmp);
    } catch (const YAML::Exception&) {
        return default_value;                 // out-of-range, etc.
    }
}

/// branch for *non-arithmetic* targets – just forward the default
template<typename T>
typename std::enable_if<!std::is_arithmetic<T>::value, T>::type
numeric_fallback(const YAML::Node&, const T& default_value) {
    return default_value;
}

// Keep legacy numeric coercions available, but never truncate or perform an
// out-of-range float-to-integer cast in the opt-in validation mode.
template<typename T>
T checked_conversion(const YAML::Node& n) {
    try {
        return n.as<T>();
    } catch (const YAML::Exception&) {
        if constexpr (std::is_integral_v<T>) {
            const long double value = n.as<long double>();
            if constexpr (std::is_same_v<T, bool>) {
                if (value == 0 || value == 1) return value != 0;
            } else {
                // Exclusive power-of-two bound also works when MSVC's long
                // double cannot represent UINT64_MAX exactly.
                const long double upper = std::ldexp(1.0L, std::numeric_limits<T>::digits);
                const long double lower = std::is_signed_v<T> ? -upper : 0;
                if (std::isfinite(value) && std::trunc(value) == value &&
                        value >= lower && value < upper) return static_cast<T>(value);
            }
        }
        throw;
    }
}

template<typename T>
std::string expected_type() {
    if constexpr (std::is_same_v<T, bool>) return "a boolean (or 0/1)";
    if constexpr (std::is_integral_v<T>) return "an integer in the requested type's range";
    if constexpr (std::is_floating_point_v<T>) return "a number in the requested type's range";
    if constexpr (std::is_same_v<T, std::string>) return "a scalar string";
    return "a value of the requested type";
}

} // namespace detail

template<typename T>
T Configuration::get(const T& default_value) const {
    if (!node_) {                      // nothing stored here
        return default_value;
    }

    // Get the cached resolved node at this level
    const YAML::Node& target_ref = get_resolved();
    YAML::Node target = target_ref;  // Copy for subsequent modifications

    /* honour an eventual "default_option" sub-key ---------------- */
    if (target.IsMap()) {
        const YAML::Node opt = target_ref["default_option"]; // Const lookup: no insertion.
        // Do not overwrite the batch definition through YAML's shared aliases.
        if (opt) { target.reset(opt); }
    }

    // Missing/null entries retain their historical default behavior. This is
    // especially important for optional parameters and batch defaults.
    if (validate_ && target && !target.IsNull()) {
        try {
            return detail::checked_conversion<T>(target);
        } catch (const YAML::Exception&) {
            invalid(detail::expected_type<T>());
        }
    }

    /* -------- 1st attempt – let yaml-cpp do the conversion ------ */
    try {
        return target.as<T>();         // success? great, we’re done.
    } catch (const YAML::Exception&) {
        /* -------- 2nd attempt – only if T is numeric ------------ */
        return detail::numeric_fallback(target, default_value);
    }
}

template<typename T>
void Configuration::set(const std::string& key, const T& value) {
    // Ensure the current node is a map; if not, convert it to one.
    if (!node_ || !node_.IsMap()) {
        node_ = YAML::Node(YAML::NodeType::Map);
    }
    node_[key] = value;
    cache_valid_ = false;
}

template<typename T>
T Configuration::get_path(const std::string& dotted_key, const T& default_value) const {
    return at_path(dotted_key).get<T>(default_value);
}


#endif // CONFIGURATION_H

// MODELINE "{{{1
// vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
// vim:foldmethod=marker
