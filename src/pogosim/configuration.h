#ifndef CONFIGURATION_H
#define CONFIGURATION_H

#include "pogosim.h"

#include <yaml-cpp/yaml.h>
#include <string>
#include <stdexcept>
#include <sstream>
#include <vector>
#include <utility>
#include <type_traits>

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
    Configuration(const YAML::Node &node);

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

    /**
     * @brief Retrieves the configuration value cast to type T.
     *
     * If the current node is defined, attempts to cast it to type T; otherwise returns default_value.
     *
     * @param T The expected type.
     * @param default_value The default value to return if the node is not defined or conversion fails.
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
    }
    catch (const YAML::Exception&) {
        return default_value;                 // out-of-range, etc.
    }
}

/// branch for *non-arithmetic* targets – just forward the default
template<typename T>
typename std::enable_if<!std::is_arithmetic<T>::value, T>::type
numeric_fallback(const YAML::Node&, const T& default_value) {
    return default_value;
}

} // namespace detail

template<typename T>
T Configuration::get(const T& default_value) const {
    if (!node_) {                      // nothing stored here
        return default_value;
    }

    /* honour an eventual "default_option" sub-key ---------------- */
    YAML::Node target = node_;
    if (target.IsMap()) {
        YAML::Node opt = target["default_option"];
        if (opt) { target = opt; }
    }

    /* -------- 1st attempt – let yaml-cpp do the conversion ------ */
    try {
        return target.as<T>();         // success? great, we’re done.
    }
    catch (const YAML::Exception&) {
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
}


#endif // CONFIGURATION_H

// MODELINE "{{{1
// vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
// vim:foldmethod=marker
