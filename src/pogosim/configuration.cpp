
#include "configuration.h"
#include <cctype>


Configuration::Configuration() : node_(YAML::Node()), cache_valid_(false) {}

Configuration::Configuration(const YAML::Node &node) : node_(node), cache_valid_(false) {}

const YAML::Node& Configuration::get_resolved() const {
    if (!cache_valid_) {
        // Reset the handle rather than assigning through a cached YAML alias.
        resolved_cache_.reset(resolve_hierarchical_default(node_));
        cache_valid_ = true;
    }
    return resolved_cache_;
}

void Configuration::load(const std::string& file_name) {
    try {
        node_ = YAML::LoadFile(file_name);
        cache_valid_ = false;
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("Error reading YAML file: " + std::string(e.what()));
    }
}

Configuration Configuration::operator[](const std::string& key) const {
    // Use cached resolved node
    const YAML::Node& self = get_resolved();
    if (validate_ && exists()) require(self.IsMap(), "a mapping");
    if (self && self[key]) {
        return child(self[key], key);
    }
    return child(YAML::Node(), key);
}

Configuration Configuration::child(const YAML::Node& node, const std::string& key) const {
    Configuration result(node);
    result.validate_ = validate_;
    if (validate_) result.path_ = path_.empty() ? key : path_ + "." + key;
    return result;
}

void Configuration::invalid(const std::string& expectation) const {
    std::string message = "Invalid configuration '" + (path_.empty() ? std::string("<root>") : path_) + "': expected " + expectation;
    const auto mark = node_.Mark();
    if (!mark.is_null()) message += " (line " + std::to_string(mark.line + 1) + ")";
    throw std::invalid_argument(message);
}

void Configuration::require(bool condition, const std::string& expectation) const {
    if (validate_ && !condition) invalid(expectation);
}


bool Configuration::exists() const {
    return node_ && !node_.IsNull();
}

std::string Configuration::summary() const {
    std::ostringstream oss;
    oss << node_;
    return oss.str();
}

std::vector<std::pair<std::string, Configuration>> Configuration::children() const {
    std::vector<std::pair<std::string, Configuration>> result;
    const YAML::Node& self = get_resolved();
    if (!self || !(self.IsMap() || self.IsSequence())) {
        return result;
    }
    if (self.IsMap()) {
        for (auto it = self.begin(); it != self.end(); ++it) {
            std::string key = it->first.as<std::string>();
            result.push_back({ key, child(it->second, key) });
        }
    } else if (self.IsSequence()) {
        for (std::size_t i = 0; i < self.size(); ++i) {
            result.push_back({ std::to_string(i), child(self[i], std::to_string(i)) });
        }
    }
    return result;
}


Configuration Configuration::at_path(const std::string& dotted_key) const {
    if (!node_) return child(YAML::Node(), dotted_key);

    auto has_unescaped_dot = [](const std::string& s) {
        bool esc = false;
        for (char ch : s) {
            if (esc) { esc = false; continue; }
            if (ch == '\\') { esc = true; continue; }
            if (ch == '.') { return true; }
        }
        return false;
    };

    // If no unescaped dot, just do a plain safe lookup
    if (!has_unescaped_dot(dotted_key)) {
        return (*this)[dotted_key];
    }

    // Start from cached resolved view
    const YAML::Node& start = get_resolved();

    // Exact key with dots wins (const lookup; no insertion)
    if (start.IsMap()) {
        YAML::Node exact = start[dotted_key]; // const operator[] called here
        if (exact) return child(exact, dotted_key);
    }

    auto split_dotted = [](const std::string& s) {
        std::vector<std::string> parts;
        std::string cur;
        cur.reserve(s.size());
        bool esc = false;
        for (char ch : s) {
            if (esc) { cur.push_back(ch); esc = false; continue; }
            if (ch == '\\') { esc = true; continue; }
            if (ch == '.') { parts.push_back(cur); cur.clear(); continue; }
            cur.push_back(ch);
        }
        parts.push_back(cur);
        return parts;
    };

    auto is_int_str = [](const std::string& s) -> bool {
        if (s.empty()) return false;
        std::size_t i = (s[0] == '-' || s[0] == '+') ? 1 : 0;
        if (i >= s.size()) return false;
        for (; i < s.size(); ++i) {
            if (!std::isdigit(static_cast<unsigned char>(s[i]))) return false;
        }
        return true;
    };

    // IMPORTANT: keep traversal on a *const* node to avoid insertions
    const YAML::Node* cur = &start;
    YAML::Node cur_val; // holds the last retrieved value to return at end
    for (const std::string& part : split_dotted(dotted_key)) {
        if (!*cur) return child(YAML::Node(), dotted_key);

        if (cur->IsMap()) {
            // const operator[] (no insertion)
            // YAML assignment mutates aliases; traversal must only rebind.
            cur_val.reset((*cur)[part]);
            if (!cur_val) return child(YAML::Node(), dotted_key);
            // Resolve hierarchical default at the *new* level before continuing
            cur_val.reset(resolve_hierarchical_default(cur_val));
            cur = &cur_val;
        } else if (cur->IsSequence() && is_int_str(part)) {
            long long idx = 0;
            try { idx = std::stoll(part); } catch (...) { return child(YAML::Node(), dotted_key); }
            if (idx < 0 || static_cast<std::size_t>(idx) >= cur->size()) return child(YAML::Node(), dotted_key);
            cur_val.reset((*cur)[static_cast<std::size_t>(idx)]); // const operator[]
            // If the sequence element itself is a map with hierarchical options,
            // resolve before descending further
            cur_val.reset(resolve_hierarchical_default(cur_val));
            cur = &cur_val;
        } else {
            if (validate_ && !cur->IsNull()) child(*cur, dotted_key).invalid("a mapping or sequence along the path");
            return child(YAML::Node(), dotted_key);
        }
    }

    return child(cur_val, dotted_key);
}

void Configuration::validate_simulator() const {
    // This is deliberately a small set of core hazards, not a closed schema.
    // New parameters are automatically type-checked at their existing get<T>()
    // call sites in strict runs; controller-specific validation stays there.
    Configuration root(*this);
    root.enable_validation();
    root.require(root.get_resolved().IsMap(), "a configuration mapping");
    auto mapping = [](const Configuration& value) {
        if (value.exists()) value.require(value.get_resolved().IsMap(), "a mapping");
    };
    auto present = [](const Configuration& value) {
        const YAML::Node& node = value.get_resolved();
        if (!node || node.IsNull()) return false;
        // A null batch default is absent too: get<T>(default) will use its
        // ordinary default, not zero merely because this wrapper is a map.
        if (node.IsMap()) {
            const YAML::Node option = node["default_option"];
            if (option) return !option.IsNull();
        }
        return true;
    };
    auto nonnegative = [&](const Configuration& parent, const char* key, bool positive = false, bool allow_infinity = false) {
        const auto value = parent[key];
        if (!present(value)) return;
        const float number = value.get<float>();
        value.require((std::isfinite(number) || (allow_infinity && number > 0)) &&
                (positive ? number > 0 : number >= 0),
                positive ? "a finite positive number" : "a nonnegative number within the allowed range");
    };
    auto positive_int = [&](const Configuration& parent, const char* key, int maximum = (std::numeric_limits<int>::max)()) {
        const auto value = parent[key];
        if (!present(value)) return;
        const int number = value.get<int>();
        value.require(number > 0 && number <= maximum, "a positive integer no greater than " + std::to_string(maximum));
    };
    for (const char* key : {"time_step", "arena_surface", "GUI_speed_up", "chessboard_distance_between_neighbors"}) nonnegative(root, key, true);
    for (const char* key : {"simulation_time", "formation_min_space_between_neighbors"}) nonnegative(root, key);
    nonnegative(root, "formation_max_space_between_neighbors", false, true);
    const auto minimum = root["formation_min_space_between_neighbors"], maximum = root["formation_max_space_between_neighbors"];
    if (present(minimum) && present(maximum)) maximum.require(maximum.get<float>() >= minimum.get<float>(), "formation_max_space_between_neighbors >= formation_min_space_between_neighbors");
    // The progress tick count is stored in uint32_t even in headless runs.
    const double ticks = std::ceil(static_cast<double>(root["simulation_time"].get(100.0f)) / root["time_step"].get(0.01f));
    root["simulation_time"].require(std::isfinite(ticks) && ticks <= (std::numeric_limits<uint32_t>::max)(), "simulation_time / time_step within the uint32 tick-count range");
    for (const char* key : {"light_map_nb_bin_x", "light_map_nb_bin_y", "data_logger_flush_row_count"}) positive_int(root, key);
    for (const char* key : {"window_width", "window_height"}) positive_int(root, key, (std::numeric_limits<uint16_t>::max)());
    for (const char* key : {"formation_attempts_per_point", "formation_max_restarts"}) {
        const auto value = root[key];
        if (present(value)) value.require(value.get<unsigned>() > 0, "a positive integer");
    }
    const auto stack = root["coroutine_stack_size"];
    if (present(stack)) stack.require(stack.get<std::size_t>() > 0, "a positive integer");
    root["seed"].get<uint32_t>();
    for (const char* key : {"GUI", "enable_data_logging", "enable_console_logging", "delete_old_files", "progress_bar", "communication_ignore_occlusions"}) root[key].get<bool>();
    for (const char* key : {"data_filename", "frames_name", "console_filename", "log_format", "arena_file"}) root[key].get<std::string>();
    for (const char* key : {"data_logger_fields", "data_logger_categories"}) root[key].get<std::vector<std::string>>();
    // Negative/zero output periods are supported disabling sentinels.
    for (const char* key : {"save_data_period", "save_video_period"}) {
        const auto value = root[key];
        if (present(value)) value.require(std::isfinite(value.get<float>()), "a finite output period");
    }
    const auto boundary = root["boundary_condition"];
    if (present(boundary)) {
        const auto name = boundary.get<std::string>();
        boundary.require(name == "solid" || name == "periodic" || name == "null", "solid or periodic");
    }
    const auto flash = root["flash_state"];
    mapping(flash);
    flash["input_file"].get<std::string>();
    flash["output_file"].get<std::string>();
    flash["create_if_missing"].get<bool>();
    mapping(root["parameters"]);
    const auto objects = root["objects"];
    mapping(objects);
    for (const auto& entry : objects.children()) {
        const auto& object = entry.second;
        mapping(object);
        const auto count = object["nb"];
        if (present(count)) count.require(count.get<int>() >= 0, "a nonnegative robot/object count");
        object["type"].get<std::string>();
        object["geometry"].get<std::string>();
        for (const char* key : {"radius", "body_width", "body_height", "side_length"}) nonnegative(object, key, true);
        for (const char* key : {"body_density", "body_friction", "body_restitution", "body_linear_damping", "body_angular_damping", "communication_radius", "temporal_noise_stddev", "linear_noise_stddev", "angular_noise_stddev", "photosensors_noise_stddev", "max_linear_speed", "max_angular_speed"}) nonnegative(object, key);
        const auto reception = object["msg_success_rate"];
        mapping(reception);
        auto reception_type = reception["type"].get<std::string>("realistic");
        for (char& ch : reception_type) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        if (reception.exists() && reception_type == "static") {
            const auto rate = reception["rate"];
            const double number = rate.get<double>(0.9);
            rate.require(std::isfinite(number) && number >= 0 && number <= 1, "a probability between 0 and 1");
        }
    }
}


YAML::Node Configuration::resolve_hierarchical_default(const YAML::Node& n) {
    if (!n || !n.IsMap()) {
        return n;
    }
    YAML::Node bho = n["batch_hierarchical_options"];
    if (!bho || !bho.IsMap()) {
        return n;
    }
    YAML::Node def = bho["default"];
    if (!def || !def.IsMap()) {
        // No usable default → behave as if no hierarchical options existed.
        return n;
    }
    // Build a merged view: copy all parent keys except the hierarchical key,
    // then overlay the default alternative's content at the same level.
    YAML::Node out(YAML::NodeType::Map);
    for (auto it = n.begin(); it != n.end(); ++it) {
        std::string k = it->first.as<std::string>();
        if (k == "batch_hierarchical_options") {
            continue;
        }
        out[k] = it->second;
    }
    for (auto it = def.begin(); it != def.end(); ++it) {
        std::string k = it->first.as<std::string>();
        out[k] = it->second;
    }
    return out;
}


// MODELINE "{{{1
// vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
// vim:foldmethod=marker
