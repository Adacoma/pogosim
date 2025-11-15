
#include "configuration.h"


Configuration::Configuration() : node_(YAML::Node()), cache_valid_(false) {}

Configuration::Configuration(const YAML::Node &node) : node_(node), cache_valid_(false) {}

const YAML::Node& Configuration::get_resolved() const {
    if (!cache_valid_) {
        resolved_cache_ = resolve_hierarchical_default(node_);
        cache_valid_ = true;
    }
    return resolved_cache_;
}

void Configuration::load(const std::string& file_name) {
    try {
        node_ = YAML::LoadFile(file_name);
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("Error reading YAML file: " + std::string(e.what()));
    }
}

Configuration Configuration::operator[](const std::string& key) const {
    // Use cached resolved node
    const YAML::Node& self = get_resolved();
    if (self && self[key]) {
        return Configuration(self[key]);
    }
    return Configuration(YAML::Node());
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
            result.push_back({ key, Configuration(it->second) });
        }
    } else if (self.IsSequence()) {
        for (std::size_t i = 0; i < self.size(); ++i) {
            result.push_back({ std::to_string(i), Configuration(self[i]) });
        }
    }
    return result;
}


Configuration Configuration::at_path(const std::string& dotted_key) const {
    if (!node_) return Configuration(YAML::Node());

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
        if (exact) return Configuration(exact);
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
        if (!*cur) return Configuration(YAML::Node());

        if (cur->IsMap()) {
            // const operator[] (no insertion)
            cur_val = (*cur)[part];
            if (!cur_val) return Configuration(YAML::Node());
            // Resolve hierarchical default at the *new* level before continuing
            cur_val = resolve_hierarchical_default(cur_val);
            cur = &cur_val;
        } else if (cur->IsSequence() && is_int_str(part)) {
            long long idx = 0;
            try { idx = std::stoll(part); } catch (...) { return Configuration(YAML::Node()); }
            if (idx < 0 || static_cast<std::size_t>(idx) >= cur->size()) return Configuration(YAML::Node());
            cur_val = (*cur)[static_cast<std::size_t>(idx)]; // const operator[]
            // If the sequence element itself is a map with hierarchical options,
            // resolve before descending further
            cur_val = resolve_hierarchical_default(cur_val);
            cur = &cur_val;
        } else {
            return Configuration(YAML::Node());
        }
    }

    return Configuration(cur_val);
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
