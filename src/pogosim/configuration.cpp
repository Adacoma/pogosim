
#include "configuration.h"


Configuration::Configuration() : node_(YAML::Node()) {}

Configuration::Configuration(const YAML::Node &node) : node_(node) {}

void Configuration::load(const std::string& file_name) {
    try {
        node_ = YAML::LoadFile(file_name);
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("Error reading YAML file: " + std::string(e.what()));
    }
}

Configuration Configuration::operator[](const std::string& key) const {
    if (node_ && node_[key])
        return Configuration(node_[key]);
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
    if (!node_ || !(node_.IsMap() || node_.IsSequence())) {
        return result;
    }
    if (node_.IsMap()) {
        for (auto it = node_.begin(); it != node_.end(); ++it) {
            std::string key = it->first.as<std::string>();
            result.push_back({ key, Configuration(it->second) });
        }
    } else if (node_.IsSequence()) {
        for (std::size_t i = 0; i < node_.size(); ++i) {
            result.push_back({ std::to_string(i), Configuration(node_[i]) });
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

    // Exact key with dots wins (const lookup; no insertion)
    if (node_.IsMap()) {
        YAML::Node exact = node_[dotted_key]; // const operator[] called here
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
    const YAML::Node* cur = &node_;
    YAML::Node cur_val; // holds the last retrieved value to return at end
    for (const std::string& part : split_dotted(dotted_key)) {
        if (!*cur) return Configuration(YAML::Node());

        if (cur->IsMap()) {
            // const operator[] (no insertion)
            cur_val = (*cur)[part];
            if (!cur_val) return Configuration(YAML::Node());
            cur = &cur_val;
        } else if (cur->IsSequence() && is_int_str(part)) {
            long long idx = 0;
            try { idx = std::stoll(part); } catch (...) { return Configuration(YAML::Node()); }
            if (idx < 0 || static_cast<std::size_t>(idx) >= cur->size()) return Configuration(YAML::Node());
            cur_val = (*cur)[static_cast<std::size_t>(idx)]; // const operator[]
            cur = &cur_val;
        } else {
            return Configuration(YAML::Node());
        }
    }

    return Configuration(cur_val);
}


// MODELINE "{{{1
// vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
// vim:foldmethod=marker
