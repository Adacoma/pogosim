
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
    return Configuration(YAML::Node());  // Return an empty configuration if key is not found.
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

// MODELINE "{{{1
// vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
// vim:foldmethod=marker
