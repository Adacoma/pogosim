#include "pogosim/configuration.h"
#undef main
#include <cstdint>
#include <functional>
#include <iostream>

// Explicit checks stay active in optimized builds, unlike assert().
static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

static void rejects(const std::function<void()>& action, const std::string& path) {
    try {
        action();
    } catch (const std::invalid_argument& error) {
        require(std::string(error.what()).find("'" + path + "'") != std::string::npos,
                "Validation error did not identify the YAML path");
        return;
    }
    throw std::runtime_error("Invalid configuration was accepted: " + path);
}

static Configuration config(const std::string& yaml, bool strict = true) {
    Configuration result(YAML::Load(yaml));
    result.enable_validation(strict);
    return result;
}

int main() {
    try {
        auto legacy = config("bad: not_a_number\nfraction: 2.5\nnew_parameter: 7", false);
        require(legacy["bad"].get(13) == 13, "Legacy conversion fallback changed");
        require(legacy["fraction"].get(0) == 2, "Legacy numeric coercion changed");
        legacy["new_parameter"].require(false, "inactive constraint");
        legacy.enable_validation();
        rejects([&] { legacy["bad"].get(13); }, "bad");
        rejects([&] { legacy["fraction"].get(0); }, "fraction");
        require(legacy["new_parameter"].get<int>() == 7, "New parameters need registration");
        require(legacy["missing"].get(17) == 17, "Missing parameter default changed");
        rejects([&] { legacy["new_parameter"].require(false, "an example domain"); }, "new_parameter");

        auto numeric = config("whole: 2.0\nscientific: 2e1\nyes: true\nzero: 0\none: 1\ntwo: 2\nnegative: -1\nlarge: 18446744073709551616.0\nmax: 18446744073709551615\ninf: .inf\nnan: .nan\noverflow: 1e100\nnull_value: null");
        require(numeric["whole"].get<int>() == 2 && numeric["scientific"].get<int>() == 20, "Exact numeric coercions rejected");
        require(numeric["yes"].get<bool>() && numeric["one"].get<bool>() && !numeric["zero"].get<bool>(), "Boolean compatibility changed");
        require(numeric["max"].get<uint64_t>() == (std::numeric_limits<uint64_t>::max)(), "Exact uint64 conversion lost precision");
        require(numeric["null_value"].get(17) == 17, "Null default changed");
        rejects([&] { numeric["two"].get<bool>(); }, "two");
        rejects([&] { numeric["negative"].get<unsigned>(); }, "negative");
        rejects([&] { numeric["large"].get<uint64_t>(); }, "large");
        rejects([&] { numeric["inf"].get<int>(); }, "inf");
        rejects([&] { numeric["nan"].get<int>(); }, "nan");
        rejects([&] { numeric["overflow"].get<float>(); }, "overflow");

        auto nested = config("parameters:\n  items: [1, invalid]\n  future_key: {default_option: 2.0, batch_options: [bad]}\nobjects:\n  robots:\n    type: pogobot\n    nb: 2\n    batch_hierarchical_options:\n      default: {radius: 26.5}\n      inactive: {nb: broken}\n");
        const auto original = nested.summary();
        require(nested.get_path<int>("parameters.items.0") == 1, "Sequence lookup failed");
        rejects([&] { nested.get_path<int>("parameters.items.1"); }, "parameters.items.1");
        require(nested.get_path<int>("parameters.future_key") == 2, "Batch scalar default failed");
        nested.validate_simulator();
        require(nested.summary() == original, "Read/validation mutated YAML batch definitions");
        auto children = nested["parameters"]["items"].children();
        rejects([&] { children[1].second.get<int>(); }, "parameters.items.1");
        auto literal = config("a.b: invalid\na: {b: 2}");
        rejects([&] { literal.get_path<int>("a.b"); }, "a.b");
        auto scalar = config("parameters: invalid");
        rejects([&] { scalar["parameters"]["new_key"].get<int>(); }, "parameters");

        // Preflight is independent of mode, and does not enable strictness on
        // the original object or inspect arbitrary controller-specific keys.
        auto permissive = config("time_step: 0.01\nparameters: {custom: anything}\nnew_section: [a, b]\nformation_max_space_between_neighbors: .inf\nsave_video_period: -1", false);
        permissive.validate_simulator();
        require(permissive["parameters"]["custom"].get(7) == 7, "Preflight altered original validation mode");
        auto null_defaults = config("time_step: {default_option: null}\nwindow_width: {default_option: null}\nformation_max_space_between_neighbors: {default_option: null}\nobjects: {robots: {radius: {default_option: null}, nb: 0}}\n");
        null_defaults.validate_simulator();
        for (const auto& entry : std::vector<std::pair<std::string, std::string>>{
                {"[]", "<root>"}, {"time_step: 0", "time_step"}, {"time_step: .nan", "time_step"},
                {"time_step: 1e-50", "time_step"}, {"simulation_time: -1", "simulation_time"},
                {"simulation_time: 1e12", "simulation_time"}, {"window_width: 65536", "window_width"},
                {"light_map_nb_bin_x: 0", "light_map_nb_bin_x"}, {"flash_state: []", "flash_state"},
                {"flash_state: {create_if_missing: maybe}", "flash_state.create_if_missing"},
                {"objects: {robots: {nb: -1}}", "objects.robots.nb"},
                {"objects: {robots: {nb: 2.5}}", "objects.robots.nb"},
                {"objects: {robots: {body_density: -1}}", "objects.robots.body_density"},
                {"objects: {robots: {body_width: 0}}", "objects.robots.body_width"},
                {"objects: {robots: {msg_success_rate: {type: STATIC, rate: 1.1}}}", "objects.robots.msg_success_rate.rate"},
                {"formation_min_space_between_neighbors: 5\nformation_max_space_between_neighbors: 2", "formation_max_space_between_neighbors"}}) {
            auto invalid = config(entry.first, false);
            rejects([&] { invalid.validate_simulator(); }, entry.second);
        }
        // Re-reading after a mutation must not use a stale resolved default.
        auto changed = config("batch_hierarchical_options: {default: {seed: 1}}");
        require(changed["seed"].get<unsigned>() == 1, "Initial default lookup failed");
        changed.set("batch_hierarchical_options", YAML::Load("{default: {seed: 2}}"));
        require(changed["seed"].get<unsigned>() == 2, "Resolved cache not invalidated");
        std::cout << "Configuration validation tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
