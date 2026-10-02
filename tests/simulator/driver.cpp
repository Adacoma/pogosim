#include "pogosim/configuration.h"
#include "pogosim/utils.h"
#undef main
#include "test_support.h"
#include <chrono>
#include <filesystem>
#include <iostream>

void test_geometry();
void test_lighting();
void test_neighbors();
void test_probabilities();
void test_factory(const std::string&, bool, const std::filesystem::path&);
void test_flash(const std::filesystem::path&);
void test_logging(const std::filesystem::path&);
void verify_simulation_output(const std::filesystem::path&, bool);

int main(int argc, char** argv) {
    try {
        check(argc == 4, "Expected case, build-tree output directory and source directory");
        const std::string mode = argv[1];
        Configuration logging_config;
        init_logger(logging_config);
        if (mode == "geometry") test_geometry();
        else if (mode == "lighting") test_lighting();
        else if (mode == "neighbors") test_neighbors();
        else if (mode == "probabilities") test_probabilities();
        else if (mode.rfind("factory_", 0) == 0) {
            const bool periodic = mode.ends_with("_periodic");
            const auto suffix = periodic ? std::string("_periodic") : std::string("_solid");
            test_factory(mode.substr(8, mode.size() - 8 - suffix.size()), periodic, argv[3]);
        } else if (mode == "verify_all" || mode == "verify_restricted") {
            verify_simulation_output(argv[2], mode == "verify_restricted");
        } else {
            // Per-invocation directories isolate parallel CTest runs and avoid
            // overwriting existing user files or assuming a POSIX temp API.
            const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
            const auto output = std::filesystem::path(argv[2]) / (mode + "-" + std::to_string(nonce));
            std::filesystem::create_directories(output);
            if (mode == "flash") test_flash(output);
            else if (mode == "logging") test_logging(output);
            else throw std::runtime_error("Unknown regression case: " + mode);
        }
        std::cout << "Passed: " << mode << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Regression failure: " << error.what() << '\n';
        return 1;
    }
}
