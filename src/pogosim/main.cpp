
#include <iostream>
#include <exception>
#include <string>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <cstdint>
#include <limits>

#include "version.h"
#include "simulator.h"
#include "main.h"
#include "utils.h"
#undef main         // We defined main() as robot_main() in pogobot.h

// Keep the existing parser entry point below while extending the simulator CLI.
static bool parse_arguments_impl(int argc, char* argv[], std::string& config_file, bool& verbose, bool& quiet, bool& do_not_show_robot_msg, bool& gui, bool& progress, bool& seed_provided, uint32_t& seed, bool& check_config, bool& strict_config) {
    verbose = false;
    quiet = false;
    do_not_show_robot_msg = false;
    gui = true;
    progress = false;
    seed_provided = false;
    seed = 0;
    config_file.clear();
    check_config = false;
    strict_config = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-c" || arg == "--config") {
            if (i + 1 < argc) {
                config_file = argv[++i];
            } else {
                std::cerr << "Error: -c requires a configuration file argument." << std::endl;
                return false;
            }
        } else if (arg == "-g" || arg == "--no-GUI") {
            gui = false;
        } else if (arg == "-v" || arg == "--verbose") {
            verbose = true;
        } else if (arg == "-q" || arg == "--quiet") {
            quiet = true;
        } else if (arg == "-nr" || arg == "--do-not-show-robot-msg") {
            do_not_show_robot_msg = true;
        } else if (arg == "-P" || arg == "--progress") {
            progress = true;
        } else if (arg == "--check-config") {
            check_config = true;
        } else if (arg == "--strict-config") {
            strict_config = true;
        } else if (arg == "-s" || arg == "--seed") {
            if (i + 1 < argc) {
                std::string seed_arg = argv[++i];
                try {
                    // accepts only a non-negative integer.
                    std::size_t consumed = 0;
                    unsigned long parsed = std::stoul(seed_arg, &consumed, 10);
                    if (consumed != seed_arg.size()) {
                        throw std::invalid_argument("trailing characters");
                    }
                    if (parsed > std::numeric_limits<uint32_t>::max()) {
                        throw std::out_of_range("seed too large");
                    }
                    seed = static_cast<uint32_t>(parsed);
                    seed_provided = true;
                } catch (const std::exception&) {
                    std::cerr << "Error: -s/--seed requires a non-negative integer argument." << std::endl;
                    return false;
                }
            } else {
                std::cerr << "Error: -s/--seed requires an integer argument." << std::endl;
                return false;
            }
        } else if (arg == "-V" || arg == "--version") {
            std::cout << "Pogosim simulator. Version " << POGOSIM_VERSION << "." << std::endl;
            return false;
        } else if (arg == "-h" || arg == "--help") {
            print_help();
            return false;
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            return false;
        }
    }

    return true;
}

bool parse_arguments(int argc, char* argv[], std::string& config_file, bool& verbose, bool& quiet, bool& do_not_show_robot_msg, bool& gui, bool& progress, bool& seed_provided, uint32_t& seed) {
    bool check_config, strict_config;
    return parse_arguments_impl(argc, argv, config_file, verbose, quiet, do_not_show_robot_msg, gui, progress, seed_provided, seed, check_config, strict_config);
}

void print_help() {
    std::cout << "Usage: pogosim [options]\n"
              << "Options:\n"
              << "  -c, --config <file>             Specify the configuration file.\n"
              << "  -g, --no-GUI                    Disable GUI mode.\n"
              << "  -v, --verbose                   Enable verbose mode.\n"
              << "  -q, --quiet                     Enable quiet mode (ouput only warning and errors on terminal).\n"
              << "  -nr, --do-not-show-robot-msg    Suppress robot messages.\n"
              << "  -s, --seed <int>                Seed the simulator RNG.\n"
              << "  -P, --progress                  Show progress output.\n"
              << "  --check-config                  Check core configuration without running.\n"
              << "  --strict-config                 Check core constraints and typed lookups.\n"
              << "  -V, --version                   Show version information.\n"
              << "  -h, --help                      Display this help message.\n";
}


int main(int argc, char** argv) {
    std::string config_file;
    bool verbose = false;
    bool quiet = false;
    bool do_not_show_robot_msg = false;
    bool gui = true;
    bool progress = false;
    bool cli_seed_provided = false;
    uint32_t cli_seed = 0;
    bool check_config = false;
    bool strict_config = false;

    // Parse command-line arguments
    if (!parse_arguments_impl(argc, argv, config_file, verbose, quiet, do_not_show_robot_msg, gui, progress, cli_seed_provided, cli_seed, check_config, strict_config)) {
        std::cerr << "Usage: " << argv[0] << " -c CONFIG_FILE [-v/-q] [-nr] [-g] [-s SEED] [-P] [-V]" << std::endl;
        return 1;
    }

    Configuration config;
    try {
        // Load configuration
        config.load(config_file);
        if (check_config || strict_config) {
            // Validate the effective CLI overrides before any SDL/controller,
            // logging output or flash-file initialization can have side effects.
            if (cli_seed_provided) config.set("seed", cli_seed);
            config.set("GUI", gui);
            config.set("progress_bar", progress);
            config.validate_simulator();
            if (check_config) {
                std::cout << "Core configuration checks passed; controller callbacks are not run." << std::endl;
                return 0;
            }
            config.enable_validation();
        }
        // Init logging
        init_logger(config);
        if (verbose) {
            glogger->info("Loaded configuration from: {}", config_file);
        }

        // Display configuration
        if (verbose)
            glogger->debug(config.summary());

        // The command line wins over a config (YAML) provided seed.
        uint32_t seed = 0;
        std::string seed_source;
        if (cli_seed_provided) {
            seed = cli_seed;
            seed_source = "command line";
        } else if (config["seed"].exists()) {
            seed = config["seed"].get<uint32_t>();
            seed_source = "configuration";
        } else {
            seed = make_random_seed();
            seed_source = "generated";
        }
        seed_random_generators(seed);
        config.set("seed", seed);
        glogger->info("Using random seed {} ({})", seed, seed_source);
    } catch (const std::exception& e) {
        std::cerr << "Unable to create configuration. Error: " << e.what() << std::endl;
        return 1;
    }

    if (gui) {
        glogger->info("GUI enabled.");
    }
    config.set("GUI", gui ? "true" : "false");
    config.set("progress_bar", progress ? "true" : "false");

    if (quiet) {
        // Quiet mode, only output warnings and error on terminal
        auto glogger_console_sink = glogger->sinks().front();
        glogger_console_sink->set_level(spdlog::level::warn);
        auto robotlogger_console_sink = robotlogger->sinks().front();
        robotlogger_console_sink->set_level(spdlog::level::warn);
    } else if (verbose) {
        // Enable verbose mode if requested
        glogger->info("Verbose mode enabled.");
        glogger->set_level(spdlog::level::debug);
        robotlogger->set_level(spdlog::level::debug);
        auto glogger_console_sink = glogger->sinks().front();
        glogger_console_sink->set_level(spdlog::level::debug);
        auto robotlogger_console_sink = robotlogger->sinks().front();
        robotlogger_console_sink->set_level(spdlog::level::debug);
    }

    if (do_not_show_robot_msg) {
        robotlogger->sinks().clear();
    }

    std::exception_ptr simulation_failure;
    try {
        // Create the simulation object
        simulation = std::make_unique<Simulation>(config);
        simulation->init_all();
        simulation->init_callbacks();

        // Launch simulation
        simulation->main_loop();
    } catch (const std::exception&) {
        // Fiber cancellation switches contexts: Boost.Context forbids doing
        // that inside a catch block. Retain the error until cleanup is done.
        simulation_failure = std::current_exception();
    }

    // Keep the global handle available while suspended controller locals
    // unwind, then destroy resources before late process teardown.
    if (simulation) simulation->stop_robot_controllers();
    simulation.reset();
    if (simulation_failure) {
        try {
            std::rethrow_exception(simulation_failure);
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            return 2;
        }
    }
    return 0;
}

// MODELINE "{{{1
// vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
// vim:foldmethod=marker
