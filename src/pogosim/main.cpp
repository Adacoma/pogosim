
#include <iostream>
#include <string>
#include <chrono>
#include <sstream>
#include <iomanip>

#include "version.h"
#include "simulator.h"
#include "main.h"
#undef main         // We defined main() as robot_main() in pogobot.h

bool parse_arguments(int argc, char* argv[], std::string& config_file, bool& verbose, bool& quiet, bool& do_not_show_robot_msg, bool& gui, bool& progress) {
    verbose = false;
    quiet = false;
    do_not_show_robot_msg = false;
    gui = true;
    progress = false;
    config_file.clear();

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

void print_help() {
    std::cout << "Usage: pogosim [options]\n"
              << "Options:\n"
              << "  -c, --config <file>             Specify the configuration file.\n"
              << "  -g, --no-GUI                    Disable GUI mode.\n"
              << "  -v, --verbose                   Enable verbose mode.\n"
              << "  -q, --verbose                   Enable quiet mode (ouput only warning and errors on terminal).\n"
              << "  -nr, --do-not-show-robot-msg    Suppress robot messages.\n"
              << "  -P, --progress                  Show progress output.\n"
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

    // Parse command-line arguments
    if (!parse_arguments(argc, argv, config_file, verbose, quiet, do_not_show_robot_msg, gui, progress)) {
        std::cerr << "Usage: " << argv[0] << " -c CONFIG_FILE [-v/-q] [-nr] [-g] [-P] [-V]" << std::endl;
        return 1;
    }

    // Init logging
    init_logger();

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

    if (gui) {
        glogger->info("GUI enabled.");
    }

    Configuration config;
    try {
        // Load configuration
        config.load(config_file);

        if (verbose) {
            glogger->info("Loaded configuration from: {}", config_file);
        }

        // Display configuration
        if (verbose)
            glogger->debug(config.summary());

        config.set("GUI", gui ? "true" : "false");
        config.set("progress_bar", progress ? "true" : "false");

        // Create the simulation object
        simulation = std::make_unique<Simulation>(config);
        simulation->init_all();
        simulation->init_callbacks();

        // Launch simulation
        simulation->main_loop();

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

// MODELINE "{{{1
// vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
// vim:foldmethod=marker
