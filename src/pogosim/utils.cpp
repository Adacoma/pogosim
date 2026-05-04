#include <iostream>
#include <string>
#include <filesystem>
#include <algorithm>
#include <cstdlib>

#include "utils.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/basic_file_sink.h"


std::shared_ptr<spdlog::logger> glogger;
std::shared_ptr<spdlog::logger> robotlogger;

// Random device and generator
std::random_device rd;
std::mt19937 rnd_gen(rd());

void seed_random_generators(uint32_t seed) {
    rnd_gen.seed(seed); // C++ rng
    std::srand(static_cast<unsigned int>(seed)); // C rng (in case of rand() usage)
}

uint32_t make_random_seed() {
    return rd();
}


void init_logger(Configuration& config) {
//    // Create a console logger with color support
//    glogger = spdlog::stdout_color_mt("console");
//    glogger->set_level(spdlog::level::info); // Set default log level
//    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v"); // Set log format

    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_level(spdlog::level::info);
    std::string const log_format_str = config["log_format"].get(std::string("default"));
    if (log_format_str == "default" || log_format_str == "date" || log_format_str == "null") {
        console_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
    } else if (log_format_str == "minimal") {
        console_sink->set_pattern("%v");
    } else {
        throw std::runtime_error("Unknown value of 'log_format'. Use either 'default' or 'minimal'.");
    }

    std::vector<spdlog::sink_ptr> sinks{console_sink};
    glogger     = std::make_shared<spdlog::logger>("global", sinks.begin(), sinks.end());
    robotlogger = std::make_shared<spdlog::logger>("robot",  sinks.begin(), sinks.end());
    glogger->set_level(spdlog::level::info);
    robotlogger->set_level(spdlog::level::info);
}

void loggers_add_file_sink(std::string const& filename) {
    ensure_directories_exist(filename);

    // Create the new sink
    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(filename, true);
    file_sink->set_level(spdlog::level::debug);
    file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");

    // Add the new sink to all loggers
    glogger->sinks().push_back(file_sink);
    robotlogger->sinks().push_back(file_sink);
}


bool string_to_bool( std::string const& str) {
    // Convert string to lowercase for case-insensitive comparison
    std::string lowerStr = str;
    std::transform(lowerStr.begin(), lowerStr.end(), lowerStr.begin(), ::tolower);

    if (lowerStr == "true" || lowerStr == "1") {
        return true;
    } else if (lowerStr == "false" || lowerStr == "0") {
        return false;
    } else {
        throw std::invalid_argument("Invalid string for boolean conversion: " + str);
    }
}

// Note: copy the string
std::string to_lowercase(std::string const& str) {
    std::string result = str; // Avoid modifying original string
    std::ranges::transform(result, result.begin(), [](unsigned char c) { return std::tolower(c); });
    return result;
}


void ensure_directories_exist(const std::string& filename) {
    try {
        std::filesystem::path filePath(filename);
        std::filesystem::path directory = filePath.parent_path();

        if (!directory.empty() && !std::filesystem::exists(directory)) {
            std::filesystem::create_directories(directory);
            glogger->info("Created directories: {}", directory.string());
        }
    } catch (const std::filesystem::filesystem_error& e) {
        glogger->error("Error creating directories for '{}': {}'", filename, e.what());
    }
}

void delete_files_with_extension(const std::string& path, const std::string& extension, bool recursive) {
    try {
        std::filesystem::path dirPath(path);

        // Check if the path exists and is a directory
        if (!std::filesystem::exists(dirPath) || !std::filesystem::is_directory(dirPath)) {
            glogger->error("Invalid directory: {}", path);
            return;
        }

        // Declare an iterator
        if (recursive) {
            for (const auto& entry : std::filesystem::recursive_directory_iterator(dirPath)) {
                if (entry.is_regular_file() && entry.path().extension() == extension) {
                    std::filesystem::remove(entry.path());
                    glogger->debug("Deleted: {}", entry.path().string());
                }
            }
        } else {
            for (const auto& entry : std::filesystem::directory_iterator(dirPath)) {
                if (entry.is_regular_file() && entry.path().extension() == extension) {
                    std::filesystem::remove(entry.path());
                    glogger->debug("Deleted: {}", entry.path().string());
                }
            }
        }
    } catch (const std::filesystem::filesystem_error& e) {
        glogger->error("Filesystem error: {}", e.what());
    } catch (const std::exception& e) {
        glogger->error("Error: {}", e.what());
    }
}


std::string resolve_path(const std::string& inputPath) {
    namespace fs = std::filesystem;

    // Convert the input path to a filesystem path
    fs::path path(inputPath);

    // 1. Check if the file is accessible from the current directory
    if (fs::exists(path) && fs::is_regular_file(path)) {
        return fs::absolute(path).string();
    }

    // 2. Check if the file is accessible relative to DATA_DIR
#ifdef DATA_DIR
    fs::path dataDirPath(DATA_DIR);
    fs::path relativeToDataDir = dataDirPath / path;

    glogger->debug("DATA_DIR: {}", DATA_DIR);
    glogger->debug("Trying to find: {}", relativeToDataDir.string());
    if (fs::exists(relativeToDataDir) && fs::is_regular_file(relativeToDataDir)) {
        return fs::absolute(relativeToDataDir).string();
    }
#endif

    // If the file is not found in either location, throw an exception
    throw std::runtime_error("Impossible to load file '" + inputPath + "'.");
}

// MODELINE "{{{1
// vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
// vim:foldmethod=marker
