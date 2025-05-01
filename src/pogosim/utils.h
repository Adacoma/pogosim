#ifndef UTILS_H
#define UTILS_H

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h> // For colored console output
#include <spdlog/fmt/ostr.h> // Enables << operator for logging

#include <box2d/box2d.h>

#include <random>
#include <algorithm>

/// Global logger for general logging.
extern std::shared_ptr<spdlog::logger> glogger;
/// Global logger for robot-specific logging.
extern std::shared_ptr<spdlog::logger> robotlogger;

/// Random device used for seeding the random generator.
extern std::random_device rd;
/// Random number generator seeded with rd.
extern std::mt19937 rnd_gen;

/**
 * @brief Check whether a vector contains a given value.
 *
 * Traverses the vector and returns `true` as soon as the first element equal
 * to @p value is found.
 *
 * @tparam T   Element type stored in the vector and of the searched value.
 * @param vec  Vector to inspect.
 * @param value Value to look for inside @p vec.
 * @return `true` if @p value is present in @p vec, otherwise `false`.
 *
 * @note Complexity: O(n) comparisons, where *n* is `vec.size()`.
 * @since C++20 — `std::find` is `constexpr`‐friendly when the iterators (and
 * `operator==`) are `constexpr`.
 */
template<typename T>
[[nodiscard]] constexpr bool contains(const std::vector<T>& vec,
                                      const T& value) {
    return std::find(vec.begin(), vec.end(), value) != vec.end();
}


/**
 * @brief Check whether every element in a `std::vector<bool>` is `true`.
 *
 * Uses `std::ranges::all_of` to test each element.  Stops early on the first
 * `false`, so the complexity is *O(n)* in the worst case, *O(1)* if the first
 * value is `false`.
 *
 * @param flags  Vector of `bool` (or any range of `bool`-convertible values).
 * @return `true` if **all** elements evaluate to `true`, otherwise `false`.
 *
 * @note Works with the `std::vector<bool>` specialisation because the proxy
 *       object returned by its iterator is convertible to `bool`.
 * @since C++20
 */
[[nodiscard]] constexpr bool all_true(const std::vector<bool>& flags) {
    return std::ranges::all_of(flags, [](bool v) { return v; });
}


/**
 * @brief Initializes the global loggers.
 *
 * This function creates a console sink with color support and attaches it to two loggers:
 * one for global logging (glogger) and one for robot-specific logging (robotlogger).
 * The log level for both loggers is set to info and a specific log message format is defined.
 */
void init_logger();

/**
 * @brief Adds a file sink to the global loggers.
 *
 * This function ensures that the directory for the given filename exists, then creates a file sink
 * that logs messages with debug level and a specified format. The file sink is then added to both
 * the global logger (glogger) and the robot logger (robotlogger).
 *
 * @param filename The file path where the log file will be created.
 */
void loggers_add_file_sink(std::string const& filename);

/**
 * @brief Converts a string to a boolean value.
 *
 * The conversion is case-insensitive. The function returns true if the input string is "true" or "1",
 * and false if it is "false" or "0". For any other input, an std::invalid_argument exception is thrown.
 *
 * @param str The input string to convert.
 * @return true if the string represents a boolean true.
 * @return false if the string represents a boolean false.
 *
 * @throws std::invalid_argument If the input string is not a valid boolean representation.
 */
bool string_to_bool( std::string const& str);

/**
 * @brief Converts a string to lowercase.
 *
 * This function creates and returns a lowercase copy of the input string.
 *
 * @param str The string to convert.
 * @return std::string A new string where all characters have been converted to lowercase.
 */
std::string to_lowercase(std::string const& str);

/**
 * @brief Ensures that the directory for a given file exists.
 *
 * This function checks the parent directory of the provided filename. If the directory does not exist,
 * it attempts to create it. Successful creation is logged using glogger. If an error occurs during directory
 * creation, the error is logged.
 *
 * @param filename The file path for which the directories should be ensured.
 */
void ensure_directories_exist(const std::string& filename);

/**
 * @brief Deletes files with a specified extension in a given directory.
 *
 * This function deletes all regular files with the specified extension within the given directory.
 * If the recursive flag is true, it searches subdirectories as well; otherwise, it only searches the
 * specified directory. Deletion operations and any errors encountered are logged.
 *
 * @param path The directory path where files will be deleted.
 * @param extension The file extension (including the dot) to match for deletion.
 * @param recursive If true, deletion is performed recursively in subdirectories.
 */
void delete_files_with_extension(const std::string& path, const std::string& extension, bool recursive = false);

/**
 * @brief Resolves an input file path to an absolute path.
 *
 * This function attempts to resolve the provided input path by first checking if the file exists
 * in the current directory. If not found and if the DATA_DIR macro is defined, it checks relative
 * to DATA_DIR. If the file is found in either location, the absolute path is returned; otherwise,
 * an exception is thrown.
 *
 * @param inputPath The relative or absolute path of the file.
 * @return std::string The resolved absolute path of the file.
 *
 * @throws std::runtime_error If the file cannot be found in the current directory or DATA_DIR.
 */
std::string resolve_path(const std::string& inputPath);


#endif // UTILS_H

// MODELINE "{{{1
// vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
// vim:foldmethod=marker
