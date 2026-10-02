#ifndef POGOSIM_REGRESSION_TEST_SUPPORT_H
#define POGOSIM_REGRESSION_TEST_SUPPORT_H

#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>

// Do not use assert(): regression checks must also execute in Release builds.
inline void check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

inline void close_to(double value, double expected, double tolerance = 1e-5) {
    check(std::isfinite(value) && std::fabs(value - expected) <= tolerance,
          "Expected " + std::to_string(expected) + ", got " + std::to_string(value));
}

inline void expect_error(const std::function<void()>& action, const std::string& diagnostic) {
    try {
        action();
    } catch (const std::runtime_error& error) {
        check(std::string(error.what()).find(diagnostic) != std::string::npos,
              "Wrong failure diagnostic: " + std::string(error.what()));
        return;
    }
    throw std::runtime_error("Expected failure: " + diagnostic);
}

#endif
