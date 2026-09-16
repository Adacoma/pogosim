#include "flash_state.h"

#include "robot.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>
#include <utility>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace pogosim::flash_state {
namespace {

// On-disk integers are little-endian. The header contains magic, format
// version, flash size, and record count. Each record contains robot ID,
// category, motor calibration memories, the fixed-size user flash, and an
// FNV-1a checksum over that record's identity and persistent data.
constexpr std::array<unsigned char, 8> archive_magic = {
    'P', 'G', 'F', 'L', 'A', 'S', 'H', '\0'
};
constexpr std::uint16_t archive_version = 1;
constexpr std::uint64_t fnv_offset_basis = 14695981039346656037ULL;
constexpr std::uint64_t fnv_prime = 1099511628211ULL;

using robot_key = std::pair<std::string, std::uint16_t>;

[[noreturn]] void archive_error(
    const std::filesystem::path& filename,
    const std::string& message
) {
    throw std::runtime_error(
        "Invalid flash-state archive '" + filename.string() + "': " + message
    );
}

void update_checksum(
    std::uint64_t& checksum,
    const unsigned char* bytes,
    std::size_t size
) {
    for (std::size_t i = 0; i < size; ++i) {
        checksum ^= bytes[i];
        checksum *= fnv_prime;
    }
}

void write_bytes(
    std::ostream& output,
    const unsigned char* bytes,
    std::size_t size,
    std::uint64_t* checksum = nullptr
) {
    output.write(
        reinterpret_cast<const char*>(bytes),
        static_cast<std::streamsize>(size)
    );
    if (!output) {
        throw std::runtime_error("Unable to write flash-state archive");
    }
    if (checksum != nullptr) {
        update_checksum(*checksum, bytes, size);
    }
}

void read_bytes(
    std::istream& input,
    unsigned char* bytes,
    std::size_t size,
    const std::filesystem::path& filename,
    const char* field,
    std::uint64_t* checksum = nullptr
) {
    input.read(reinterpret_cast<char*>(bytes), static_cast<std::streamsize>(size));
    if (!input) {
        archive_error(filename, std::string("truncated while reading ") + field);
    }
    if (checksum != nullptr) {
        update_checksum(*checksum, bytes, size);
    }
}

template<typename T>
void write_unsigned_le(
    std::ostream& output,
    T value,
    std::uint64_t* checksum = nullptr
) {
    static_assert(std::is_unsigned_v<T>);
    std::array<unsigned char, sizeof(T)> bytes{};
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = static_cast<unsigned char>(value & static_cast<T>(0xFF));
        value >>= 8;
    }
    write_bytes(output, bytes.data(), bytes.size(), checksum);
}

template<typename T>
T read_unsigned_le(
    std::istream& input,
    const std::filesystem::path& filename,
    const char* field,
    std::uint64_t* checksum = nullptr
) {
    static_assert(std::is_unsigned_v<T>);
    std::array<unsigned char, sizeof(T)> bytes{};
    read_bytes(input, bytes.data(), bytes.size(), filename, field, checksum);

    T value = 0;
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        value |= static_cast<T>(bytes[i]) << (8 * i);
    }
    return value;
}

std::map<robot_key, PogobotObject*> index_robots(
    const std::vector<std::shared_ptr<PogobotObject>>& robots
) {
    std::map<robot_key, PogobotObject*> result;
    for (const auto& robot : robots) {
        const robot_key key{robot->category, robot->id};
        if (!result.emplace(key, robot.get()).second) {
            throw std::runtime_error(
                "Cannot persist flash state: duplicate robot identity ('" +
                robot->category + "', " + std::to_string(robot->id) + ")"
            );
        }
    }
    return result;
}

void write_archive(
    const std::filesystem::path& filename,
    const std::vector<std::shared_ptr<PogobotObject>>& robots
) {
    if (robots.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("Too many robots for the flash-state archive format");
    }

    // Sorting by identity makes archives deterministic independently of vector order.
    const auto indexed_robots = index_robots(robots);
    std::ofstream output(filename, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error(
            "Unable to open flash-state archive for writing: '" +
            filename.string() + "'"
        );
    }

    write_bytes(output, archive_magic.data(), archive_magic.size());
    write_unsigned_le(output, archive_version);
    write_unsigned_le(
        output,
        static_cast<std::uint32_t>(flash_memory_authorized_section_size)
    );
    write_unsigned_le(output, static_cast<std::uint32_t>(indexed_robots.size()));

    for (const auto& [key, robot] : indexed_robots) {
        const auto& [category, id] = key;
        if (category.size() > std::numeric_limits<std::uint16_t>::max()) {
            throw std::runtime_error(
                "Robot category is too long for the flash-state archive: '" +
                category + "'"
            );
        }

        std::uint64_t checksum = fnv_offset_basis;
        write_unsigned_le(output, id, &checksum);
        write_unsigned_le(
            output,
            static_cast<std::uint16_t>(category.size()),
            &checksum
        );
        write_bytes(
            output,
            reinterpret_cast<const unsigned char*>(category.data()),
            category.size(),
            &checksum
        );
        write_bytes(output, robot->motor_dir_mem, 3, &checksum);
        for (std::uint16_t value : robot->motor_power_mem) {
            write_unsigned_le(output, value, &checksum);
        }
        write_bytes(
            output,
            robot->flash_memory_authorized_section,
            flash_memory_authorized_section_size,
            &checksum
        );
        write_unsigned_le(output, checksum);
    }

    output.flush();
    if (!output) {
        throw std::runtime_error(
            "Unable to finish writing flash-state archive: '" +
            filename.string() + "'"
        );
    }
}

} // namespace

void load(
    const std::filesystem::path& filename,
    const std::vector<std::shared_ptr<PogobotObject>>& robots
) {
    std::ifstream input(filename, std::ios::binary);
    if (!input) {
        throw std::runtime_error(
            "Unable to open flash-state archive for reading: '" +
            filename.string() + "'"
        );
    }

    std::array<unsigned char, archive_magic.size()> magic{};
    read_bytes(input, magic.data(), magic.size(), filename, "archive header");
    if (magic != archive_magic) {
        archive_error(filename, "unrecognized file signature");
    }

    const auto version = read_unsigned_le<std::uint16_t>(
        input, filename, "format version"
    );
    if (version != archive_version) {
        archive_error(
            filename,
            "unsupported format version " + std::to_string(version)
        );
    }

    const auto flash_size = read_unsigned_le<std::uint32_t>(
        input, filename, "flash size"
    );
    if (flash_size != flash_memory_authorized_section_size) {
        archive_error(
            filename,
            "flash size is " + std::to_string(flash_size) + ", expected " +
            std::to_string(flash_memory_authorized_section_size)
        );
    }

    const auto record_count = read_unsigned_le<std::uint32_t>(
        input, filename, "robot count"
    );
    if (record_count != robots.size()) {
        archive_error(
            filename,
            "robot count is " + std::to_string(record_count) + ", expected " +
            std::to_string(robots.size())
        );
    }

    auto indexed_robots = index_robots(robots);
    std::map<robot_key, bool> restored;
    // One temporary image bounds loader memory independently of robot count.
    std::vector<unsigned char> flash(flash_memory_authorized_section_size);

    for (std::uint32_t record = 0; record < record_count; ++record) {
        std::uint64_t checksum = fnv_offset_basis;
        const auto id = read_unsigned_le<std::uint16_t>(
            input, filename, "robot ID", &checksum
        );
        const auto category_size = read_unsigned_le<std::uint16_t>(
            input, filename, "category length", &checksum
        );
        std::string category(category_size, '\0');
        read_bytes(
            input,
            reinterpret_cast<unsigned char*>(category.data()),
            category.size(),
            filename,
            "robot category",
            &checksum
        );

        const robot_key key{category, id};
        const auto robot_it = indexed_robots.find(key);
        if (robot_it == indexed_robots.end()) {
            archive_error(
                filename,
                "no current robot matches ('" + category + "', " +
                std::to_string(id) + ")"
            );
        }
        if (!restored.emplace(key, true).second) {
            archive_error(
                filename,
                "duplicate robot record ('" + category + "', " +
                std::to_string(id) + ")"
            );
        }

        std::array<unsigned char, 3> motor_directions{};
        std::array<std::uint16_t, 3> motor_powers{};
        read_bytes(
            input,
            motor_directions.data(),
            motor_directions.size(),
            filename,
            "motor direction memory",
            &checksum
        );
        for (auto& value : motor_powers) {
            value = read_unsigned_le<std::uint16_t>(
                input, filename, "motor power memory", &checksum
            );
        }
        read_bytes(
            input,
            flash.data(),
            flash.size(),
            filename,
            "user flash",
            &checksum
        );
        const auto stored_checksum = read_unsigned_le<std::uint64_t>(
            input, filename, "record checksum"
        );
        if (stored_checksum != checksum) {
            archive_error(
                filename,
                "checksum mismatch for robot ('" + category + "', " +
                std::to_string(id) + ")"
            );
        }

        // Commit a record only after all of its bytes and checksum are valid.
        PogobotObject* robot = robot_it->second;
        std::memcpy(robot->motor_dir_mem, motor_directions.data(), 3);
        std::copy(motor_powers.begin(), motor_powers.end(), robot->motor_power_mem);
        std::memcpy(
            robot->flash_memory_authorized_section,
            flash.data(),
            flash.size()
        );
    }

    char trailing_byte = 0;
    if (input.read(&trailing_byte, 1)) {
        archive_error(filename, "trailing data after the final robot record");
    }
    if (!input.eof()) {
        archive_error(filename, "I/O error after the final robot record");
    }
}

void save_atomic(
    const std::filesystem::path& filename,
    const std::vector<std::shared_ptr<PogobotObject>>& robots
) {
    const auto parent = filename.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }

    // Keep the temporary file on the same filesystem so rename is atomic.
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    std::filesystem::path temporary = filename;
    temporary += ".tmp." + std::to_string(nonce);

    try {
        write_archive(temporary, robots);
#ifdef _WIN32
        // std::filesystem::rename cannot replace an existing file on Windows.
        // MoveFileExW preserves the same-path import/export workflow while
        // keeping replacement within one filesystem operation.
        if (!MoveFileExW(
                temporary.c_str(),
                filename.c_str(),
                MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH
            )) {
            const std::error_code error(
                static_cast<int>(GetLastError()),
                std::system_category()
            );
            throw std::filesystem::filesystem_error(
                "Unable to replace flash-state archive",
                temporary,
                filename,
                error
            );
        }
#else
        std::filesystem::rename(temporary, filename);
#endif
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw;
    }
}

} // namespace pogosim::flash_state
