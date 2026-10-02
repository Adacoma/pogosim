#include "pogosim/robot.h"
#include "pogosim/flash_state.h"
#include "pogosim/data_logger.h"
#include "pogosim/version.h"
#undef main
#include "test_support.h"
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>

namespace {
template<typename T> T arrow_value(arrow::Result<T> result) {
    check(result.ok(), result.status().ToString());
    return std::move(result).ValueOrDie();
}

std::vector<unsigned char> read_bytes(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    check(file.good(), "Cannot read test archive");
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

void write_bytes(const std::filesystem::path& path, const std::vector<unsigned char>& bytes) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    check(file.good(), "Cannot write test archive");
}

std::shared_ptr<PogobotObject> flash_robot(uint16_t id) {
    Configuration config(YAML::Load("{geometry: disk, radius: 1, magnetometer_enabled: false, msg_success_rate: {type: static, rate: 1}}"));
    auto robot = std::make_shared<PogobotObject>(nullptr, id, 0, 0, 16, config, "robots", false);
    // Never read fresh indeterminate flash. Every byte used by these tests is
    // explicitly initialized before calling the persistence implementation.
    std::fill(std::begin(robot->flash_memory_authorized_section), std::end(robot->flash_memory_authorized_section), static_cast<unsigned char>(id + 30));
    robot->motor_dir_mem[0] = id & 1;
    robot->motor_power_mem[0] = 200 + id;
    return robot;
}

std::shared_ptr<arrow::ipc::RecordBatchFileReader> reader(const std::filesystem::path& path) {
    return arrow_value(arrow::ipc::RecordBatchFileReader::Open(arrow_value(arrow::io::ReadableFile::Open(path.string()))));
}
}

void test_flash(const std::filesystem::path& directory) {
    using namespace pogosim::flash_state;
    const auto empty = directory / "nested/empty.pgflash";
    check(create_empty_if_missing(empty), "Missing archive was not created");
    check(!create_empty_if_missing(empty), "Existing archive was replaced");
    check(std::filesystem::file_size(empty) == 18, "Fresh-state marker contains unexpected data");
    auto first = flash_robot(0), second = flash_robot(1);
    check(!load(empty, {first}), "Empty archive was not recognized");
    check(first->flash_memory_authorized_section[0] == 30, "Empty archive fabricated flash contents");
    const auto full = directory / "full.pgflash";
    save_atomic(full, {second, first});
    const auto original = read_bytes(full);
    const auto ordered = directory / "ordered.pgflash";
    save_atomic(ordered, {first, second});
    check(read_bytes(ordered) == original, "Archive depends on vector order");
    first->flash_memory_authorized_section[0] = 99;
    first->motor_power_mem[0] = 999;
    check(load(full, {first}), "Subset import did not restore its record");
    check(first->flash_memory_authorized_section[0] == 30 && first->motor_power_mem[0] == 200,
          "Flash or motor calibration was not restored");
    check(first->flash_memory_authorized_section[flash_memory_authorized_section_size - 1] == 30,
          "Full v3 flash region was not restored");
    first->flash_memory_authorized_section[flash_memory_authorized_section_size - 1] = 42;
    save_atomic(full, {first}, full); // Same path must retain the unused second record.
    std::fill(std::begin(second->flash_memory_authorized_section), std::end(second->flash_memory_authorized_section), 0);
    check(load(full, {first, second}) && second->flash_memory_authorized_section[0] == 31,
          "Subset same-path export lost the unused record");
    check(first->flash_memory_authorized_section[flash_memory_authorized_section_size - 1] == 42,
          "Subset export discarded the updated last page");

    const auto invalid = directory / "invalid.pgflash";
    auto reject_bytes = [&](std::vector<unsigned char> bytes, const std::string& message) {
        write_bytes(invalid, bytes);
        expect_error([&] { load(invalid, {first}); }, message);
    };
    auto bytes = original; bytes[0] ^= 1; reject_bytes(bytes, "signature");
    bytes = original; bytes[8] = 2; reject_bytes(bytes, "unsupported format version");
    bytes = original; bytes[10] ^= 1; reject_bytes(bytes, "flash size");
    bytes = original; bytes.resize(7); reject_bytes(bytes, "truncated");
    bytes = original; bytes.pop_back(); reject_bytes(bytes, "truncated");
    bytes = original; bytes.push_back(0); reject_bytes(bytes, "trailing data");
    bytes = original;
    const std::size_t record_size = 2 + 2 + std::string("robots").size() + 3 + 6 + flash_memory_authorized_section_size + 8;
    std::copy_n(original.begin() + 18, record_size, bytes.begin() + 18 + record_size);
    reject_bytes(bytes, "duplicate robot record");
    bytes = original; bytes[40] ^= 1;
    first->flash_memory_authorized_section[0] = 77;
    reject_bytes(bytes, "checksum mismatch");
    check(first->flash_memory_authorized_section[0] == 77, "A bad-checksum record changed current flash");
    auto absent = flash_robot(2);
    expect_error([&] { load(full, {absent}); }, "missing current robot");
    expect_error([&] { load(full, {first, second, absent}); }, "expected at least");
    first->category = "wrong-category";
    expect_error([&] { load(full, {first}); }, "missing current robot");
    first->category = "robots";
    expect_error([&] { save_atomic(directory / "duplicate.pgflash", {first, first}); }, "duplicate robot identity");

    // A corrupt imported source must not destroy an existing destination or
    // leave a temporary archive behind while preserving its unused records.
    write_bytes(invalid, bytes);
    const auto before_failure = read_bytes(full);
    expect_error([&] { save_atomic(full, {first}, invalid); }, "checksum mismatch");
    check(read_bytes(full) == before_failure, "Failed export replaced the destination");
    for (const auto& entry : std::filesystem::directory_iterator(directory))
        check(entry.path().filename().string().find(".tmp.") == std::string::npos, "Failed export left temporary data");
    save_atomic(empty, {first}, empty);
    check(load(empty, {first}), "Empty same-path archive did not become a full archive");
}

void test_logging(const std::filesystem::path& directory) {
    expect_error([] { DataLogger bad(0); }, "strictly positive");
    expect_error([] { DataLogger bad(-1); }, "strictly positive");
    const auto path = directory / "typed.feather";
    {
        DataLogger logger(2);
        expect_error([&] { logger.save_row(); }, "must be opened");
        expect_error([&] { logger.flush(); }, "must be opened");
        expect_error([&] { logger.open_file(path.string()); }, "Schema is empty");
        expect_error([&] { logger.add_metadata("", "bad"); }, "must not be empty");
        logger.add_metadata("fixture", "regression");
        expect_error([&] { logger.add_metadata("fixture", "duplicate"); }, "already exists");
        logger.add_field("i8", arrow::int8()); logger.add_field("i16", arrow::int16());
        logger.add_field("i32", arrow::int32()); logger.add_field("i64", arrow::int64());
        logger.add_field("f32", arrow::float32()); logger.add_field("f64", arrow::float64());
        logger.add_field("text", arrow::utf8()); logger.add_field("flag", arrow::boolean());
        logger.add_field("half", arrow::float16());
        expect_error([&] { logger.add_field("i32", arrow::int32()); }, "already exists");
        logger.add_field("i32", arrow::int32(), true);
        expect_error([&] { logger.set_value("i32", int32_t{0}); }, "must be opened");
        logger.open_file(path.string());
        expect_error([&] { logger.add_metadata("late", "bad"); }, "after the file");
        expect_error([&] { logger.add_field("late", arrow::int32()); }, "after the file");
        expect_error([&] { logger.set_logged_fields({"i32"}); }, "after the file");
        expect_error([&] { logger.set_value("unknown", int32_t{0}); }, "does not exist");
        for (int32_t row = 0; row < 5; ++row) {
            check(!logger.column_value_already_set("i32"), "Row values leaked into the next row");
            logger.set_value("i32", row);
            check(logger.column_value_already_set("i32"), "Column-set tracking failed");
            if (row == 0) {
                logger.set_value("i8", int8_t{-7}); logger.set_value("i16", int16_t{-1234});
                logger.set_value("i64", int64_t{9007199254740993LL});
                logger.set_value("f32", 1.25f); logger.set_value("f64", 1.0 / 3.0);
                logger.set_value("text", std::string("pogobot-\xc3\xa9")); logger.set_value("flag", true);
            }
            if (row == 0) logger.set_value_float16("half", 1.5f);
            if (row == 1) logger.set_value_float16("half", -0.0f);
            if (row == 2) logger.set_value_float16("half", std::numeric_limits<float>::infinity());
            if (row == 3) logger.set_value_float16("half", std::numeric_limits<float>::quiet_NaN());
            logger.save_row();
        }
        // The fifth row stays buffered: destruction must flush it and close
        // the IPC footer before the reader can successfully open this file.
    }
    auto file = reader(path);
    check(file->num_record_batches() == 3, "Buffered logging lost batch boundaries or the final partial batch");
    check(file->schema()->num_fields() == 9, "Typed schema changed");
    check(arrow_value(file->schema()->metadata()->Get("program_version")) == POGOSIM_VERSION, "Version metadata lost");
    check(arrow_value(file->schema()->metadata()->Get("fixture")) == "regression", "Custom metadata lost");
    int64_t seen = 0;
    for (int batch_index = 0; batch_index < file->num_record_batches(); ++batch_index) {
        const auto batch = arrow_value(file->ReadRecordBatch(batch_index));
        check(batch->ValidateFull().ok(), "Invalid Arrow record batch");
        const auto ids = std::static_pointer_cast<arrow::Int32Array>(batch->GetColumnByName("i32"));
        const auto half = std::static_pointer_cast<arrow::HalfFloatArray>(batch->GetColumnByName("half"));
        for (int64_t row = 0; row < batch->num_rows(); ++row, ++seen) {
            check(!ids->IsNull(row) && ids->Value(row) == seen, "Buffered rows reordered or lost precision");
            if (seen == 0) {
                check(std::static_pointer_cast<arrow::Int8Array>(batch->GetColumnByName("i8"))->Value(row) == -7, "int8 value changed");
                check(std::static_pointer_cast<arrow::Int16Array>(batch->GetColumnByName("i16"))->Value(row) == -1234, "int16 value changed");
                check(std::static_pointer_cast<arrow::Int64Array>(batch->GetColumnByName("i64"))->Value(row) == 9007199254740993LL, "int64 rounded through a float");
                close_to(std::static_pointer_cast<arrow::FloatArray>(batch->GetColumnByName("f32"))->Value(row), 1.25);
                close_to(std::static_pointer_cast<arrow::DoubleArray>(batch->GetColumnByName("f64"))->Value(row), 1.0 / 3, 1e-12);
                check(std::static_pointer_cast<arrow::StringArray>(batch->GetColumnByName("text"))->GetString(row) == "pogobot-\xc3\xa9", "UTF-8 value changed");
                check(std::static_pointer_cast<arrow::BooleanArray>(batch->GetColumnByName("flag"))->Value(row), "Boolean value changed");
                check(half->Value(row) == 0x3e00, "Normal float16 conversion changed");
            } else {
                for (const char* key : {"i8", "i16", "i64", "f32", "f64", "text", "flag"})
                    check(batch->GetColumnByName(key)->IsNull(row), "Unset column retained the previous row's value");
                if (seen == 1) check(half->Value(row) == 0x8000, "float16 negative zero changed");
                if (seen == 2) check(half->Value(row) == 0x7c00, "float16 infinity changed");
                if (seen == 3) check((half->Value(row) & 0x7c00) == 0x7c00 && (half->Value(row) & 0x03ff), "float16 NaN changed");
                if (seen == 4) check(half->IsNull(row), "Unset float16 is not null");
            }
        }
    }
    check(seen == 5, "Final buffered row missing");
    const auto filtered = directory / "filtered.feather";
    {
        DataLogger logger(2);
        logger.set_logged_fields({"keep"});
        logger.add_field("keep", arrow::int32()); logger.add_field("exclude", arrow::int32());
        check(!logger.column_exists("exclude"), "Excluded field remains in the schema");
        logger.open_file(filtered.string());
        logger.set_value("exclude", int32_t{99}); // Intentionally a supported no-op.
        logger.set_value("keep", int32_t{42}); logger.save_row();
    }
    auto filtered_reader = reader(filtered);
    check(filtered_reader->schema()->num_fields() == 1, "Logged-field filter changed");
}

void verify_simulation_output(const std::filesystem::path& path, bool restricted) {
    auto file = reader(path);
    const auto schema = file->schema();
    const auto config = YAML::Load(arrow_value(schema->metadata()->Get("configuration")));
    check(config["seed"].as<unsigned>() == 17, "Effective CLI seed not recorded");
    check(!YAML::Load(arrow_value(schema->metadata()->Get("arena_polygons"))).IsNull(), "Arena metadata missing");
    check(arrow_value(schema->metadata()->Get("program_version")) == POGOSIM_VERSION, "Simulation version missing");
    check(schema->GetFieldIndex("controller_steps") >= 0, "User-defined field missing");
    if (restricted) check(schema->num_fields() == 4 && schema->GetFieldIndex("x") < 0 && schema->GetFieldIndex("controller_tag") < 0, "Field filtering failed");
    else check(schema->GetFieldIndex("x") >= 0 && schema->GetFieldIndex("controller_tag") >= 0, "Default schema lost fields");
    std::map<int32_t, float> previous_time;
    int64_t rows = 0;
    for (int i = 0; i < file->num_record_batches(); ++i) {
        const auto batch = arrow_value(file->ReadRecordBatch(i));
        check(batch->ValidateFull().ok(), "Invalid simulation batch");
        const auto times = std::static_pointer_cast<arrow::FloatArray>(batch->GetColumnByName("time"));
        const auto ids = std::static_pointer_cast<arrow::Int32Array>(batch->GetColumnByName("robot_id"));
        const auto categories = std::static_pointer_cast<arrow::StringArray>(batch->GetColumnByName("robot_category"));
        const auto steps = std::static_pointer_cast<arrow::Int32Array>(batch->GetColumnByName("controller_steps"));
        for (int64_t row = 0; row < batch->num_rows(); ++row) {
            check(!times->IsNull(row) && !ids->IsNull(row) && !steps->IsNull(row), "Simulation/user fields became null");
            const auto id = ids->Value(row);
            const float time = times->Value(row);
            check(id == 0 || id == 1, "Unexpected robot in output");
            check(time >= 0 && time < 0.031f && time >= previous_time[id], "Time column is invalid or reordered");
            previous_time[id] = time;
            check(steps->Value(row) >= 0 && steps->Value(row) <= 32, "Controller state was not exported");
            if (restricted) check(id == 0 && categories->GetString(row) == "robots", "Category filter leaked peers");
            else {
                const auto tags = std::static_pointer_cast<arrow::StringArray>(batch->GetColumnByName("controller_tag"));
                check(tags->GetString(row) == (id == 0 ? "source" : "receiver"), "User state was swapped between robots");
            }
            ++rows;
        }
    }
    check(rows >= 8 && previous_time.size() == (restricted ? 1u : 2u), "Simulation output is incomplete");
    check(file->num_record_batches() > 1, "Simulation buffer-flush setting was ignored");
}
