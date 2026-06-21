
#include "data_logger.h"
#include "utils.h"
#include "version.h"
#include <bit>     // C++20
#include <cmath>   // std::fabs, std::isnan, std::isinf
#include <cstring> // std::memcpy


DataLogger::DataLogger(int64_t flush_row_count) : flush_row_count_(flush_row_count) {
    if (flush_row_count_ <= 0) {
        throw std::runtime_error("flush_row_count must be strictly positive.");
    }
}


// ---- float32 -> binary16 helper ----
DataLogger::half_float_t DataLogger::float_to_half_bits(float f) {
    // IEEE-754 binary32 layout
    uint32_t w;
    static_assert(sizeof(float) == sizeof(uint32_t));
    std::memcpy(&w, &f, sizeof(w));

    const uint32_t sign = (w >> 31) & 0x1;
    const int32_t exp = ((w >> 23) & 0xFF) - 127;  // unbiased
    uint32_t mant = (w & 0x7FFFFF);

    // NaN / Inf handling
    if ((w & 0x7F800000u) == 0x7F800000u) {
        // Inf or NaN -> preserve sign; map to half Inf/NaN
        const uint16_t half_sign = static_cast<uint16_t>(sign) << 15;
        if ((w & 0x007FFFFFu) == 0) {
            // Infinity
            return half_sign | static_cast<uint16_t>(0x7C00);
        } else {
            // NaN: quiet NaN payload
            return half_sign | static_cast<uint16_t>(0x7E00);
        }
    }

    // Underflow to zero or subnormal half
    if (exp < -14) {
        // Too small for normal half; may be subnormal or zero
        // Convert by shifting the mantissa with implicit 1
        if (exp < -24) {
            // Underflow to signed zero
            return static_cast<uint16_t>(sign << 15);
        }
        // Subnormal half: exp = -14, shift mant accordingly
        uint32_t shifted = (mant | 0x00800000u) >> static_cast<uint32_t>(-exp - 14 + 1);
        // Round to nearest, ties to even (add 1 if LSBs indicate rounding)
        uint16_t half_mant = static_cast<uint16_t>((shifted + 1) >> 1);
        return static_cast<uint16_t>((sign << 15) | half_mant);
    }

    // Overflow to Inf
    if (exp > 15) {
        return static_cast<uint16_t>((sign << 15) | 0x7C00);
    }

    // Normal half number
    uint16_t half_exp = static_cast<uint16_t>(exp + 15);
    // Take top 10 bits of mantissa with rounding
    uint32_t half_mant_rounded = (mant + 0x00001000u) >> 13; // add 0.5 ulp then shift

    // Handle rounding overflow in mantissa
    if (half_mant_rounded == 0x400) { // 11 bits set -> carry into exponent
        half_mant_rounded = 0;
        ++half_exp;
        if (half_exp >= 0x1F) {
            // becomes Inf
            return static_cast<uint16_t>((sign << 15) | 0x7C00);
        }
    }

    return static_cast<uint16_t>((sign << 15) | (half_exp << 10) | (half_mant_rounded & 0x3FF));
}


// Destructor to flush and close file
DataLogger::~DataLogger() {
    if (file_opened_) {
        try {
            flush();
        } catch (const std::exception& e) {
            glogger->warn("Failed to flush buffered rows before closing Feather writer: {}", e.what());
        }

        if (!writer_->Close().ok()) {
            glogger->warn("Failed to close Feather writer properly");
        }
        if (!outfile_->Close().ok()) {
            glogger->warn("Failed to close file properly");
        }
    }
}

void DataLogger::add_metadata(const std::string& key, const std::string& value) {
    if (file_opened_) {
        throw std::runtime_error("Cannot add metadata after the file has been opened.");
    }
    if (key.empty()) {
        throw std::runtime_error("Metadata key must not be empty.");
    }
    if (user_metadata_.contains(key)) {
        throw std::runtime_error("Metadata key '" + key + "' already exists.");
    }
    user_metadata_[key] = value;    // Overwrite if the key already exists
}

void DataLogger::set_logged_fields(const std::vector<std::string>& field_names) {
    if (file_opened_) {
        throw std::runtime_error("Cannot configure logged fields after the file has been opened.");
    }
    logged_fields_ = std::unordered_set<std::string>(field_names.begin(), field_names.end());
}

// Add fields dynamically before opening the file
void DataLogger::add_field(const std::string& name, std::shared_ptr<arrow::DataType> type, bool ignore_existing_name) {
    if (file_opened_) {
        throw std::runtime_error("Cannot add fields after the file has been opened.");
    }
    if (!field_is_logged(name)) {
        return;
    }
    if (column_indices_.find(name) != column_indices_.end()) {
        if (ignore_existing_name)
            return;
        throw std::runtime_error("Field '" + name + "' already exists in the schema.");
    }
    fields_.push_back(arrow::field(name, type));
    column_indices_[name] = fields_.size() - 1;  // Store index for quick lookup
}

// Open the file after schema is defined
void DataLogger::open_file(const std::string& filename) {
    if (fields_.empty()) {
        throw std::runtime_error("Schema is empty. Please add fields before opening the file.");
    }

    // Define schema
    schema_ = arrow::schema(fields_);

    // Assemble built-in and user metadata
    std::vector<std::string> keys = {"program_version"};
    std::vector<std::string> values = {POGOSIM_VERSION};
    for (const auto& kv : user_metadata_) {
        keys.push_back(kv.first);
        values.push_back(kv.second);
    }
    schema_ = schema_->WithMetadata(
        std::make_shared<arrow::KeyValueMetadata>(std::move(keys), std::move(values)));

    // Check if parent directory exists
    ensure_directories_exist(filename);

    // Open file for writing
    auto outfile_result = arrow::io::FileOutputStream::Open(filename);
    if (!outfile_result.ok()) {
        throw std::runtime_error("Failed to open file for writing");
    }
    outfile_ = *outfile_result;

    // Set up Feather writer with Zstd compression
    arrow::ipc::IpcWriteOptions options = arrow::ipc::IpcWriteOptions::Defaults();
    options.codec = *arrow::util::Codec::Create(arrow::Compression::ZSTD);

    auto writer_result = arrow::ipc::MakeFileWriter(outfile_, schema_, options);
    if (!writer_result.ok()) {
        throw std::runtime_error("Failed to create Feather writer");
    }
    writer_ = *writer_result;

    // Initialize persistent builders used for buffered writing
    initialize_builders();

    file_opened_ = true;
    buffered_row_count_ = 0;

    // Initialize row values (set to empty so missing values are written as null)
    reset_row();
}

// Set value of a specific column in the current row (Overloaded for different types)
void DataLogger::set_value(const std::string& column_name, int64_t value) {
    if (!field_is_logged(column_name)) {
        return;
    }
    check_column(column_name);
    row_values_[column_name] = value;
}

void DataLogger::set_value(const std::string& column_name, int32_t value) {
    if (!field_is_logged(column_name)) {
        return;
    }
    check_column(column_name);
    row_values_[column_name] = value;
}

void DataLogger::set_value(const std::string& column_name, int16_t value) {
    if (!field_is_logged(column_name)) {
        return;
    }
    check_column(column_name);
    row_values_[column_name] = value;
}

void DataLogger::set_value(const std::string& column_name, int8_t value) {
    if (!field_is_logged(column_name)) {
        return;
    }
    check_column(column_name);
    row_values_[column_name] = value;
}

void DataLogger::set_value(const std::string& column_name, float value) {
    if (!field_is_logged(column_name)) {
        return;
    }
    check_column(column_name);
    row_values_[column_name] = value;
}

void DataLogger::set_value(const std::string& column_name, double value) {
    if (!field_is_logged(column_name)) {
        return;
    }
    check_column(column_name);
    row_values_[column_name] = value;
}

void DataLogger::set_value(const std::string& column_name, const std::string& value) {
    if (!field_is_logged(column_name)) {
        return;
    }
    check_column(column_name);
    row_values_[column_name] = value;
}

void DataLogger::set_value(const std::string& column_name, bool value) {
    if (!field_is_logged(column_name)) {
        return;
    }
    check_column(column_name);
    row_values_[column_name] = value;
}

void DataLogger::set_value_float16(const std::string& column_name, float value) {
    if (!field_is_logged(column_name)) {
        return;
    }
    check_column(column_name);
    row_values_[column_name] = float_to_half_bits(value);
}

void DataLogger::save_row() {
    if (!file_opened_) {
        throw std::runtime_error("File must be opened before saving data.");
    }

//    // Check if the row is complete (all columns values were provided)
//    std::vector<std::string> missing;
//    for (const auto& field : fields_) {
//        if (!row_values_.contains(field->name())) {
//            missing.push_back(field->name());
//        }
//    }
//    if (!missing.empty()) {
//        std::ostringstream oss;
//        oss << "Incomplete row. Missing values for: ";
//        for (size_t i = 0; i < missing.size(); ++i) {
//            oss << "'" << missing[i] << (i + 1 == missing.size() ? "'" : "', ");
//        }
//        throw std::runtime_error(oss.str());
//    }

    // Append the current row to the persistent builders
    append_current_row_to_builders();
    ++buffered_row_count_;

    // Write a record batch only when enough rows have been accumulated
    if (buffered_row_count_ >= flush_row_count_) {
        flush();
    }

    // Reset row values for next iteration
    reset_row();
}

void DataLogger::flush() {
    if (!file_opened_) {
        throw std::runtime_error("File must be opened before flushing data.");
    }

    // Nothing buffered: nothing to do
    if (buffered_row_count_ == 0) {
        return;
    }

    // Finalize arrays from persistent builders
    std::vector<std::shared_ptr<arrow::Array>> data_arrays;
    data_arrays.reserve(builders_.size());

    for (size_t i = 0; i < builders_.size(); ++i) {
        std::shared_ptr<arrow::Array> array;
        if (!builders_[i]->Finish(&array).ok()) {
            throw std::runtime_error("Failed to finalize data array.");
        }
        data_arrays.push_back(array);
    }

    // Combine into a RecordBatch
    auto batch = arrow::RecordBatch::Make(schema_, buffered_row_count_, data_arrays);

    // Write batch to the Feather file
    if (!writer_->WriteRecordBatch(*batch).ok()) {
        throw std::runtime_error("Failed to write record batch.");
    }

    // Recreate fresh persistent builders for the next buffered chunk
    initialize_builders();
    buffered_row_count_ = 0;
}

bool DataLogger::column_exists(const std::string& column_name) {
    return column_indices_.find(column_name) != column_indices_.end();
}

bool DataLogger::column_value_already_set(const std::string& column_name) {
    check_column(column_name);
    return row_values_.find(column_name) != row_values_.end();
}

bool DataLogger::field_is_logged(const std::string& column_name) const {
    return !logged_fields_ || logged_fields_->contains(column_name);
}

void DataLogger::check_column(const std::string& column_name) {
    if (!file_opened_) {
        throw std::runtime_error("File must be opened before setting values.");
    }
    if (column_indices_.find(column_name) == column_indices_.end()) {
        throw std::runtime_error("Column '" + column_name + "' does not exist.");
    }
}

void DataLogger::reset_row() {
    row_values_.clear();
}

void DataLogger::initialize_builders() {
    builders_.clear();
    builders_.resize(fields_.size());

    for (size_t i = 0; i < fields_.size(); ++i) {
        auto field_type = fields_[i]->type();

        if (field_type->id() == arrow::Type::INT64) {
            builders_[i] = std::make_shared<arrow::Int64Builder>();
            if (!std::static_pointer_cast<arrow::Int64Builder>(builders_[i])->Reserve(flush_row_count_).ok()) {
                throw std::runtime_error("Failed to reserve INT64 builder capacity.");
            }
        } else if (field_type->id() == arrow::Type::INT32) {
            builders_[i] = std::make_shared<arrow::Int32Builder>();
            if (!std::static_pointer_cast<arrow::Int32Builder>(builders_[i])->Reserve(flush_row_count_).ok()) {
                throw std::runtime_error("Failed to reserve INT32 builder capacity.");
            }
        } else if (field_type->id() == arrow::Type::INT16) {
            builders_[i] = std::make_shared<arrow::Int16Builder>();
            if (!std::static_pointer_cast<arrow::Int16Builder>(builders_[i])->Reserve(flush_row_count_).ok()) {
                throw std::runtime_error("Failed to reserve INT16 builder capacity.");
            }
        } else if (field_type->id() == arrow::Type::INT8) {
            builders_[i] = std::make_shared<arrow::Int8Builder>();
            if (!std::static_pointer_cast<arrow::Int8Builder>(builders_[i])->Reserve(flush_row_count_).ok()) {
                throw std::runtime_error("Failed to reserve INT8 builder capacity.");
            }
        } else if (field_type->id() == arrow::Type::FLOAT) {
            builders_[i] = std::make_shared<arrow::FloatBuilder>();
            if (!std::static_pointer_cast<arrow::FloatBuilder>(builders_[i])->Reserve(flush_row_count_).ok()) {
                throw std::runtime_error("Failed to reserve FLOAT builder capacity.");
            }
        } else if (field_type->id() == arrow::Type::DOUBLE) {
            builders_[i] = std::make_shared<arrow::DoubleBuilder>();
            if (!std::static_pointer_cast<arrow::DoubleBuilder>(builders_[i])->Reserve(flush_row_count_).ok()) {
                throw std::runtime_error("Failed to reserve DOUBLE builder capacity.");
            }
        } else if (field_type->id() == arrow::Type::STRING) {
            builders_[i] = std::make_shared<arrow::StringBuilder>();
            if (!std::static_pointer_cast<arrow::StringBuilder>(builders_[i])->Reserve(flush_row_count_).ok()) {
                throw std::runtime_error("Failed to reserve STRING builder capacity.");
            }
        } else if (field_type->id() == arrow::Type::BOOL) {
            builders_[i] = std::make_shared<arrow::BooleanBuilder>();
            if (!std::static_pointer_cast<arrow::BooleanBuilder>(builders_[i])->Reserve(flush_row_count_).ok()) {
                throw std::runtime_error("Failed to reserve BOOL builder capacity.");
            }
        } else if (field_type->id() == arrow::Type::HALF_FLOAT) {
            builders_[i] = std::make_shared<arrow::HalfFloatBuilder>();
            if (!std::static_pointer_cast<arrow::HalfFloatBuilder>(builders_[i])->Reserve(flush_row_count_).ok()) {
                throw std::runtime_error("Failed to reserve HALF_FLOAT builder capacity.");
            }
        } else {
            throw std::runtime_error("Unsupported data type for column: " + fields_[i]->name());
        }
    }
}

void DataLogger::append_current_row_to_builders() {
    for (size_t i = 0; i < fields_.size(); ++i) {
        auto field_name = fields_[i]->name();
        auto field_type = fields_[i]->type();
        arrow::Status status;

        // Check if the field exists in row_values_, otherwise append null
        if (row_values_.find(field_name) == row_values_.end()) {
            // Handle missing data gracefully
            if (field_type->id() == arrow::Type::INT64) {
                status = std::static_pointer_cast<arrow::Int64Builder>(builders_[i])->AppendNull();
            } else if (field_type->id() == arrow::Type::INT32) {
                status = std::static_pointer_cast<arrow::Int32Builder>(builders_[i])->AppendNull();
            } else if (field_type->id() == arrow::Type::INT16) {
                status = std::static_pointer_cast<arrow::Int16Builder>(builders_[i])->AppendNull();
            } else if (field_type->id() == arrow::Type::INT8) {
                status = std::static_pointer_cast<arrow::Int8Builder>(builders_[i])->AppendNull();
            } else if (field_type->id() == arrow::Type::FLOAT) {
                status = std::static_pointer_cast<arrow::FloatBuilder>(builders_[i])->AppendNull();
            } else if (field_type->id() == arrow::Type::DOUBLE) {
                status = std::static_pointer_cast<arrow::DoubleBuilder>(builders_[i])->AppendNull();
            } else if (field_type->id() == arrow::Type::STRING) {
                status = std::static_pointer_cast<arrow::StringBuilder>(builders_[i])->AppendNull();
            } else if (field_type->id() == arrow::Type::BOOL) {
                status = std::static_pointer_cast<arrow::BooleanBuilder>(builders_[i])->AppendNull();
            } else if (field_type->id() == arrow::Type::HALF_FLOAT) {
                status = std::static_pointer_cast<arrow::HalfFloatBuilder>(builders_[i])->AppendNull();
            } else {
                throw std::runtime_error("Unsupported data type for column: " + field_name);
            }
        } else {
            // Append actual values
            if (field_type->id() == arrow::Type::INT64) {
                status = std::static_pointer_cast<arrow::Int64Builder>(builders_[i])->Append(
                    std::get<int64_t>(row_values_[field_name])
                );
            } else if (field_type->id() == arrow::Type::INT32) {
                status = std::static_pointer_cast<arrow::Int32Builder>(builders_[i])->Append(
                    std::get<int32_t>(row_values_[field_name])
                );
            } else if (field_type->id() == arrow::Type::INT16) {
                status = std::static_pointer_cast<arrow::Int16Builder>(builders_[i])->Append(
                    std::get<int16_t>(row_values_[field_name])
                );
            } else if (field_type->id() == arrow::Type::INT8) {
                status = std::static_pointer_cast<arrow::Int8Builder>(builders_[i])->Append(
                    std::get<int8_t>(row_values_[field_name])
                );
            } else if (field_type->id() == arrow::Type::FLOAT) {
                status = std::static_pointer_cast<arrow::FloatBuilder>(builders_[i])->Append(
                    std::get<float>(row_values_[field_name])
                );
            } else if (field_type->id() == arrow::Type::DOUBLE) {
                status = std::static_pointer_cast<arrow::DoubleBuilder>(builders_[i])->Append(
                    std::get<double>(row_values_[field_name])
                );
            } else if (field_type->id() == arrow::Type::STRING) {
                status = std::static_pointer_cast<arrow::StringBuilder>(builders_[i])->Append(
                    std::get<std::string>(row_values_[field_name])
                );
            } else if (field_type->id() == arrow::Type::BOOL) {
                status = std::static_pointer_cast<arrow::BooleanBuilder>(builders_[i])->Append(
                    std::get<bool>(row_values_[field_name])
                );
            } else if (field_type->id() == arrow::Type::HALF_FLOAT) {
                status = std::static_pointer_cast<arrow::HalfFloatBuilder>(builders_[i])->Append(
                    std::get<half_float_t>(row_values_[field_name])
                );
            } else {
                throw std::runtime_error("Unsupported data type for column: " + field_name);
            }
        }

        // Check if append operation was successful
        if (!status.ok()) {
            throw std::runtime_error("Failed to append value for column: " + field_name);
        }
    }
}
