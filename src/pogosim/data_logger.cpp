
#include "data_logger.h"
#include "utils.h"
#include "version.h"


// Destructor to close file
DataLogger::~DataLogger() {
    if (file_opened_) {
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
    user_metadata_[key] = value;    // Overwrite if the key already exists
}

// Add fields dynamically before opening the file
void DataLogger::add_field(const std::string& name, std::shared_ptr<arrow::DataType> type) {
    if (file_opened_) {
        throw std::runtime_error("Cannot add fields after the file has been opened.");
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
    std::vector<std::string> keys   = {"program_version"};
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

    file_opened_ = true;

    // Initialize row values (set to NaN)
    reset_row();
}

// Set value of a specific column in the current row (Overloaded for different types)
void DataLogger::set_value(const std::string& column_name, int64_t value) {
    check_column(column_name);
    row_values_[column_name] = value;
}
void DataLogger::set_value(const std::string& column_name, int32_t value) {
    check_column(column_name);
    row_values_[column_name] = value;
}
void DataLogger::set_value(const std::string& column_name, int16_t value) {
    check_column(column_name);
    row_values_[column_name] = value;
}
void DataLogger::set_value(const std::string& column_name, int8_t value) {
    check_column(column_name);
    row_values_[column_name] = value;
}
void DataLogger::set_value(const std::string& column_name, double value) {
    check_column(column_name);
    row_values_[column_name] = value;
}
void DataLogger::set_value(const std::string& column_name, const std::string& value) {
    check_column(column_name);
    row_values_[column_name] = value;
}
void DataLogger::set_value(const std::string& column_name, bool value) {
    check_column(column_name);
    row_values_[column_name] = value;
}


void DataLogger::save_row() {
    if (!file_opened_) {
        throw std::runtime_error("File must be opened before saving data.");
    }

    // Create builders dynamically based on the schema
    std::vector<std::shared_ptr<arrow::ArrayBuilder>> builders(fields_.size());

    for (size_t i = 0; i < fields_.size(); ++i) {
        auto field_type = fields_[i]->type();
        if (field_type->id() == arrow::Type::INT64) {
            builders[i] = std::make_shared<arrow::Int64Builder>();
        } else if (field_type->id() == arrow::Type::INT32) {
            builders[i] = std::make_shared<arrow::Int32Builder>();
        } else if (field_type->id() == arrow::Type::INT16) {
            builders[i] = std::make_shared<arrow::Int16Builder>();
        } else if (field_type->id() == arrow::Type::INT8) {
            builders[i] = std::make_shared<arrow::Int8Builder>();
        } else if (field_type->id() == arrow::Type::DOUBLE) {
            builders[i] = std::make_shared<arrow::DoubleBuilder>();
        } else if (field_type->id() == arrow::Type::STRING) {
            builders[i] = std::make_shared<arrow::StringBuilder>();
        } else if (field_type->id() == arrow::Type::BOOL) {
            builders[i] = std::make_shared<arrow::BooleanBuilder>();
        } else {
            throw std::runtime_error("Unsupported data type for column: " + fields_[i]->name());
        }
    }

    // Append data for each field
    for (size_t i = 0; i < fields_.size(); ++i) {
        auto field_name = fields_[i]->name();
        auto field_type = fields_[i]->type();
        arrow::Status status;

        // Check if the field exists in row_values_, otherwise use default values
        if (row_values_.find(field_name) == row_values_.end()) {
            // Handle missing data gracefully
            if (field_type->id() == arrow::Type::INT64) {
                status = std::static_pointer_cast<arrow::Int64Builder>(builders[i])->AppendNull();
            } else if (field_type->id() == arrow::Type::INT32) {
                status = std::static_pointer_cast<arrow::Int32Builder>(builders[i])->AppendNull();
            } else if (field_type->id() == arrow::Type::INT16) {
                status = std::static_pointer_cast<arrow::Int16Builder>(builders[i])->AppendNull();
            } else if (field_type->id() == arrow::Type::INT8) {
                status = std::static_pointer_cast<arrow::Int8Builder>(builders[i])->AppendNull();
            } else if (field_type->id() == arrow::Type::DOUBLE) {
                status = std::static_pointer_cast<arrow::DoubleBuilder>(builders[i])->AppendNull();
            } else if (field_type->id() == arrow::Type::STRING) {
                status = std::static_pointer_cast<arrow::StringBuilder>(builders[i])->AppendNull();
            } else if (field_type->id() == arrow::Type::BOOL) {
                status = std::static_pointer_cast<arrow::BooleanBuilder>(builders[i])->AppendNull();
            }
        } else {
            // Append actual values
            if (field_type->id() == arrow::Type::INT64) {
                status = std::static_pointer_cast<arrow::Int64Builder>(builders[i])->Append(
                    std::get<int64_t>(row_values_[field_name])
                );
            } else if (field_type->id() == arrow::Type::INT32) {
                status = std::static_pointer_cast<arrow::Int32Builder>(builders[i])->Append(
                    std::get<int32_t>(row_values_[field_name])
                );
            } else if (field_type->id() == arrow::Type::INT16) {
                status = std::static_pointer_cast<arrow::Int16Builder>(builders[i])->Append(
                    std::get<int16_t>(row_values_[field_name])
                );
            } else if (field_type->id() == arrow::Type::INT8) {
                status = std::static_pointer_cast<arrow::Int8Builder>(builders[i])->Append(
                    std::get<int8_t>(row_values_[field_name])
                );
            } else if (field_type->id() == arrow::Type::DOUBLE) {
                status = std::static_pointer_cast<arrow::DoubleBuilder>(builders[i])->Append(
                    std::get<double>(row_values_[field_name])
                );
            } else if (field_type->id() == arrow::Type::STRING) {
                status = std::static_pointer_cast<arrow::StringBuilder>(builders[i])->Append(
                    std::get<std::string>(row_values_[field_name])
                );
            } else if (field_type->id() == arrow::Type::BOOL) {
                status = std::static_pointer_cast<arrow::BooleanBuilder>(builders[i])->Append(
                    std::get<bool>(row_values_[field_name])
                );
            }
        }

        // Check if append operation was successful
        if (!status.ok()) {
            throw std::runtime_error("Failed to append value for column: " + field_name);
        }
    }

    // Finalize arrays
    std::vector<std::shared_ptr<arrow::Array>> data_arrays;

    for (size_t i = 0; i < builders.size(); ++i) {
        std::shared_ptr<arrow::Array> array;
        if (!builders[i]->Finish(&array).ok()) {
            throw std::runtime_error("Failed to finalize data array.");
        }
        data_arrays.push_back(array);
    }

    // Combine into a RecordBatch
    auto batch = arrow::RecordBatch::Make(schema_, 1, data_arrays);

    // Write batch to the Feather file
    if (!writer_->WriteRecordBatch(*batch).ok()) {
        throw std::runtime_error("Failed to write record batch.");
    }

    // Reset row values for next iteration
    reset_row();
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
    for (const auto& field : fields_) {
        row_values_[field->name()] = int64_t(0);
    }
}

