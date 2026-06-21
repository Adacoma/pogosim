
#ifndef DATA_LOGGER_H
#define DATA_LOGGER_H

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <arrow/ipc/api.h>
#include <arrow/util/compression.h>
#include <optional>
#include <unordered_set>
#include <unordered_map>
#include <variant>
#include <vector>

/**
 * @brief DataLogger class for writing data to a Feather file using Apache Arrow.
 *
 * This class allows dynamic creation of a schema by adding fields prior to opening the file.
 * Once the schema is defined, data rows can be built by setting individual column values and
 * saved row-by-row into a Feather file using Zstd compression.
 *
 * Internally, rows are buffered and written in larger record batches so that compression is
 * applied on larger chunks of data, which significantly reduces file size and improves write
 * performance compared with writing one record batch per row.
 */
class DataLogger {
public:
    /**
     * @brief Default constructor.
     *
     * Uses a default buffered batch size.
     */
    DataLogger() = default;

    /**
     * @brief Constructor with explicit buffered batch size.
     *
     * @param flush_row_count Number of rows to buffer before writing a record batch.
     *
     * @throw std::runtime_error if flush_row_count is zero.
     */
    explicit DataLogger(int64_t flush_row_count);

    /**
     * @brief Destructor.
     *
     * Flushes any remaining buffered rows, then closes the Feather writer and the output file
     * if they are open. Warnings are logged if flushing or closing fails.
     */
    virtual ~DataLogger();

    /**
     * @brief Checks if the specified column exists
     *
     * @param column_name The name of the column to check.
     *
     * @return whether the column exists
     */
    bool column_exists(const std::string& column_name);

    /**
     * @brief Checks if the specified column value has been set for the current row
     *
     * @param column_name The name of the column to check.
     *
     * @return whether the column value has been set for the current row
     * @throw std::runtime_error if the file is not open or the column does not exist.
     */
    bool column_value_already_set(const std::string& column_name);

    /**
     * @brief Adds arbitrary string metadata to be embedded in the Feather file.
     *
     * This can be called any time **before** `open_file()`.
     * If the same key is supplied more than once the value is overwritten.
     *
     * @param key   The metadata key (UTF-8, non-empty).
     * @param value The metadata value (UTF-8, may be empty).
     *
     * @throw std::runtime_error if the file has already been opened.
     */
    void add_metadata(const std::string& key, const std::string& value);

    /**
     * @brief Restricts the output schema to the listed field names.
     */
    void set_logged_fields(const std::vector<std::string>& field_names);

    /**
     * @brief Adds a new field to the schema.
     *
     * Adds a field with the specified name and data type to the internal schema. This must be called
     * before the file is opened.
     *
     * @param name The name of the field.
     * @param type The Arrow data type for the field.
     * @param ignore_existing_name Whether to ignore an existing field with the same name. If False, throw std::runtime_error.
     *
     * @throw std::runtime_error if called after the file has been opened.
     * @throw std::runtime_error if ignore_existing_name==false and field name already exists
     */
    void add_field(const std::string& name, std::shared_ptr<arrow::DataType> type, bool ignore_existing_name = false);

    /**
     * @brief Opens the output file for writing.
     *
     * Constructs the schema from the added fields, attaches custom metadata (such as the program version),
     * ensures that the parent directory exists, and opens the file for writing. A Feather writer is then
     * created with Zstd compression. Internal persistent builders are initialized, and the current row values
     * are reset.
     *
     * @param filename The path to the file to be written.
     *
     * @throw std::runtime_error if no fields have been added or if file or writer initialization fails.
     */
    void open_file(const std::string& filename);

    /**
     * @brief Sets the value for a specified column in the current row.
     *
     * Overloaded method for int64_t values.
     *
     * @param column_name The name of the column.
     * @param value The int64_t value to set.
     *
     * @throw std::runtime_error if the file is not open or the column does not exist.
     */
    void set_value(const std::string& column_name, int64_t value);

    /**
     * @brief Sets the value for a specified column in the current row.
     *
     * Overloaded method for int32_t values.
     *
     * @param column_name The name of the column.
     * @param value The int32_t value to set.
     *
     * @throw std::runtime_error if the file is not open or the column does not exist.
     */
    void set_value(const std::string& column_name, int32_t value);

    /**
     * @brief Sets the value for a specified column in the current row.
     *
     * Overloaded method for int16_t values.
     *
     * @param column_name The name of the column.
     * @param value The int16_t value to set.
     *
     * @throw std::runtime_error if the file is not open or the column does not exist.
     */
    void set_value(const std::string& column_name, int16_t value);

    /**
     * @brief Sets the value for a specified column in the current row.
     *
     * Overloaded method for int8_t values.
     *
     * @param column_name The name of the column.
     * @param value The int8_t value to set.
     *
     * @throw std::runtime_error if the file is not open or the column does not exist.
     */
    void set_value(const std::string& column_name, int8_t value);

    /**
     * @brief Sets the value for a specified column in the current row.
     *
     * Overloaded method for double values.
     *
     * @param column_name The name of the column.
     * @param value The double value to set.
     *
     * @throw std::runtime_error if the file is not open or the column does not exist.
     */
    void set_value(const std::string& column_name, double value);

    /**
     * @brief Sets the value for a specified column in the current row.
     *
     * Overloaded method for float values.
     *
     * @param column_name The name of the column.
     * @param value The float value to set.
     *
     * @throw std::runtime_error if the file is not open or the column does not exist.
     */
    void set_value(const std::string& column_name, float value);

    /**
     * @brief Sets the value for a specified column in the current row.
     *
     * Overloaded method for string values.
     *
     * @param column_name The name of the column.
     * @param value The string value to set.
     *
     * @throw std::runtime_error if the file is not open or the column does not exist.
     */
    void set_value(const std::string& column_name, const std::string& value);

    /**
     * @brief Sets the value for a specified column in the current row.
     *
     * Overloaded method for boolean values.
     *
     * @param column_name The name of the column.
     * @param value The boolean value to set.
     *
     * @throw std::runtime_error if the file is not open or the column does not exist.
     */
    void set_value(const std::string& column_name, bool value);

    /**
     * @brief Sets the value for a specified column in the current row.
     *
     * Not overloaded method (like set_value methods), to avoid automatic promotions to double
     *
     * @param column_name The name of the column.
     * @param value The float value to convert to Arrow float16 raw payload.
     *
     * @throw std::runtime_error if the file is not open or the column does not exist.
     */
    void set_value_float16(const std::string& column_name, float value);

    /**
     * @brief Saves the current row to the Feather file.
     *
     * Appends the current row's values (or nulls if missing) to persistent Arrow array builders.
     * Once enough rows have been accumulated, a record batch is finalized and written to the file.
     * After appending, the row is reset to its default state.
     *
     * @throw std::runtime_error if any step in appending data or writing the record batch fails.
     */
    void save_row();

    /**
     * @brief Flushes all currently buffered rows to the Feather file.
     *
     * If no rows are buffered, this is a no-op. This may be called manually, but is also called
     * automatically by the destructor before closing the file.
     *
     * @throw std::runtime_error if finalizing arrays or writing the record batch fails.
     */
    void flush();

private:
    // Store Arrow float16 values as their raw 16-bit representation
    using half_float_t = arrow::NumericBuilder<arrow::HalfFloatType>::value_type;

    /// Vector of Arrow fields representing the schema.
    std::vector<std::shared_ptr<arrow::Field>> fields_;
    /// The Arrow schema constructed from the fields.
    std::shared_ptr<arrow::Schema> schema_;
    /// Output stream for writing data to the file.
    std::shared_ptr<arrow::io::OutputStream> outfile_;
    /// Feather writer for writing record batches.
    std::shared_ptr<arrow::ipc::RecordBatchWriter> writer_;
    /// Persistent builders used to buffer multiple rows before writing.
    std::vector<std::shared_ptr<arrow::ArrayBuilder>> builders_;
    /// Mapping from column names to their index positions in the schema.
    std::unordered_map<std::string, size_t> column_indices_;
    /// Current row values stored as a variant of supported types.
    std::unordered_map<std::string, std::variant<int64_t, int32_t, int16_t, int8_t, float, double, std::string, bool, half_float_t>> row_values_;
    /// Flag indicating whether the file has been opened.
    bool file_opened_ = false;
    /// Stores user-supplied metadata until the file is opened.
    std::unordered_map<std::string, std::string> user_metadata_;
    /// Optional allow-list of fields that should be written.
    std::optional<std::unordered_set<std::string>> logged_fields_;
    /// Number of rows to buffer before flushing a record batch.
    int64_t flush_row_count_ = 1024;
    /// Number of rows currently accumulated in builders_.
    int64_t buffered_row_count_ = 0;

    /**
     * @brief Checks if the specified column exists and if the file is open.
     *
     * Verifies that the file has been opened and that the column is defined in the schema.
     *
     * @param column_name The name of the column to check.
     *
     * @throw std::runtime_error if the file is not open or the column does not exist.
     */
    void check_column(const std::string& column_name);

    /**
     * @brief Returns true if a field should be part of the schema/output.
     */
    bool field_is_logged(const std::string& column_name) const;

    /**
     * @brief Resets the current row values to default.
     *
     * Clears all row values so that missing values in the next row are written as null.
     */
    void reset_row();

    /**
     * @brief Creates persistent builders matching the schema.
     *
     * Allocates one builder per field and reserves capacity according to the configured flush size.
     *
     * @throw std::runtime_error if an unsupported type is encountered.
     */
    void initialize_builders();

    /**
     * @brief Appends the current row to the persistent builders.
     *
     * For each field, appends either the current row value or a null if the value is missing.
     *
     * @throw std::runtime_error if an append operation fails.
     */
    void append_current_row_to_builders();

    // Convert float32 -> IEEE-754 half payload (16-bit)
    static half_float_t float_to_half_bits(float f);
};

#endif // DATA_LOGGER_H
