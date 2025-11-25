
#ifndef DATA_LOGGER_H
#define DATA_LOGGER_H

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <arrow/ipc/api.h>
#include <arrow/util/compression.h>
#include <unordered_map>
#include <vector>

/**
 * @brief DataLogger class for writing data to a Feather file using Apache Arrow.
 *
 * This class allows dynamic creation of a schema by adding fields prior to opening the file.
 * Once the schema is defined, data rows can be built by setting individual column values and
 * saved row-by-row into a Feather file using Zstd compression.
 */
class DataLogger {
public:
    /**
     * @brief Default constructor.
     */
    DataLogger() = default;

    /**
     * @brief Destructor.
     *
     * Closes the Feather writer and the output file if they are open. Warnings are logged if either
     * the writer or the file cannot be closed properly.
     */
    virtual ~DataLogger();

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
     * @brief Adds a new field to the schema.
     *
     * Adds a field with the specified name and data type to the internal schema. This must be called
     * before the file is opened.
     *
     * @param name The name of the field.
     * @param type The Arrow data type for the field.
     *
     * @throw std::runtime_error if called after the file has been opened.
     */
    void add_field(const std::string& name, std::shared_ptr<arrow::DataType> type);

    /**
     * @brief Opens the output file for writing.
     *
     * Constructs the schema from the added fields, attaches custom metadata (such as the program version),
     * ensures that the parent directory exists, and opens the file for writing. A Feather writer is then
     * created with Zstd compression. The current row values are initialized.
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
     * @param value The boolean value to set.
     *
     * @throw std::runtime_error if the file is not open or the column does not exist.
     */
    void set_value_float16(const std::string& column_name, float value);


    /**
     * @brief Saves the current row to the Feather file.
     *
     * Constructs Arrow array builders for each column based on the field type, appends the current row's
     * values (or nulls if missing), finalizes the arrays, and creates a record batch which is then written
     * to the file. After writing, the row is reset to its default state.
     *
     * @throw std::runtime_error if any step in appending data, finalizing arrays, or writing the record batch fails.
     */
    void save_row();

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
    /// Mapping from column names to their index positions in the schema.
    std::unordered_map<std::string, size_t> column_indices_;
    /// Current row values stored as a variant of supported types.
    std::unordered_map<std::string, std::variant<int64_t, int32_t, int16_t, int8_t, double, std::string, bool, half_float_t>> row_values_;
    /// Flag indicating whether the file has been opened.
    bool file_opened_ = false;
    /// Stores user-supplied metadata until the file is opened.
    std::unordered_map<std::string, std::string> user_metadata_;

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
     * @brief Resets the current row values to default.
     *
     * Initializes all row values for each field to a default value (zero for numeric types).
     */
    void reset_row();

    // Convert float32 -> IEEE-754 half payload (16-bit)
    static half_float_t float_to_half_bits(float f);
};


#endif // DATA_LOGGER_H

