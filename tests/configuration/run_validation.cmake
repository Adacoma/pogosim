# Exercise the public CLI through a real controller, never a mock parser.
set(TEST_DIR "${CMAKE_CURRENT_BINARY_DIR}/configuration-cli")
file(MAKE_DIRECTORY "${TEST_DIR}")
set(CONFIG "${TEST_DIR}/check.yaml")
set(FLASH "${TEST_DIR}/must-not-create.pgflash")
set(OUTPUT "${TEST_DIR}/must-not-create.feather")
set(SENTINEL "${TEST_DIR}/keep.png")
file(WRITE "${SENTINEL}" "keep this existing output")
file(WRITE "${CONFIG}" "seed: invalid\narena_file: does-not-exist.csv\ntime_step: 0.01\nGUI: true\nenable_data_logging: true\ndelete_old_files: true\nframes_name: '${TEST_DIR}/keep'\ndata_filename: '${OUTPUT}'\nflash_state: {input_file: '${FLASH}', output_file: '${FLASH}'}\nparameters: {custom_unknown_key: anything}\n")

function(run_case expected_status expected_message)
    execute_process(COMMAND "${PROGRAM}" ${ARGN}
        WORKING_DIRECTORY "${SOURCE_DIR}" RESULT_VARIABLE status
        OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 15)
    if(NOT "${status}" STREQUAL "${expected_status}")
        message(FATAL_ERROR "Unexpected status ${status}: ${output}${error}")
    endif()
    string(FIND "${output}${error}" "${expected_message}" found)
    if(found EQUAL -1)
        message(FATAL_ERROR "Missing '${expected_message}': ${output}${error}")
    endif()
endfunction()

# Explicit seed overrides the invalid YAML seed, just as in normal runs.
run_case(0 "Core configuration checks passed" -c "${CONFIG}" --check-config -s 7)
run_case(1 "Invalid configuration 'seed'" -c "${CONFIG}" --check-config)
if(EXISTS "${FLASH}" OR EXISTS "${OUTPUT}")
    message(FATAL_ERROR "Check-only mode created flash/output files")
endif()
file(READ "${SENTINEL}" sentinel_content)
if(NOT sentinel_content STREQUAL "keep this existing output")
    message(FATAL_ERROR "Check-only mode deleted/changed an existing output")
endif()
file(WRITE "${CONFIG}" "time_step: 0\n")
run_case(1 "Invalid configuration 'time_step'" -c "${CONFIG}" --check-config)
run_case(1 "Invalid configuration 'time_step'" -c "${CONFIG}" --strict-config -g)

# A valid strict run also exercises the C init_from_configuration bridge.
file(READ "${NORMAL_CONFIG}" normal)
file(WRITE "${CONFIG}" "${normal}")
run_case(0 "" -c "${CONFIG}" --strict-config -g)
string(REPLACE "test_mode: 0" "test_mode: invalid" invalid_controller "${normal}")
file(WRITE "${CONFIG}" "${invalid_controller}")
run_case(2 "Invalid configuration 'parameters.test_mode'" -c "${CONFIG}" --strict-config -g)
# The same invalid controller value retains its old fallback without the flag.
run_case(0 "" -c "${CONFIG}" -g)
