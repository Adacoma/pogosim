set(TEST_DIR "${BUILD_DIR}/regression/installed-consumer")
set(prefix "${TEST_DIR}/prefix with spaces")
function(checked_command)
    execute_process(COMMAND ${ARGV} RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
    if(NOT "${status}" STREQUAL "0")
        message(FATAL_ERROR "Installed-consumer command failed (${status}): ${output}${error}")
    endif()
endfunction()
checked_command("${CMAKE_COMMAND}" --install "${BUILD_DIR}" --prefix "${prefix}" --config "${CONFIG}")
checked_command("${CMAKE_COMMAND}"
    -S "${SOURCE_DIR}/tests/simulator/installed_consumer" -B "${TEST_DIR}/build" -G "${GENERATOR}"
    "-DCMAKE_BUILD_TYPE=${CONFIG}" "-DCMAKE_C_COMPILER=${C_COMPILER}" "-DCMAKE_CXX_COMPILER=${CXX_COMPILER}"
    "-DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN}" "-DVCPKG_TARGET_TRIPLET=${TRIPLET}"
    "-Dyaml-cpp_DIR=${YAML_DIR}" "-Dbox2d_DIR=${BOX2D_DIR}" "-DArrow_DIR=${ARROW_DIR}"
    "-DCMAKE_C_FLAGS=${C_FLAGS}" "-DCMAKE_CXX_FLAGS=${CXX_FLAGS}"
    "-DCMAKE_EXE_LINKER_FLAGS=${LINK_FLAGS}"
    "-DINSTALL_PREFIX=${prefix}" "-DINSTALLED_LIBRARY=${prefix}/${INSTALL_LIBDIR}/${LIBRARY_NAME}"
    "-DCONTROLLER_SOURCE=${SOURCE_DIR}/tests/robot_coroutine/legacy_controller.c"
    "-DCPP_CONTROLLER_SOURCE=${CPP_CONTROLLER_SOURCE}"
    "-DSIMULATION_CONFIG=${BUILD_DIR}/coroutine-0.yaml" "-DSOURCE_DIR=${SOURCE_DIR}")
checked_command("${CMAKE_COMMAND}" --build "${TEST_DIR}/build" --config "${CONFIG}" --parallel 2)
checked_command("${CMAKE_CTEST_COMMAND}" --test-dir "${TEST_DIR}/build" -C "${CONFIG}" --output-on-failure)
