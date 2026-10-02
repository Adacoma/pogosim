# Keep test infrastructure separate from the installed runtime build.
function(pogosim_regression_executable target)
    add_executable(${target} ${ARGN})
    target_include_directories(${target} PRIVATE "${CMAKE_SOURCE_DIR}/src")
    target_link_libraries(${target} PRIVATE pogosim
        ${SDL2_GFX_LIBRARIES} ${SDL2TTF_LIBRARY} Arrow::arrow_shared)
    if(NOT MSVC)
        target_link_libraries(${target} PRIVATE box2d)
    endif()
endfunction()

pogosim_regression_executable(test_simulator_models
    tests/simulator/driver.cpp tests/simulator/models.cpp
    tests/simulator/factories.cpp tests/simulator/persistence.cpp
    tests/simulator/stub_controller.c)
pogosim_regression_executable(test_simulator_regression
    tests/simulator/controller.c tests/simulator/controller_bridge.cpp)
# Reuse the hardware-compatible example unchanged on every CMake toolchain.
pogosim_regression_executable(test_flash_archive_controller examples/test_flash_state/main.c)

set(REGRESSION_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/regression")
set(MODEL_CASES geometry lighting neighbors probabilities flash logging)
foreach(type pogobot pogobject pogowall membrane rectmembrane passive_object
        active_object static_light rotating_ray_of_light alternating_rays_of_light)
    foreach(boundary solid periodic)
        list(APPEND MODEL_CASES "factory_${type}_${boundary}")
    endforeach()
endforeach()
foreach(case IN LISTS MODEL_CASES)
    add_test(NAME "simulator_${case}" COMMAND test_simulator_models
        "${case}" "${REGRESSION_OUTPUT}" "${CMAKE_SOURCE_DIR}")
    set_tests_properties("simulator_${case}" PROPERTIES
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        ENVIRONMENT "SDL_VIDEODRIVER=dummy" TIMEOUT 30)
endforeach()

foreach(case communication_delivery communication_drop logging_all logging_restricted
        periodic_neighbors_and_wrap solid_distant_neighbors schema_callback_failure
        export_callback_failure startup_yaml startup_missing_arena startup_unknown_object
        startup_unknown_geometry startup_empty_output startup_output_directory
        startup_empty_console startup_unknown_magnetometer)
    add_test(NAME "simulator_${case}" COMMAND ${CMAKE_COMMAND}
        "-DPROGRAM=$<TARGET_FILE:test_simulator_regression>"
        "-DVERIFIER=$<TARGET_FILE:test_simulator_models>"
        "-DSOURCE_DIR=${CMAKE_SOURCE_DIR}" "-DCASE=${case}"
        -P "${CMAKE_SOURCE_DIR}/tests/simulator/run_scenario.cmake")
    set_tests_properties("simulator_${case}" PROPERTIES
        ENVIRONMENT "SDL_VIDEODRIVER=dummy" TIMEOUT 45)
endforeach()

add_test(NAME simulator_flash_example_round_trips COMMAND ${CMAKE_COMMAND}
    "-DPROGRAM=$<TARGET_FILE:test_flash_archive_controller>"
    "-DSOURCE_DIR=${CMAKE_SOURCE_DIR}"
    -P "${CMAKE_SOURCE_DIR}/tests/simulator/run_flash_example.cmake")
set_tests_properties(simulator_flash_example_round_trips PROPERTIES
    ENVIRONMENT "SDL_VIDEODRIVER=dummy" TIMEOUT 90)

# Install under an isolated prefix, then configure a separate legacy consumer
# with no source-tree headers or Pogosim target interface. Pass the existing
# toolchain and package paths, including pinned Box2D on native MSVC.
add_test(NAME simulator_installed_legacy_consumer COMMAND ${CMAKE_COMMAND}
    "-DBUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}" "-DSOURCE_DIR=${CMAKE_SOURCE_DIR}"
    "-DCONFIG=$<CONFIG>" "-DGENERATOR=${CMAKE_GENERATOR}"
    "-DC_COMPILER=${CMAKE_C_COMPILER}" "-DCXX_COMPILER=${CMAKE_CXX_COMPILER}"
    "-DTOOLCHAIN=${CMAKE_TOOLCHAIN_FILE}" "-DTRIPLET=${VCPKG_TARGET_TRIPLET}"
    "-DINSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}" "-DLIBRARY_NAME=$<TARGET_FILE_NAME:pogosim>"
    "-DYAML_DIR=${yaml-cpp_DIR}" "-DBOX2D_DIR=${box2d_DIR}" "-DARROW_DIR=${Arrow_DIR}"
    "-DC_FLAGS=${CMAKE_C_FLAGS}" "-DCXX_FLAGS=${CMAKE_CXX_FLAGS}"
    "-DLINK_FLAGS=${CMAKE_EXE_LINKER_FLAGS}"
    -P "${CMAKE_SOURCE_DIR}/tests/simulator/run_installed_consumer.cmake")
set_tests_properties(simulator_installed_legacy_consumer PROPERTIES
    ENVIRONMENT "SDL_VIDEODRIVER=dummy" TIMEOUT 180)
