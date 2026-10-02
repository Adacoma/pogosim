cmake_minimum_required(VERSION 3.16)
project(pogosim VERSION 1.0.0 LANGUAGES CXX C)
include(GNUInstallDirs)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/modules" ${CMAKE_MODULE_PATH})

# Set C++ and C standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Vcpkg provides a current Boost package config on MSVC. Keep legacy
# FindBoost on established Unix builds to preserve their existing discovery.
if(MSVC)
    find_package(Boost 1.70 CONFIG REQUIRED COMPONENTS context)
else()
    if(POLICY CMP0167)
        cmake_policy(SET CMP0167 OLD)
    endif()
    set(Boost_NO_BOOST_CMAKE ON)
    find_package(Boost 1.70 REQUIRED COMPONENTS context)
endif()

# Find required packages
find_package(SDL2 REQUIRED)
find_package(SDL2_gfx REQUIRED)
find_package(SDL2_ttf REQUIRED)
# The Unix toolchains install Box2D headers in their default compiler search
# paths. MSVC needs the package target to discover the separately pinned build.
if(MSVC)
    find_package(box2d CONFIG REQUIRED)
endif()
find_package(yaml-cpp REQUIRED)
# yaml-cpp 0.7 (Ubuntu 22.04) exports only the unnamespaced installed
# target; newer packages export yaml-cpp::yaml-cpp. Link a real target so
# includes, static/shared definitions and library locations are inherited.
if(TARGET yaml-cpp::yaml-cpp)
    set(POGOSIM_YAML_CPP_TARGET yaml-cpp::yaml-cpp)
elseif(TARGET yaml-cpp)
    set(POGOSIM_YAML_CPP_TARGET yaml-cpp)
else()
    message(FATAL_ERROR "yaml-cpp package did not provide a supported CMake target")
endif()
find_package(spdlog REQUIRED)
find_package(Arrow REQUIRED)
find_package(fmt REQUIRED)


# Collect source files
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/*.c"
)
list(FILTER SOURCES EXCLUDE REGEX "/(libs|tests|pogobot-sdk|examples)/.*$")

# Create library target
add_library(${PROJECT_NAME} ${SOURCES})

if(APPLE)
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
        # Apple Silicon (M1/M2/M3) - Homebrew uses /opt/homebrew
        set(HOMEBREW_PREFIX "/opt/homebrew")
    else()
        # Intel Mac - Homebrew uses /usr/local
        set(HOMEBREW_PREFIX "/usr/local")
    endif()

    set(SDL2_INCLUDE_DIRS "${HOMEBREW_PREFIX}/include/SDL2")
    set(SDL2_LIB_DIR "${HOMEBREW_PREFIX}/lib")

    include_directories(${SDL2_INCLUDE_DIRS})
    link_directories(${SDL2_LIB_DIR})
endif()

link_directories("/usr/local/lib")

# Include directories
target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        /usr/local/include
        ${Boost_INCLUDE_DIRS}
        ${SDL2_INCLUDE_DIRS}
        ${SDL2_gfx_INCLUDE_DIRS}
        ${SDL2_ttf_INCLUDE_DIRS}
        #        ${BOX2D_INCLUDE_DIRS}
        ${YAML_CPP_INCLUDE_DIR}
        ${spdlog_INCLUDE_DIRS}
        ${ARROW_INCLUDE_DIR}

)

target_compile_definitions(${PROJECT_NAME}
    PUBLIC
        DATA_DIR="${CMAKE_INSTALL_FULL_DATADIR}/${PROJECT_NAME}"
        # MSVC exposes the standard math constants only when this is defined
        # before <cmath>; command-line target definitions satisfy that rule.
        $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:_USE_MATH_DEFINES>
)

# Link libraries
target_link_libraries(${PROJECT_NAME}
    PUBLIC
        Boost::context
        SDL2::SDL2
        ${POGOSIM_YAML_CPP_TARGET}
        spdlog::spdlog
        fmt::fmt
)

if(MSVC)
    target_link_libraries(${PROJECT_NAME} PUBLIC box2d::box2d)
endif()

# Set compile options
#target_compile_options(${PROJECT_NAME}
#    PRIVATE
#        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wunused-parameter -O3 -msse4.1 -mpclmul -march=native>
#        $<$<CXX_COMPILER_ID:MSVC>:/W4 /O2 /arch:AVX2>
#)

#target_compile_options(${PROJECT_NAME}
#    PRIVATE
#        $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -Wunused-parameter -O3 -msse4.1 -mpclmul -march=native>
#        $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -Wunused-parameter -O3 -msse4.1 -mpclmul -march=native>
#        $<$<CXX_COMPILER_ID:MSVC>:/W4 /O2 /arch:AVX2>
#)

if(APPLE)
    target_compile_options(${PROJECT_NAME}
        PRIVATE
        $<$<CXX_COMPILER_ID:GNU,AppleClang,Clang>:
        $<$<CONFIG:Release>:-Wall -Wextra -Wunused-parameter -Og -march=native -DNDEBUG -DFPNG_NO_SSE -L/usr/local/lib>
        $<$<CONFIG:Debug>:-Wall -Wextra -Wunused-parameter -g -Og -DDEBUG -DFPNG_NO_SSE -L/usr/local/lib>
        >
    )
else()
    target_compile_options(${PROJECT_NAME}
        PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:
            $<$<CONFIG:Release>:-Wall -Wextra -Wunused-parameter -O3 -msse4.1 -mpclmul -march=native -DNDEBUG>
            $<$<CONFIG:Debug>:-Wall -Wextra -Wunused-parameter -g -Og -DDEBUG>
        >
        $<$<CXX_COMPILER_ID:MSVC>:
            $<$<CONFIG:Release>:/W4 /O2 /arch:AVX2 /DNDEBUG>
            $<$<CONFIG:Debug>:/W4 /Od /Zi /DDEBUG>
        >
    )
endif()

# Boost.Context may unwind through the C API when a suspended controller is
# stopped. MSVC must not assume extern "C" calls are non-throwing, or use /GL.
target_compile_options(${PROJECT_NAME} PUBLIC
    $<$<COMPILE_LANG_AND_ID:C,GNU,Clang,AppleClang>:-fexceptions>)
if(MSVC)
    target_compile_options(${PROJECT_NAME} PUBLIC
        $<$<COMPILE_LANGUAGE:CXX>:/EHs>
        $<$<COMPILE_LANGUAGE:CXX>:/EHc->
        /GL-)
endif()

# A standalone runtime test exercises mixed C/C++ stacks without SDL/physics.
include(CTest)
if(BUILD_TESTING)
    add_executable(test_robot_coroutine
        tests/robot_coroutine/runtime_test.cpp
        tests/robot_coroutine/nested_calls.c
        src/pogosim/robot_coroutine.cpp)
    target_include_directories(test_robot_coroutine PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(test_robot_coroutine PRIVATE Boost::context)
    target_compile_options(test_robot_coroutine PRIVATE
        $<$<COMPILE_LANG_AND_ID:C,GNU,Clang,AppleClang>:-fexceptions>)
    if(MSVC)
        target_compile_options(test_robot_coroutine PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/EHs>
            $<$<COMPILE_LANGUAGE:CXX>:/EHc-> /GL-)
    endif()
    add_test(NAME robot_coroutine_runtime COMMAND test_robot_coroutine)
    set_tests_properties(robot_coroutine_runtime PROPERTIES TIMEOUT 15)

    # Link a real simulator/controller fixture on every toolchain, including
    # MSVC. Example Makefiles remain unchanged as the user-facing build path.
    add_executable(test_controller_scheduling
        tests/robot_coroutine/controller.c
        tests/robot_coroutine/controller_bridge.cpp)
    target_include_directories(test_controller_scheduling PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(test_controller_scheduling PRIVATE pogosim
        ${SDL2_GFX_LIBRARIES} ${SDL2TTF_LIBRARY} Arrow::arrow_shared)
    if(NOT MSVC)
        target_link_libraries(test_controller_scheduling PRIVATE box2d)
    endif()
    # Generate isolated configs/results in the build tree, not the worktree.
    set(SCHEDULING_CASE_NAMES normal end_sleep setup_sleep step_exception
        negative_sleep init_exception export_sleep invalid_stack early_stop
        init_cancellation global_exception)
    foreach(TEST_MODE RANGE 0 10)
        list(GET SCHEDULING_CASE_NAMES ${TEST_MODE} TEST_CASE_NAME)
        set(STACK_SIZE 131072)
        set(SIMULATION_TIME 0.09)
        set(LOG_DATA false)
        set(DATA_FILENAME "${CMAKE_CURRENT_BINARY_DIR}/coroutine-${TEST_MODE}.feather")
        set(EXPECTED_STATUS 2)
        set(EXPECTED_MESSAGE "Sleep is only supported inside")
        if(TEST_MODE EQUAL 0 OR TEST_MODE EQUAL 8 OR TEST_MODE EQUAL 9)
            set(EXPECTED_STATUS 0)
            set(EXPECTED_MESSAGE "")
            if(TEST_MODE EQUAL 9)
                set(SIMULATION_TIME 0)
            endif()
        elseif(TEST_MODE EQUAL 3 OR TEST_MODE EQUAL 5 OR TEST_MODE EQUAL 10)
            set(EXPECTED_MESSAGE "fixture controller exception")
        elseif(TEST_MODE EQUAL 4)
            set(EXPECTED_MESSAGE "msleep duration must be nonnegative")
        elseif(TEST_MODE EQUAL 6)
            set(LOG_DATA true)
        elseif(TEST_MODE EQUAL 7)
            set(STACK_SIZE 0)
            set(EXPECTED_MESSAGE "Invalid coroutine_stack_size")
        endif()
        set(TEST_CONFIG "${CMAKE_CURRENT_BINARY_DIR}/coroutine-${TEST_MODE}.yaml")
        configure_file(tests/robot_coroutine/simulation.yaml.in "${TEST_CONFIG}" @ONLY)
        add_test(NAME controller_scheduling_${TEST_CASE_NAME} COMMAND ${CMAKE_COMMAND}
            "-DPROGRAM=$<TARGET_FILE:test_controller_scheduling>"
            "-DCONFIG=${TEST_CONFIG}" "-DEXPECTED_STATUS=${EXPECTED_STATUS}"
            "-DEXPECTED_MESSAGE=${EXPECTED_MESSAGE}"
            -P "${CMAKE_SOURCE_DIR}/tests/robot_coroutine/run_simulation.cmake")
        set_tests_properties(controller_scheduling_${TEST_CASE_NAME} PROPERTIES
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            ENVIRONMENT "SDL_VIDEODRIVER=dummy" TIMEOUT 15)
    endforeach()
endif()

# Install rules
install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}  # Shared libraries (.so, .dylib)
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}  # Static libraries (.a)
    #RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}  # Executables (if any)
)

# Install header files
install(DIRECTORY ${CMAKE_SOURCE_DIR}/src/  # Path to the header files
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}  # Install location
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"  # Only install .h and .hpp files
)

# Install fonts directory
install(DIRECTORY ${CMAKE_SOURCE_DIR}/fonts/  # Source directory
    DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/fonts  # Install location
    FILES_MATCHING PATTERN "*"  # Install all files
)

# Install arenas directory
install(DIRECTORY ${CMAKE_SOURCE_DIR}/arenas/  # Source directory
    DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/arenas  # Install location
    FILES_MATCHING PATTERN "*.csv"  # Install all files
)

# Generate and install CMake config files
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${PROJECT_NAME}ConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)


###### BUILD EXAMPLES ######
# Find all subdirectories in examples
file(GLOB EXAMPLE_DIRS RELATIVE ${CMAKE_SOURCE_DIR}/examples ${CMAKE_SOURCE_DIR}/examples/*)

# Create a custom target that will build all examples
add_custom_target(examples #ALL
#    DEPENDS ${PROJECT_NAME}
    COMMENT "Building all examples"
)

# Example directories use GNU-style Makefiles. Resolve make through CMake so
# generator-specific syntax such as $(MAKE) is not written into Ninja files.
find_program(POGOSIM_MAKE_EXECUTABLE NAMES gmake make)
if(POGOSIM_MAKE_EXECUTABLE)
    # For each example directory
    foreach(EXAMPLE ${EXAMPLE_DIRS})
        if(IS_DIRECTORY ${CMAKE_SOURCE_DIR}/examples/${EXAMPLE})
            # Create a custom target for this specific example
            add_custom_target(example_${EXAMPLE}
                COMMAND ${CMAKE_COMMAND} -E echo "Building example: ${EXAMPLE}"
                COMMAND ${POGOSIM_MAKE_EXECUTABLE} -C ${CMAKE_SOURCE_DIR}/examples/${EXAMPLE} clean
                COMMAND ${POGOSIM_MAKE_EXECUTABLE} -C ${CMAKE_SOURCE_DIR}/examples/${EXAMPLE} sim
                DEPENDS ${PROJECT_NAME}
                USES_TERMINAL
                COMMENT "Building example ${EXAMPLE}"
            )

            # Add this example target as a dependency of the main examples target
            add_dependencies(examples example_${EXAMPLE})
        endif()
    endforeach()
else()
    message(STATUS "GNU make not found; per-example build targets are unavailable")
endif()
