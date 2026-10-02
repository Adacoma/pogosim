cmake_minimum_required(VERSION 3.16)
project(pogosim_installed_consumer LANGUAGES C CXX)
set(CMAKE_C_STANDARD 11)
# Reuse dependency discovery modules, but never source-tree simulator headers.
list(APPEND CMAKE_MODULE_PATH "${SOURCE_DIR}/cmake/modules")
find_package(SDL2 REQUIRED)
find_package(SDL2_gfx REQUIRED)
find_package(SDL2_ttf REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(spdlog REQUIRED)
find_package(fmt REQUIRED)
find_package(Arrow REQUIRED)
if(TARGET yaml-cpp::yaml-cpp)
    set(YAML_TARGET yaml-cpp::yaml-cpp)
else()
    set(YAML_TARGET yaml-cpp)
endif()
configure_file("${CONTROLLER_SOURCE}" "${CMAKE_CURRENT_BINARY_DIR}/controller.c" COPYONLY)
add_executable(installed_consumer "${CMAKE_CURRENT_BINARY_DIR}/controller.c")
set_target_properties(installed_consumer PROPERTIES LINKER_LANGUAGE CXX)
# Deliberately use only installed headers and the raw installed archive, with
# the pre-coroutine dependency list. Do not add Boost.Context here.
target_include_directories(installed_consumer PRIVATE "${INSTALL_PREFIX}/include")
target_link_libraries(installed_consumer PRIVATE "${INSTALLED_LIBRARY}"
    SDL2::SDL2 ${YAML_TARGET} spdlog::spdlog fmt::fmt
    ${SDL2_GFX_LIBRARIES} ${SDL2TTF_LIBRARY} Arrow::arrow_shared)
if(MSVC)
    find_package(box2d CONFIG REQUIRED)
    target_link_libraries(installed_consumer PRIVATE box2d::box2d)
else()
    if(APPLE)
        # Match legacy Makefiles: pinned Box2D lives in /usr/local, while
        # Homebrew libraries may live under /opt/homebrew on Apple Silicon.
        target_link_directories(installed_consumer PRIVATE /usr/local/lib /opt/homebrew/lib)
    endif()
    target_link_libraries(installed_consumer PRIVATE box2d)
endif()
include(CTest)
add_test(NAME installed_controller COMMAND installed_consumer -c "${SIMULATION_CONFIG}" -g -nr)
set_tests_properties(installed_controller PROPERTIES WORKING_DIRECTORY "${SOURCE_DIR}"
    ENVIRONMENT "SDL_VIDEODRIVER=dummy" TIMEOUT 20)
