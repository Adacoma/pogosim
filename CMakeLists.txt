cmake_minimum_required(VERSION 3.16)
project(pogosim VERSION 1.0.0 LANGUAGES CXX C)
include(GNUInstallDirs)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/modules" ${CMAKE_MODULE_PATH})

# Set C++ and C standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Use legacy FindBoost instead of BoostConfig.cmake (avoids per‑component configs like boost_systemConfig.cmake)
if(POLICY CMP0167)
  cmake_policy(SET CMP0167 OLD)
endif()
set(Boost_NO_BOOST_CMAKE ON)

# Find required packages
find_package(Boost 1.70 REQUIRED)
find_package(SDL2 REQUIRED)
find_package(SDL2_gfx REQUIRED)
find_package(SDL2_ttf REQUIRED)
#find_package(Box2D REQUIRED)
find_package(yaml-cpp REQUIRED)
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
)

# Link libraries
target_link_libraries(${PROJECT_NAME}
    PUBLIC
        #Boost::system
        SDL2::SDL2
        #        Box2D::Box2D
        yaml-cpp
        spdlog::spdlog
        fmt::fmt
)

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

# For each example directory
foreach(EXAMPLE ${EXAMPLE_DIRS})
    if(IS_DIRECTORY ${CMAKE_SOURCE_DIR}/examples/${EXAMPLE})
        # Create a custom target for this specific example
        add_custom_target(example_${EXAMPLE}
            COMMAND ${CMAKE_COMMAND} -E echo "Building example: ${EXAMPLE}"
            #COMMAND make clean
            #COMMAND make sim
            #WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/examples/${EXAMPLE}
            COMMAND $(MAKE) -C ${CMAKE_SOURCE_DIR}/examples/${EXAMPLE} clean
            COMMAND $(MAKE) -C ${CMAKE_SOURCE_DIR}/examples/${EXAMPLE} sim
            DEPENDS ${PROJECT_NAME}
            USES_TERMINAL
            COMMENT "Building example ${EXAMPLE}"
        )
        
        # Add this example target as a dependency of the main examples target
        add_dependencies(examples example_${EXAMPLE})
    endif()
endforeach()

