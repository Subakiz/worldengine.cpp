cmake_minimum_required(VERSION 3.25)
project(WorldEngine VERSION 0.1.0 LANGUAGES CXX C)

# Enforce C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(PLAYWORLD_BUILD_TESTS "Build verification tests" ON)

# Compiler optimization and warnings configuration
if(MSVC)
    add_compile_options(/W4 /O2 /permissive- /utf-8)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -O3)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64|arm64")
        # Ensure ARM NEON and standard extensions are active
        add_compile_options(-march=armv8-a+crc)
    endif()
endif()

# Core static library
set(PLAYWORLD_CORE_SOURCES
    src/core/pwmf_parser.cpp
    src/core/tensor.cpp
    src/core/action_subsystem.cpp
    src/core/voxel_grid.cpp
    src/core/scheduler.cpp
    src/core/world_engine.cpp
)

add_library(playworld_core STATIC ${PLAYWORLD_CORE_SOURCES})

target_include_directories(playworld_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Test model generator and verification CLI
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_test_model.cpp")
    add_executable(generate_test_model tools/generate_test_model.cpp)
    target_link_libraries(generate_test_model PRIVATE playworld_core)
endif()

# Standalone Native Benchmark CLI
add_executable(worldengine-bench src/main_bench.cpp)
target_link_libraries(worldengine-bench PRIVATE playworld_core)
set_target_properties(worldengine-bench PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Automated testing harness integration
if(PLAYWORLD_BUILD_TESTS)
    enable_testing()

    if(TARGET generate_test_model)
        add_test(NAME M1_Verification_Test COMMAND generate_test_model)
    endif()
endif()

