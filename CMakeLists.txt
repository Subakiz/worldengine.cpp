cmake_minimum_required(VERSION 3.25)
project(WorldEngine VERSION 0.1.0 LANGUAGES CXX C)

# Enforce C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Standard installation directory definitions and package helpers
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Options
option(PLAYWORLD_BUILD_TESTS "Build verification tests" ON)
option(PLAYWORLD_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

# Compiler optimization and warnings configuration
if(MSVC)
    add_compile_options(/W4 /O2 /permissive- /utf-8)
    if(PLAYWORLD_ENABLE_SANITIZERS)
        add_compile_options(/fsanitize=address)
    endif()
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
    if(PLAYWORLD_ENABLE_SANITIZERS)
        add_compile_options(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
        add_link_options(-fsanitize=address,undefined)
    else()
        add_compile_options(-O3)
    endif()
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64|arm64")
        # Ensure ARM NEON and standard extensions are active
        add_compile_options(-march=armv8-a+crc)
    endif()
endif()

if(WIN32)
    add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
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
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/worldengine>
)

# Ensure C++20 feature requirement propagates transitively to consumers
target_compile_features(playworld_core PUBLIC cxx_std_20)

if(WIN32)
    target_link_libraries(playworld_core PUBLIC psapi)
endif()

# Configure output naming and export identity for static library
set_target_properties(playworld_core PROPERTIES
    OUTPUT_NAME "worldengine"
    EXPORT_NAME "worldengine"
    POSITION_INDEPENDENT_CODE ON
)

# Provide alias targets for internal and build-tree usage
add_library(worldengine ALIAS playworld_core)
add_library(WorldEngine::worldengine ALIAS playworld_core)
add_library(WorldEngine::playworld_core ALIAS playworld_core)

# Backward-compatibility copy in build tree so scripts/run_all_tests.sh finds libplayworld_core.a
add_custom_command(TARGET playworld_core POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:playworld_core>
        ${CMAKE_CURRENT_BINARY_DIR}/libplayworld_core.a
    BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/libplayworld_core.a
    COMMENT "Ensuring build-tree backward compatibility: libplayworld_core.a"
    VERBATIM
)

if(PLAYWORLD_ENABLE_SANITIZERS AND NOT MSVC)
    target_compile_options(playworld_core PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer -g)
    target_link_options(playworld_core INTERFACE -fsanitize=address,undefined)
endif()

# Test model generator and verification CLI
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_test_model.cpp")
    add_executable(generate_test_model tools/generate_test_model.cpp)
    target_link_libraries(generate_test_model PRIVATE playworld_core)
    set_target_properties(generate_test_model PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# Standalone Native Benchmark CLI
add_executable(worldengine-bench src/main_bench.cpp)
target_link_libraries(worldengine-bench PRIVATE playworld_core)
if(WIN32)
    target_link_libraries(worldengine-bench PRIVATE psapi)
endif()
set_target_properties(worldengine-bench PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Standalone CLI alias executable (worldengine_cli)
add_executable(worldengine_cli src/main_bench.cpp)
target_link_libraries(worldengine_cli PRIVATE playworld_core)
if(WIN32)
    target_link_libraries(worldengine_cli PRIVATE psapi)
endif()
set_target_properties(worldengine_cli PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    EXPORT_NAME "worldengine_cli"
)

# Automated testing harness integration
if(PLAYWORLD_BUILD_TESTS)
    enable_testing()

    if(TARGET generate_test_model)
        add_test(NAME M1_Verification_Test COMMAND generate_test_model)
    endif()

    # Unit Tests
    add_executable(test_pwmf tests/unit/test_pwmf.cpp)
    target_link_libraries(test_pwmf PRIVATE playworld_core)
    set_target_properties(test_pwmf PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    add_test(NAME test_pwmf COMMAND test_pwmf)

    add_executable(test_quant tests/unit/test_quant.cpp)
    target_link_libraries(test_quant PRIVATE playworld_core)
    set_target_properties(test_quant PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    add_test(NAME test_quant COMMAND test_quant)

    add_executable(test_ring_buffer tests/unit/test_ring_buffer.cpp)
    target_link_libraries(test_ring_buffer PRIVATE playworld_core)
    set_target_properties(test_ring_buffer PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    add_test(NAME test_ring_buffer COMMAND test_ring_buffer)

    add_executable(test_voxel_grid tests/unit/test_voxel_grid.cpp)
    target_link_libraries(test_voxel_grid PRIVATE playworld_core)
    set_target_properties(test_voxel_grid PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    add_test(NAME test_voxel_grid COMMAND test_voxel_grid)

    add_executable(test_scheduler tests/unit/test_scheduler.cpp)
    target_link_libraries(test_scheduler PRIVATE playworld_core)
    set_target_properties(test_scheduler PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    add_test(NAME test_scheduler COMMAND test_scheduler)

    # Integration Tests
    add_executable(test_benchmark_cli tests/integration/test_benchmark_cli.cpp)
    target_link_libraries(test_benchmark_cli PRIVATE playworld_core)
    set_target_properties(test_benchmark_cli PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    if(NOT MSVC)
        target_compile_options(test_benchmark_cli PRIVATE -Wno-unused-variable)
    endif()
    add_test(NAME test_benchmark_cli COMMAND test_benchmark_cli)

    add_executable(test_spatial_ssim tests/integration/test_spatial_ssim.cpp)
    target_link_libraries(test_spatial_ssim PRIVATE playworld_core)
    set_target_properties(test_spatial_ssim PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    add_test(NAME test_spatial_ssim COMMAND test_spatial_ssim)

    add_executable(test_memory_leak tests/integration/test_memory_leak.cpp)
    target_link_libraries(test_memory_leak PRIVATE playworld_core)
    set_target_properties(test_memory_leak PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    add_test(NAME test_memory_leak COMMAND test_memory_leak)

    # M2 Adversarial Model Fuzzing & Malformed Binary Suite
    add_executable(test_pwmf_fuzz tests/fuzz/test_pwmf_fuzz.cpp)
    target_link_libraries(test_pwmf_fuzz PRIVATE playworld_core)
    set_target_properties(test_pwmf_fuzz PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    add_test(NAME test_pwmf_fuzz COMMAND test_pwmf_fuzz)

    # M3 High-Contention Concurrency & Threading Stress Suite
    add_executable(test_concurrency_stress tests/stress/test_concurrency_stress.cpp)
    target_link_libraries(test_concurrency_stress PRIVATE playworld_core)
    set_target_properties(test_concurrency_stress PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    add_test(NAME test_concurrency_stress COMMAND test_concurrency_stress)

    # M4 Long-Horizon 5,000-Step Soak & Numerical Drift Simulation
    add_executable(test_soak_simulation tests/stress/test_soak_simulation.cpp)
    target_link_libraries(test_soak_simulation PRIVATE playworld_core)
    set_target_properties(test_soak_simulation PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    add_test(NAME test_soak_simulation COMMAND test_soak_simulation)

    # Convenient unified test target for CTest verification
    add_custom_target(run_all_tests_native
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        DEPENDS test_pwmf test_quant test_ring_buffer test_voxel_grid test_scheduler
                test_benchmark_cli test_spatial_ssim test_memory_leak
                test_pwmf_fuzz test_concurrency_stress test_soak_simulation
        COMMENT "Executing all native test suites via CTest"
    )
endif()

# ==============================================================================
# Standard Installation and CPack Packaging Rules (Requirement R1)
# ==============================================================================

# 1. Install Targets (Runtime binaries to bin/, static library to lib/)
install(TARGETS playworld_core worldengine-bench worldengine_cli
    EXPORT WorldEngineTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Provide backward-compatible libplayworld_core.a copy in lib/
install(FILES "$<TARGET_FILE:playworld_core>"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RENAME "libplayworld_core.a"
)

# 2. Install Public Headers to include/worldengine/ and include/playworld/
install(DIRECTORY include/playworld/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/worldengine
    FILES_MATCHING PATTERN "*.h"
)

install(DIRECTORY include/playworld/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/playworld
    FILES_MATCHING PATTERN "*.h"
)

# 3. Export Targets and CMake Package Configuration
set(WORLDENGINE_CMAKE_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/WorldEngine")

install(EXPORT WorldEngineTargets
    FILE WorldEngineTargets.cmake
    NAMESPACE WorldEngine::
    DESTINATION "${WORLDENGINE_CMAKE_INSTALL_DIR}"
)

# Generate WorldEngineConfigVersion.cmake
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/WorldEngineConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

# Configure WorldEngineConfig.cmake from template
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/WorldEngineConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/WorldEngineConfig.cmake"
    INSTALL_DESTINATION "${WORLDENGINE_CMAKE_INSTALL_DIR}"
    PATH_VARS CMAKE_INSTALL_INCLUDEDIR CMAKE_INSTALL_LIBDIR CMAKE_INSTALL_BINDIR
)

# Install Package Configuration files
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/WorldEngineConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/WorldEngineConfigVersion.cmake"
    DESTINATION "${WORLDENGINE_CMAKE_INSTALL_DIR}"
)

# 4. CPack Configuration for Standalone Compressed Archives (.tar.gz and .zip)
set(CPACK_PACKAGE_NAME "WorldEngine")
set(CPACK_PACKAGE_VENDOR "PlayWorld Team")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Zero-dependency C++20 and WebGPU runtime for interactive neural world models")
set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
    set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
    set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
endif()
set(CPACK_PACKAGE_CONTACT "https://github.com/playworld/worldengine.cpp")
set(CPACK_PACKAGE_HOMEPAGE_URL "https://github.com/playworld/worldengine.cpp")

string(TOLOWER "${CMAKE_SYSTEM_NAME}" PLAYWORLD_SYS_NAME)
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" PLAYWORLD_SYS_ARCH)
if(PLAYWORLD_SYS_ARCH MATCHES "x86_64|amd64")
    set(PLAYWORLD_SYS_ARCH "x86_64")
elseif(PLAYWORLD_SYS_ARCH MATCHES "arm64|aarch64")
    set(PLAYWORLD_SYS_ARCH "arm64")
endif()

set(CPACK_SYSTEM_NAME "${PLAYWORLD_SYS_NAME}-${PLAYWORLD_SYS_ARCH}")
set(CPACK_PACKAGE_FILE_NAME "worldengine-v${PROJECT_VERSION}-${PLAYWORLD_SYS_NAME}-${PLAYWORLD_SYS_ARCH}")
set(CPACK_GENERATOR "TGZ;ZIP")
set(CPACK_SOURCE_GENERATOR "TGZ;ZIP")
set(CPACK_SOURCE_IGNORE_FILES
    "/.git/"
    "/build/"
    "/build-.*/"
    "/dist/"
    "/\\\\.agents/"
    "\\\\.DS_Store"
)

include(CPack)

