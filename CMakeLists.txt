# C++ build entry point for the native ISTHMUS implementation.
cmake_minimum_required(VERSION 3.22)

project(isthmus_cpp VERSION 0.1.0 LANGUAGES CXX)

# Load standard install-directory variables and helper utilities for generating
# a consumable CMake package configuration for downstream users.
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Request C++20 for all targets in this project.
# - REQUIRED: fail configuration if the compiler can't do C++20
# - EXTENSIONS OFF: prefer -std=c++20 over -std=gnu++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# User-configurable toggles (similar to Makefile variables).
# Example: cmake -S . -B build -DISTHMUS_BUILD_TESTS=OFF
option(ISTHMUS_BUILD_TESTS "Build C++ tests" ON)

# Define the core library target.
# CMake will compile these sources and produce a library (static or shared,
# depending on the platform/defaults or any future configuration).
add_library(isthmus_cpp
    cpp/src/geometry.cpp
    cpp/src/motion_mapping.cpp
    cpp/src/io.cpp
    cpp/src/marching_windows.cpp
)

# Export the public include path for the library.
# PUBLIC means:
# - used when compiling the library itself
# - also visible to targets that link against this library
target_include_directories(isthmus_cpp
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
)

# Enable strict warnings on the library target.
# PRIVATE means these flags apply only to this target (not automatically to
# tests/examples that link it).
if (MSVC)
    target_compile_options(isthmus_cpp PRIVATE /W4 /permissive-)
else()
    target_compile_options(isthmus_cpp PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Require C++20 as a *usage requirement*.
# PUBLIC propagates the requirement to anything that links isthmus_cpp.
target_compile_features(isthmus_cpp PUBLIC cxx_std_20)

if(ISTHMUS_BUILD_EXAMPLES)
    # Example executable that links against the library.
    add_executable(isthmus_corner_demo cpp/examples/corner_demo.cpp)
    target_link_libraries(isthmus_corner_demo PRIVATE isthmus_cpp)
endif()

if(ISTHMUS_BUILD_TESTS)
    # Enable CTest integration in this build tree.
    enable_testing()

    # Test executable built with a tiny custom harness.
    # (Keeps the native build dependency-light: no third-party test framework.)
    add_executable(isthmus_cpp_tests
        cpp/tests/test_main.cpp
        cpp/tests/test_geometry.cpp
        cpp/tests/test_io.cpp
        cpp/tests/test_motion_mapping.cpp
    )
    target_link_libraries(isthmus_cpp_tests PRIVATE isthmus_cpp)

    # Register the test with CTest so `ctest` will run it.
    add_test(NAME isthmus_cpp_tests COMMAND isthmus_cpp_tests)
endif()
