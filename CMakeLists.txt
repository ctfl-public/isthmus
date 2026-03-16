# C++ build entry point for the native ISTHMUS implementation.
cmake_minimum_required(VERSION 3.22)

project(isthmus_cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ISTHMUS_BUILD_TESTS "Build C++ tests" ON)
option(ISTHMUS_BUILD_EXAMPLES "Build C++ examples" ON)

# The native library is kept small and dependency-light so downstream solvers
# can integrate it easily while the implementation surface is still growing.
add_library(isthmus_cpp
    cpp/src/geometry.cpp
    cpp/src/motion_mapping.cpp
    cpp/src/io.cpp
    cpp/src/marching_windows.cpp
)

target_include_directories(isthmus_cpp
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
)

# Use strict warnings early because most of the code implements numerical logic
# where silent mistakes are harder to spot than API-level failures.
if (MSVC)
    target_compile_options(isthmus_cpp PRIVATE /W4 /permissive-)
else()
    target_compile_options(isthmus_cpp PRIVATE -Wall -Wextra -Wpedantic)
endif()

target_compile_features(isthmus_cpp PUBLIC cxx_std_20)

if(ISTHMUS_BUILD_EXAMPLES)
    add_executable(isthmus_corner_demo cpp/examples/corner_demo.cpp)
    target_link_libraries(isthmus_corner_demo PRIVATE isthmus_cpp)
endif()

if(ISTHMUS_BUILD_TESTS)
    enable_testing()

    # The project currently uses a tiny custom harness to keep the native build
    # simple and avoid forcing a third-party test framework on every consumer.
    add_executable(isthmus_cpp_tests
        cpp/tests/test_main.cpp
        cpp/tests/test_geometry.cpp
        cpp/tests/test_motion_mapping.cpp
    )
    target_link_libraries(isthmus_cpp_tests PRIVATE isthmus_cpp)

    add_test(NAME isthmus_cpp_tests COMMAND isthmus_cpp_tests)
endif()
