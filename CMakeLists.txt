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
    cpp/src/marching_cubes.cpp
    cpp/src/io.cpp
    cpp/src/utilities.cpp
    cpp/src/marching_windows.cpp
)

# Provide a namespaced alias so downstream CMake consumers can link against a
# stable imported-style target name.
# It gives the existing target isthmus_cpp a second, namespaced name: isthmus_cpp::isthmus_cpp
add_library(isthmus_cpp::isthmus_cpp ALIAS isthmus_cpp)

# Export the public include path for the library.
# PUBLIC means:
# - used when compiling the library itself
# - also visible to targets that link against this library
# - In other words, these headers are part of the library’s public interface.
# The two $<...> generator expressions ensure the correct include path is used for both build-tree and install-tree consumers.
# - Build-tree consumers will see the path to the source directory (where the headers are located during development).
# - Install-tree consumers will see the path to the installation include directory (where the headers will be installed).
target_include_directories(isthmus_cpp
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpp/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
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
        cpp/tests/test_utilities.cpp
    )
    target_link_libraries(isthmus_cpp_tests PRIVATE isthmus_cpp)

    # Register the test with CTest so `ctest` will run it.
    add_test(NAME isthmus_cpp_tests COMMAND isthmus_cpp_tests)
endif()

# Generate versioned package metadata in the build tree so standalone example
# projects can consume a locally built ISTHMUS++ tree without installation.
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/isthmus_cppConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

# Configure the package entry point that will load the exported target set for
# both build-tree and install-tree consumers.
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cpp/cmake/isthmus_cppConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/isthmus_cppConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/isthmus_cpp"
)

# Export the build-tree targets directly so external standalone examples can
# point `isthmus_cpp_DIR` at the main build directory and immediately link the
# already-built library target.
export(
    TARGETS isthmus_cpp
    FILE "${CMAKE_CURRENT_BINARY_DIR}/isthmus_cppTargets.cmake"
    NAMESPACE isthmus_cpp::
)

# Install the library and publish the exported target set so external projects
# can locate ISTHMUS++ through a normal `find_package()` call after install.
install(
    TARGETS isthmus_cpp
    EXPORT isthmus_cppTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)

# Install the public headers exactly as they are laid out under `cpp/include`
# so downstream includes stay unchanged.
install(
    DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cpp/include/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

# Install the generated package entry points and exported targets into the
# conventional CMake package directory.
install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/isthmus_cppConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/isthmus_cppConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/isthmus_cpp"
)

# Install the exported target definitions with a namespace so consumers link
# via `isthmus_cpp::isthmus_cpp`.
install(
    EXPORT isthmus_cppTargets
    FILE isthmus_cppTargets.cmake
    NAMESPACE isthmus_cpp::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/isthmus_cpp"
)

