cmake_minimum_required(VERSION 3.22)

# Declare this example as its own standalone CMake project so it can consume a
# previously built or installed ISTHMUS++ package.
project(isthmus_surface_export_demo LANGUAGES CXX)

# Locate the reusable ISTHMUS++ package before creating the demo target that
# links against it.
find_package(isthmus_cpp CONFIG REQUIRED)

# Build the standalone surface-export demo from the sources stored in this
# directory so the example remains self-contained.
add_executable(isthmus_surface_export_demo
    main.cpp
)

# Link the example against the exported native ISTHMUS++ library that provides
# the marching-windows pipeline and mesh writers.
target_link_libraries(isthmus_surface_export_demo PRIVATE isthmus_cpp::isthmus_cpp)
