cmake_minimum_required(VERSION 3.22)

# Declare this example as its own standalone CMake project so it can be
# configured independently from the main ISTHMUS++ source tree.
project(isthmus_corner_demo LANGUAGES CXX)

# Locate a previously built or installed ISTHMUS++ package before defining the
# demo target that consumes it.
find_package(isthmus_cpp CONFIG REQUIRED)

# Build the standalone corner-demo executable from this example directory.
add_executable(isthmus_corner_demo
    main.cpp
)

# Link against the exported library target that exposes the marching-windows
# implementation and public headers used by the example.
target_link_libraries(isthmus_corner_demo PRIVATE isthmus_cpp::isthmus_cpp)
