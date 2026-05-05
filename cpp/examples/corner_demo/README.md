# Corner Demo

This standalone example constructs a simple 2D voxelized square, runs the
native ISTHMUS++ marching-windows pipeline, and prints how many corner fill
fractions and surface voxels were produced.

## What It Demonstrates

- How a consumer project includes `isthmus/marching_windows.hpp`
- How to configure a small `DomainConfig`
- How to build a `VoxelSet` in user code
- How to call `isthmus::MarchingWindows::run`

## Prerequisite: Build ISTHMUS++

Configure and build the main library first from the repository root:

```bash
cmake -S . -B build-wsl
cmake --build build-wsl -j
```

That build generates a CMake package in `build-wsl/` that this standalone
example can consume directly.

## Build This Example Against The Build Tree

From the repository root:

```bash
cmake -S cpp/examples/corner_demo -B build-corner-demo -Disthmus_cpp_DIR="$PWD/build-wsl"
cmake --build build-corner-demo -j
```

## Run

```bash
./build-corner-demo/isthmus_corner_demo
```

## Use Pattern In Your Own Project

The minimal CMake pattern is:

```cmake
find_package(isthmus_cpp CONFIG REQUIRED)
add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE isthmus_cpp::isthmus_cpp)
```
