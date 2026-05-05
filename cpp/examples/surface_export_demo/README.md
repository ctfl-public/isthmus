# Surface Export Demo

This standalone example reconstructs a small synthetic 3D voxel cube, builds a
native surface mesh, and writes both `surface_cube.surf` and
`surface_cube.vtp`.

## What It Demonstrates

- How a consumer project includes `isthmus/marching_windows.hpp`
- How to request surface generation through `RunOptions`
- How to export native mesh output through `isthmus/io.hpp`
- How to write results to a caller-selected output directory

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
cmake -S cpp/examples/surface_export_demo -B build-surface-export-demo -Disthmus_cpp_DIR="$PWD/build-wsl"
cmake --build build-surface-export-demo -j
```

## Run

Use the default output directory:

```bash
./build-surface-export-demo/isthmus_surface_export_demo
```

Write into a caller-selected output directory:

```bash
./build-surface-export-demo/isthmus_surface_export_demo /tmp/isthmus-surface-demo
```

## Use Pattern In Your Own Project

The minimal CMake pattern is:

```cmake
find_package(isthmus_cpp CONFIG REQUIRED)
add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE isthmus_cpp::isthmus_cpp)
```
