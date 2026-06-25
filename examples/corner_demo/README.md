# Corner Demo

This standalone example constructs a simple 2D voxelized square, runs the ISTHMUS marching-windows pipeline, and prints how many corner fill fractions and surface voxels were produced.

## What It Demonstrates

- How a consumer project includes `isthmus/marching_windows.hpp`
- How to configure a small `RunOptions` object
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

## Build With CMake

From the repository root:

```bash
cmake -S examples/corner_demo -B examples/corner_demo/build -Disthmus_cpp_DIR="$PWD/build-wsl"
cmake --build examples/corner_demo/build -j
```

## Build With Make

From the repository root:

```bash
make -C examples/corner_demo
```

The local `Makefile` links against `build-wsl/libisthmus_cpp.a` by default.
Override `ISTHMUS_BUILD_DIR=/path/to/build` if your native library archive
lives somewhere else.

## Run

```bash
./examples/corner_demo/build/isthmus_corner_demo
```

## Use Pattern In Your Own Project

The minimal CMake pattern is:

```cmake
find_package(isthmus_cpp CONFIG REQUIRED)
add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE isthmus_cpp::isthmus_cpp)
```
