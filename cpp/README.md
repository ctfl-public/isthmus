# C++ ISTHMUS Guide

This directory contains the C++ implementation of ISTHMUS. The goal of
this code is to provide a reusable library that downstream C++ solvers can
link against directly while preserving the marching-windows behavior that
ISTHMUS relies on.

## Current Coverage

The native code currently provides:

- A CMake library target: `isthmus_cpp`
- A public entry point: `isthmus::MarchingWindows`
- Geometry utilities used in mapping
- Native data-loading utilities for TIFF voxel stacks and legacy ownership files
- 3D surface extraction with post-reconstruction cleanup
- 3D flux association between surface triangles and voxels
- SPARTA-style output writers for surface meshes and voxel ownership files
- A lightweight test harness and standalone demo programs

The native code does not yet provide:

- Marching-squares extraction in 2D
- GPU acceleration


## Build and Test

The commands below assume a WSL/Linux toolchain and use `build-wsl` as an out-of-source build directory.

```bash
cmake -S . -B build-wsl
```
Generates the build system in `build-wsl/` and configures the project using the source tree in the repository root.

```bash
cmake --build build-wsl -j
```
Compiles the library and the native tests. The `-j` flag allows the build tool to use parallel jobs.

```bash
ctest --test-dir build-wsl --output-on-failure
```
Runs the compiled native test suite from the build directory and prints detailed output for any failing test.
