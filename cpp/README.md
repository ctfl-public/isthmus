# Native C++ ISTHMUS Developer Guide

This directory contains the native C++ implementation that is being built up alongside the established Python codebase. The goal of this code is to provide a reusable library that downstream C++ solvers can link against directly while preserving the marching-windows behavior that ISTHMUS already relies on.

## What the C++ Code Does Today

The native code currently covers the part of marching windows that turns a sparse list of occupied voxels into the scalar field that later surface extraction will consume.

At a high level, the implemented flow is:

1. Validate the user-supplied marching-windows domain.
2. Expand the sparse voxel list into the regular voxel lattice used by the algorithm.
3. Classify voxels by boundary depth.
   - Solid voxels are layered inward from the exterior surface.
   - Ghost voxels are layered outward into the nearby void.
4. Convert those depth layers into voxel weights.
5. Identify which voxel faces are exposed to the surrounding void.
6. Split voxel volume among corner neighborhoods.
7. Compute one fill fraction per corner neighborhood.

Those corner fill fractions are the scalar values that later marching-cubes or marching-squares extraction will consume.

## Current Coverage

The native code currently provides:

- A CMake library target: `isthmus_cpp`
- A public entry point: `isthmus::MarchingWindows`
- Shared input and output types for C++ callers
- Geometry utilities used by the implemented and planned mapping stages
- Early motion-mapping stages up through corner fill fraction generation
- SPARTA-style output writers for surface meshes and voxel ownership files
- A lightweight test harness and a pair of standalone demo programs

The native code does not yet provide:

- Marching-cubes extraction in 3D
- Marching-squares extraction in 2D
- Flux mapping from surface elements back to voxels
- GPU acceleration

## How To Read This Code

- `include/isthmus/`
  - Public-facing API and shared data structures.
  - Start here if you want to understand what a caller is expected to provide and what a run returns.
- `uml.md`
  - A current UML-style class diagram for the native C++ implementation.
  - Use this as a quick map of the public API, internal grid types, geometry helpers, IO writers, and the native surface backend.
- `src/`
  - Implementation of the current algorithm stages.
  - `motion_mapping.cpp` is the most important file for understanding the existing native behavior.
  - `geometry.cpp` contains the reusable geometric kernels that later flux mapping and surface extraction will build on.
- `tests/`
  - Small native tests that protect the currently implemented behavior.
  - These tests focus on geometric correctness and agreement with existing verification cases.
- `examples/`
  - One subdirectory per standalone native example, each with its own `CMakeLists.txt`.
  - See `examples/README.md` for the list of examples and consumer-style build steps.

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

## Use From Standalone Example Projects

```bash
cmake -S cpp/examples/corner_demo -B build-corner-demo -Disthmus_cpp_DIR="$PWD/build-wsl"
cmake --build build-corner-demo -j
```
Configures and builds the standalone corner-demo project against the already
built ISTHMUS++ package exported from `build-wsl/`.

```bash
./build-corner-demo/isthmus_corner_demo
```
Runs the corner demo after it has been built as a separate consumer project.

```bash
cmake -S cpp/examples/surface_export_demo -B build-surface-export-demo -Disthmus_cpp_DIR="$PWD/build-wsl"
cmake --build build-surface-export-demo -j
```
Configures and builds the standalone surface-export demo against the already
built ISTHMUS++ package exported from `build-wsl/`.

```bash
./build-surface-export-demo/isthmus_surface_export_demo
```
Runs the native 3D surface export example. It reconstructs a small synthetic
voxel cube, writes `surface_cube.surf` and `surface_cube.vtp` into
`surface_export_demo_output/`, and prints the absolute output paths. The
`.vtp` file opens directly in ParaView.

```bash
./build-surface-export-demo/isthmus_surface_export_demo /tmp/isthmus-surface-demo
```
Runs the same example but writes the exported files into the user-specified
output directory.


