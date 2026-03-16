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
- A lightweight test harness and a small demo program

The native code does not yet provide:

- Marching-cubes extraction in 3D
- Marching-squares extraction in 2D
- Flux mapping from surface elements back to voxels
- GPU acceleration

## How To Read This Code

- `include/isthmus/`
  - Public-facing API and shared data structures.
  - Start here if you want to understand what a caller is expected to provide and what a run returns.
- `src/`
  - Implementation of the current algorithm stages.
  - `motion_mapping.cpp` is the most important file for understanding the existing native behavior.
  - `geometry.cpp` contains the reusable geometric kernels that later flux mapping and surface extraction will build on.
- `tests/`
  - Small native tests that protect the currently implemented behavior.
  - These tests focus on geometric correctness and agreement with existing verification cases.
- `examples/`
  - Minimal programs that exercise the code without requiring the full Python workflow.

## Build and Test

The commands below assume a WSL/Linux toolchain and use `build-wsl` as an out-of-source build directory.

```bash
cmake -S . -B build-wsl
```
Generates the build system in `build-wsl/` and configures the project using the source tree in the repository root.

```bash
cmake --build build-wsl -j
```
Compiles the library, the demo executable, and the native tests. The `-j` flag allows the build tool to use parallel jobs.

```bash
ctest --test-dir build-wsl --output-on-failure
```
Runs the compiled native test suite from the build directory and prints detailed output for any failing test.

## Demo

```bash
./build-wsl/isthmus_corner_demo
```
Runs the current example program. It constructs a simple voxelized shape, executes the implemented motion-mapping stages, and reports how many corner fill values and surface voxels were generated.

## Relationship to the Python Code

The Python codebase is still the broader reference implementation because it already contains the full workflow, including surface extraction and flux mapping. The C++ code in this directory should be read as a native implementation of the same underlying algorithm, not as a thin wrapper around the Python modules.

Where tests or comments mention existing verification cases, they are pointing to already-known behavior that the native code is expected to reproduce. The intent is to keep the algorithm understandable on its own while still checking that it agrees with the established implementation.
