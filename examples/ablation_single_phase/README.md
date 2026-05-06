# Single-Phase Ablation Demo

This standalone example runs the single-phase ablation workflow.

It reconstructs a 3D surface, builds triangle-to-voxel flux ownership, applies
a constant ablation mass to every triangle, and performs three
ablation updates.
For each step it writes:

- `grids/grid_<step>.surf`
- `grids/grid_<step>.vtp`
- `voxel_tri/triangle_voxels_<step>.dat`
- `voxel_data/voxel_data_<step>.dat`
- `volFrac.dat`

## What It Demonstrates

- How a consumer project includes `isthmus/marching_windows.hpp`
- How to use marching windows to reconstruct a surface mesh and flux association from a voxel state
- How to drive a multi-step ablation loop (with a constant per-triangle mass rate instead of external reaction files)
- How to report conservation, dropped-triangle, and dropped-mass diagnostics.

## Prerequisite: Build ISTHMUS++

Configure and build the main library first from the repository root:

```bash
cmake -S . -B build-wsl
cmake --build build-wsl -j
```

## Build With CMake

From the repository root:

```bash
cmake -S examples/ablation_single_phase -B examples/ablation_single_phase/build -Disthmus_cpp_DIR="$PWD/build-wsl"
cmake --build examples/ablation_single_phase/build -j
```

## Build With Make

From the repository root:

```bash
make -C examples/ablation_single_phase
```

The local `Makefile` links against `build-wsl/libisthmus_cpp.a` by default.
Override `ISTHMUS_BUILD_DIR=/path/to/build` if your native library archive
lives somewhere else.

## Run

From build directory:
```bash
./ablation_demo
```

Write into a custom output directory:
```bash
./ablation_demo {custom_output_directory}
```

Override the constant per-triangle ablation mass explicitly:
```bash
./ablation_demo /tmp/isthmus-ablation-single-phase 2.0e-14
```

## Notes

- The example writes `.surf` and `.vtp` grid files instead of STL.
- The example is fully standalone: it loads the bundled `sample1.tif` from this directory.
- The default constant triangle ablation mass is `1.5e-14` kg per triangle per
  step. Pass a different second CLI argument if you want a stronger or weaker
  erosion rate.
