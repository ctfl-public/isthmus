# C++ Examples

Each example in this directory is a standalone project. Each one keeps its own
local child build directory at `examples/<example>/build/`, and each one
also provides a GNU `Makefile` for users who prefer `make` over CMake.

## Available Examples

- `corner_demo/`
  - Builds a minimal 2D marching-windows consumer.
  - See [`corner_demo/README.md`](corner_demo/README.md) for build and run steps.

- `surface_export_demo/`
  - Builds a 3D mesh-export consumer that writes `.surf` and `.vtp` files.
  - See [`surface_export_demo/README.md`](surface_export_demo/README.md) for build and run steps.

- `ablation_single_phase/`
  - Builds a native single-phase ablation walkthrough.
  - See [`ablation_single_phase/README.md`](ablation_single_phase/README.md) for build and run steps.

