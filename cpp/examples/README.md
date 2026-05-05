# Native C++ Examples

Each example in this directory is a standalone CMake project. The main
ISTHMUS++ build does not compile these examples automatically.

## Available Examples

- `corner_demo/`
  - Builds a minimal 2D marching-windows consumer.
  - See `corner_demo/README.md` for build and run steps.
- `surface_export_demo/`
  - Builds a 3D mesh-export consumer that writes `.surf` and `.vtp` files.
  - See `surface_export_demo/README.md` for build and run steps.

## General Workflow

1. Build the main ISTHMUS++ library from the repository root.
2. Point an example at that build with `-Disthmus_cpp_DIR=/path/to/main/build`.
3. Configure and build the example in its own out-of-source build directory.

## Build The Main Library First

```bash
cmake -S . -B build-wsl
cmake --build build-wsl -j
```

## Then Build Any Example

Corner demo:

```bash
cmake -S cpp/examples/corner_demo -B build-corner-demo -Disthmus_cpp_DIR="$PWD/build-wsl"
cmake --build build-corner-demo -j
```

Surface export demo:

```bash
cmake -S cpp/examples/surface_export_demo -B build-surface-export-demo -Disthmus_cpp_DIR="$PWD/build-wsl"
cmake --build build-surface-export-demo -j
```
