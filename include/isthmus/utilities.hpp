// Reusable helpers for file and volume workflows.
#pragma once

#include <filesystem>
#include <optional>

#include "isthmus/types.hpp"

namespace isthmus::utilities {

/*
 * Return type for extracting a local voxel block from a TIFF volume.
 *
 * The sliced voxel coordinates are emitted as a regular `VoxelSet` so callers
 * can feed them directly into the public marching-windows interface. The
 * limits field stays in physical space.
 */
struct VoxelSliceResult {
    VoxelSet voxels;
    std::array<std::array<double, kMaxDims>, 2> limits{{{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}};
};

/*
 * Load all active voxels from a narrow 8-bit grayscale TIFF stack.
 *
 * Each `1` voxel in the image stack is interpreted as an occupied lattice cell
 * and then scaled by `voxel_size` to produce the marching input points.
 */
VoxelSet load_active_voxels_from_tiff(
    const std::filesystem::path& tiff_path,
    double voxel_size);

/*
 * Slice a production TIFF stack into a local voxel block.
 *
 * The slice extents are expressed in lattice coordinates, not physical space.
 * The returned voxel coordinates are local to the extracted block and are then
 * scaled by `voxel_size`.
 */
VoxelSliceResult tiff_slicer(
    const std::filesystem::path& tiff_path,
    double x,
    double y,
    double z,
    double length,
    double voxel_size,
    double lb = 5.0,
    std::optional<double> height = std::nullopt);

/*
 * Parse the legacy triangle-to-voxel ownership file format back into the
 * in-memory association structure.
 *
 * This exists so tools and examples can round-trip the same ownership
 * artifacts written by the legacy association format.
 */
FluxAssociation read_flux_association(const std::filesystem::path& path);

}  // namespace isthmus::utilities
