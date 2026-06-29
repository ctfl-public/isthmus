// Reusable helpers for file and volume workflows.
#pragma once

#include <filesystem>
#include <optional>
#include <cstdint>

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
 * Return type for label-preserving TIFF volume import.
 *
 * The voxels field contains every nonzero TIFF voxel. Each VoxelRecord's
 * material_tag is the integer grayscale value encoded as decimal text.
 * Dimensions are stored in native `(z, y, x)` TIFF stack order.
 */
struct LabeledTiffVoxelSet {
    VoxelSet voxels;
    std::array<std::size_t, 3> dims{{0u, 0u, 0u}};
};

/*
 * Load all active voxels from a narrow 8-bit grayscale TIFF stack.
 *
 * Each voxel with value of 1 in the image stack is interpreted as an occupied lattice cell
 * and then scaled by `voxel_size` to produce the marching input points.
 * 
 * It fills centroids in (Z, Y, X) order: 
 *  centroid[0] = tiff_page × voxel_size, 
 *  centroid[1] = tiff_row × voxel_size,
 *  centroid[2] = tiff_column × voxel_size
 */
VoxelSet load_active_voxels_from_tiff(
    const std::filesystem::path& tiff_path,
    double voxel_size);

/*
 * Load all nonzero voxels from a narrow 8-bit grayscale TIFF stack while
 * preserving each voxel's grayscale value as its material tag.
 *
 * Coordinates follow the same `(Z, Y, X)` convention as
 * load_active_voxels_from_tiff.
 */
LabeledTiffVoxelSet load_labeled_voxels_from_tiff(
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
