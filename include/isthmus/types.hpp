#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace isthmus {

// The current implementation works in 2D and 3D only.
constexpr std::size_t kMaxDims = 3;

enum class Dimension : std::size_t {
    D2 = 2,
    D3 = 3
};

/*
 * Requests optional algorithm stages in addition to the corner-fill work that
 * is already implemented.
 *
 * `build_surface`: run the surface extraction 
 * `build_flux_association`: compute surface-voxel ownership fractions for flux mapping. 
 * `write_diagnostics`: reserved for future structured debug output.
 */
struct RunOptions {
    bool build_surface = false;
    bool build_flux_association = false;
    bool write_diagnostics = false;
};

/*
 * Describes the physical marching-windows domain.
 */
struct DomainConfig {
    // Dimension of the marching grid, 2 or 3.
    Dimension dimension = Dimension::D3;

    // Lower and upper coordinate bounds of the marching grid in each dimension.
    std::array<std::array<double, kMaxDims>, 2> limits{{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}};

    // Number of marching cells along each active axis.
    std::array<std::size_t, kMaxDims> cell_counts{{1, 1, 1}};

    // Edge length of voxel in the caller's model.
    double voxel_size = 1.0;

    // Whether to apply depth-based weighting to the corner fill field.
    bool weighting = true;

    // Isosurface value for the marching cubes/squares algorithm.
    double iso_value = 0.5;
};

/*
 * Represents one occupied voxel from the caller's solid model.
 *
 * The code expects voxel locations as centroids in physical space, not
 * as integer lattice indices.
 */
struct VoxelRecord {
    std::array<double, kMaxDims> centroid{{0.0, 0.0, 0.0}};
    std::size_t original_id = 0;
    std::optional<std::string> material_tag;
};

// Collection of occupied voxels supplied to a MarchingWindows run from the caller.
struct VoxelSet {
    std::vector<VoxelRecord> voxels;
};

/*
 * Surface connectivity produced by the reconstruction stage.
 *
 * The current code does not populate this yet, but the structure is
 * part of the stable result shape so downstream code can be written against it.
 */
struct SurfaceMesh {
    std::vector<std::array<double, 3>> vertices;
    std::vector<std::array<std::size_t, 3>> triangles;
    std::vector<std::array<std::size_t, 2>> segments;
};

/*
 * Ownership fractions for one surface element.
 *
 * Once flux mapping is implemented, each surface triangle or line element will
 * own a list of voxel ids and conservative scalar fractions that sum to one.
 */
struct FluxElementOwnership {
    std::size_t element_id = 0;
    std::vector<std::size_t> voxel_ids;
    std::vector<double> scalar_fractions;
};

// Container for all surface-element ownership records from one run.
struct FluxAssociation {
    std::vector<FluxElementOwnership> elements;
};

/*
 * Metadata for voxels that remain on the detected outer boundary after depth
 * classification.
 *
 * These are the voxels whose faces are candidates for later coupling to the
 * reconstructed surface.
 */
struct SurfaceVoxelInfo {
    std::size_t original_id = 0;
    std::array<int, kMaxDims> voxel_indices{{0, 0, 0}};
    std::array<double, kMaxDims> centroid{{0.0, 0.0, 0.0}};
    int depth = 0;
    double weight = 1.0;
};

/*
 * Complete in-memory result of a MarchingWindows run.
 *
 * Today the reliably populated outputs are the validated domain, the corner
 * fill fractions, the dimensions of that corner field, and the list of surface
 * voxels found during depth classification.
 */
struct MarchingWindowsResult {
    DomainConfig domain;
    std::vector<double> corner_fill_fractions;
    std::array<std::size_t, kMaxDims> corner_dims{{0, 0, 0}};
    std::vector<SurfaceVoxelInfo> surface_voxels;
    SurfaceMesh surface_mesh;
    FluxAssociation flux_association;
};

}  // namespace isthmus
