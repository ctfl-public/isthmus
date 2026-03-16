#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace isthmus {

// The native implementation currently works in 2D and 3D only.
constexpr std::size_t kMaxDims = 3;

enum class Dimension : std::size_t {
    D2 = 2,
    D3 = 3
};

/*
 * Requests optional algorithm stages in addition to the corner-fill work that
 * is already implemented.
 *
 * `build_surface` asks the library to run the surface extraction stage once a
 * native marching-cubes or marching-squares backend exists.
 * `build_flux_association` asks the library to compute surface-element to
 * voxel ownership fractions once flux mapping is available.
 * `write_diagnostics` is reserved for future structured debug output.
 */
struct RunOptions {
    bool build_surface = false;
    bool build_flux_association = false;
    bool write_diagnostics = false;
};

/*
 * Describes the physical marching-windows domain.
 *
 * `limits` stores the lower and upper coordinate bounds of the marching grid.
 * `cell_counts` gives the number of marching cells along each active axis.
 * `voxel_size` is the edge length of one solid voxel in the caller's model.
 * `weighting` enables the depth-based weighting used to smooth the later
 * corner fill field near the solid boundary.
 */
struct DomainConfig {
    Dimension dimension = Dimension::D3;
    std::array<std::array<double, kMaxDims>, 2> limits{{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}};
    std::array<std::size_t, kMaxDims> cell_counts{{1, 1, 1}};
    double voxel_size = 1.0;
    bool weighting = true;
};

/*
 * Represents one occupied voxel from the caller's solid model.
 *
 * The native code expects voxel locations as centroids in physical space, not
 * as integer lattice indices.
 */
struct VoxelRecord {
    std::array<double, kMaxDims> centroid{{0.0, 0.0, 0.0}};
    std::size_t original_id = 0;
    std::optional<std::string> material_tag;
};

// Collection of occupied voxels supplied to a MarchingWindows run.
struct VoxelSet {
    std::vector<VoxelRecord> voxels;
};

/*
 * Surface connectivity produced by the reconstruction stage.
 *
 * The current native code does not populate this yet, but the structure is
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
 * Complete in-memory result of a native MarchingWindows run.
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
