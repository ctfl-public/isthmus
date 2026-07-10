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
 * User-facing settings for one MarchingWindows run.
 *
 * `dimension`: 2D or 3D marching domain.
 * `voxel_size`: edge length of one input voxel.
 * `marching_voxel_ratio`: marching cell length divided by voxel_size.
 * `weighting`: apply depth-based weighting to the corner fill field.
 * `iso_value`: isosurface value for marching cubes/squares.
 * `edge_clamp`: minimum interpolation fraction kept between a surface vertex
 *   and the marching-grid corners of its edge, in [0, 0.5). It bounds the
 *   smallest edge length and triangle area of the extracted mesh to
 *   ~edge_clamp * marching cell length, which DSMC/SPARTA cut-cell tolerances
 *   require. 0 disables the clamp and allows degenerate sliver triangles.
 * `build_surface`: run the surface extraction
 * `build_flux_association`: compute surface-voxel ownership fractions for flux mapping.
 * `min_component_volume_voxels`: de-noising threshold in units of one voxel
 *   volume. Closed surface components (floating shells and sealed cavities
 *   alike) with |enclosed volume| below this are removed: the voxel grid
 *   cannot resolve features smaller than a voxel, so such shells are
 *   interpolation artifacts of the weighting/iso stages, not data. Set 0 to
 *   keep every component.
 * `remove_sealed_pores`: additionally remove ALL sealed cavities regardless
 *   of size. Off by default because enclosed porosity is real information
 *   for many consumers (density, conductivity, porosity statistics).
 *   DSMC/SPARTA drivers should enable it: sealed pores are unreachable by
 *   particles, are wrongly counted as flow volume, and a pore contained in a
 *   single grid cell poisons SPARTA's inside/outside cell marking.
 * `verbose`: print progress messages to stdout at each major stage.
 */
struct RunOptions {
    Dimension dimension = Dimension::D3;
    double voxel_size = 1.0;
    double marching_voxel_ratio = 0.0;
    bool weighting = true;
    double iso_value = 0.5;
    double edge_clamp = 0.01;
    bool build_surface = true;
    bool build_flux_association = true;
    double min_component_volume_voxels = 0.1;
    bool remove_sealed_pores = false;
    bool verbose = false;
};

/*
 * Resolved physical marching-windows domain.
 */
struct DomainConfig {
    Dimension dimension = Dimension::D3;
    double voxel_size = 1.0;
    double marching_voxel_ratio = 0.0;
    bool weighting = true;
    double iso_value = 0.5;
    double edge_clamp = 0.01;
    std::array<std::array<double, kMaxDims>, 2> limits{{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}};
    std::array<std::size_t, kMaxDims> cell_counts{{1, 1, 1}};
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
