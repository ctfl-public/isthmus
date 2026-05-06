/*
 * Internal 3D flux-mapping helpers for ISTHMUS.
 *
 * This header stays private to the implementation because it depends on
 * internal voxel-grid data structures that are not part of the public API.
 */
#pragma once

#include <vector>

#include "isthmus/types.hpp"
#include "isthmus/voxels.hpp"

namespace isthmus::flux_mapping {

/*
 * Build triangle-to-voxel ownership fractions for an already reconstructed 3D
 * surface mesh.
 *
 * The implementation bins triangles and surface voxels into the marching-
 * domain cell grid, inspects voxels from neighboring cells, projects exposed
 * voxel faces onto triangle planes, measures overlap area, and normalizes the
 * accumulated areas into conservative ownership fractions per triangle.
 */
FluxAssociation build_flux_association_3d(
    const DomainConfig& domain,
    const SurfaceMesh& mesh,
    const std::vector<VoxelCell>& voxel_grid);

}  // namespace isthmus::flux_mapping
