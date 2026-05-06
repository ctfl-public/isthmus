/*
 * Internal surface-mesh cleanup helpers for the ISTHMUS pipeline.
 *
 * This private helper repairs the raw marching-cubes mesh before flux mapping
 * and export so downstream stages see a deterministic and production-safe
 * surface.
 */
#pragma once

#include "isthmus/types.hpp"

namespace isthmus::mesh_cleanup {

/*
 * Clean a raw 3D marching-cubes mesh using the established sequence:
 *
 * 1. Merge duplicate vertices implied by degenerate faces.
 * 2. Merge near-duplicate vertices within a voxel-size-scaled tolerance.
 * 3. Drop repeated-vertex triangles.
 * 4. Remove or repair low-area degenerate triangles to preserve connectivity.
 *
 * The returned mesh remains deterministic so downstream SPARTA exports keep a
 * stable triangle order across repeated runs.
 */
SurfaceMesh clean_surface_mesh_3d(
    const SurfaceMesh& raw_mesh,
    double voxel_size);

}  // namespace isthmus::mesh_cleanup
