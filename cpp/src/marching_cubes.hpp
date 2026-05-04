/*
 * Internal 3D marching-cubes helpers for the native ISTHMUS port.
 *
 * This header is intentionally private to the implementation. Public headers
 * that downstream callers include live under `cpp/include/isthmus`, while this
 * file stays next to the backend source because it only exists to wire the
 * internal marching-cubes stage into motion mapping.
 */
#pragma once

#include <vector>

#include "isthmus/types.hpp"

namespace isthmus::marching_cubes {

/*
 * Reconstruct the 0.5 isosurface from the already-computed corner fill field.
 *
 * The input scalar field uses the native x-fastest flattened layout described
 * by `corner_dims`. The returned mesh is expressed directly in physical
 * coordinates so downstream callers and writers do not need any additional
 * index-space transformation.
 */
SurfaceMesh extract_surface_mesh_3d(
    const DomainConfig& domain,
    const std::vector<double>& corner_fill_fractions,
    const std::array<std::size_t, kMaxDims>& corner_dims,
    double iso_value = 0.5);

}  // namespace isthmus::marching_cubes
