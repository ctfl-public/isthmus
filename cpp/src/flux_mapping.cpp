/*
 * Native 3D flux-mapping implementation for the ISTHMUS pipeline.
 *
 * This stage associates each reconstructed surface triangle with the surface
 * voxels whose exposed faces overlap the triangle when viewed along the
 * triangle normal.
 */
#include "flux_mapping.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <optional>
#include <vector>

#include "isthmus/geometry.hpp"

namespace isthmus::flux_mapping {

namespace {

/*
 * Hold precomputed triangle data so the ownership loop can reuse geometric
 * quantities instead of rebuilding them for every candidate voxel face.
 */
struct TriangleData {
    std::size_t triangle_id = 0;
    std::array<geometry::Vec3, 3> vertices{};
    geometry::Vec3 centroid{{0.0, 0.0, 0.0}};
    geometry::Vec3 normal{{0.0, 0.0, 0.0}};
    std::array<geometry::Vec3, 3> clip_plane_normals{};
    double area = 0.0;
    double epsilon = 0.0;
    bool valid = true;
};

/*
 * Convert the public dimension enum into a raw active-dimension count.
 *
 * This helper keeps the cell-binning logic aligned with the rest of the native
 * code, which stores dimensions as enums publicly but uses integers
 * internally for loops and indexing.
 */
std::size_t active_dims(Dimension d) {
    return static_cast<std::size_t>(d);
}

/*
 * Compute one marching-domain cell length per active direction.
 *
 * Flux mapping uses the marching-domain cell grid for neighborhood lookup, so
 * this helper reconstructs the same physical cell spacing used elsewhere in
 * the pipeline.
 */
std::array<double, kMaxDims> cell_lengths(const DomainConfig& domain) {
    std::array<double, kMaxDims> out{{0.0, 0.0, 0.0}};
    const auto dims = active_dims(domain.dimension);
    for (std::size_t i = 0; i < dims; ++i) {
        out[i] = (domain.limits[1][i] - domain.limits[0][i]) /
                 static_cast<double>(domain.cell_counts[i]);
    }
    return out;
}

/*
 * Flatten a structured-grid index using the native x-fastest convention.
 *
 * The ownership stage stores per-cell buckets in flat vectors for cache
 * locality, so this helper converts structured indices into flat positions.
 */
std::size_t flatten_index(
    const std::array<int, kMaxDims>& index,
    const std::array<std::size_t, kMaxDims>& dims,
    std::size_t ndims) {
    std::size_t flat = 0;
    std::size_t stride = 1;
    for (std::size_t i = 0; i < ndims; ++i) {
        flat += static_cast<std::size_t>(index[i]) * stride;
        stride *= dims[i];
    }
    return flat;
}

/*
 * Check whether a structured cell index lies inside the marching-domain grid.
 *
 * Neighbor lookup visits the 26 adjacent cells around each triangle-owning
 * cell, so a bounds check is required before flattening candidate indices.
 */
bool valid_index(
    const std::array<int, kMaxDims>& index,
    const std::array<std::size_t, kMaxDims>& dims,
    std::size_t ndims) {
    for (std::size_t i = 0; i < ndims; ++i) {
        if (index[i] < 0 || static_cast<std::size_t>(index[i]) >= dims[i]) {
            return false;
        }
    }
    return true;
}

/*
 * Clamp a physical-space point into its owning marching-domain cell.
 *
 * Triangle centroids and voxel centroids are guaranteed to live in or on the
 * marching domain, but exact-boundary floating-point values still need safe
 * clamping so they map into the last valid cell instead of falling one past
 * the end.
 */
std::array<int, kMaxDims> point_to_cell_index(
    const std::array<double, kMaxDims>& point,
    const DomainConfig& domain,
    const std::array<double, kMaxDims>& lengths) {
    std::array<int, kMaxDims> out{{0, 0, 0}};
    const auto dims = active_dims(domain.dimension);
    for (std::size_t i = 0; i < dims; ++i) {
        const double scaled = (point[i] - domain.limits[0][i]) / lengths[i];
        const auto upper = static_cast<int>(domain.cell_counts[i] - 1);
        out[i] = std::clamp(static_cast<int>(scaled), 0, upper);
    }
    return out;
}

/*
 * Compute the arithmetic centroid of a triangle in physical coordinates.
 *
 * The centroid is used only for cell bucketing, so a simple average of the
 * three vertices is sufficient for the cell-bucketing stage.
 */
std::array<double, kMaxDims> triangle_centroid(const std::array<geometry::Vec3, 3>& vertices) {
    return {{
        (vertices[0][0] + vertices[1][0] + vertices[2][0]) / 3.0,
        (vertices[0][1] + vertices[1][1] + vertices[2][1]) / 3.0,
        (vertices[0][2] + vertices[1][2] + vertices[2][2]) / 3.0
    }};
}

/*
 * Normalize a 3D vector when it has a meaningful length.
 *
 * Triangle normals drive both back-face rejection and the clipping-plane
 * construction, so a graceful “no normal available” result is more useful to
 * the tolerant production path than throwing an exception.
 */
std::optional<geometry::Vec3> normalize(const geometry::Vec3& v) {
    const double len = geometry::norm(v);
    if (len <= 0.0) {
        return std::nullopt;
    }
    return geometry::scale(v, 1.0 / len);
}

/*
 * Build the inward-facing edge half-space normals used for triangle clipping.
 *
 * Each normal is orthogonal to one triangle edge and lies in the triangle
 * plane. The existing Sutherland-Hodgman helper then uses these three
 * half-spaces to clip a projected quadrilateral down to the triangle overlap.
 */
std::array<geometry::Vec3, 3> triangle_clip_plane_normals(
    const std::array<geometry::Vec3, 3>& vertices,
    const geometry::Vec3& normal,
    bool& valid) {
    std::array<geometry::Vec3, 3> out{};
    valid = true;
    for (std::size_t i = 0; i < 3; ++i) {
        const auto edge = geometry::subtract(vertices[i], vertices[(i + 2) % 3]);
        const auto normalized = normalize(geometry::cross(edge, normal));
        if (!normalized.has_value()) {
            valid = false;
            return out;
        }
        out[i] = *normalized;
    }
    return out;
}

/*
 * Precompute reusable geometric data for every triangle in the surface mesh.
 *
 * The ownership walk may test the same triangle against many voxel faces, so
 * this stage computes the triangle plane, clipping planes, centroid, and
 * tolerances once up front.
 */
std::vector<TriangleData> build_triangle_data(const SurfaceMesh& mesh) {
    std::vector<TriangleData> out;
    out.reserve(mesh.triangles.size());

    for (std::size_t triangle_id = 0; triangle_id < mesh.triangles.size(); ++triangle_id) {
        const auto& tri = mesh.triangles[triangle_id];

        TriangleData data{};
        data.triangle_id = triangle_id;
        data.vertices = {{
            mesh.vertices[tri[0]],
            mesh.vertices[tri[1]],
            mesh.vertices[tri[2]]
        }};
        data.centroid = triangle_centroid(data.vertices);
        const auto edge_u = geometry::subtract(data.vertices[1], data.vertices[0]);
        const auto edge_v = geometry::subtract(data.vertices[2], data.vertices[0]);
        const auto raw_normal = geometry::cross(edge_u, edge_v);
        const double raw_normal_norm = geometry::norm(raw_normal);
        data.area = 0.5 * raw_normal_norm;

        /*
         * Large production-style surfaces can still contain a few numerically
         * degenerate triangles. The ownership stage skips those triangles
         * rather than aborting the entire run because they carry no physical
         * area to distribute.
         */
        if (raw_normal_norm <= 1e-24 || data.area <= 1e-24) {
            data.valid = false;
            out.push_back(data);
            continue;
        }

        data.normal = geometry::scale(raw_normal, 1.0 / raw_normal_norm);

        /*
         * Scale the clipping tolerance by the largest triangle span rather
         * than by area so very skinny but still valid production triangles
         * remain stable.
         */
        const double span_x = std::max({data.vertices[0][0], data.vertices[1][0], data.vertices[2][0]}) -
                              std::min({data.vertices[0][0], data.vertices[1][0], data.vertices[2][0]});
        const double span_y = std::max({data.vertices[0][1], data.vertices[1][1], data.vertices[2][1]}) -
                              std::min({data.vertices[0][1], data.vertices[1][1], data.vertices[2][1]});
        const double span_z = std::max({data.vertices[0][2], data.vertices[1][2], data.vertices[2][2]}) -
                              std::min({data.vertices[0][2], data.vertices[1][2], data.vertices[2][2]});
        const double max_span = std::max({span_x, span_y, span_z});
        data.epsilon = std::max(1e-12, 1e-4 * max_span);

        bool clip_planes_valid = true;
        data.clip_plane_normals = triangle_clip_plane_normals(data.vertices, data.normal, clip_planes_valid);
        if (!clip_planes_valid) {
            data.valid = false;
        }
        out.push_back(data);
    }

    return out;
}

/*
 * Project one exposed voxel face orthogonally onto the triangle plane.
 *
 * The projected quadrilateral is the subject polygon used in the later
 * triangle-clipping step.
 */
std::vector<geometry::Vec3> project_face_onto_triangle_plane(
    const VoxelFace3D& face,
    const TriangleData& triangle) {
    std::vector<geometry::Vec3> projected;
    projected.reserve(face.corners.size());
    for (const auto& corner : face.corners) {
        const auto delta = geometry::subtract(corner, triangle.vertices[0]);
        const double signed_distance = geometry::dot(delta, triangle.normal);
        projected.push_back(
            geometry::subtract(corner, geometry::scale(triangle.normal, signed_distance)));
    }
    return projected;
}

/*
 * Measure the area shared by one projected voxel face and one triangle.
 *
 * A zero or tiny overlap is treated as numerically insignificant and dropped
 * before it contributes to voxel ownership totals.
 */
double projected_overlap_area(const VoxelFace3D& face, const TriangleData& triangle) {
    const auto projected_face = project_face_onto_triangle_plane(face, triangle);
    const auto clipped = geometry::clip_polygon_sutherland_hodgman(
        projected_face,
        triangle.clip_plane_normals,
        triangle.vertices,
        triangle.epsilon);

    if (clipped.size() < 3) {
        return 0.0;
    }

    const auto oriented = geometry::orient_polygon_xy(clipped, triangle.normal);
    return geometry::polygon_area(oriented);
}

/*
 * Produce the fixed 3x3x3 neighborhood offsets used by the ownership walk.
 *
 * Flux mapping gathers surface voxels from the current triangle cell plus its
 * 26 neighbors.
 */
std::vector<std::array<int, kMaxDims>> neighbor_offsets_3d() {
    std::vector<std::array<int, kMaxDims>> out;
    out.reserve(27);
    for (int z = -1; z <= 1; ++z) {
        for (int y = -1; y <= 1; ++y) {
            for (int x = -1; x <= 1; ++x) {
                out.push_back({{x, y, z}});
            }
        }
    }
    return out;
}

}  // namespace

FluxAssociation build_flux_association_3d(
    const DomainConfig& domain,
    const SurfaceMesh& mesh,
    const std::vector<VoxelCell>& voxel_grid) {
    /*
     * Reject non-3D requests at the module boundary so the call site can keep
     * using a simple dimension-based dispatch in MotionMapper.
     */
    if (domain.dimension != Dimension::D3) {
        throw NotImplementedError("3D flux mapping requires a 3D marching domain");
    }

    /*
     * Empty meshes simply produce empty ownership data, which keeps the
     * result internally consistent and avoids special cases in callers.
     */
    if (mesh.triangles.empty()) {
        return {};
    }

    /*
     * Build the triangle geometry cache before the neighborhood walk so later
     * face tests can reuse normals, areas, and clipping planes.
     */
    const auto triangles = build_triangle_data(mesh);

    /*
     * Reconstruct the marching-domain cell grid, then bucket both triangles
     * and surface voxels into that grid.
     */
    const auto lengths = cell_lengths(domain);
    const std::array<std::size_t, kMaxDims> cell_dims = domain.cell_counts;
    const std::size_t cell_count = cell_dims[0] * cell_dims[1] * cell_dims[2];

    std::vector<std::vector<std::size_t>> triangle_buckets(cell_count);
    std::vector<std::vector<const VoxelCell*>> voxel_buckets(cell_count);

    /*
     * Bucket only detected surface voxels because interior and void voxels do
     * not participate in surface-element ownership.
     */
    for (const auto& voxel : voxel_grid) {
        if (!voxel.surface) {
            continue;
        }
        const auto cell_index = point_to_cell_index(voxel.centroid, domain, lengths);
        voxel_buckets[flatten_index(cell_index, cell_dims, 3)].push_back(&voxel);
    }

    /*
     * Bucket triangles by centroid in the same marching-domain cell lattice
     * so each occupied cell can later search neighboring surface voxels.
     */
    for (const auto& triangle : triangles) {
        if (!triangle.valid) {
            continue;
        }
        const auto cell_index = point_to_cell_index(triangle.centroid, domain, lengths);
        triangle_buckets[flatten_index(cell_index, cell_dims, 3)].push_back(triangle.triangle_id);
    }

    /*
     * Prepare the final result shape up front so every mesh triangle receives
     * one ownership record even if a later error interrupts the run.
     */
    FluxAssociation association;
    association.elements.resize(mesh.triangles.size());
    for (std::size_t i = 0; i < association.elements.size(); ++i) {
        association.elements[i].element_id = i;
    }

    /*
     * Reuse the same fixed 27-cell stencil for every populated triangle cell.
     * This keeps the stencil fixed across every populated triangle cell.
     */
    const auto neighbor_offsets = neighbor_offsets_3d();

    /*
     * Walk the domain cell grid and only do detailed work for cells that
     * actually contain triangles.
     */
    for (std::size_t flat_cell = 0; flat_cell < triangle_buckets.size(); ++flat_cell) {
        const auto& resident_triangles = triangle_buckets[flat_cell];
        if (resident_triangles.empty()) {
            continue;
        }

        /*
         * Decode the flat cell index back into structured coordinates so the
         * 3x3x3 neighborhood can be enumerated around the current cell.
         */
        std::array<int, kMaxDims> base_index{{
            static_cast<int>(flat_cell % cell_dims[0]),
            static_cast<int>((flat_cell / cell_dims[0]) % cell_dims[1]),
            static_cast<int>(flat_cell / (cell_dims[0] * cell_dims[1]))
        }};

        /*
         * Gather pointers to all surface voxels from the current and adjacent
         * marching cells before testing them against the resident triangles.
         */
        std::vector<const VoxelCell*> candidate_voxels;
        for (const auto& offset : neighbor_offsets) {
            const std::array<int, kMaxDims> neighbor_index{{
                base_index[0] + offset[0],
                base_index[1] + offset[1],
                base_index[2] + offset[2]
            }};
            if (!valid_index(neighbor_index, cell_dims, 3)) {
                continue;
            }
            const auto neighbor_flat = flatten_index(neighbor_index, cell_dims, 3);
            candidate_voxels.insert(
                candidate_voxels.end(),
                voxel_buckets[neighbor_flat].begin(),
                voxel_buckets[neighbor_flat].end());
        }

        /*
         * Evaluate every triangle in the current cell against all candidate
         * surface voxels and accumulate one overlap area total per voxel id.
         */
        for (const auto triangle_id : resident_triangles) {
            const auto& triangle = triangles[triangle_id];
            auto& element = association.elements[triangle_id];
            std::map<std::size_t, double> accumulated_area;

            /*
             * Discard tiny overlap slivers relative to triangle size as
             * numerical noise instead of treating them as meaningful
             * ownership.
             */
            const double area_tolerance = triangle.area * 1e-6;

            for (const auto* voxel : candidate_voxels) {
                for (const auto& face : voxel->faces3d) {
                    if (!face.exposed) {
                        continue;
                    }
                    if (geometry::dot(face.normal, triangle.normal) <= 0.0) {
                        continue;
                    }

                    const double overlap = projected_overlap_area(face, triangle);
                    if (overlap <= area_tolerance) {
                        continue;
                    }

                    accumulated_area[voxel->original_id] += overlap;
                }
            }

            /*
             * A small population of triangles can lose their overlap after
             * clipping. Leave those ownership entries empty instead of
             * aborting the whole production run.
             */
            double total_area = 0.0;
            for (const auto& [voxel_id, overlap] : accumulated_area) {
                (void)voxel_id;
                total_area += overlap;
            }
            if (total_area <= area_tolerance) {
                continue;
            }

            /*
             * Normalize the accumulated areas into conservative fractions whose
             * sum is one for the current triangle.
             */
            for (const auto& [voxel_id, overlap] : accumulated_area) {
                element.voxel_ids.push_back(voxel_id);
                element.scalar_fractions.push_back(overlap / total_area);
            }
        }
    }

    return association;
}

}  // namespace isthmus::flux_mapping
