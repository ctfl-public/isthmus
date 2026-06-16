/*
 * 3D surface-mesh cleanup for the ISTHMUS pipeline.
 *
 * This file merges duplicate vertices, drops repeated-vertex faces, and
 * repairs a narrow class of degenerate-triangle connectivity failures so the
 * mesh is robust enough for flux mapping and production-scale ablation
 * runs.
 */
#include "mesh_cleanup.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>

#include "isthmus/geometry.hpp"

namespace isthmus::mesh_cleanup {

namespace {

/*
 * Simple Union-Find helper used by the duplicate-vertex merge passes.
 *
 * This class groups vertex indices that should be treated as the same point.
 * Call `unite(a, b)` when two indices refer to the same vertex, and call
 * `find(x)` to get the group's leader (the chosen representative index).
 *
 * The implementation always keeps the smallest index in a group as the
 * leader. That makes the cleaned mesh deterministic: the same input yields
 * the same vertex chosen to survive each merge.
 *
 * Example:
 *  - Start: parent = [0,1,2,3]
 *  - `unite(2, 0)` -> group {0,2} with leader 0
 *  - `unite(3, 1)` -> group {1,3} with leader 1
 *  - `unite(2, 3)` -> groups merge, leader becomes 0 (smallest index)
 *
 * Notes:
 *  - `find` shortens links on the way back so future lookups are faster.
 *  - Callers must pass valid indices; there are no bounds checks here.
 *  - The class is not thread-safe for concurrent `find`/`unite` calls.
 */
class UnionFind {
public:
    explicit UnionFind(std::size_t n) : parent_(n) {
        for (std::size_t i = 0; i < n; ++i) {
            parent_[i] = i;
        }
    }

    std::size_t find(std::size_t value) {
        if (parent_[value] != value) {
            parent_[value] = find(parent_[value]);
        }
        return parent_[value];
    }

    void unite(std::size_t lhs, std::size_t rhs) {
        lhs = find(lhs);
        rhs = find(rhs);
        if (lhs == rhs) {
            return;
        }
        if (lhs < rhs) {
            parent_[rhs] = lhs;
        } else {
            parent_[lhs] = rhs;
        }
    }

private:
    std::vector<std::size_t> parent_;
};

/*
 * Hashable integer key used by the near-duplicate spatial bins.
 *
 * A tiny voxel-size-scaled bin width keeps comparisons local while still
 * preserving deterministic traversal order.
 */
struct QuantizedKey {
    std::int64_t x = 0;
    std::int64_t y = 0;
    std::int64_t z = 0;

    bool operator==(const QuantizedKey& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct QuantizedKeyHash {
    std::size_t operator()(const QuantizedKey& key) const {
        const auto hx = std::hash<std::int64_t>{}(key.x);
        const auto hy = std::hash<std::int64_t>{}(key.y);
        const auto hz = std::hash<std::int64_t>{}(key.z);
        return hx ^ (hy << 1u) ^ (hz << 2u);
    }
};

/*
 * Keep together one low-area triangle and the longest-edge signature used by
 * the connectivity-repair pass.
 */
struct DegenerateTriangleRecord {
    std::array<std::size_t, 3> triangle{};
    std::array<std::size_t, 2> longest_edge{};
    std::size_t original_order = 0;
};

/*
 * Measure Euclidean distance between two mesh vertices.
 *
 * Euclidean distance is the right metric for the near-duplicate merge stage.
 */
double distance(
    const std::array<double, 3>& lhs,
    const std::array<double, 3>& rhs) {
    const auto dx = lhs[0] - rhs[0];
    const auto dy = lhs[1] - rhs[1];
    const auto dz = lhs[2] - rhs[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

/*
 * Detect exact coordinate equality between two vertices.
 *
 * This matches the first marching-cubes cleanup stage, which merges vertices
 * when a face explicitly references multiple identical points.
 */
bool exactly_equal(
    const std::array<double, 3>& lhs,
    const std::array<double, 3>& rhs) {
    return lhs[0] == rhs[0] &&
           lhs[1] == rhs[1] &&
           lhs[2] == rhs[2];
}

/*
 * Quantize a physical-space vertex into the near-duplicate spatial grid.
 *
 * The grid spacing equals the duplicate tolerance, so any true duplicate must
 * live in the same or an immediately adjacent bin.
 */
QuantizedKey quantize(
    const std::array<double, 3>& vertex,
    double epsilon) {
    return {
        static_cast<std::int64_t>(std::floor(vertex[0] / epsilon)),
        static_cast<std::int64_t>(std::floor(vertex[1] / epsilon)),
        static_cast<std::int64_t>(std::floor(vertex[2] / epsilon))
    };
}

/*
 * Remap vertices through a union-find structure and drop any faces that end up
 * with repeated vertex ids after the merge.
 *
 * This helper also compacts away unused vertices so later cleanup passes work
 * on the smallest possible mesh.
 */
SurfaceMesh compress_mesh(
    const std::vector<std::array<double, 3>>& vertices,
    const std::vector<std::array<std::size_t, 3>>& triangles,
    UnionFind& uf) {
    SurfaceMesh out;
    if (vertices.empty()) {
        return out;
    }

    std::vector<std::size_t> root_to_new(vertices.size(), std::numeric_limits<std::size_t>::max());
    std::vector<std::size_t> old_to_new(vertices.size(), std::numeric_limits<std::size_t>::max());

    /*
     * Assign a compact vertex id to each surviving union-find root. The root
     * id itself determines which coordinate survives the merge.
     */
    for (std::size_t i = 0; i < vertices.size(); ++i) {
        const auto root = uf.find(i);
        if (root_to_new[root] == std::numeric_limits<std::size_t>::max()) {
            root_to_new[root] = out.vertices.size();
            out.vertices.push_back(vertices[root]);
        }
        old_to_new[i] = root_to_new[root];
    }

    /*
     * Remap every face and drop any that collapse onto fewer than three unique
     * vertices after the merge.
     */
    for (const auto& tri : triangles) {
        const std::array<std::size_t, 3> remapped{{
            old_to_new[tri[0]],
            old_to_new[tri[1]],
            old_to_new[tri[2]]
        }};
        if (remapped[0] == remapped[1] ||
            remapped[1] == remapped[2] ||
            remapped[0] == remapped[2]) {
            continue;
        }
        out.triangles.push_back(remapped);
    }

    return out;
}

/*
 * Compact a mesh that already has final face connectivity but may still carry
 * now-unused vertices from earlier cleanup steps.
 */
SurfaceMesh compact_used_vertices(const SurfaceMesh& mesh) {
    SurfaceMesh out;
    if (mesh.vertices.empty() || mesh.triangles.empty()) {
        return out;
    }

    std::vector<std::size_t> old_to_new(mesh.vertices.size(), std::numeric_limits<std::size_t>::max());
    for (const auto& tri : mesh.triangles) {
        for (const auto vertex_id : tri) {
            if (old_to_new[vertex_id] == std::numeric_limits<std::size_t>::max()) {
                old_to_new[vertex_id] = out.vertices.size();
                out.vertices.push_back(mesh.vertices[vertex_id]);
            }
        }
    }

    out.triangles.reserve(mesh.triangles.size());
    for (const auto& tri : mesh.triangles) {
        out.triangles.push_back({{
            old_to_new[tri[0]],
            old_to_new[tri[1]],
            old_to_new[tri[2]]
        }});
    }

    return out;
}

/*
 * Merge vertices that are exactly duplicated within a face-degeneracy pattern.
 *
 * This matches the bundled `remove_degenerate_faces()` pre-pass behavior.
 */
SurfaceMesh remove_exact_duplicate_vertices(const SurfaceMesh& mesh) {
    UnionFind uf(mesh.vertices.size());

    /*
     * Only exact coordinate duplicates are merged in this stage, and only when
     * a face explicitly demonstrates that those duplicates exist.
     */
    for (const auto& tri : mesh.triangles) {
        const auto& v0 = mesh.vertices[tri[0]];
        const auto& v1 = mesh.vertices[tri[1]];
        const auto& v2 = mesh.vertices[tri[2]];

        if (exactly_equal(v0, v1)) {
            uf.unite(tri[0], tri[1]);
        }
        if (exactly_equal(v0, v2)) {
            uf.unite(tri[0], tri[2]);
        }
        if (exactly_equal(v1, v2)) {
            uf.unite(tri[1], tri[2]);
        }
    }

    return compress_mesh(mesh.vertices, mesh.triangles, uf);
}

/*
 * Merge vertices that are merely near-duplicates in physical space.
 *
 * This stage uses a deterministic spatial hash so production meshes stay
 * tractable while preserving the same voxel-size-scaled merge tolerance.
 */
SurfaceMesh remove_near_duplicate_vertices(
    const SurfaceMesh& mesh,
    double vertex_epsilon) {
    if (mesh.vertices.empty() || vertex_epsilon <= 0.0) {
        return mesh;
    }

    UnionFind uf(mesh.vertices.size());
    std::unordered_map<QuantizedKey, std::vector<std::size_t>, QuantizedKeyHash> bins;
    bins.reserve(mesh.vertices.size());

    /*
     * Insert vertices in original order so the smallest representative id
     * remains stable across runs.
     */
    for (std::size_t vertex_id = 0; vertex_id < mesh.vertices.size(); ++vertex_id) {
        const auto key = quantize(mesh.vertices[vertex_id], vertex_epsilon);

        /*
         * True duplicates can only live in the same or adjacent bins, so a
         * 3x3x3 neighborhood search keeps the pass efficient.
         */
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const QuantizedKey neighbor_key{
                        key.x + dx,
                        key.y + dy,
                        key.z + dz
                    };
                    const auto it = bins.find(neighbor_key);
                    if (it == bins.end()) {
                        continue;
                    }

                    for (const auto other_id : it->second) {
                        if (distance(mesh.vertices[vertex_id], mesh.vertices[other_id]) <= vertex_epsilon) {
                            uf.unite(vertex_id, other_id);
                        }
                    }
                }
            }
        }

        bins[key].push_back(vertex_id);
    }

    return compress_mesh(mesh.vertices, mesh.triangles, uf);
}

/*
 * Normalize a longest-edge vertex pair into ascending order.
 *
 * The connectivity-repair stage treats degenerate triangles that share the
 * same longest edge as duplicates, so the edge signature must be order-free.
 */
std::array<std::size_t, 2> canonicalize_edge(
    std::array<std::size_t, 2> edge) {
    if (edge[0] > edge[1]) {
        std::swap(edge[0], edge[1]);
    }
    return edge;
}

/*
 * Compute the area of one indexed mesh triangle.
 */
double triangle_area(
    const SurfaceMesh& mesh,
    const std::array<std::size_t, 3>& tri) {
    const std::array<geometry::Vec3, 3> vertices{{
        mesh.vertices[tri[0]],
        mesh.vertices[tri[1]],
        mesh.vertices[tri[2]]
    }};
    return geometry::triangle_area(vertices);
}

/*
 * Determine whether a full triangle contains both endpoints of a candidate
 * degenerate edge.
 */
bool triangle_contains_edge(
    const std::array<std::size_t, 3>& tri,
    const std::array<std::size_t, 2>& edge) {
    const bool has_a = tri[0] == edge[0] || tri[1] == edge[0] || tri[2] == edge[0];
    const bool has_b = tri[0] == edge[1] || tri[1] == edge[1] || tri[2] == edge[1];
    return has_a && has_b;
}

/*
 * Extract the one vertex in a triangle that is not part of a supplied edge.
 */
std::size_t triangle_vertex_not_in_edge(
    const std::array<std::size_t, 3>& tri,
    const std::array<std::size_t, 2>& edge) {
    for (const auto vertex_id : tri) {
        if (vertex_id != edge[0] && vertex_id != edge[1]) {
            return vertex_id;
        }
    }
    return tri[0];
}

/*
 * Run low-area triangle filtering and connectivity repair.
 *
 * The pass repairs the remaining “degenerate + full triangle” quad case 
 * by replacing them with two nondegenerate ones.
 */
SurfaceMesh repair_degenerate_triangles(
    const SurfaceMesh& mesh,
    double area_epsilon) {
    SurfaceMesh out;
    out.vertices = mesh.vertices;

    std::vector<std::array<std::size_t, 3>> full_triangles;
    std::vector<DegenerateTriangleRecord> degenerate_triangles;
    full_triangles.reserve(mesh.triangles.size());
    degenerate_triangles.reserve(mesh.triangles.size());

    /*
     * Separate the mesh into clearly valid faces and low-area faces that may
     * still carry useful connectivity information.
     */
    for (std::size_t triangle_order = 0; triangle_order < mesh.triangles.size(); ++triangle_order) {
        const auto& tri = mesh.triangles[triangle_order];
        const double area = triangle_area(mesh, tri);
        if (area < area_epsilon) {
            const std::array<geometry::Vec3, 3> vertices{{
                mesh.vertices[tri[0]],
                mesh.vertices[tri[1]],
                mesh.vertices[tri[2]]
            }};
            const auto longest_side = geometry::longest_triangle_side(vertices);
            degenerate_triangles.push_back({
                tri,
                canonicalize_edge({{
                    tri[longest_side[0]],
                    tri[longest_side[1]]
                }}),
                triangle_order
            });
        } else {
            full_triangles.push_back(tri);
        }
    }

    /*
     * Repair the surviving single degenerates by replacing the associated full
     * triangle with two new triangles that span the same quadrilateral region.
     */
    bool repaired = false;
    for (std::size_t i = 0; i < degenerate_triangles.size(); ++i) {

        const auto& degenerate = degenerate_triangles[i];
        const auto& edge = degenerate.longest_edge;
        const auto replacement_vertex_from_degenerate =
            triangle_vertex_not_in_edge(degenerate.triangle, edge);

        repaired = false;
        for (std::size_t full_id = 0; full_id < full_triangles.size(); ++full_id) {
            auto full_triangle = full_triangles[full_id];
            if (!triangle_contains_edge(full_triangle, edge)) {
                continue;
            }

            const auto replacement_vertex_from_full =
                triangle_vertex_not_in_edge(full_triangle, edge);

            /*
             * The replacement topology is:
             *   full ACM  -> BCM
             *   degen ABC -> ABM
             * using the canonicalized edge endpoints as A and C.
             */
            for (auto& vertex_id : full_triangle) {
                if (vertex_id == edge[0]) {
                    vertex_id = replacement_vertex_from_degenerate;
                }
            }

            auto second_triangle = degenerate.triangle;
            for (auto& vertex_id : second_triangle) {
                if (vertex_id == edge[1]) {
                    vertex_id = replacement_vertex_from_full;
                }
            }

            full_triangles[full_id] = full_triangle;
            full_triangles.push_back(second_triangle);
            repaired = true;
            break;
        }

        // if reached this point, means the degenerate triangle does not 
        // share its longest edge with any full triangle and cannot be repaired, 
        // so we keep it as is to keep triangle connectivity.
        if (!repaired) {
            full_triangles.push_back(degenerate.triangle);
        }
    }

    // move the repaired full triangles into the output mesh.
    // using std::move to avoid unnecessary copying.
    out.triangles = std::move(full_triangles);

    return compact_used_vertices(out);
}

}  // namespace

SurfaceMesh clean_surface_mesh_3d(
    const SurfaceMesh& raw_mesh,
    double min_cell_length) {
    /*
     * The cleanup stage is a no-op for empty meshes because there is no
     * topology to repair or vertices to merge.
     */
    if (raw_mesh.vertices.empty() || raw_mesh.triangles.empty()) {
        return raw_mesh;
    }

    /*
     * Use a vertex tolerance that scales with the marching-cubes cell size,
     * because near-duplicate checks operate on the extracted surface mesh.
     */
    const double vertex_epsilon = 1e-7 * min_cell_length;
    const double area_epsilon = 0.5 * std::pow(vertex_epsilon, 2);

    /*
     * Apply the bundled marching-cubes style degenerate-face removal first so
     * obviously duplicated vertices collapse before the ISTHMUS-specific
     * near-duplicate and connectivity-repair logic runs.
     */
    auto mesh = remove_exact_duplicate_vertices(raw_mesh);

    /*
     * Merge any remaining near-duplicate vertices in physical space after the
     * exact-duplicate cleanup stage.
     */
    mesh = remove_near_duplicate_vertices(mesh, vertex_epsilon);

    /*
     * Finally remove and repair low-area triangles so the flux mapper sees a
     * production-safe cleaned topology.
     */
    mesh = repair_degenerate_triangles(mesh, area_epsilon);
    return mesh;
}

}  // namespace isthmus::mesh_cleanup
