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
#include <iostream>

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

struct RepairStats {
    std::size_t degenerate_found = 0;
    std::size_t quad_flipped = 0;
    std::size_t kept_for_topology = 0;
};

/*
 * Canonical 64-bit key for one undirected mesh edge.
 *
 * Vertex ids fit in 32 bits for every mesh this pipeline produces (about one
 * million vertices), so packing both endpoints into one integer gives cheap
 * hashing for the edge-count maps used by the topology guards.
 */
using EdgeKey = std::uint64_t;

EdgeKey edge_key(std::size_t a, std::size_t b) {
    if (a > b) {
        std::swap(a, b);
    }
    return (static_cast<std::uint64_t>(a) << 32u) | static_cast<std::uint64_t>(b);
}

/*
 * Count how many triangles reference each undirected edge.
 *
 * In a closed manifold mesh every edge has exactly two incident triangles;
 * one incident triangle marks a boundary edge and three or more mark a
 * non-manifold edge.
 */
std::unordered_map<EdgeKey, int> build_edge_counts(
    const std::vector<std::array<std::size_t, 3>>& triangles) {
    std::unordered_map<EdgeKey, int> counts;
    counts.reserve(triangles.size() * 3u);
    for (const auto& tri : triangles) {
        ++counts[edge_key(tri[0], tri[1])];
        ++counts[edge_key(tri[1], tri[2])];
        ++counts[edge_key(tri[2], tri[0])];
    }
    return counts;
}

struct EdgeHealth {
    std::size_t non_manifold = 0;
    std::size_t boundary = 0;
};

EdgeHealth edge_health(const std::vector<std::array<std::size_t, 3>>& triangles) {
    EdgeHealth health;
    for (const auto& [key, count] : build_edge_counts(triangles)) {
        if (count > 2) {
            ++health.non_manifold;
        } else if (count == 1) {
            ++health.boundary;
        }
    }
    return health;
}

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
    UnionFind& uf,
    std::vector<std::size_t>* old_to_new_out = nullptr) {
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

    if (old_to_new_out != nullptr) {
        *old_to_new_out = std::move(old_to_new);
    }

    return out;
}

/*
 * Merge candidate vertex pairs without damaging mesh topology.
 *
 * Blind duplicate-vertex merging can fuse separate triangle fans that only
 * touch in space, which is what created the non-manifold edges and vertices
 * observed after cleanup on tight microstructure meshes. This driver applies
 * the requested merges, checks the resulting edge health, and when the merge
 * introduced new non-manifold or boundary edges it freezes every vertex
 * involved in a bad edge and retries without those merges. Merges that are
 * topologically safe still happen; unsafe ones are skipped.
 */
SurfaceMesh merge_vertices_topology_safe(
    const SurfaceMesh& mesh,
    const std::vector<std::pair<std::size_t, std::size_t>>& candidate_pairs,
    std::size_t& frozen_vertices_out) {
    const EdgeHealth before = edge_health(mesh.triangles);
    std::vector<char> frozen(mesh.vertices.size(), 0);
    frozen_vertices_out = 0;

    constexpr int kMaxGuardIterations = 20;
    for (int iteration = 0; iteration < kMaxGuardIterations; ++iteration) {
        UnionFind uf(mesh.vertices.size());
        bool any_union = false;
        for (const auto& [a, b] : candidate_pairs) {
            if (frozen[a] == 0 && frozen[b] == 0) {
                uf.unite(a, b);
                any_union = true;
            }
        }

        std::vector<std::size_t> old_to_new;
        SurfaceMesh merged = compress_mesh(mesh.vertices, mesh.triangles, uf, &old_to_new);
        if (!any_union) {
            return merged;
        }

        const EdgeHealth after = edge_health(merged.triangles);
        if (after.non_manifold <= before.non_manifold && after.boundary <= before.boundary) {
            return merged;
        }

        /*
         * The merge made topology worse somewhere. Mark every vertex touching
         * a bad edge in the merged mesh, then freeze the source vertices that
         * map onto those, so the next iteration skips the offending merges.
         */
        std::vector<char> bad_new_vertex(merged.vertices.size(), 0);
        for (const auto& [key, count] : build_edge_counts(merged.triangles)) {
            if (count == 2) {
                continue;
            }
            bad_new_vertex[static_cast<std::size_t>(key >> 32u)] = 1;
            bad_new_vertex[static_cast<std::size_t>(key & 0xffffffffu)] = 1;
        }
        for (std::size_t old_id = 0; old_id < old_to_new.size(); ++old_id) {
            if (bad_new_vertex[old_to_new[old_id]] != 0 && frozen[old_id] == 0) {
                frozen[old_id] = 1;
                ++frozen_vertices_out;
            }
        }
    }

    /*
     * Guard-iteration budget exhausted: give up on merging entirely rather
     * than return a mesh with damaged topology.
     */
    UnionFind identity(mesh.vertices.size());
    return compress_mesh(mesh.vertices, mesh.triangles, identity);
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
SurfaceMesh remove_exact_duplicate_vertices(
    const SurfaceMesh& mesh,
    std::size_t& frozen_vertices_out) {
    std::vector<std::pair<std::size_t, std::size_t>> candidate_pairs;

    /*
     * Only exact coordinate duplicates are merged in this stage, and only when
     * a face explicitly demonstrates that those duplicates exist.
     */
    for (const auto& tri : mesh.triangles) {
        const auto& v0 = mesh.vertices[tri[0]];
        const auto& v1 = mesh.vertices[tri[1]];
        const auto& v2 = mesh.vertices[tri[2]];

        if (exactly_equal(v0, v1)) {
            candidate_pairs.emplace_back(tri[0], tri[1]);
        }
        if (exactly_equal(v0, v2)) {
            candidate_pairs.emplace_back(tri[0], tri[2]);
        }
        if (exactly_equal(v1, v2)) {
            candidate_pairs.emplace_back(tri[1], tri[2]);
        }
    }

    return merge_vertices_topology_safe(mesh, candidate_pairs, frozen_vertices_out);
}

/*
 * Merge vertices that are merely near-duplicates in physical space.
 *
 * This stage uses a deterministic spatial hash so production meshes stay
 * tractable while preserving the same voxel-size-scaled merge tolerance.
 */
SurfaceMesh remove_near_duplicate_vertices(
    const SurfaceMesh& mesh,
    double vertex_epsilon,
    std::size_t& frozen_vertices_out) {
    if (mesh.vertices.empty() || vertex_epsilon <= 0.0) {
        frozen_vertices_out = 0;
        return mesh;
    }

    std::vector<std::pair<std::size_t, std::size_t>> candidate_pairs;
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
                            candidate_pairs.emplace_back(vertex_id, other_id);
                        }
                    }
                }
            }
        }

        bins[key].push_back(vertex_id);
    }

    return merge_vertices_topology_safe(mesh, candidate_pairs, frozen_vertices_out);
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
    double area_epsilon,
    RepairStats& stats) {
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
            ++stats.degenerate_found;
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
     *
     * Each flip is validated against the global edge-count map before it is
     * applied. Without those guards this pass could reuse one full triangle
     * for several repairs, duplicate faces, or create non-manifold and
     * boundary edges — the exact damage observed downstream in SPARTA.
     */
    auto edge_counts = build_edge_counts(mesh.triangles);
    std::vector<char> full_triangle_consumed(full_triangles.size(), 0);

    for (std::size_t i = 0; i < degenerate_triangles.size(); ++i) {

        const auto& degenerate = degenerate_triangles[i];
        const auto& edge = degenerate.longest_edge;
        const auto replacement_vertex_from_degenerate =
            triangle_vertex_not_in_edge(degenerate.triangle, edge);

        /*
         * The flip removes edge A-C and introduces edge B-M. It is only valid
         * when A-C is an ordinary manifold edge shared by exactly this
         * degenerate and one full triangle, and when B-M does not already
         * exist (otherwise the flip would create a non-manifold edge).
         */
        const auto shared_edge_it = edge_counts.find(edge_key(edge[0], edge[1]));
        const int shared_edge_count = shared_edge_it == edge_counts.end() ? 0 : shared_edge_it->second;

        bool repaired = false;
        if (shared_edge_count == 2) {
            const std::size_t original_full_count = full_triangle_consumed.size();
            for (std::size_t full_id = 0; full_id < original_full_count; ++full_id) {
                if (full_triangle_consumed[full_id] != 0) {
                    continue;
                }
                auto full_triangle = full_triangles[full_id];
                if (!triangle_contains_edge(full_triangle, edge)) {
                    continue;
                }

                const auto replacement_vertex_from_full =
                    triangle_vertex_not_in_edge(full_triangle, edge);

                if (replacement_vertex_from_full == replacement_vertex_from_degenerate) {
                    continue;
                }
                if (edge_counts.count(edge_key(
                        replacement_vertex_from_degenerate,
                        replacement_vertex_from_full)) != 0) {
                    continue;
                }

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

                /*
                 * Both replacement triangles must clear the degeneracy
                 * threshold, or the flip only relocates the sliver.
                 */
                if (triangle_area(mesh, full_triangle) < area_epsilon ||
                    triangle_area(mesh, second_triangle) < area_epsilon) {
                    continue;
                }

                full_triangles[full_id] = full_triangle;
                full_triangles.push_back(second_triangle);
                full_triangle_consumed[full_id] = 1;
                full_triangle_consumed.push_back(1);

                /*
                 * Keep the edge-count map in sync with the flip: the shared
                 * edge A-C loses both faces and the new diagonal B-M gains two.
                 */
                edge_counts[edge_key(edge[0], edge[1])] -= 2;
                edge_counts[edge_key(
                    replacement_vertex_from_degenerate,
                    replacement_vertex_from_full)] += 2;

                repaired = true;
                ++stats.quad_flipped;
                break;
            }
        }

        // if reached this point, means the degenerate triangle does not
        // share its longest edge with exactly one full triangle in a way that
        // can be flipped safely, so we keep it as is to preserve connectivity.
        if (!repaired) {
            full_triangles.push_back(degenerate.triangle);
            // Kept degenerates must never serve as flip partners later.
            full_triangle_consumed.push_back(1);
            ++stats.kept_for_topology;
        }
    }

    // move the repaired full triangles into the output mesh.
    // using std::move to avoid unnecessary copying.
    out.triangles = std::move(full_triangles);

    return compact_used_vertices(out);
}

struct ComponentFilterStats {
    std::size_t components = 0;
    std::size_t pores_removed = 0;
    std::size_t specks_removed = 0;
};

/*
 * Drop closed enclosed cavity and floating solid specks.
 *
 * Marching cubes on XRCT voxel data produces, besides the main surface, small
 * closed shells: enclosed cavities (pores sealed inside the solid) and
 * floating solid specks. Signed volume distinguishes them, because triangle
 * normals point from solid into void: a speck's normals face outward
 * (positive volume) while a cavity's normals face inward (negative volume).
 *
 *  - Cavities are removed unconditionally. No particle can ever reach a
 *    sealed pore, and their near-zero enclosed flow volumes are exactly what
 *    broke SPARTA's cut3d/flood-fill marking. If ablation later opens a
 *    pore, the next surface rebuild reconnects it to the main component and
 *    it reappears naturally.
 *  - Specks are removed only when smaller than min_speck_volume, which
 *    filters sub-voxel weighting artifacts while keeping real free-standing
 *    material.
 */
SurfaceMesh remove_trapped_components(
    const SurfaceMesh& mesh,
    double min_speck_volume,
    ComponentFilterStats& stats) {
    if (mesh.vertices.empty() || mesh.triangles.empty()) {
        return mesh;
    }

    /*
     * Identify connected components using Union-Find.
     *  by uniting every triangle's three vertices, each component's root is the
     *  smallest vertex id in that component. The connected triangle are all connected 
     *  to the same root, so we can accumulate each component's signed volume.
     */
    UnionFind uf(mesh.vertices.size());
    for (const auto& tri : mesh.triangles) {
        uf.unite(tri[0], tri[1]);
        uf.unite(tri[0], tri[2]);
    }

    /*
     * Accumulate each component's signed volume with the divergence theorem.
     * The absolute values are only meaningful for closed components, which is
     * what marching cubes produces for isolated shells.
     */
    std::unordered_map<std::size_t, double> component_volume;
    std::unordered_map<std::size_t, std::size_t> component_triangles;
    for (const auto& tri : mesh.triangles) {
        const auto& a = mesh.vertices[tri[0]];
        const auto& b = mesh.vertices[tri[1]];
        const auto& c = mesh.vertices[tri[2]];
        const double signed_volume =
            (a[0] * (b[1] * c[2] - b[2] * c[1]) -
             a[1] * (b[0] * c[2] - b[2] * c[0]) +
             a[2] * (b[0] * c[1] - b[1] * c[0])) / 6.0;
        const auto root = uf.find(tri[0]);
        component_volume[root] += signed_volume;
        ++component_triangles[root];
    }
    stats.components = component_volume.size();

    /*
     * The dominant component (largest triangle count) is always kept, even if
     * its signed volume were negative: orientation conventions belong to the
     * winding checks, not to this filter.
     */
    std::size_t main_root = uf.find(mesh.triangles.front()[0]);
    for (const auto& [root, count] : component_triangles) {
        if (count > component_triangles[main_root]) {
            main_root = root;
        }
    }

    std::unordered_map<std::size_t, bool> keep;
    for (const auto& [root, volume] : component_volume) {
        if (root == main_root) {
            keep[root] = true;
        } else if (volume < 0.0) {
            keep[root] = false;
            ++stats.pores_removed;
        } else if (volume < min_speck_volume) {
            keep[root] = false;
            ++stats.specks_removed;
        } else {
            keep[root] = true;
        }
    }

    if (stats.pores_removed == 0 && stats.specks_removed == 0) {
        return mesh;
    }

    SurfaceMesh out;
    out.vertices = mesh.vertices;
    out.triangles.reserve(mesh.triangles.size());
    for (const auto& tri : mesh.triangles) {
        if (keep[uf.find(tri[0])]) {
            out.triangles.push_back(tri);
        }
    }
    return compact_used_vertices(out);
}

}  // namespace

SurfaceMesh clean_surface_mesh_3d(
    const SurfaceMesh& raw_mesh,
    double min_cell_length,
    const RunOptions& options) {
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
    if (options.verbose) std::cout << "\t\tremoving exact vertices...\n";
    {
        const auto v_before = raw_mesh.vertices.size();
        const auto t_before = raw_mesh.triangles.size();
        std::size_t frozen_exact = 0;
        auto mesh = remove_exact_duplicate_vertices(raw_mesh, frozen_exact);
        if (options.verbose) {
            std::cout << "\t\t  merged " << v_before << " -> " << mesh.vertices.size()
                      << " vertices, dropped " << (t_before - mesh.triangles.size()) << " triangles"
                      << " (" << frozen_exact << " vertices kept for topology)\n";
        }

        /*
         * Merge any remaining near-duplicate vertices in physical space after the
         * exact-duplicate cleanup stage.
         */
        if (options.verbose) std::cout << "\t\tremoving near-duplicate vertices...\n";
        {
            const auto v2_before = mesh.vertices.size();
            const auto t2_before = mesh.triangles.size();
            std::size_t frozen_near = 0;
            mesh = remove_near_duplicate_vertices(mesh, vertex_epsilon, frozen_near);
            if (options.verbose) {
                std::cout << "\t\t  merged " << v2_before << " -> " << mesh.vertices.size()
                          << " vertices, dropped " << (t2_before - mesh.triangles.size()) << " triangles"
                          << " (" << frozen_near << " vertices kept for topology)\n";
            }
        }

        /*
         * Finally remove and repair low-area triangles so the flux mapper sees a
         * production-safe cleaned topology.
         */
        if (options.verbose) std::cout << "\t\trepairing degenerate triangles...\n";
        RepairStats repair_stats;
        mesh = repair_degenerate_triangles(mesh, area_epsilon, repair_stats);
        if (options.verbose) {
            std::cout << "\t\t  found " << repair_stats.degenerate_found << " degenerate"
                      << "; " << repair_stats.quad_flipped << " quad-flipped"
                      << " and " << repair_stats.kept_for_topology << " kept for topology\n";
        }

        /*
         * Finally drop sealed cavities and sub-voxel floating shells, which
         * DSMC solvers cannot mark or use.
         */
        if (options.remove_trapped_components) {
            if (options.verbose) std::cout << "\t\tremoving trapped components...\n";
            const double voxel_volume = std::pow(options.voxel_size, 3);
            const double min_speck_volume =
                options.min_speck_volume_voxels * voxel_volume;
            ComponentFilterStats component_stats;
            mesh = remove_trapped_components(mesh, min_speck_volume, component_stats);
            if (options.verbose) {
                std::cout << "\t\t  " << component_stats.components << " components; removed "
                          << component_stats.pores_removed << " sealed pores and "
                          << component_stats.specks_removed << " sub-threshold specks\n";
            }
        }
        return mesh;
    }
}

}  // namespace isthmus::mesh_cleanup
