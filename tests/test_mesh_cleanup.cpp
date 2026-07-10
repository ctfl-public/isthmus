#include "test_framework.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <set>
#include <vector>

#include "isthmus/marching_windows.hpp"
#include "isthmus/voxels.hpp"
#include "flux_mapping.hpp"
#include "mesh_cleanup.hpp"

namespace {

/*
 * Build one synthetic surface voxel with the same face layout used by the
 * native motion-mapping path.
 */
isthmus::VoxelCell make_surface_voxel(
    const std::array<double, 3>& centroid,
    double voxel_size,
    std::size_t original_id) {
    isthmus::VoxelCell voxel{};
    voxel.centroid = centroid;
    voxel.original_id = original_id;
    voxel.surface = true;

    const std::array<double, 3> lo{{
        centroid[0] - 0.5 * voxel_size,
        centroid[1] - 0.5 * voxel_size,
        centroid[2] - 0.5 * voxel_size
    }};

    const std::array<std::array<double, 3>, 8> corners{{
        {{lo[0], lo[1], lo[2]}},
        {{lo[0] + voxel_size, lo[1], lo[2]}},
        {{lo[0], lo[1] + voxel_size, lo[2]}},
        {{lo[0] + voxel_size, lo[1] + voxel_size, lo[2]}},
        {{lo[0], lo[1], lo[2] + voxel_size}},
        {{lo[0] + voxel_size, lo[1], lo[2] + voxel_size}},
        {{lo[0], lo[1] + voxel_size, lo[2] + voxel_size}},
        {{lo[0] + voxel_size, lo[1] + voxel_size, lo[2] + voxel_size}}
    }};

    /*
     * Mark every face as exposed so the flux-mapping tests can focus on
     * geometric overlap behavior rather than surface-voxel classification.
     */
    voxel.faces3d = {
        {std::array<std::array<double, 3>, 4>{{corners[2], corners[0], corners[4], corners[6]}}, {{-1.0, 0.0, 0.0}}, true},
        {std::array<std::array<double, 3>, 4>{{corners[1], corners[3], corners[7], corners[5]}}, {{1.0, 0.0, 0.0}}, true},
        {std::array<std::array<double, 3>, 4>{{corners[0], corners[1], corners[5], corners[4]}}, {{0.0, -1.0, 0.0}}, true},
        {std::array<std::array<double, 3>, 4>{{corners[3], corners[2], corners[6], corners[7]}}, {{0.0, 1.0, 0.0}}, true},
        {std::array<std::array<double, 3>, 4>{{corners[2], corners[3], corners[1], corners[0]}}, {{0.0, 0.0, -1.0}}, true},
        {std::array<std::array<double, 3>, 4>{{corners[4], corners[5], corners[7], corners[6]}}, {{0.0, 0.0, 1.0}}, true}
    };

    return voxel;
}

/*
 * Convert one indexed triangle into a sorted list of explicit vertex
 * coordinates so topology-repair tests do not depend on compacted vertex ids.
 */
std::array<std::array<double, 3>, 3> sorted_triangle_points(
    const isthmus::SurfaceMesh& mesh,
    const std::array<std::size_t, 3>& tri) {
    std::array<std::array<double, 3>, 3> out{{
        mesh.vertices[tri[0]],
        mesh.vertices[tri[1]],
        mesh.vertices[tri[2]]
    }};
    std::sort(out.begin(), out.end());
    return out;
}

/*
 * Normalize one explicit triangle description into the same sorted point order
 * used for actual repaired triangles.
 */
std::array<std::array<double, 3>, 3> make_sorted_triangle_points(
    std::array<double, 3> a,
    std::array<double, 3> b,
    std::array<double, 3> c) {
    std::array<std::array<double, 3>, 3> out{{a, b, c}};
    std::sort(out.begin(), out.end());
    return out;
}

/*
 * Build a medium-scale cylindrical voxel set that is closer to the ablation
 * geometry than the tiny cube fixtures used by the existing surface tests.
 */
isthmus::VoxelSet make_voxel_cylinder(
    double voxel_size,
    int sample_length,
    int sample_diameter) {
    isthmus::VoxelSet voxels;
    const double center = 0.5 * static_cast<double>(sample_diameter - 1);
    const double radius = center - 1.0;
    std::size_t original_id = 0;

    for (int x = 0; x < sample_length; ++x) {
        for (int y = 0; y < sample_diameter; ++y) {
            for (int z = 0; z < sample_diameter; ++z) {
                const double dy = static_cast<double>(y) - center;
                const double dz = static_cast<double>(z) - center;
                if ((dy * dy) + (dz * dz) <= radius * radius) {
                    isthmus::VoxelRecord record;
                    record.centroid = {
                        static_cast<double>(x) * voxel_size,
                        static_cast<double>(y) * voxel_size,
                        static_cast<double>(z) * voxel_size
                    };
                    record.original_id = original_id++;
                    voxels.voxels.push_back(record);
                }
            }
        }
    }

    return voxels;
}

}  // namespace

TEST_CASE(test_mesh_cleanup_merges_near_duplicate_vertices_and_drops_repeated_face) {
    using namespace isthmus;

    /*
     * Case:
     * Feed the cleanup stage a mesh where two vertices lie within the
     * near-duplicate tolerance and one triangle therefore collapses.
     *
     * Sketch:
     *   v0 ~ v1 ---- v2
     *      \          |
     *       \         |
     *        \        |
     *          \      |
     *             v3
     *
     * Expected outcome:
     * The near-duplicate merge should collapse `v0` and `v1`, remove the
     * repeated-vertex triangle, and preserve the one remaining valid face.
     */
    SurfaceMesh mesh;
    mesh.vertices = {
        {0.0, 0.0, 0.0},
        {5.0e-8, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0}
    };
    mesh.triangles = {
        {{0, 1, 2}},
        {{1, 2, 3}}
    };

    const auto cleaned = mesh_cleanup::clean_surface_mesh_3d(mesh, 1.0, {});

    CHECK(cleaned.vertices.size() == 3);
    CHECK(cleaned.triangles.size() == 1);
    CHECK(cleaned.triangles[0][0] != cleaned.triangles[0][1]);
    CHECK(cleaned.triangles[0][1] != cleaned.triangles[0][2]);
    CHECK(cleaned.triangles[0][0] != cleaned.triangles[0][2]);
}

TEST_CASE(test_mesh_cleanup_removes_paired_degenerate_triangles) {
    using namespace isthmus;

    /*
     * Case:
     * Supply two collinear triangles that share the same longest edge and have
     * no associated full triangle to repair.
     *
     * Sketch:
     *   A -- B ---- C
     *   A ----- D - C
     *
     * Expected outcome:
     * The paired-degenerate cancellation should remove both low-area triangles
     * entirely.
     */
    SurfaceMesh mesh;
    mesh.vertices = {
        {0.0, 0.0, 0.0},
        {0.25, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.75, 0.0, 0.0}
    };
    mesh.triangles = {
        {{0, 1, 2}},
        {{0, 3, 2}}
    };

    const auto cleaned = mesh_cleanup::clean_surface_mesh_3d(mesh, 1.0, {});
    CHECK(cleaned.triangles.empty());
}

TEST_CASE(test_mesh_cleanup_repairs_degenerate_triangle_connectivity) {
    using namespace isthmus;

    /*
     * Case:
     * Supply the repair configuration where a degenerate triangle shares its
     * longest edge with a single full triangle.
     *
     * Sketch:
     *   A --- B --- C
     *   |           /
     *   |          /
     *   |         /
     *   M--------
     *
     * Expected outcome:
     * The cleanup stage should replace the one full triangle and one
     * degenerate triangle with two nondegenerate triangles that span the same
     * quadrilateral region.
     */
    SurfaceMesh mesh;
    mesh.vertices = {
        {0.0, 0.0, 0.0},
        {0.5, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0}
    };
    mesh.triangles = {
        {{0, 1, 2}},
        {{0, 2, 3}}
    };

    const auto cleaned = mesh_cleanup::clean_surface_mesh_3d(mesh, 1.0, {});

    CHECK(cleaned.triangles.size() == 2);

    std::set<std::array<std::array<double, 3>, 3>> actual;
    for (const auto& tri : cleaned.triangles) {
        actual.insert(sorted_triangle_points(cleaned, tri));
    }

    const std::set<std::array<std::array<double, 3>, 3>> expected{
        make_sorted_triangle_points({{0.0, 0.0, 0.0}}, {{0.0, 1.0, 0.0}}, {{0.5, 0.0, 0.0}}),
        make_sorted_triangle_points({{0.0, 1.0, 0.0}}, {{0.5, 0.0, 0.0}}, {{1.0, 0.0, 0.0}})
    };
    CHECK(actual == expected);
}

/*
 * Append an axis-aligned cube shell with outward normals (solid inside).
 */
void append_cube_shell(isthmus::SurfaceMesh& mesh, double lo, double hi) {
    const std::size_t base = mesh.vertices.size();
    mesh.vertices.insert(mesh.vertices.end(), {
        {lo, lo, lo}, {hi, lo, lo}, {hi, hi, lo}, {lo, hi, lo},
        {lo, lo, hi}, {hi, lo, hi}, {hi, hi, hi}, {lo, hi, hi}
    });
    const std::array<std::array<std::size_t, 3>, 12> faces{{
        {{0, 3, 2}}, {{0, 2, 1}},
        {{4, 5, 6}}, {{4, 6, 7}},
        {{0, 1, 5}}, {{0, 5, 4}},
        {{3, 7, 6}}, {{3, 6, 2}},
        {{0, 4, 7}}, {{0, 7, 3}},
        {{1, 2, 6}}, {{1, 6, 5}}
    }};
    for (const auto& tri : faces) {
        mesh.triangles.push_back({{tri[0] + base, tri[1] + base, tri[2] + base}});
    }
}

/*
 * Append an octahedron shell. Outward normals model a floating solid speck;
 * inverted normals model a sealed cavity. Enclosed volume is (4/3) r^3.
 */
void append_octahedron(
    isthmus::SurfaceMesh& mesh,
    const std::array<double, 3>& center,
    double radius,
    bool inverted) {
    const std::size_t base = mesh.vertices.size();
    const auto [cx, cy, cz] = center;
    mesh.vertices.insert(mesh.vertices.end(), {
        {cx + radius, cy, cz}, {cx - radius, cy, cz},
        {cx, cy + radius, cz}, {cx, cy - radius, cz},
        {cx, cy, cz + radius}, {cx, cy, cz - radius}
    });
    std::array<std::array<std::size_t, 3>, 8> faces{{
        {{0, 2, 4}}, {{2, 1, 4}}, {{1, 3, 4}}, {{3, 0, 4}},
        {{2, 0, 5}}, {{1, 2, 5}}, {{3, 1, 5}}, {{0, 3, 5}}
    }};
    for (auto& tri : faces) {
        if (inverted) {
            std::swap(tri[1], tri[2]);
        }
        mesh.triangles.push_back({{tri[0] + base, tri[1] + base, tri[2] + base}});
    }
}

TEST_CASE(test_mesh_cleanup_component_filter_denoises_but_keeps_real_pores) {
    using namespace isthmus;

    /*
     * Case:
     * Main surface = cube shell. Add a sub-resolution cavity and speck
     * (octahedra with |volume| = 0.0107 voxel volumes, below the default
     * 0.1 threshold) plus one real-sized cavity (volume 10.7 voxel volumes).
     *
     * Expected outcome with default options (remove_sealed_pores = false):
     * both sub-resolution shells are removed as noise, the real cavity is
     * preserved because enclosed porosity is legitimate data.
     */
    SurfaceMesh mesh;
    append_cube_shell(mesh, 0.0, 10.0);
    append_octahedron(mesh, {{5.0, 5.0, 5.0}}, 2.0, true);    // real cavity
    append_octahedron(mesh, {{2.0, 5.0, 5.0}}, 0.2, true);    // noise cavity
    append_octahedron(mesh, {{20.0, 20.0, 20.0}}, 0.2, false); // noise speck

    RunOptions options;
    options.voxel_size = 1.0;
    const auto cleaned = mesh_cleanup::clean_surface_mesh_3d(mesh, 1.0, options);

    CHECK(cleaned.triangles.size() == 12u + 8u);
}

TEST_CASE(test_mesh_cleanup_remove_sealed_pores_drops_all_cavities) {
    using namespace isthmus;

    /*
     * Case:
     * Same geometry idea, but with remove_sealed_pores enabled (the DSMC
     * driver setting) and a real-sized speck added.
     *
     * Expected outcome:
     * every cavity goes regardless of size; the real-sized speck survives
     * because it is legitimate free-standing material.
     */
    SurfaceMesh mesh;
    append_cube_shell(mesh, 0.0, 10.0);
    append_octahedron(mesh, {{5.0, 5.0, 5.0}}, 2.0, true);     // real cavity
    append_octahedron(mesh, {{2.0, 5.0, 5.0}}, 0.2, true);     // noise cavity
    append_octahedron(mesh, {{20.0, 20.0, 20.0}}, 2.0, false); // real speck

    RunOptions options;
    options.voxel_size = 1.0;
    options.remove_sealed_pores = true;
    const auto cleaned = mesh_cleanup::clean_surface_mesh_3d(mesh, 1.0, options);

    CHECK(cleaned.triangles.size() == 12u + 8u);
}

TEST_CASE(test_flux_mapping_tolerates_degenerate_surface_triangle) {
    using namespace isthmus;

    /*
     * Case:
     * Ask the internal flux mapper to process a mesh that contains a repeated-
     * vertex triangle.
     *
     * Sketch:
     *   triangle = [v0, v1, v1]
     *
     * Expected outcome:
     * The production-tolerant flux path should not throw. Instead it should
     * return one empty ownership entry for the invalid triangle.
    */
    VoxelSet domain_voxels;
    domain_voxels.voxels.push_back({{0.5, 0.5, 0.5}, 0});
    RunOptions options;
    options.dimension = Dimension::D3;
    options.voxel_size = 1.0;
    options.marching_voxel_ratio = 1.0;
    options.weighting = false;
    options.build_surface = false;
    options.build_flux_association = false;
    MarchingWindows mw;
    const auto domain_result = mw.run(domain_voxels, options);
    const auto& domain = domain_result.domain;

    SurfaceMesh mesh;
    mesh.vertices = {
        {1.0, 1.0, 1.0},
        {1.5, 1.0, 1.0}
    };
    mesh.triangles = {
        {{0, 1, 1}}
    };

    std::vector<VoxelCell> voxel_grid;
    voxel_grid.push_back(make_surface_voxel({0.5, 0.5, 0.5}, options.voxel_size, 7));

    const auto association = flux_mapping::build_flux_association_3d(domain, mesh, voxel_grid);
    CHECK(association.elements.size() == 1);
    CHECK(association.elements[0].voxel_ids.empty());
    CHECK(association.elements[0].scalar_fractions.empty());
}

TEST_CASE(test_flux_mapping_tolerates_triangle_with_no_usable_overlap) {
    using namespace isthmus;

    /*
     * Case:
     * Ask the internal flux mapper to process a valid triangle that lies far
     * from the exposed faces of the only candidate surface voxel.
     *
     * Sketch:
     *   voxel near origin, triangle near opposite domain corner
     *
     * Expected outcome:
     * The flux mapper should complete and leave the triangle ownership entry
     * empty rather than throwing a hard error.
    */
    VoxelSet domain_voxels;
    domain_voxels.voxels.push_back({{0.5, 0.5, 0.5}, 0});
    RunOptions options;
    options.dimension = Dimension::D3;
    options.voxel_size = 1.0;
    options.marching_voxel_ratio = 1.0;
    options.weighting = false;
    options.build_surface = false;
    options.build_flux_association = false;
    MarchingWindows mw;
    const auto domain_result = mw.run(domain_voxels, options);
    const auto& domain = domain_result.domain;

    SurfaceMesh mesh;
    mesh.vertices = {
        {1.5, 1.5, 1.5},
        {1.75, 1.5, 1.5},
        {1.5, 1.75, 1.5}
    };
    mesh.triangles = {
        {{0, 1, 2}}
    };

    std::vector<VoxelCell> voxel_grid;
    voxel_grid.push_back(make_surface_voxel({0.5, 0.5, 0.5}, options.voxel_size, 3));

    const auto association = flux_mapping::build_flux_association_3d(domain, mesh, voxel_grid);
    CHECK(association.elements.size() == 1);
    CHECK(association.elements[0].voxel_ids.empty());
    CHECK(association.elements[0].scalar_fractions.empty());
}

TEST_CASE(test_medium_scale_cylinder_surface_and_flux_run_completes) {
    using namespace isthmus;

    /*
     * Case:
     * Run the public 3D surface-and-flux path on a medium-scale cylindrical
     * voxel body that is qualitatively closer to the ablation example than the
     * tiny cube fixtures.
     *
     * Sketch:
     *   elongated cylinder-like voxel body with buffered marching domain
     *
     * Expected outcome:
     * The run should complete without exceptions, produce one ownership entry
     * per triangle, and keep the number of empty entries bounded well below a
     * catastrophic fraction of the surface.
     */
    constexpr double voxel_size = 3.3757e-6;
    constexpr int sample_length = 30;
    constexpr int sample_diameter = 28;

    RunOptions options;
    options.dimension = Dimension::D3;
    options.voxel_size = voxel_size;
    options.marching_voxel_ratio = 1.35;
    options.weighting = false;
    options.build_surface = true;
    options.build_flux_association = true;

    MarchingWindows mw;
    const auto result = mw.run(make_voxel_cylinder(voxel_size, sample_length, sample_diameter), options);

    CHECK(!result.surface_mesh.triangles.empty());
    CHECK(result.flux_association.elements.size() == result.surface_mesh.triangles.size());

    std::size_t empty_count = 0;
    for (const auto& element : result.flux_association.elements) {
        if (element.voxel_ids.empty()) {
            ++empty_count;
        }
    }

    CHECK(empty_count <= 64);
}
