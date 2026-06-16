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
    DomainConfig domain;
    domain.dimension = Dimension::D3;
    domain.limits = {{{0.0, 0.0, 0.0}, {2.0, 2.0, 2.0}}};
    domain.cell_counts = {{2, 2, 2}};
    domain.voxel_size = 1.0;
    domain.weighting = false;

    SurfaceMesh mesh;
    mesh.vertices = {
        {1.0, 1.0, 1.0},
        {1.5, 1.0, 1.0}
    };
    mesh.triangles = {
        {{0, 1, 1}}
    };

    std::vector<VoxelCell> voxel_grid;
    voxel_grid.push_back(make_surface_voxel({0.5, 0.5, 0.5}, domain.voxel_size, 7));

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
    DomainConfig domain;
    domain.dimension = Dimension::D3;
    domain.limits = {{{0.0, 0.0, 0.0}, {2.0, 2.0, 2.0}}};
    domain.cell_counts = {{2, 2, 2}};
    domain.voxel_size = 1.0;
    domain.weighting = false;

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
    voxel_grid.push_back(make_surface_voxel({0.5, 0.5, 0.5}, domain.voxel_size, 3));

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
    constexpr int buffer = 5;

    DomainConfig domain;
    domain.dimension = Dimension::D3;
    domain.limits = {{{-buffer * voxel_size, -buffer * voxel_size, -buffer * voxel_size},
                      {(sample_length + buffer) * voxel_size,
                       (sample_diameter + buffer) * voxel_size,
                       (sample_diameter + buffer) * voxel_size}}};
    domain.cell_counts = {{static_cast<std::size_t>(sample_length),
                           static_cast<std::size_t>(sample_diameter),
                           static_cast<std::size_t>(sample_diameter)}};
    domain.voxel_size = voxel_size;
    domain.weighting = false;

    RunOptions options;
    options.build_surface = true;
    options.build_flux_association = true;

    MarchingWindows mw;
    const auto result = mw.run(domain, make_voxel_cylinder(voxel_size, sample_length, sample_diameter), options);

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
