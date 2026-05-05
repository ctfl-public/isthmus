#include "test_framework.hpp"

#include <algorithm>
#include <array>
#include <cmath>

#include "isthmus/marching_windows.hpp"
#include "../src/marching_cubes.hpp"

namespace {

/*
 * Compare two 3D points component-wise with a small absolute tolerance.
 *
 * The surface-geometry tests use this helper because marching-cubes output is
 * deterministic here, but floating-point coordinates should still be compared
 * with tolerance rather than exact bitwise equality.
 */
bool points_match(
    const std::array<double, 3>& lhs,
    const std::array<double, 3>& rhs,
    double epsilon) {
    return std::abs(lhs[0] - rhs[0]) <= epsilon &&
           std::abs(lhs[1] - rhs[1]) <= epsilon &&
           std::abs(lhs[2] - rhs[2]) <= epsilon;
}

/*
 * Compare two triangles as unordered sets of vertices.
 *
 * The winding order is not important for these regression tests. What matters
 * is that the extractor returns the expected geometric triangle in space.
 */
bool triangle_matches_unordered(
    const std::array<std::array<double, 3>, 3>& expected,
    const std::array<std::array<double, 3>, 3>& actual,
    double epsilon) {
    std::array<bool, 3> matched{{false, false, false}};
    for (const auto& expected_vertex : expected) {
        bool found = false;
        for (std::size_t i = 0; i < actual.size(); ++i) {
            if (!matched[i] && points_match(expected_vertex, actual[i], epsilon)) {
                matched[i] = true;
                found = true;
                break;
            }
        }
        if (!found) {
            return false;
        }
    }
    return true;
}

isthmus::VoxelSet make_voxel_square(double voxel_size) {
    isthmus::VoxelSet voxels;
    std::size_t id = 0;
    const double cube_lo = -4.0 * voxel_size;
    for (int j = 0; j < 8; ++j) {
        for (int i = 0; i < 8; ++i) {
            isthmus::VoxelRecord record;
            record.centroid = {
                cube_lo + (0.5 + i) * voxel_size,
                cube_lo + (0.5 + j) * voxel_size,
                0.0
            };
            record.original_id = id++;
            voxels.voxels.push_back(record);
        }
    }
    return voxels;
}

isthmus::VoxelSet make_voxel_cube(double cube_side_length, int voxels_per_axis) {
    isthmus::VoxelSet voxels;
    const double voxel_size = cube_side_length / static_cast<double>(voxels_per_axis);
    const double cube_lo = -0.5 * cube_side_length;
    std::size_t id = 0;

    for (int k = 0; k < voxels_per_axis; ++k) {
        for (int j = 0; j < voxels_per_axis; ++j) {
            for (int i = 0; i < voxels_per_axis; ++i) {
                isthmus::VoxelRecord record;
                record.centroid = {
                    cube_lo + (0.5 + static_cast<double>(i)) * voxel_size,
                    cube_lo + (0.5 + static_cast<double>(j)) * voxel_size,
                    cube_lo + (0.5 + static_cast<double>(k)) * voxel_size
                };
                record.original_id = id++;
                voxels.voxels.push_back(record);
            }
        }
    }

    return voxels;
}

}  // namespace

TEST_CASE(test_smoke_run_returns_corner_fill_fractions) {
    using namespace isthmus;
    DomainConfig domain;
    domain.dimension = Dimension::D2;
    domain.limits = {{{-5.0, -5.0, 0.0}, {5.0, 5.0, 0.0}}};
    domain.cell_counts = {{10, 10, 1}};
    domain.voxel_size = 2.0 / 3.0;
    domain.weighting = true;

    /*
     * Case:
     * Run the 2D marching-windows pipeline on a simple centered voxel square
     * using the standard weighted corner-fill configuration.
     *
     * Sketch:
     *   +-----------+
     *   |           |
     *   |  ######   |
     *   |  ######   |
     *   |  ######   |
     *   |           |
     *   +-----------+
     *   grid domain with a centered filled square
     *
     * Expected outcome:
     * The solver should complete successfully, populate the full corner grid
     * for the 10x10 cell domain, and mark at least some voxels as belonging to
     * the reconstructed surface band.
     */
    MarchingWindows mw;
    const auto result = mw.run(domain, make_voxel_square(domain.voxel_size));
    CHECK(result.corner_fill_fractions.size() == 121);
    CHECK(!result.surface_voxels.empty());
}

TEST_CASE(test_2d_corner_fill_profile_matches_expected_centerline_profile) {
    using namespace isthmus;
    DomainConfig domain;
    domain.dimension = Dimension::D2;
    domain.limits = {{{-5.0, -5.0, 0.0}, {5.0, 5.0, 0.0}}};
    domain.cell_counts = {{10, 10, 1}};
    domain.voxel_size = 2.0 / 3.0;
    domain.weighting = true;

    /*
     * Case:
     * Run the centered 2D voxel-square fixture and sample the horizontal
     * centerline of the computed corner-fill field.
     *
     * Sketch:
     *   boundary -> center
     *   0, 1/12, 7/18, 13/18, 140/144, 1
     *   o----o----o----o----o----o
     *
     *   The profile rises step by step as the sample moves from empty outer
     *   corners into the fully covered center of the square.
     *
     * Expected outcome:
     * The fill fractions should match the known stepped profile from the
     * verification fixture, progressing from empty corners at the boundary to
     * fully filled corners at the center with the expected intermediate values.
     */
    MarchingWindows mw;
    const auto result = mw.run(domain, make_voxel_square(domain.voxel_size));

    const std::array<double, 6> expected{{0.0, 1.0 / 12.0, 7.0 / 18.0, 13.0 / 18.0, 140.0 / 144.0, 1.0}};
    auto corner_at = [&](int i, int j) {
        return result.corner_fill_fractions[static_cast<std::size_t>(j) * result.corner_dims[0] + static_cast<std::size_t>(i)];
    };

    CHECK_CLOSE(corner_at(0, 5), expected[0], 1e-9);
    CHECK_CLOSE(corner_at(1, 5), expected[1], 1e-9);
    CHECK_CLOSE(corner_at(2, 5), expected[2], 1e-9);
    CHECK_CLOSE(corner_at(3, 5), expected[3], 1e-9);
    CHECK_CLOSE(corner_at(4, 5), expected[4], 1e-9);
    CHECK_CLOSE(corner_at(5, 5), expected[5], 1e-9);
}

TEST_CASE(test_3d_corner_fill_profile_matches_expected_centerline_profile) {
    using namespace isthmus;

    /*
     * Case:
     * Run the weighted 3D voxel-cube fixture in a 10x10x10 marching grid and
     * sample the centerline of corner-fill fractions along the x-axis while
     * keeping y = 0 and z = 0.
     *
     * Sketch:
     *   boundary -> center
     *   0, 1/12, 7/18, 13/18, 140/144, 1
     *   o----o----o----o----o----o
     *
     *   The 3D profile should rise in the same stepped pattern as the weighted
     *   centered solid grows from empty exterior corners to the fully covered
     *   center of the cube.
     *
     * Expected outcome:
     * The sampled centerline should match the known six-value progression from
     * the outer boundary to the center of the cube.
     */
    DomainConfig domain;
    domain.dimension = Dimension::D3;
    domain.limits = {{{-5.0, -5.0, -5.0}, {5.0, 5.0, 5.0}}};
    domain.cell_counts = {{10, 10, 10}};
    domain.voxel_size = 2.0 / 3.0;
    domain.weighting = true;

    MarchingWindows mw;
    const auto result = mw.run(domain, make_voxel_cube(8.0 * domain.voxel_size, 8));

    const std::array<double, 6> expected{{0.0, 1.0 / 12.0, 7.0 / 18.0, 13.0 / 18.0, 140.0 / 144.0, 1.0}};

    /*
     * Corner data is flattened with x as the fastest-varying axis, then y,
     * then z. Fixing y = 5 and z = 5 selects the centerline through the cube.
     */
    auto corner_at = [&](int i, int j, int k) {
        const auto nx = result.corner_dims[0];
        const auto ny = result.corner_dims[1];
        return result.corner_fill_fractions[
            static_cast<std::size_t>(k) * nx * ny +
            static_cast<std::size_t>(j) * nx +
            static_cast<std::size_t>(i)];
    };

    CHECK_CLOSE(corner_at(0, 5, 5), expected[0], 1e-9);
    CHECK_CLOSE(corner_at(1, 5, 5), expected[1], 1e-9);
    CHECK_CLOSE(corner_at(2, 5, 5), expected[2], 1e-9);
    CHECK_CLOSE(corner_at(3, 5, 5), expected[3], 1e-9);
    CHECK_CLOSE(corner_at(4, 5, 5), expected[4], 1e-9);
    CHECK_CLOSE(corner_at(5, 5, 5), expected[5], 1e-9);
}

TEST_CASE(test_3d_surface_request_returns_non_empty_mesh_in_domain_bounds) {
    using namespace isthmus;

    /*
     * Case:
     * Run 3D surface extraction for a tiny 2x2x2 voxel cube centered inside a
     * 4x4x4 marching grid.
     *
     * Sketch:
     *        outer marching grid
     *     +---------------------+
     *     |        _____        |
     *     |       /____/|       |
     *     |       |####|/       |
     *     |                     |
     *     +---------------------+
     *     small filled cube centered in a larger domain
     *
     * Expected outcome:
     * Native 3D surface extraction should produce a non-empty triangle mesh
     * whose triangle indices are valid, whose triangles are non-degenerate, and
     * whose vertices remain inside the configured domain bounds.
     */
    DomainConfig domain;
    domain.dimension = Dimension::D3;
    domain.limits = {{{-2.0e-6, -2.0e-6, -2.0e-6}, {2.0e-6, 2.0e-6, 2.0e-6}}};
    domain.cell_counts = {{4, 4, 4}};
    domain.weighting = false;

    const double marching_grid_length = 4.0e-6;
    const double cube_side_length = std::cbrt(0.75) * (marching_grid_length / 4.0);
    domain.voxel_size = cube_side_length / 2.0;

    RunOptions options;
    options.build_surface = true;

    MarchingWindows mw;
    const auto result = mw.run(domain, make_voxel_cube(cube_side_length, 2), options);

    CHECK(!result.surface_mesh.vertices.empty());
    CHECK(!result.surface_mesh.triangles.empty());
    CHECK(result.surface_mesh.segments.empty());

    for (const auto& tri : result.surface_mesh.triangles) {
        CHECK(tri[0] < result.surface_mesh.vertices.size());
        CHECK(tri[1] < result.surface_mesh.vertices.size());
        CHECK(tri[2] < result.surface_mesh.vertices.size());
        CHECK(tri[0] != tri[1]);
        CHECK(tri[1] != tri[2]);
        CHECK(tri[0] != tri[2]);
    }

    for (const auto& vertex : result.surface_mesh.vertices) {
        CHECK(vertex[0] >= domain.limits[0][0] - 1e-12);
        CHECK(vertex[0] <= domain.limits[1][0] + 1e-12);
        CHECK(vertex[1] >= domain.limits[0][1] - 1e-12);
        CHECK(vertex[1] <= domain.limits[1][1] + 1e-12);
        CHECK(vertex[2] >= domain.limits[0][2] - 1e-12);
        CHECK(vertex[2] <= domain.limits[1][2] + 1e-12);
    }
}

TEST_CASE(test_3d_surface_request_matches_expected_diamond_triangles) {
    using namespace isthmus;

    /*
     * Case:
     * Run 3D surface extraction for the centered 2x2x2 voxel-cube fixture and
     * compare the returned mesh against the exact expected diamond triangles.
     *
     * Sketch:
     *           z
     *           |
     *           *
     *         * | *
     *       *---+---*  y
     *         * | *
     *           *
     *          /
     *         x
     *
     *   Each octant should contain exactly one triangle built from the three
     *   axis intercepts that bound that octant.
     *
     * Expected outcome:
     * The extractor should return exactly the eight analytical triangles of
     * the symmetric diamond surface, regardless of triangle winding order.
     */
    DomainConfig domain;
    domain.dimension = Dimension::D3;
    domain.limits = {{{-2.0e-6, -2.0e-6, -2.0e-6}, {2.0e-6, 2.0e-6, 2.0e-6}}};
    domain.cell_counts = {{4, 4, 4}};
    domain.weighting = false;

    const double marching_grid_length = 4.0e-6;
    const double cube_side_length = std::cbrt(0.75) * (marching_grid_length / 4.0);
    domain.voxel_size = cube_side_length / 2.0;

    RunOptions options;
    options.build_surface = true;

    MarchingWindows mw;
    const auto result = mw.run(domain, make_voxel_cube(cube_side_length, 2), options);

    const double lc = (marching_grid_length / 4.0) / 3.0;
    const std::array<std::array<std::array<double, 3>, 3>, 8> expected_triangles{{
        {{{{-lc, 0.0, 0.0}}, {{0.0, -lc, 0.0}}, {{0.0, 0.0, -lc}}}},
        {{{{-lc, 0.0, 0.0}}, {{0.0, -lc, 0.0}}, {{0.0, 0.0, lc}}}},
        {{{{-lc, 0.0, 0.0}}, {{0.0, lc, 0.0}}, {{0.0, 0.0, -lc}}}},
        {{{{-lc, 0.0, 0.0}}, {{0.0, lc, 0.0}}, {{0.0, 0.0, lc}}}},
        {{{{lc, 0.0, 0.0}}, {{0.0, -lc, 0.0}}, {{0.0, 0.0, -lc}}}},
        {{{{lc, 0.0, 0.0}}, {{0.0, -lc, 0.0}}, {{0.0, 0.0, lc}}}},
        {{{{lc, 0.0, 0.0}}, {{0.0, lc, 0.0}}, {{0.0, 0.0, -lc}}}},
        {{{{lc, 0.0, 0.0}}, {{0.0, lc, 0.0}}, {{0.0, 0.0, lc}}}}
    }};

    CHECK(result.surface_mesh.triangles.size() == expected_triangles.size());

    /*
     * Convert the indexed mesh into explicit triangle coordinates so each
     * output triangle can be matched against one of the analytical targets.
     */
    std::vector<std::array<std::array<double, 3>, 3>> actual_triangles;
    actual_triangles.reserve(result.surface_mesh.triangles.size());
    for (const auto& tri : result.surface_mesh.triangles) {
        actual_triangles.push_back({{
            result.surface_mesh.vertices[tri[0]],
            result.surface_mesh.vertices[tri[1]],
            result.surface_mesh.vertices[tri[2]]
        }});
    }

    std::array<bool, 8> matched{{false, false, false, false, false, false, false, false}};
    for (const auto& expected_triangle : expected_triangles) {
        bool found = false;
        for (std::size_t i = 0; i < actual_triangles.size(); ++i) {
            if (!matched[i] && triangle_matches_unordered(expected_triangle, actual_triangles[i], 1e-12)) {
                matched[i] = true;
                found = true;
                break;
            }
        }
        CHECK(found);
    }
}

TEST_CASE(test_internal_lewiner_backend_handles_ambiguous_case_with_center_vertex) {
    using namespace isthmus;

    /*
     * Case:
     * Feed the native marching-cubes backend a single-cell scalar field that
     * maps to ambiguous cube index 5, where only corners v0 and v2 are above
     * the iso-threshold.
     *
     * Sketch:
     *   top face:     bottom face:
     *   v7 ---- v6    v3 ---- v2*
     *   |        |    |        |
     *   |        |    |        |
     *   v4 ---- v5    v0*--- v1
     *
     *   * = above isovalue
     *   The active corners sit on a diagonal, which creates the ambiguous
     *   saddle configuration.
     *
     * Expected outcome:
     * The Lewiner-style resolver should emit the ambiguous-face tiling with a
     * center vertex, producing exactly six vertices and four triangles instead
     * of the older classic two-triangle split.
     */
    DomainConfig domain;
    domain.dimension = Dimension::D3;
    domain.limits = {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}};
    domain.cell_counts = {{1, 1, 1}};
    domain.voxel_size = 1.0;
    domain.weighting = false;

    const std::vector<double> corner_fill_fractions{
        1.0, 0.0,
        0.0, 1.0,
        0.0, 0.0,
        0.0, 0.0
    };

    const auto mesh = marching_cubes::extract_surface_mesh_3d(
        domain,
        corner_fill_fractions,
        {{2, 2, 2}},
        0.5);

    CHECK(mesh.vertices.size() == 6);
    CHECK(mesh.triangles.size() == 4);
}
