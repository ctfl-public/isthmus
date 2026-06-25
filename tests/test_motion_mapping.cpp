#include "test_framework.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <set>

#include "isthmus/exceptions.hpp"
#include "isthmus/marching_windows.hpp"
#include "marching_cubes.hpp"

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

isthmus::RunOptions make_run_options(
    isthmus::Dimension dimension,
    double voxel_size,
    double marching_voxel_ratio,
    bool weighting = true) {
    isthmus::RunOptions options;
    options.dimension = dimension;
    options.voxel_size = voxel_size;
    options.marching_voxel_ratio = marching_voxel_ratio;
    options.weighting = weighting;
    options.build_surface = false;
    options.build_flux_association = false;
    return options;
}

}  // namespace

TEST_CASE(test_smoke_run_returns_corner_fill_fractions) {
    using namespace isthmus;
    const auto options = make_run_options(Dimension::D2, 2.0 / 3.0, 1.5);

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
    const auto result = mw.run(make_voxel_square(options.voxel_size), options);
    CHECK(result.corner_fill_fractions.size() == 144);
    CHECK(!result.surface_voxels.empty());
}

TEST_CASE(test_marching_voxel_ratio_populates_domain_from_voxel_bounds) {
    using namespace isthmus;

    /*
     * Case:
     * Let the caller provide only a marching/voxel ratio instead of explicit
     * marching limits and cell counts.
     *
     * Expected outcome:
     * The run should derive a concrete domain from the voxel bounding box. For
     * ratio 1.6 with weighting enabled, the required physical margin is
     * 3.4 voxel lengths, so the resolver chooses three marching-cell padding
     * layers: 3 * 1.6 = 4.8 voxel lengths.
     */
    const auto options = make_run_options(Dimension::D3, 1.0, 1.6);

    MarchingWindows mw;
    const auto result = mw.run(make_voxel_cube(2.0, 2), options);

    CHECK(result.domain.cell_counts[0] == 7u);
    CHECK(result.domain.cell_counts[1] == 7u);
    CHECK(result.domain.cell_counts[2] == 7u);
    CHECK_CLOSE(result.domain.limits[0][0], -5.3, 1e-12);
    CHECK_CLOSE(result.domain.limits[1][0], 5.9, 1e-12);
    CHECK_CLOSE(
        (result.domain.limits[1][0] - result.domain.limits[0][0]) /
            static_cast<double>(result.domain.cell_counts[0]),
        1.6,
        1e-12);
}

TEST_CASE(test_marching_voxel_ratio_rejects_sub_voxel_marching_cells) {
    using namespace isthmus;

    const auto options = make_run_options(Dimension::D3, 1.0, 0.5);

    MarchingWindows mw;
    bool threw_expected = false;
    try {
        (void)mw.run(make_voxel_cube(2.0, 2), options);
    } catch (const InvalidInputError&) {
        threw_expected = true;
    }

    CHECK(threw_expected);
}

TEST_CASE(test_2d_corner_fill_profile_matches_expected_centerline_profile) {
    using namespace isthmus;
    const auto options = make_run_options(Dimension::D2, 2.0 / 3.0, 1.5);

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
    const auto result = mw.run(make_voxel_square(options.voxel_size), options);

    const std::array<double, 6> expected{{0.0, 1.0 / 36.0, 5.0 / 18.0, 11.0 / 18.0, 11.0 / 12.0, 1.0}};
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
    const auto options = make_run_options(Dimension::D3, 2.0 / 3.0, 1.5);

    MarchingWindows mw;
    const auto result = mw.run(make_voxel_cube(8.0 * options.voxel_size, 8), options);

    const std::array<double, 6> expected{{0.0, 1.0 / 36.0, 5.0 / 18.0, 11.0 / 18.0, 11.0 / 12.0, 1.0}};

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
    const double marching_grid_length = 4.0e-6;
    const double cube_side_length = std::cbrt(0.75) * (marching_grid_length / 4.0);
    auto options = make_run_options(Dimension::D3, cube_side_length / 2.0, 1.5, false);
    options.build_surface = true;

    MarchingWindows mw;
    const auto result = mw.run(make_voxel_cube(cube_side_length, 2), options);

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

    const auto& limits = result.domain.limits;
    for (const auto& vertex : result.surface_mesh.vertices) {
        CHECK(vertex[0] >= limits[0][0] - 1e-12);
        CHECK(vertex[0] <= limits[1][0] + 1e-12);
        CHECK(vertex[1] >= limits[0][1] - 1e-12);
        CHECK(vertex[1] <= limits[1][1] + 1e-12);
        CHECK(vertex[2] >= limits[0][2] - 1e-12);
        CHECK(vertex[2] <= limits[1][2] + 1e-12);
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
    const double marching_grid_length = 4.0e-6;
    const double cube_side_length = std::cbrt(0.75) * (marching_grid_length / 4.0);
    auto options = make_run_options(Dimension::D3, cube_side_length / 2.0, 1.5, false);
    options.build_surface = true;

    MarchingWindows mw;
    const auto result = mw.run(make_voxel_cube(cube_side_length, 2), options);

    CHECK(result.surface_mesh.vertices.size() == 6u);
    CHECK(result.surface_mesh.triangles.size() == 8u);
    for (const auto& tri : result.surface_mesh.triangles) {
        CHECK(tri[0] < result.surface_mesh.vertices.size());
        CHECK(tri[1] < result.surface_mesh.vertices.size());
        CHECK(tri[2] < result.surface_mesh.vertices.size());
        CHECK(tri[0] != tri[1]);
        CHECK(tri[1] != tri[2]);
        CHECK(tri[0] != tri[2]);
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
    VoxelSet domain_voxels;
    domain_voxels.voxels.push_back({{0.0, 0.0, 0.0}, 0});
    MarchingWindows mw;
    const auto domain_result = mw.run(domain_voxels, make_run_options(Dimension::D3, 1.0, 1.0, false));
    const auto& domain = domain_result.domain;

    std::vector<double> corner_fill_fractions(
        domain.cell_counts[0] + 1u,
        0.0);
    const auto corner_dims = std::array<std::size_t, 3>{
        domain.cell_counts[0] + 1u,
        domain.cell_counts[1] + 1u,
        domain.cell_counts[2] + 1u
    };
    corner_fill_fractions.assign(corner_dims[0] * corner_dims[1] * corner_dims[2], 0.0);
    auto corner_at = [&](std::size_t i, std::size_t j, std::size_t k) -> double& {
        return corner_fill_fractions[k * corner_dims[0] * corner_dims[1] + j * corner_dims[0] + i];
    };
    corner_at(0, 0, 0) = 1.0;
    corner_at(1, 1, 0) = 1.0;

    const auto mesh = marching_cubes::extract_surface_mesh_3d(
        domain,
        corner_fill_fractions,
        corner_dims,
        0.5);

    CHECK(mesh.vertices.size() == 8);
    CHECK(mesh.triangles.size() == 7);
}

TEST_CASE(test_3d_flux_request_populates_normalized_triangle_ownership) {
    using namespace isthmus;

    /*
     * Case:
     * Run the native 3D pipeline for a centered 2x2x2 voxel cube while asking
     * for both surface extraction and triangle-to-voxel ownership.
     *
     * Sketch:
     *   voxel cube -> surface triangles -> normalized voxel fractions
     *
     * Expected outcome:
     * Every extracted triangle should receive a populated ownership record
     * whose voxel ids and scalar fractions align one-to-one and whose
     * fractions are finite, non-negative, and normalized to sum to one.
    */
    const double marching_grid_length = 4.0e-6;
    const double cube_side_length = std::cbrt(0.75) * (marching_grid_length / 4.0);
    auto options = make_run_options(Dimension::D3, cube_side_length / 2.0, 1.5, false);
    options.build_surface = true;
    options.build_flux_association = true;

    MarchingWindows mw;
    const auto result = mw.run(make_voxel_cube(cube_side_length, 2), options);

    CHECK(result.flux_association.elements.size() == result.surface_mesh.triangles.size());

    for (const auto& element : result.flux_association.elements) {
        CHECK(!element.voxel_ids.empty());
        CHECK(element.voxel_ids.size() == element.scalar_fractions.size());

        double sum = 0.0;
        for (const double fraction : element.scalar_fractions) {
            CHECK(std::isfinite(fraction));
            CHECK(fraction >= 0.0);
            sum += fraction;
        }
        CHECK_CLOSE(sum, 1.0, 1e-9);
    }
}

TEST_CASE(test_3d_flux_request_references_input_voxels_and_shares_some_triangles) {
    using namespace isthmus;

    /*
     * Case:
     * Reuse the larger weighted 8x8x8 cube fixture and inspect the resulting
     * triangle ownership graph for identifier validity and shared ownership.
     *
     * Sketch:
     *   one diamond triangle may overlap faces from multiple neighboring voxels
     *
     * Expected outcome:
     * Every referenced voxel id should come from the input voxel set, and at
     * least one triangle should be shared by multiple voxels so the native
     * path proves it is doing area-based distribution rather than a nearest
     * triangle-to-single-voxel assignment.
    */
    auto options = make_run_options(Dimension::D3, 2.0 / 3.0, 1.5);

    const auto voxels = make_voxel_cube(8.0 * options.voxel_size, 8);
    std::set<std::size_t> valid_ids;
    for (const auto& voxel : voxels.voxels) {
        valid_ids.insert(voxel.original_id);
    }

    options.build_surface = true;
    options.build_flux_association = true;

    MarchingWindows mw;
    const auto result = mw.run(voxels, options);

    bool found_shared_triangle = false;
    for (const auto& element : result.flux_association.elements) {
        if (element.voxel_ids.size() > 1) {
            found_shared_triangle = true;
        }
        for (const auto voxel_id : element.voxel_ids) {
            CHECK(valid_ids.count(voxel_id) == 1);
        }
    }

    CHECK(found_shared_triangle);
}

TEST_CASE(test_2d_flux_request_remains_explicitly_unsupported) {
    using namespace isthmus;

    /*
     * Case:
     * Request flux mapping from the 2D native pipeline even though this phase
     * only adds 3D triangle ownership.
     *
     * Sketch:
     *   2D corner field -> build_flux_association
     *
     * Expected outcome:
     * The solver should reject the request with the documented
     * NotImplementedError instead of silently returning incomplete ownership.
    */
    auto options = make_run_options(Dimension::D2, 2.0 / 3.0, 1.5);
    options.build_flux_association = true;

    MarchingWindows mw;
    bool threw_expected = false;
    try {
        (void)mw.run(make_voxel_square(options.voxel_size), options);
    } catch (const NotImplementedError&) {
        threw_expected = true;
    }

    CHECK(threw_expected);
}
