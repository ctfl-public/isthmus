#include "test_framework.hpp"

#include "isthmus/marching_windows.hpp"

namespace {

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

}  // namespace

TEST_CASE(test_smoke_run_returns_corner_fill_fractions) {
    using namespace isthmus;
    DomainConfig domain;
    domain.dimension = Dimension::D2;
    domain.limits = {{{-5.0, -5.0, 0.0}, {5.0, 5.0, 0.0}}};
    domain.cell_counts = {{10, 10, 1}};
    domain.voxel_size = 2.0 / 3.0;
    domain.weighting = true;

    // Protects the basic end-to-end contract that a valid run produces a corner field and surface voxels.
    MarchingWindows mw;
    const auto result = mw.run(domain, make_voxel_square(domain.voxel_size));
    CHECK(result.corner_fill_fractions.size() == 121);
    CHECK(!result.surface_voxels.empty());
}

TEST_CASE(test_2d_corner_fill_profile_matches_python_verification_case) {
    using namespace isthmus;
    DomainConfig domain;
    domain.dimension = Dimension::D2;
    domain.limits = {{{-5.0, -5.0, 0.0}, {5.0, 5.0, 0.0}}};
    domain.cell_counts = {{10, 10, 1}};
    domain.voxel_size = 2.0 / 3.0;
    domain.weighting = true;

    // Protects the weighted corner-fill math against a known verification profile.
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

TEST_CASE(test_surface_request_is_honest_about_unimplemented_backend) {
    using namespace isthmus;
    DomainConfig domain;
    domain.dimension = Dimension::D2;
    domain.limits = {{{-5.0, -5.0, 0.0}, {5.0, 5.0, 0.0}}};
    domain.cell_counts = {{10, 10, 1}};
    domain.voxel_size = 2.0 / 3.0;
    domain.weighting = true;

    RunOptions options;
    options.build_surface = true;

    // Protects the API contract that unavailable stages fail explicitly rather than silently doing partial work.
    MarchingWindows mw;
    bool threw = false;
    try {
        (void)mw.run(domain, make_voxel_square(domain.voxel_size), options);
    } catch (const NotImplementedError&) {
        threw = true;
    }
    CHECK(threw);
}
