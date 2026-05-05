#include <cmath>
#include <filesystem>
#include <iostream>

#include "isthmus/io.hpp"
#include "isthmus/marching_windows.hpp"

namespace {

/**
 * Build the same compact synthetic voxel cube used by the native surface
 * tests "test_3d_surface_request_matches_expected_diamond_triangles".
 */
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

int main(int argc, char** argv) {
    using namespace isthmus;

    /**
     * Mirror the existing 3D verification setup so the demo produces a tiny
     * known-good surface without needing any external voxel input files.
     */
    DomainConfig domain;
    domain.dimension = Dimension::D3;
    domain.limits = {{{-2.0e-6, -2.0e-6, -2.0e-6}, {2.0e-6, 2.0e-6, 2.0e-6}}};
    domain.cell_counts = {{4, 4, 4}};
    domain.weighting = false;

    /**
     * This is the same voxel size derivation used by the regression tests, so
     * the example surface remains aligned with the native reference case.
     */
    const double marching_grid_length = 4.0e-6;
    const double cube_side_length = std::cbrt(0.75) * (marching_grid_length / 4.0);
    domain.voxel_size = cube_side_length / 2.0;

    RunOptions options;
    options.build_surface = true;

    /**
     * Write into a user-supplied directory when one is passed on the command
     * line, otherwise keep the historical default output folder name.
     */
    const std::filesystem::path output_dir = argc > 1
        ? std::filesystem::path(argv[1])
        : std::filesystem::path("surface_export_demo_output");
    std::filesystem::create_directories(output_dir);

    /**
     * Run the native reconstruction on the synthetic cube and prepare both
     * export file paths before writing the mesh to disk.
     */
    MarchingWindows mw;
    const auto result = mw.run(domain, make_voxel_cube(cube_side_length, 2), options);

    const std::filesystem::path surf_path = output_dir / "surface_cube.surf";
    const std::filesystem::path vtp_path = output_dir / "surface_cube.vtp";

    /**
     * Write both formats on every run so users can compare the legacy solver
     * surface file with the ParaView-native visualization file immediately.
     */
    io::write_sparta_surface(result.surface_mesh, domain.dimension, surf_path);
    io::write_vtp_surface(result.surface_mesh, vtp_path);

    std::cout << "Exported " << result.surface_mesh.vertices.size()
              << " vertices and " << result.surface_mesh.triangles.size()
              << " triangles.\n";
    std::cout << "SPARTA surface: " << std::filesystem::absolute(surf_path) << '\n';
    std::cout << "ParaView VTP: " << std::filesystem::absolute(vtp_path) << '\n';
    return 0;
}
