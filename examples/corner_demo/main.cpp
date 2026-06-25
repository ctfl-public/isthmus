#include <iostream>

#include "isthmus/marching_windows.hpp"

int main() {
    using namespace isthmus;

    /**
     * Configure a simple 2D marching-windows domain that is large enough to
     * hold the synthetic voxelized square built below.
     */
    RunOptions options;
    options.dimension = Dimension::D2;
    options.voxel_size = 2.0 / 3.0;
    options.marching_voxel_ratio = 1.5;
    options.weighting = true;
    options.build_surface = false;
    options.build_flux_association = false;

    /**
     * Build a small synthetic voxelized square and run the implemented
     * motion-mapping stages on it.
     */
    VoxelSet voxels;
    std::size_t original_id = 0;
    for (int j = 0; j < 8; ++j) {
        for (int i = 0; i < 8; ++i) {
            VoxelRecord record;
            record.centroid = {
                (-4.0 * options.voxel_size) + (0.5 + i) * options.voxel_size,
                (-4.0 * options.voxel_size) + (0.5 + j) * options.voxel_size,
                0.0
            };
            record.original_id = original_id++;
            voxels.voxels.push_back(record);
        }
    }

    /**
     * Execute the native pipeline and report a compact summary that confirms
     * the example ran and produced meaningful reconstruction data.
     */
    MarchingWindows mw;
    const auto result = mw.run(voxels, options);
    std::cout << "Computed " << result.corner_fill_fractions.size()
              << " corner fill fractions and found "
              << result.surface_voxels.size() << " surface voxels.\n";
    return 0;
}
