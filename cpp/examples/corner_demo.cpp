#include <iostream>

#include "isthmus/marching_windows.hpp"

int main() {
    using namespace isthmus;

    DomainConfig domain;
    domain.dimension = Dimension::D2;
    domain.limits = {{{-5.0, -5.0, 0.0}, {5.0, 5.0, 0.0}}};
    domain.cell_counts = {{10, 10, 1}};
    domain.voxel_size = 2.0 / 3.0;
    domain.weighting = true;

    // Build a small synthetic voxelized square and run the implemented motion-mapping stages on it.
    VoxelSet voxels;
    std::size_t original_id = 0;
    for (int j = 0; j < 8; ++j) {
        for (int i = 0; i < 8; ++i) {
            VoxelRecord record;
            record.centroid = {
                (-4.0 * domain.voxel_size) + (0.5 + i) * domain.voxel_size,
                (-4.0 * domain.voxel_size) + (0.5 + j) * domain.voxel_size,
                0.0
            };
            record.original_id = original_id++;
            voxels.voxels.push_back(record);
        }
    }

    MarchingWindows mw;
    const auto result = mw.run(domain, voxels);
    std::cout << "Computed " << result.corner_fill_fractions.size()
              << " corner fill fractions and found "
              << result.surface_voxels.size() << " surface voxels.\n";
    return 0;
}
