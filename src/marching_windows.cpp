#include "isthmus/marching_windows.hpp"

namespace isthmus {

MarchingWindowsResult MarchingWindows::run(
    const DomainConfig& domain_config,
    const VoxelSet& voxel_set,
    const RunOptions& run_options) const {
    return motion_mapper_.run(domain_config, voxel_set, run_options);
}

}  // namespace isthmus
