#include "isthmus/marching_windows.hpp"

namespace isthmus {

MarchingWindowsResult MarchingWindows::run(
    const DomainConfig& domain,
    const VoxelSet& voxels,
    const RunOptions& options) const {
    return motion_mapper_.run(domain, voxels, options);
}

}  // namespace isthmus
