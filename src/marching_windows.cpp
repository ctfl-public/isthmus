#include "isthmus/marching_windows.hpp"

namespace isthmus {

MarchingWindowsResult MarchingWindows::run(
    const VoxelSet& voxel_set,
    const RunOptions& run_options) const {
    return motion_mapper_.run(voxel_set, run_options);
}

}  // namespace isthmus
