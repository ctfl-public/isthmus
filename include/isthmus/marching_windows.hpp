#pragma once

#include "isthmus/motion_mapping.hpp"
#include "isthmus/types.hpp"

namespace isthmus {

/*
 * Public entry point for the library.
 *
 * A caller provides the marching-windows domain and a set of occupied voxel
 * centroids. The class then executes the implemented algorithm stages and
 * returns all results in memory so downstream codes can consume them
 * without going through intermediate files.
 */
class MarchingWindows {
public:
    // Execute one marching-windows pass and return all populated result data.
    MarchingWindowsResult run(
        const DomainConfig& domain_config,
        const VoxelSet& voxel_set,
        const RunOptions& run_options = {}) const;

private:
    MotionMapper motion_mapper_;
};

}  // namespace isthmus
