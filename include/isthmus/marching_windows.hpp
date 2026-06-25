#pragma once

#include "isthmus/motion_mapping.hpp"
#include "isthmus/types.hpp"

namespace isthmus {

/*
 * Public entry point for the library.
 *
 * A caller provides physical marching settings through RunOptions and a set of
 * occupied voxel centroids. The class derives the concrete marching domain,
 * executes the implemented algorithm stages, and returns all results in memory
 * so downstream codes can consume them without going through intermediate
 * files.
 */
class MarchingWindows {
public:
    // Execute one marching-windows pass and return all populated result data.
    MarchingWindowsResult run(
        const VoxelSet& voxel_set,
        const RunOptions& run_options) const;

private:
    MotionMapper motion_mapper_;
};

}  // namespace isthmus
