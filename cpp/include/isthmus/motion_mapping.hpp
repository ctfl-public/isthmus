#pragma once

#include <vector>

#include "isthmus/types.hpp"
#include "isthmus/voxels.hpp"

namespace isthmus {

/*
 * Implements the motion-mapping stages that are already available natively.
 *
 * The current implementation stops after generating the corner fill field that
 * later surface extraction will consume. Even so, the code already performs the
 * core marching-windows preparation steps: validation, voxel lattice creation,
 * boundary-depth classification, weighting, exposed-face tagging, and corner
 * volume splitting.
 */
class MotionMapper {
public:
    MarchingWindowsResult run(
        const DomainConfig& domain,
        const VoxelSet& voxels,
        const RunOptions& options) const;

private:
    static void validate_inputs(const DomainConfig& domain, const VoxelSet& voxels);
    static std::vector<VoxelCell> build_voxel_grid(const DomainConfig& domain, const VoxelSet& voxels);
    static void classify_voxels(const DomainConfig& domain, std::vector<VoxelCell>& grid);
    static void assign_exposed_faces(const DomainConfig& domain, std::vector<VoxelCell>& grid);
    static std::vector<CornerData> build_corner_grid(
        const DomainConfig& domain,
        const std::vector<VoxelCell>& voxels,
        std::array<std::size_t, kMaxDims>& out_dims);
    static std::vector<SurfaceVoxelInfo> collect_surface_voxels(const std::vector<VoxelCell>& voxels);
};

}  // namespace isthmus
