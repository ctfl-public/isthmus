#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include "isthmus/exceptions.hpp"
#include "isthmus/types.hpp"

namespace isthmus {

// One quadrilateral face of a 3D voxel together with its outward normal.
struct VoxelFace3D {
    std::array<std::array<double, 3>, 4> corners{};
    std::array<double, 3> normal{{0.0, 0.0, 0.0}};
    bool exposed = false;
};

// One edge of a 2D voxel together with its outward normal.
struct VoxelFace2D {
    std::array<std::array<double, 2>, 2> corners{};
    std::array<double, 2> normal{{0.0, 0.0}};
    bool exposed = false;
};

/*
 * Internal voxel record used after the sparse user input has been expanded onto
 * the regular lattice that marching windows operates on.
 *
 * `type` stores boundary depth: surface voxels start at 0, solid interior
 * layers count upward, and ghost voxels in the surrounding void count downward.
 */
struct VoxelCell {
    std::array<double, kMaxDims> centroid{{0.0, 0.0, 0.0}};
    std::array<int, kMaxDims> indices{{0, 0, 0}};
    std::size_t flattened_index = 0;
    std::size_t original_id = static_cast<std::size_t>(-1);
    int type = -1;
    bool finalized = false;
    bool surface = false;
    double weight = 0.0;
    std::vector<VoxelFace3D> faces3d;
    std::vector<VoxelFace2D> faces2d;
};

/*
 * Internal corner record used to build the scalar field that later marching
 * cubes or marching squares will sample.
 */
struct CornerData {
    std::array<double, kMaxDims> position{{0.0, 0.0, 0.0}};
    std::array<int, kMaxDims> indices{{0, 0, 0}};
    double volume = 0.0;
    int inside = -1;
    std::vector<std::size_t> owned_voxel_indices;
};

/*
 * Shared regular-grid indexing helper.
 *
 * The code stores structured data in flat vectors for cache-friendly traversal,
 * so this helper centralizes the conversion between integer lattice indices and
 * their flattened x-fastest representation.
 */
template <std::size_t Dims>
class RegularGrid {
public:
    using IndexArray = std::array<int, Dims>;
    using SizeArray = std::array<std::size_t, Dims>;

    explicit RegularGrid(SizeArray dims) : dims_(dims) {}

    [[nodiscard]] std::size_t element_index(const IndexArray& index) const {
        std::size_t stride = 1;
        std::size_t flat = 0;
        for (std::size_t d = 0; d < Dims; ++d) {
            if (index[d] < 0 || static_cast<std::size_t>(index[d]) >= dims_[d]) {
                throw InvalidInputError("Grid index out of bounds");
            }
            flat += static_cast<std::size_t>(index[d]) * stride;
            stride *= dims_[d];
        }
        return flat;
    }

    [[nodiscard]] IndexArray indices(std::size_t flat) const {
        IndexArray out{};
        for (std::size_t d = 0; d < Dims; ++d) {
            out[d] = static_cast<int>(flat % dims_[d]);
            flat /= dims_[d];
        }
        return out;
    }

    [[nodiscard]] bool valid(const IndexArray& index) const {
        for (std::size_t d = 0; d < Dims; ++d) {
            if (index[d] < 0 || static_cast<std::size_t>(index[d]) >= dims_[d]) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] const SizeArray& dims() const { return dims_; }

private:
    SizeArray dims_;
};

using RegularGrid2D = RegularGrid<2>;
using RegularGrid3D = RegularGrid<3>;

}  // namespace isthmus
