/*
 * Motion-mapping stages that are already implemented natively.
 *
 * This file covers the preparation work that marching windows performs before
 * surface extraction: validating the domain, expanding sparse occupied voxels
 * into the regular voxel lattice, classifying boundary depth, assigning voxel
 * weights, identifying exposed faces, and building the corner fill field.
 */
#include "isthmus/motion_mapping.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "isthmus/exceptions.hpp"
#include "marching_cubes.hpp"

namespace isthmus {

namespace {

// Convert the public dimension enum into a simple loop bound for shared helpers.
std::size_t active_dims(Dimension d) {
    return static_cast<std::size_t>(d);
}

// Compute the physical size of one marching-window cell in each active direction.
std::array<double, kMaxDims> cell_lengths(const DomainConfig& domain) {
    std::array<double, kMaxDims> out{{0.0, 0.0, 0.0}};
    const auto dims = active_dims(domain.dimension);
    for (std::size_t i = 0; i < dims; ++i) {
        out[i] = (domain.limits[1][i] - domain.limits[0][i]) /
                 static_cast<double>(domain.cell_counts[i]);
    }
    return out;
}

double max_component(const std::array<double, kMaxDims>& values, std::size_t dims) {
    double out = values[0];
    for (std::size_t i = 1; i < dims; ++i) {
        out = std::max(out, values[i]);
    }
    return out;
}

// Convert a physical centroid position into integer voxel-lattice coordinates.
std::array<int, kMaxDims> to_index(
    const std::array<double, kMaxDims>& point,
    const std::array<double, kMaxDims>& low,
    double voxel_size,
    std::size_t dims) {
    std::array<int, kMaxDims> out{{0, 0, 0}};
    for (std::size_t i = 0; i < dims; ++i) {
        out[i] = static_cast<int>(std::llround((point[i] - low[i]) / voxel_size));
    }
    return out;
}

/*
 * Flatten structured-grid indices with x as the fastest-varying direction.
 *
 * Keeping this ordering explicit matters because the same layout is used for
 * voxel lattices, corner lattices, and the parity tests that compare against
 * established marching-windows behavior.
 */
std::size_t flatten_index(
    const std::array<int, kMaxDims>& index,
    const std::array<std::size_t, kMaxDims>& dims,
    std::size_t ndims) {
    std::size_t stride = 1;
    std::size_t flat = 0;
    for (std::size_t i = 0; i < ndims; ++i) {
        flat += static_cast<std::size_t>(index[i]) * stride;
        stride *= dims[i];
    }
    return flat;
}

bool valid_index(
    const std::array<int, kMaxDims>& index,
    const std::array<std::size_t, kMaxDims>& dims,
    std::size_t ndims) {
    for (std::size_t i = 0; i < ndims; ++i) {
        if (index[i] < 0 || static_cast<std::size_t>(index[i]) >= dims[i]) {
            return false;
        }
    }
    return true;
}

/*
 * Generate the face-sharing neighbor offsets used by marching windows.
 *
 * Depth classification only allows movement through faces, not through edges or
 * corners. That is what makes depth correspond to the number of cardinal steps
 * from a voxel to the outer boundary.
 */
std::vector<std::array<int, kMaxDims>> cardinal_neighbors(std::size_t dims) {
    std::vector<std::array<int, kMaxDims>> offsets;
    for (std::size_t d = 0; d < dims; ++d) {
        std::array<int, kMaxDims> neg{{0, 0, 0}};
        std::array<int, kMaxDims> pos{{0, 0, 0}};
        neg[d] = -1;
        pos[d] = 1;
        offsets.push_back(neg);
        offsets.push_back(pos);
    }
    return offsets;
}

/*
 * Build the six axis-aligned faces of a voxel once that voxel has been marked
 * as part of the outer surface. Those faces become the geometric support for
 * later visibility checks and flux association.
 */
void add_3d_faces(VoxelCell& voxel, double voxel_size) {
    const auto lo = std::array<double, 3>{
        voxel.centroid[0] - 0.5 * voxel_size,
        voxel.centroid[1] - 0.5 * voxel_size,
        voxel.centroid[2] - 0.5 * voxel_size
    };
    const std::array<std::array<double, 3>, 8> cs{{
        {lo[0], lo[1], lo[2]},
        {lo[0] + voxel_size, lo[1], lo[2]},
        {lo[0], lo[1] + voxel_size, lo[2]},
        {lo[0] + voxel_size, lo[1] + voxel_size, lo[2]},
        {lo[0], lo[1], lo[2] + voxel_size},
        {lo[0] + voxel_size, lo[1], lo[2] + voxel_size},
        {lo[0], lo[1] + voxel_size, lo[2] + voxel_size},
        {lo[0] + voxel_size, lo[1] + voxel_size, lo[2] + voxel_size}
    }};

    voxel.faces3d = {
        {{{cs[2], cs[0], cs[4], cs[6]}}, {{-1.0, 0.0, 0.0}}, false},
        {{{cs[1], cs[3], cs[7], cs[5]}}, {{1.0, 0.0, 0.0}}, false},
        {{{cs[0], cs[1], cs[5], cs[4]}}, {{0.0, -1.0, 0.0}}, false},
        {{{cs[3], cs[2], cs[6], cs[7]}}, {{0.0, 1.0, 0.0}}, false},
        {{{cs[2], cs[3], cs[1], cs[0]}}, {{0.0, 0.0, -1.0}}, false},
        {{{cs[4], cs[5], cs[7], cs[6]}}, {{0.0, 0.0, 1.0}}, false}
    };
}

// 2D uses edges instead of quadrilateral faces, but they serve the same role in the algorithm.
void add_2d_faces(VoxelCell& voxel, double voxel_size) {
    const auto lo = std::array<double, 2>{
        voxel.centroid[0] - 0.5 * voxel_size,
        voxel.centroid[1] - 0.5 * voxel_size
    };
    const std::array<std::array<double, 2>, 4> cs{{
        {lo[0], lo[1]},
        {lo[0] + voxel_size, lo[1]},
        {lo[0], lo[1] + voxel_size},
        {lo[0] + voxel_size, lo[1] + voxel_size}
    }};

    voxel.faces2d = {
        {{{cs[2], cs[0]}}, {{-1.0, 0.0}}, false},
        {{{cs[1], cs[3]}}, {{1.0, 0.0}}, false},
        {{{cs[0], cs[1]}}, {{0.0, -1.0}}, false},
        {{{cs[3], cs[2]}}, {{0.0, 1.0}}, false}
    };
}

}  // namespace

/*
 * Validate that the marching grid is physically meaningful for the supplied
 * voxel set.
 *
 * The grid must have positive extent, positive cell counts, and cells at least
 * as large as the solid voxels. The voxel set must also sit far enough inside
 * the marching domain that the weighting and ghost-voxel logic have room to
 * build the auxiliary lattice around it.
 */
void MotionMapper::validate_inputs(const DomainConfig& domain, const VoxelSet& voxels) {
    const auto dims = active_dims(domain.dimension);
    if (dims != 2 && dims != 3) {
        throw InvalidInputError("Dimension must be 2 or 3");
    }
    if (voxels.voxels.empty()) {
        throw InvalidInputError("Voxel set must not be empty");
    }
    if (domain.voxel_size <= 0.0) {
        throw InvalidInputError("Voxel size must be positive");
    }

    const auto cl = cell_lengths(domain);
    for (std::size_t i = 0; i < dims; ++i) {
        if (domain.limits[1][i] <= domain.limits[0][i]) {
            throw InvalidInputError("Invalid grid limits");
        }
        if (domain.cell_counts[i] == 0) {
            throw InvalidInputError("Cell counts must be positive");
        }
        if (cl[i] < domain.voxel_size) {
            throw InvalidInputError("Voxel size is larger than marching-window grid cell dimension");
        }
    }

    std::array<double, kMaxDims> min_vox = voxels.voxels.front().centroid;
    std::array<double, kMaxDims> max_vox = voxels.voxels.front().centroid;
    for (const auto& voxel : voxels.voxels) {
        for (std::size_t i = 0; i < dims; ++i) {
            min_vox[i] = std::min(min_vox[i], voxel.centroid[i]);
            max_vox[i] = std::max(max_vox[i], voxel.centroid[i]);
        }
    }

    const double lmax = domain.weighting
        ? 1.5 * max_component(cl, dims) + domain.voxel_size
        : 0.5 * (max_component(cl, dims) + domain.voxel_size);

    for (std::size_t i = 0; i < dims; ++i) {
        const double lo = domain.limits[0][i] + lmax;
        const double hi = domain.limits[1][i] - lmax;
        if (lo >= hi) {
            throw InvalidInputError("Insufficient buffer added to marching windows grid");
        }
        if (min_vox[i] < lo || max_vox[i] > hi) {
            throw InvalidInputError("Insufficient buffer added to marching windows grid for voxel set");
        }
    }
}

/*
 * Expand the sparse occupied-voxel list from the caller's into the regular voxel lattice 
 * used by marching windows.
 *
 * The lattice is intentionally larger than the occupied region. That extra
 * buffer is what allows the algorithm to classify both solid interior layers
 * and ghost layers in the surrounding void before surface extraction happens.
 */
std::vector<VoxelCell> MotionMapper::build_voxel_grid(const DomainConfig& domain, const VoxelSet& voxels) {
    const auto dims = active_dims(domain.dimension);
    const auto cl = cell_lengths(domain); // cell lengths
    const double cv_ratio = max_component(cl, dims) / domain.voxel_size; // ratio of marching cell size to voxel size
    const double buffer = std::ceil((3.0 * cv_ratio / 2.0) + 0.5) * domain.voxel_size; // buffer size

    // First find the bounding box of the occupied voxel set, then expand it by the buffer to get the marching grid bounds.
    std::array<double, kMaxDims> xlo = voxels.voxels.front().centroid; // low bound of voxel lattice 
    std::array<double, kMaxDims> xhi = voxels.voxels.front().centroid; // high bound of voxel lattice
    for (const auto& voxel : voxels.voxels) {
        for (std::size_t i = 0; i < dims; ++i) {
            xlo[i] = std::min(xlo[i], voxel.centroid[i]);
            xhi[i] = std::max(xhi[i], voxel.centroid[i]);
        }
    }
    // Expand bounds once after the reduction is complete.
    for (std::size_t i = 0; i < dims; ++i) {
        xlo[i] -= 2.0 * buffer;
        xhi[i] += 2.0 * buffer;
    }

    /*
     * Calculate the size and origin of the voxel lattice so that the first occupied 
     * voxel is aligned to a lattice point, then build the lattice around it.
     */
    std::array<double, kMaxDims> grid_lo{{0.0, 0.0, 0.0}}; // lower corner of the voxel grid
    std::array<std::size_t, kMaxDims> grid_dims{{1, 1, 1}}; // number of voxels in the voxel grid along each dimension
    const auto first = voxels.voxels.front().centroid; // centroid of the first occupied voxel
    for (std::size_t i = 0; i < dims; ++i) {
        // number of voxels from the first occupied voxel to the lower bound of the voxel grid
        const double nlo = std::ceil((first[i] - xlo[i]) / domain.voxel_size); 
        grid_lo[i] = first[i] - nlo * domain.voxel_size;
        // number of voxels from the first occupied voxel to the upper bound of the voxel grid
        const double nhi = nlo + std::ceil((xhi[i] - first[i]) / domain.voxel_size); 
        grid_dims[i] = static_cast<std::size_t>(nhi) + 1;
    }

    /* 
     * Build the voxel lattice and initialize it with the input voxels. 
     * The lattice is initialized with all voxels as void candidates (type -1) until we mark the occupied ones as depth 0.
     */
    std::size_t total = 1;
    for (std::size_t i = 0; i < dims; ++i) {
        total *= grid_dims[i];
    }
    std::vector<VoxelCell> grid(total);
    for (std::size_t flat = 0; flat < total; ++flat) {
        VoxelCell voxel{};
        voxel.flattened_index = flat;
        std::size_t tmp = flat;
        for (std::size_t i = 0; i < dims; ++i) {
            const int idx = static_cast<int>(tmp % grid_dims[i]);
            tmp /= grid_dims[i];
            voxel.indices[i] = idx;
            voxel.centroid[i] = grid_lo[i] + static_cast<double>(idx) * domain.voxel_size;
        }
        grid[flat] = voxel;
    }

    // Occupied input voxels start at depth 0. All untouched lattice locations remain void candidates.
    for (std::size_t oid = 0; oid < voxels.voxels.size(); ++oid) {
        const auto index = to_index(voxels.voxels[oid].centroid, grid_lo, domain.voxel_size, dims);
        const auto flat = flatten_index(index, grid_dims, dims);
        auto& voxel = grid[flat];
        voxel.original_id = voxels.voxels[oid].original_id;
        voxel.type = 0;
        voxel.weight = 1.0;
    }

    return grid;
}

/*
 * Assign boundary depth and weights to every voxel in the auxiliary lattice.
 *
 * A voxel `type` of 0 means that voxel still lies on the solid boundary.
 * Positive types count inward through solid material.
 * Negative types count outward through ghost voxels in the surrounding void.
 *
 * Once depths are known, the weighting formula compresses those layers into the
 * [0, 1] range so corner fill fractions vary smoothly across the boundary.
 */
void MotionMapper::classify_voxels(const DomainConfig& domain, std::vector<VoxelCell>& grid) {
    const auto dims = active_dims(domain.dimension);
    const auto cl = cell_lengths(domain);
    const double cv_ratio = max_component(cl, dims) / domain.voxel_size;
    const auto neighbors = cardinal_neighbors(dims);

    std::array<std::size_t, kMaxDims> grid_dims{{1, 1, 1}};
    for (const auto& voxel : grid) {
        for (std::size_t i = 0; i < dims; ++i) {
            grid_dims[i] = std::max(grid_dims[i], static_cast<std::size_t>(voxel.indices[i] + 1));
        }
    }

    /*
     * A solid voxel can move from one layer to the next only if every face-
     * sharing neighbor is at least as deep in the solid. Otherwise it remains
     * part of the outer surface and is finalized at its current depth.
     */
    auto surrounded_solid = [&](std::size_t flat) {
        auto& voxel = grid[flat];
        bool surrounded = true;
        for (const auto& offset : neighbors) {
            auto idx = voxel.indices;
            for (std::size_t d = 0; d < dims; ++d) {
                idx[d] += offset[d];
            }
            if (valid_index(idx, grid_dims, dims)) {
                auto& neighbor = grid[flatten_index(idx, grid_dims, dims)];
                if (neighbor.type < voxel.type) {
                    surrounded = false;
                    break;
                }
            } else {
                surrounded = false;
                break;
            }
        }
        if (surrounded) {
            voxel.type += 1;
        } else {
            voxel.finalized = true;
        }
    };

    /*
     * Ghost voxels follow the same layering idea in the void region. A ghost
     * voxel becomes one layer farther from the surface only if every face-
     * sharing neighbor is already at least that far into the void.
     */
    auto surrounded_void = [&](std::size_t flat) {
        auto& voxel = grid[flat];
        bool surrounded = true;
        for (const auto& offset : neighbors) {
            auto idx = voxel.indices;
            for (std::size_t d = 0; d < dims; ++d) {
                idx[d] += offset[d];
            }
            if (valid_index(idx, grid_dims, dims)) {
                auto& neighbor = grid[flatten_index(idx, grid_dims, dims)];
                if (neighbor.type > voxel.type) {
                    surrounded = false;
                    break;
                }
            }
        }
        if (surrounded) {
            voxel.type -= 1;
        } else {
            voxel.finalized = true;
        }
    };

    if (domain.weighting) {
        const int w_max = static_cast<int>(std::ceil((3.0 * cv_ratio / 2.0) - 0.5));
        const int w_min = static_cast<int>(std::floor(-(3.0 * cv_ratio / 2.0) - 0.5));
        int level = 0;
        while (level <= w_max || (-level - 1) >= w_min) {
            for (std::size_t i = 0; i < grid.size(); ++i) {
                auto& voxel = grid[i];
                if (voxel.finalized) {
                    continue;
                }
                if (voxel.type == level) {
                    surrounded_solid(i);
                    if (level == 0 && voxel.type == 0) {
                        // Depth-0 solid voxels form the geometric support of the exposed surface.
                        voxel.surface = true;
                    }
                } else if (voxel.type == -(level + 1)) {
                    surrounded_void(i);
                }
            }
            ++level;
            if (level > 1000) {
                throw IsthmusError("Voxel weighting iteration did not converge");
            }
        }
    } else {
        for (std::size_t i = 0; i < grid.size(); ++i) {
            auto& voxel = grid[i];
            if (voxel.type == 0) {
                surrounded_solid(i);
                if (voxel.type == 0) {
                    voxel.surface = true;
                }
            }
        }
    }

    for (auto& voxel : grid) {
        if (domain.weighting) {
            const double dvox = 0.5 + static_cast<double>(voxel.type);
            voxel.weight = std::clamp(0.5 * (1.0 + dvox * (2.0 / (3.0 * cv_ratio))), 0.0, 1.0);
        } else {
            voxel.weight = voxel.type < 0 ? 0.0 : 1.0;
        }

        if (voxel.surface) {
            if (dims == 3) {
                add_3d_faces(voxel, domain.voxel_size);
            } else {
                add_2d_faces(voxel, domain.voxel_size);
            }
        }
    }

    assign_exposed_faces(domain, grid);
}

/*
 * Mark which faces of each surface voxel are actually exposed to the nearby
 * void region.
 *
 * Only those exposed faces matter later when surface elements are matched back
 * to voxels during flux mapping.
 */
void MotionMapper::assign_exposed_faces(const DomainConfig& domain, std::vector<VoxelCell>& grid) {
    const auto dims = active_dims(domain.dimension);
    std::array<std::size_t, kMaxDims> grid_dims{{1, 1, 1}};
    for (const auto& voxel : grid) {
        for (std::size_t i = 0; i < dims; ++i) {
            grid_dims[i] = std::max(grid_dims[i], static_cast<std::size_t>(voxel.indices[i] + 1));
        }
    }
    const auto neighbors = cardinal_neighbors(dims);

    for (auto& voxel : grid) {
        if (!voxel.surface) {
            continue;
        }
        for (std::size_t face_index = 0; face_index < neighbors.size(); ++face_index) {
            auto idx = voxel.indices;
            for (std::size_t d = 0; d < dims; ++d) {
                idx[d] += neighbors[face_index][d];
            }
            bool exposed = true;
            if (valid_index(idx, grid_dims, dims)) {
                const auto& neighbor = grid[flatten_index(idx, grid_dims, dims)];
                exposed = neighbor.type < 0;
            }
            if (dims == 3) {
                voxel.faces3d[face_index].exposed = exposed;
            } else {
                voxel.faces2d[face_index].exposed = exposed;
            }
        }
    }
}

/*
 * Build the corner-centered scalar field used by marching windows.
 *
 * Every corner owns the region of space that is closer to that corner than to
 * any neighboring corner. Voxels that lie entirely inside one corner region
 * contribute their full weighted volume there. Voxels that straddle corner
 * boundaries are split so total volume is conserved.
 */
std::vector<CornerData> MotionMapper::build_corner_grid(
    const DomainConfig& domain,
    const std::vector<VoxelCell>& voxels,
    std::array<std::size_t, kMaxDims>& out_dims) {
    const auto dims = active_dims(domain.dimension);
    const auto cl = cell_lengths(domain);
    for (std::size_t i = 0; i < dims; ++i) {
        out_dims[i] = domain.cell_counts[i] + 1;
    }
    if (dims == 2) {
        out_dims[2] = 1;
    }

    std::size_t total = 1;
    for (std::size_t i = 0; i < dims; ++i) {
        total *= out_dims[i];
    }

    /*
     * Treat the domain corner lattice as an explicit bounded grid before any
     * voxel-to-corner ownership mapping happens.
     *
     * The auxiliary voxel lattice extends beyond the marching domain on
     * purpose, because weighted ghost voxels outside the solid are needed for
     * depth classification. Those exterior voxels must not be allowed to index
     * directly into the bounded domain corner array.
     */
    const RegularGrid3D corner_grid(out_dims);

    std::vector<CornerData> corners(total);
    for (std::size_t flat = 0; flat < total; ++flat) {
        CornerData corner{};
        std::size_t tmp = flat;
        for (std::size_t i = 0; i < dims; ++i) {
            const int idx = static_cast<int>(tmp % out_dims[i]);
            tmp /= out_dims[i];
            corner.indices[i] = idx;
            corner.position[i] = domain.limits[0][i] + cl[i] * static_cast<double>(idx);
        }
        corners[flat] = corner;
    }

    std::vector<std::array<int, kMaxDims>> active_flags;
    if (dims == 3) {
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                for (int k = 0; k < 2; ++k) {
                    active_flags.push_back({i, j, k});
                }
            }
        }
    } else {
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                active_flags.push_back({i, j, 0});
            }
        }
    }

    std::array<double, kMaxDims> min_pen_dist{{0.0, 0.0, 0.0}};
    for (std::size_t i = 0; i < dims; ++i) {
        min_pen_dist[i] = 0.5 * (cl[i] - domain.voxel_size);
    }

    const double voxel_volume = dims == 3
        ? domain.voxel_size * domain.voxel_size * domain.voxel_size
        : domain.voxel_size * domain.voxel_size;

    for (std::size_t vi = 0; vi < voxels.size(); ++vi) {
        const auto& voxel = voxels[vi];
        if (voxel.weight <= 1e-6) {
            continue;
        }

        std::array<int, kMaxDims> base_index{{0, 0, 0}};
        for (std::size_t i = 0; i < dims; ++i) {
            base_index[i] = static_cast<int>(
                std::llround((voxel.centroid[i] - domain.limits[0][i]) / cl[i]));
        }

        /*
         * Skip auxiliary voxels whose centers lie outside the bounded marching
         * corner lattice.
         *
         * Those voxels belong to the buffer region that supports weighting, but
         * they do not own a domain corner and therefore must not participate in
         * the flattened corner indexing below.
         */
        if (!corner_grid.valid(base_index)) {
            continue;
        }

        const auto base_flat = corner_grid.element_index(base_index);
        corners[base_flat].owned_voxel_indices.push_back(vi);

        const auto& corner = corners[base_flat];
        std::array<double, kMaxDims> diff{{0.0, 0.0, 0.0}};
        std::array<double, kMaxDims> abs_diff{{0.0, 0.0, 0.0}};
        bool unique = true;
        for (std::size_t i = 0; i < dims; ++i) {
            diff[i] = voxel.centroid[i] - corner.position[i];
            abs_diff[i] = std::abs(diff[i]);
            if (abs_diff[i] >= min_pen_dist[i]) {
                unique = false;
            }
        }

        if (unique) {
            corners[base_flat].volume += voxel_volume * voxel.weight;
            continue;
        }

        std::array<double, kMaxDims> penetration{{0.0, 0.0, 0.0}};
        std::array<int, kMaxDims> pen_flag{{0, 0, 0}};
        for (std::size_t i = 0; i < dims; ++i) {
            penetration[i] = abs_diff[i] - min_pen_dist[i];
            if (penetration[i] > 0.0) {
                pen_flag[i] = diff[i] > 0.0 ? 1 : -1;
            } else {
                penetration[i] = 0.0;
            }
        }

        /*
         * Each active_flag selects one part of the voxel volume split. Together
         * these pieces cover every corner neighborhood penetrated by the voxel,
         * which preserves total weighted volume while building the scalar field.
         */
        for (const auto& active : active_flags) {
            std::array<double, kMaxDims> lengths{{1.0, 1.0, 1.0}};
            std::array<int, kMaxDims> target = base_index;
            for (std::size_t i = 0; i < dims; ++i) {
                lengths[i] = active[i] ? penetration[i] : domain.voxel_size - penetration[i];
                target[i] += active[i] * pen_flag[i];
            }

            /*
             * Guard split targets for the same reason as the base index above:
             * a voxel near the domain boundary can straddle a corner region
             * outside the bounded marching lattice. That exterior portion is
             * intentionally discarded instead of indexing past the vector.
             */
            if (!corner_grid.valid(target)) {
                continue;
            }

            corners[corner_grid.element_index(target)].volume +=
                lengths[0] * lengths[1] * (dims == 3 ? lengths[2] : 1.0) * voxel.weight;
        }
    }

    const double region_measure = cl[0] * cl[1] * (dims == 3 ? cl[2] : 1.0);
    for (auto& corner : corners) {
        corner.volume /= region_measure;
        // The 0.5 threshold is the fill contour later surface extraction will follow.
        corner.inside = corner.volume >= 0.5 ? 1 : 0;
    }
    return corners;
}

// Extract only the surface-facing voxel
std::vector<SurfaceVoxelInfo> MotionMapper::collect_surface_voxels(const std::vector<VoxelCell>& voxels) {
    std::vector<SurfaceVoxelInfo> out;
    for (const auto& voxel : voxels) {
        if (!voxel.surface) {
            continue;
        }
        SurfaceVoxelInfo info{};
        info.original_id = voxel.original_id;
        info.voxel_indices = voxel.indices;
        info.centroid = voxel.centroid;
        info.depth = voxel.type;
        info.weight = voxel.weight;
        out.push_back(info);
    }
    return out;
}

/*
 * Execute the native motion-mapping stages that are currently available.
 *
 * The result already contains the validated domain, the corner fill field, and
 * the surface voxels. If the caller requests later stages such as surface
 * extraction or flux mapping, the function fails explicitly until those native
 * backends are added.
 */
MarchingWindowsResult MotionMapper::run(
    const DomainConfig& domain,
    const VoxelSet& voxels,
    const RunOptions& options) const {

    validate_inputs(domain, voxels);

    // initialize the marching-window lattice and 
    // classify every voxel as solid, void, or surface with a depth and weight
    auto voxel_grid = build_voxel_grid(domain, voxels);
    classify_voxels(domain, voxel_grid);
    MarchingWindowsResult result{};
    result.domain = domain;

    // store surface voxels separately for later use in flux mapping.
    result.surface_voxels = collect_surface_voxels(voxel_grid);

    // Build the corner fill field with weighted volumes of the nearby voxels to each corner.
    auto corners = build_corner_grid(domain, voxel_grid, result.corner_dims);
    result.corner_fill_fractions.reserve(corners.size());
    for (const auto& corner : corners) {
        result.corner_fill_fractions.push_back(corner.volume);
    }

    if (options.build_surface) {
        /*
         * The next native milestone is 3D surface reconstruction from the
         * already-populated corner field. 2D marching-squares parity is still
         * deferred, so keep that contract explicit for callers.
         */
        if (domain.dimension == Dimension::D2) {
            throw NotImplementedError("2D surface extraction is not implemented yet in the C++ port");
        }

        result.surface_mesh = marching_cubes::extract_surface_mesh_3d(
            result.domain,
            result.corner_fill_fractions,
            result.corner_dims);
    }
    if (options.build_flux_association) {
        throw NotImplementedError(
            "Flux mapping is not implemented yet in the C++ port scaffold");
    }

    return result;
}

}  // namespace isthmus
