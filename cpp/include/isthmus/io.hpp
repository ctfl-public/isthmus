// File writers kept separate from the in-memory C++ algorithm core.
#pragma once

#include <filesystem>

#include "isthmus/types.hpp"

namespace isthmus::io {

// Write the current in-memory surface representation using the SPARTA surface format.
void write_sparta_surface(
    const SurfaceMesh& mesh,
    Dimension dimension,
    const std::filesystem::path& output_path);

// Write triangle-to-voxel or line-to-voxel ownership in the legacy association format.
void write_flux_association(
    const FluxAssociation& association,
    Dimension dimension,
    const std::filesystem::path& output_path);

}  // namespace isthmus::io
