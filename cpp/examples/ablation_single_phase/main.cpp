#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "isthmus/io.hpp"
#include "isthmus/marching_windows.hpp"
#include "isthmus/utilities.hpp"

namespace {

/**
 * This record stores one active voxel together with the cumulative material
 * mass that has already been removed from it by prior ablation steps.
 */
struct AblationVoxelRecord {
    std::array<double, 3> centroid{{0.0, 0.0, 0.0}};
    double removed_mass = 0.0;
};

/**
 * These summary quantities are reported after each ablation step so the native
 * example exposes the same robustness diagnostics even though the driving
 * ablation load is now synthetic and constant.
 */
struct AblationStepStats {
    double total_requested_mass = 0.0;
    double mapped_mass = 0.0;
    double dropped_mass = 0.0;
    std::size_t empty_triangle_count = 0;
};

/**
 * Convert the current ablation state into the public `VoxelSet` consumed by
 * the native marching-windows interface.
 */
isthmus::VoxelSet make_voxel_set(const std::vector<AblationVoxelRecord>& state) {
    isthmus::VoxelSet voxels;
    voxels.voxels.reserve(state.size());

    for (std::size_t i = 0; i < state.size(); ++i) {
        isthmus::VoxelRecord record;
        record.centroid = state[i].centroid;
        record.original_id = i;
        voxels.voxels.push_back(record);
    }

    return voxels;
}

/*
 * Compute the scalar material volume fraction from the corner-fill field.
 */
double compute_volume_fraction(
    const std::vector<double>& corner_fill_fractions,
    const std::array<std::size_t, 3>& cell_counts) {
    double total = 0.0;
    for (const double value : corner_fill_fractions) {
        total += value;
    }

    const double denom = static_cast<double>(cell_counts[0] * cell_counts[1] * cell_counts[2]);
    return total / denom;
}

/**
 * Parse one optional non-negative command-line override while preserving a
 * strict and explicit failure mode for malformed user input.
 */
double parse_nonnegative_double(
    const char* text,
    const char* label) {
    const std::string value_text = text == nullptr ? std::string() : std::string(text);
    if (value_text.empty()) {
        throw std::runtime_error(std::string("Missing value for ") + label);
    }

    std::size_t parsed_chars = 0u;
    const double value = std::stod(value_text, &parsed_chars);
    if (parsed_chars != value_text.size() || !std::isfinite(value) || value < 0.0) {
        throw std::runtime_error(std::string("Invalid non-negative numeric value for ") + label + ": " + value_text);
    }

    return value;
}

/**
 * Write the current voxel positions plus cumulative removed mass using the
 * established four-column CSV-style format.
 */
void write_voxel_data(
    const std::filesystem::path& path,
    const std::vector<AblationVoxelRecord>& state) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to open voxel-state file: " + path.string());
    }

    for (const auto& voxel : state) {
        out << voxel.centroid[0] << ','
            << voxel.centroid[1] << ','
            << voxel.centroid[2] << ','
            << voxel.removed_mass << '\n';
    }
}

/**
 * Write the current material volume fraction into the standard `volFrac.dat`
 * scalar file used by this workflow.
 */
void write_volume_fraction(
    const std::filesystem::path& path,
    double volume_fraction) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to open volume-fraction file: " + path.string());
    }
    out << volume_fraction << '\n';
}

/**
 * Export the reconstructed surface for one ablation step in both native mesh
 * formats used by the C++ tree.
 */
void write_surface_outputs(
    const std::filesystem::path& grid_dir,
    const isthmus::SurfaceMesh& mesh,
    std::size_t step) {
    const auto surf_path = grid_dir / ("grid_" + std::to_string(step) + ".surf");
    const auto vtp_path = grid_dir / ("grid_" + std::to_string(step) + ".vtp");

    isthmus::io::write_sparta_surface(mesh, isthmus::Dimension::D3, surf_path);
    isthmus::io::write_vtp_surface(mesh, vtp_path);
}

/**
 * Count how many flux-association entries are empty after the tolerant native
 * robustness pass.
 */
std::size_t count_empty_flux_elements(const isthmus::FluxAssociation& association) {
    std::size_t count = 0u;
    for (const auto& element : association.elements) {
        if (element.voxel_ids.empty()) {
            ++count;
        }
    }
    return count;
}

/**
 * Apply one constant-mass ablation step to the current voxel state using the
 * native triangle-to-voxel ownership map.
 *
 * Every triangle requests the same scalar mass removal. Empty ownership entries
 * are treated as dropped triangle mass rather than as hard failures so the
 * example continues to exercise the tolerant native production path.
 */
AblationStepStats ablate_voxels(
    std::vector<AblationVoxelRecord>& state,
    const isthmus::FluxAssociation& association,
    double volume_fraction,
    const std::array<std::array<double, 3>, 2>& limits,
    double sample_density,
    double triangle_mass_rate) {
    AblationStepStats stats{};

    const double volume =
        (limits[1][0] - limits[0][0]) *
        (limits[1][1] - limits[0][1]) *
        (limits[1][2] - limits[0][2]);
    const double mass_c = volume_fraction * volume * sample_density;
    const double mass_per_voxel = mass_c / static_cast<double>(state.size());
    stats.total_requested_mass =
        triangle_mass_rate * static_cast<double>(association.elements.size());

    std::vector<double> removed_mass(state.size(), 0.0);
    for (std::size_t i = 0; i < state.size(); ++i) {
        removed_mass[i] = state[i].removed_mass;
    }

    /*
     * Map the synthetic constant triangle mass onto voxels through the native
     * flux association. Any empty ownership entry counts as dropped mass by
     * design.
     */
    for (std::size_t triangle_id = 0; triangle_id < association.elements.size(); ++triangle_id) {
        const auto& element = association.elements[triangle_id];
        if (element.voxel_ids.empty()) {
            stats.dropped_mass += triangle_mass_rate;
            ++stats.empty_triangle_count;
            continue;
        }

        for (std::size_t i = 0; i < element.voxel_ids.size(); ++i) {
            const auto voxel_id = element.voxel_ids[i];
            if (voxel_id >= removed_mass.size()) {
                stats.dropped_mass += element.scalar_fractions[i] * triangle_mass_rate;
                continue;
            }

            const double contribution = element.scalar_fractions[i] * triangle_mass_rate;
            removed_mass[voxel_id] += contribution;
            stats.mapped_mass += contribution;
        }
    }

    /*
     * Remove fully ablated voxels and keep cumulative removed mass for the
     * survivors so later steps continue the recession history correctly.
     */
    std::vector<AblationVoxelRecord> next_state;
    next_state.reserve(state.size());
    for (std::size_t i = 0; i < state.size(); ++i) {
        if (removed_mass[i] > mass_per_voxel) {
            continue;
        }

        AblationVoxelRecord voxel = state[i];
        voxel.removed_mass = removed_mass[i];
        next_state.push_back(voxel);
    }

    state = std::move(next_state);
    return stats;
}

}  // namespace

int main(int argc, char** argv) {
    using namespace isthmus;

    /*
     * Resolve the example-local directory directly from this source file
     * location where TIFF input file is located.
     */
    const std::filesystem::path data_dir = std::filesystem::path(__FILE__).parent_path();
    const std::filesystem::path output_dir = argc > 1
        ? std::filesystem::path(argv[1])
        : std::filesystem::path("output");

    /*
     * Use the production single-phase physical setup.
     */
    constexpr int width = 200;
    constexpr int height = 100;
    constexpr int buffer = 5;
    constexpr std::size_t nsteps = 3;
    constexpr double voxel_size = 3.3757e-6;
    constexpr double sample_density = 1800.0;
    constexpr double default_triangle_mass_rate = 1.5e-14; 

    /*
     * Allow callers to tune the synthetic constant ablation strength without
     * having to rebuild the standalone example.
     */
    const double triangle_mass_rate = argc > 2
        ? parse_nonnegative_double(argv[2], "triangle_mass_rate")
        : default_triangle_mass_rate;

    DomainConfig domain;
    domain.dimension = Dimension::D3;
    domain.limits = {{{-buffer * voxel_size, -buffer * voxel_size, -buffer * voxel_size},
                      {(height + buffer) * voxel_size,
                       (width + buffer) * voxel_size,
                       (width + buffer) * voxel_size}}};
    domain.cell_counts = {{static_cast<std::size_t>(height),
                           static_cast<std::size_t>(width),
                           static_cast<std::size_t>(width)}};
    domain.voxel_size = voxel_size;
    domain.weighting = false;

    RunOptions options;
    options.build_surface = true;
    options.build_flux_association = true;

    /**
     * Create the standard output layout inside the caller-selected directory.
     */
    const auto grid_dir = output_dir / "grids";
    const auto voxel_data_dir = output_dir / "voxel_data";
    const auto voxel_tri_dir = output_dir / "voxel_tri";
    std::filesystem::create_directories(grid_dir);
    std::filesystem::create_directories(voxel_data_dir);
    std::filesystem::create_directories(voxel_tri_dir);

    /**
     * Load the production sample and iterate the native
     * “surface -> flux -> ablate” loop.
     */
    const auto initial_voxels =
        utilities::load_active_voxels_from_tiff(data_dir / "sample1.tif", voxel_size);
    std::vector<AblationVoxelRecord> state;
    state.reserve(initial_voxels.voxels.size());
    for (const auto& voxel : initial_voxels.voxels) {
        state.push_back(AblationVoxelRecord{voxel.centroid, 0.0});
    }
    std::cout << "Loaded " << state.size() << " active voxels from "
              << (data_dir / "sample1.tif") << '\n';
    std::cout << "Using constant triangle ablation mass of "
              << triangle_mass_rate << " kg per triangle per step\n";

    MarchingWindows marching_windows;

    for (std::size_t step = 0; step <= nsteps; ++step) {
        std::cout << "Step " << step << "/" << nsteps << '\n';

        const auto result = marching_windows.run(domain, make_voxel_set(state), options);
        const double volume_fraction =
            compute_volume_fraction(result.corner_fill_fractions, domain.cell_counts);
        const std::size_t empty_flux_count =
            count_empty_flux_elements(result.flux_association);

        write_surface_outputs(grid_dir, result.surface_mesh, step);
        write_voxel_data(
            voxel_data_dir / ("voxel_data_" + std::to_string(step) + ".dat"),
            state);
        io::write_flux_association(
            result.flux_association,
            domain.dimension,
            voxel_tri_dir / ("triangle_voxels_" + std::to_string(step) + ".dat"));
        write_volume_fraction(output_dir / "volFrac.dat", volume_fraction);

        std::cout << "  Surface triangles: " << result.surface_mesh.triangles.size() << '\n';
        std::cout << "  Surface voxels: " << result.surface_voxels.size() << '\n';
        std::cout << "  Empty flux entries: " << empty_flux_count << '\n';
        std::cout << "  Volume fraction: " << volume_fraction << '\n';

        if (step == nsteps) {
            continue;
        }

        const auto stats = ablate_voxels(
            state,
            result.flux_association,
            volume_fraction,
            domain.limits,
            sample_density,
            triangle_mass_rate);

        const double mapped_mass_error_percent = stats.total_requested_mass > 0.0
            ? 100.0 * (stats.total_requested_mass - stats.mapped_mass) / stats.total_requested_mass
            : 0.0;
        const double dropped_mass_percent = stats.total_requested_mass > 0.0
            ? 100.0 * stats.dropped_mass / stats.total_requested_mass
            : 0.0;

        std::cout << "  Dropped triangle count: " << stats.empty_triangle_count << '\n';
        std::cout << "  Dropped ablation mass (%): " << dropped_mass_percent << '\n';
        std::cout << "  Mass conservation error after mapped mass (%): "
                  << mapped_mass_error_percent << '\n';
        std::cout << "  Remaining active voxels after ablation: " << state.size() << '\n';
    }

    /**
     * Report the output directory explicitly so the example behaves like the
     * other native demos and is easy to inspect after it finishes.
     */
    std::cout << "Wrote ablation outputs to " << std::filesystem::absolute(output_dir) << '\n';
    return EXIT_SUCCESS;
}
