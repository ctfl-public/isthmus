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
#include "isthmus/geometry.hpp"
#include "isthmus/marching_windows.hpp"
#include "isthmus/utilities.hpp"

namespace {

/**
 * This record stores one active voxel together with the cumulative material
 * mass that has already been removed from it by prior ablation steps.
 */
struct AblationVoxelRecord {
    // Voxel center in domain coordinates, in meters.
    std::array<double, 3> centroid{{0.0, 0.0, 0.0}};
    // Cumulative mass removed from this voxel across all prior steps, in kg.
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
    double total_fill = 0.0;
    for (const double value : corner_fill_fractions) {
        total_fill += value;
    }

    const double total_cell_count = static_cast<double>(cell_counts[0] * cell_counts[1] * cell_counts[2]);
    return total_fill / total_cell_count;
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
 * Compute the physical surface area of every triangle in one reconstructed
 * surface mesh.
 *
 * The returned vector is indexed exactly like `mesh.triangles` so the ablation
 * update can reuse the native flux-association indexing without extra lookup
 * structures.
 */
std::vector<double> compute_triangle_areas(const isthmus::SurfaceMesh& mesh) {
    std::vector<double> triangle_areas;
    triangle_areas.reserve(mesh.triangles.size());

    for (const auto& triangle : mesh.triangles) {
        /**
         * Rebuild the explicit vertex triplet expected by the shared geometry
         * helper so this example stays aligned with the library's own area
         * convention.
         */
        const std::array<std::array<double, 3>, 3> vertices{{
            mesh.vertices[triangle[0]],
            mesh.vertices[triangle[1]],
            mesh.vertices[triangle[2]]
        }};
        triangle_areas.push_back(isthmus::geometry::triangle_area(vertices));
    }

    return triangle_areas;
}

/**
 * Apply one area-based ablation step to the current voxel state using the
 * native triangle-to-voxel ownership map.
 *
 * Every triangle now removes mass in proportion to its physical area, using a
 * constant surface mass flux in kilograms per square meter per step. Empty
 * ownership entries are still treated as dropped triangle mass rather than as
 * hard failures so the example continues to exercise the tolerant native
 * production path.
 */
AblationStepStats ablate_voxels(
    std::vector<AblationVoxelRecord>& voxel_state,
    const isthmus::SurfaceMesh& surface_mesh,
    const isthmus::FluxAssociation& association,
    double volume_fraction,
    const std::array<std::array<double, 3>, 2>& limits,
    double sample_density,
    double surface_mass_flux) {
    AblationStepStats stats{};

    const double domain_volume =
        (limits[1][0] - limits[0][0]) *
        (limits[1][1] - limits[0][1]) *
        (limits[1][2] - limits[0][2]);
    const double current_material_mass = volume_fraction * domain_volume * sample_density;
    const double mass_per_voxel = current_material_mass / static_cast<double>(voxel_state.size());

    /**
     * Precompute one physical triangle area per surface element so every
     * ablation request is scaled by reconstructed surface area instead of raw
     * triangle count.
     */
    const std::vector<double> triangle_areas = compute_triangle_areas(surface_mesh);

    std::vector<double> removed_mass(voxel_state.size(), 0.0);
    for (std::size_t i = 0; i < voxel_state.size(); ++i) {
        removed_mass[i] = voxel_state[i].removed_mass;
    }

    /*
     * Map the area-scaled triangle mass onto voxels through the native flux
     * association. Any empty ownership entry counts as dropped mass by design.
     */
    for (std::size_t triangle_id = 0; triangle_id < association.elements.size(); ++triangle_id) {
        const auto& element = association.elements[triangle_id];
        /**
         * Guard against unexpected indexing mismatch so a malformed mesh or
         * association cannot read beyond the reconstructed triangle list.
         */
        const double triangle_area = triangle_id < triangle_areas.size()
            ? triangle_areas[triangle_id]
            : 0.0;
        const double triangle_mass = surface_mass_flux * triangle_area;
        stats.total_requested_mass += triangle_mass;

        if (element.voxel_ids.empty()) {
            stats.dropped_mass += triangle_mass;
            ++stats.empty_triangle_count;
            continue;
        }

        for (std::size_t i = 0; i < element.voxel_ids.size(); ++i) {
            const auto voxel_id = element.voxel_ids[i];
            if (voxel_id >= removed_mass.size()) {
                stats.dropped_mass += element.scalar_fractions[i] * triangle_mass;
                continue;
            }

            const double contribution = element.scalar_fractions[i] * triangle_mass;
            removed_mass[voxel_id] += contribution;
            stats.mapped_mass += contribution;
        }
    }

    /*
     * Remove fully ablated voxels and keep cumulative removed mass for the
     * survivors so later steps continue the recession history correctly.
     */
    std::vector<AblationVoxelRecord> next_state;
    next_state.reserve(voxel_state.size());
    for (std::size_t i = 0; i < voxel_state.size(); ++i) {
        if (removed_mass[i] > mass_per_voxel) {
            continue;
        }

        AblationVoxelRecord voxel = voxel_state[i];
        voxel.removed_mass = removed_mass[i];
        next_state.push_back(voxel);
    }

    voxel_state = std::move(next_state);
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
     * Sample configuration
     */
    constexpr int image_width = 200;    // pixels
    constexpr int image_height = 100;   // pixels
    constexpr int buffer = 5;   // pixels of zero-value padding to add around the input image to ensure a closed marching domain
    constexpr std::size_t n_ablation_steps = 3;
    constexpr double voxel_size = 3.3757e-6;  // meters per voxel edge
    constexpr double sample_density = 1800.0;  // kg per cubic meter
    /**
     * Scale the historical per-triangle default by one voxel-face area so the
     * updated area-based model starts near the previous magnitude on roughly
     * voxel-sized surface elements.
     */
    constexpr double default_surface_mass_flux =
        4e-14 / (voxel_size * voxel_size);  // kg per square meter per step

    /*
     * Allow callers to tune the constant ablation without
     * having to rebuild this example.
     */
    const double surface_mass_flux = argc > 2
        ? parse_nonnegative_double(argv[2], "surface_mass_flux")
        : default_surface_mass_flux;

    /*
     * Configure the marching windows domain.
     */
    DomainConfig domain;
    domain.dimension = Dimension::D3;
    domain.limits = {{{-buffer * voxel_size, -buffer * voxel_size, -buffer * voxel_size},
                      {(image_height + buffer) * voxel_size,
                       (image_width + buffer) * voxel_size,
                       (image_width + buffer) * voxel_size}}};
    domain.cell_counts = {{static_cast<std::size_t>(image_height),
                           static_cast<std::size_t>(image_width),
                           static_cast<std::size_t>(image_width)}};
    domain.voxel_size = voxel_size;
    domain.weighting = false;
    domain.iso_value = 0.5;

    /*
     * Configure the run options.
     */
    RunOptions options;
    options.build_surface = true;
    options.build_flux_association = true;

    /*
     * Create the standard output layout inside the output directory.
     */
    const auto grid_dir = output_dir / "grids";
    const auto voxel_data_dir = output_dir / "voxel_data";
    const auto voxel_tri_dir = output_dir / "voxel_tri";
    std::filesystem::create_directories(grid_dir);
    std::filesystem::create_directories(voxel_data_dir);
    std::filesystem::create_directories(voxel_tri_dir);

    /**
     * Load the tiff sample into the initial voxel state.
     */
    const auto initial_voxels = utilities::load_active_voxels_from_tiff(data_dir / "sample1.tif", voxel_size);
    std::vector<AblationVoxelRecord> voxel_state; // this is the main state vector that gets updated across ablation steps
    voxel_state.reserve(initial_voxels.voxels.size()); // pre-allocate the state vector
    for (const auto& voxel : initial_voxels.voxels) {
        // Initialize the voxel_state with zero removed mass since no ablation has occurred yet.
        voxel_state.push_back(AblationVoxelRecord{voxel.centroid, 0.0});
    }
    // Report the initial voxel size and configuration
    std::cout << "Loaded " << voxel_state.size() << " active voxels from "
              << (data_dir / "sample1.tif") << '\n';
    std::cout << "Using constant surface ablation flux of "
              << surface_mass_flux << " kg per square meter per step\n";

    /*
     * Declare the marching windows instance
     */
    MarchingWindows marching_windows;

    /*
     * Ablation loop:
     * For each step, 
     * run marching windows on the current voxel state, 
     * write all outputs, 
     * then apply one ablation update to produce the next voxel state. 
     */
    for (std::size_t step = 0; step <= n_ablation_steps; ++step) {
        std::cout << "Step " << step << "/" << n_ablation_steps << '\n';

        // Run marching windows on the current voxel state
        const auto step_result = marching_windows.run(domain, make_voxel_set(voxel_state), options);
        
        // Compute diagnostics before the ablation update
        const double volume_fraction =
            compute_volume_fraction(step_result.corner_fill_fractions, domain.cell_counts);
        const std::size_t empty_flux_count =
            count_empty_flux_elements(step_result.flux_association);

        // Write all outputs for the current step
        write_surface_outputs(grid_dir, step_result.surface_mesh, step);
        write_voxel_data(
            voxel_data_dir / ("voxel_data_" + std::to_string(step) + ".dat"),
            voxel_state);
        io::write_flux_association(
            step_result.flux_association,
            domain.dimension,
            voxel_tri_dir / ("triangle_voxels_" + std::to_string(step) + ".dat"));
        write_volume_fraction(output_dir / "volFrac.dat", volume_fraction);

        // Report diagnostics for the current step
        // "Surface triangles": the number of triangles in the surface mesh
        // "Surface voxels": the number of voxels that were marked as belonging to the surface
        // "Empty flux entries": the number of surface triangles that received no voxel ownership
        // "Volume fraction": the overall material volume fraction computed from the corner-fill field
        std::cout << "  Surface triangles: " << step_result.surface_mesh.triangles.size() << '\n';
        std::cout << "  Surface voxels: " << step_result.surface_voxels.size() << '\n';
        std::cout << "  Empty flux entries: " << empty_flux_count << '\n';
        std::cout << "  Volume fraction: " << volume_fraction << '\n';

        // Skip the ablation update on the last step
        if (step == n_ablation_steps) {
            continue;
        }

        // Apply one ablation update to produce the next voxel state and report diagnostics for the ablation step
        const auto stats = ablate_voxels(
            voxel_state,
            step_result.surface_mesh,
            step_result.flux_association,
            volume_fraction,
            domain.limits,
            sample_density,
            surface_mass_flux);

        // Report ablation diagnostics for the current step
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
        std::cout << "  Remaining active voxels after ablation: " << voxel_state.size() << '\n';
    }

    /**
     * Report the output directory explicitly so the example behaves like the
     * other native demos and is easy to inspect after it finishes.
     */
    std::cout << "Wrote ablation outputs to " << std::filesystem::absolute(output_dir) << '\n';
    return EXIT_SUCCESS;
}
