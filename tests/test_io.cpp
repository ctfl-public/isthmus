#include "test_framework.hpp"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

#include "isthmus/io.hpp"
#include "isthmus/marching_windows.hpp"

namespace {

/*
 * Build a tiny triangle mesh with nontrivial coordinates so the writer tests
 * can check both topology tags and serialized floating-point point data.
 */
isthmus::SurfaceMesh make_sample_mesh() {
    isthmus::SurfaceMesh mesh;
    mesh.vertices = {
        {0.0, 0.0, 0.0},
        {1.5, 0.0, 0.25},
        {0.0, 2.0, 0.5}
    };
    mesh.triangles = {{{0, 1, 2}}};
    return mesh;
}

/*
 * Build a compact centered voxel cube so the writer tests exercise the native
 * surface reconstruction path with a real extracted mesh.
 */
isthmus::VoxelSet make_voxel_cube(double cube_side_length, int voxels_per_axis) {
    isthmus::VoxelSet voxels;
    const double voxel_size = cube_side_length / static_cast<double>(voxels_per_axis);
    const double cube_lo = -0.5 * cube_side_length;
    std::size_t id = 0;

    for (int k = 0; k < voxels_per_axis; ++k) {
        for (int j = 0; j < voxels_per_axis; ++j) {
            for (int i = 0; i < voxels_per_axis; ++i) {
                isthmus::VoxelRecord record;
                record.centroid = {
                    cube_lo + (0.5 + static_cast<double>(i)) * voxel_size,
                    cube_lo + (0.5 + static_cast<double>(j)) * voxel_size,
                    cube_lo + (0.5 + static_cast<double>(k)) * voxel_size
                };
                record.original_id = id++;
                voxels.voxels.push_back(record);
            }
        }
    }

    return voxels;
}

/*
 * Read the full contents of a small test artifact into memory for substring
 * checks against the expected VTK XML tags and array payload fragments.
 */
std::string read_file(const std::filesystem::path& path) {
    std::ifstream in(path);
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

}  // namespace

TEST_CASE(test_write_vtp_surface_creates_expected_xml_sections) {
    using namespace isthmus;

    /*
     * Case:
     * Serialize a tiny hand-built triangle mesh to VTP so the writer can be
     * checked independently of any marching-windows reconstruction logic.
     *
     * Sketch:
     *     v2
     *     /\
     *    /  \
     *   /____\
     *  v0    v1
     *
     *   one triangle, three points, one polygon
     *
     * Expected outcome:
     * The writer should create the output file and include the expected VTK XML
     * container tags, point section, connectivity metadata, and the correct
     * point and polygon counts for the sample mesh.
     */
    const std::filesystem::path output_path = std::filesystem::temp_directory_path() / "isthmus_test_surface_writer.vtp";
    std::filesystem::remove(output_path);

    io::write_vtp_surface(make_sample_mesh(), output_path);
    CHECK(std::filesystem::exists(output_path));

    const auto contents = read_file(output_path);
    CHECK(contents.find("<VTKFile type=\"PolyData\"") != std::string::npos);
    CHECK(contents.find("<PolyData>") != std::string::npos);
    CHECK(contents.find("<Points>") != std::string::npos);
    CHECK(contents.find("Name=\"connectivity\"") != std::string::npos);
    CHECK(contents.find("Name=\"offsets\"") != std::string::npos);
    CHECK(contents.find("Name=\"triangle_id\"") != std::string::npos);
    CHECK(contents.find("Name=\"vtk_cell_type\"") != std::string::npos);
    CHECK(contents.find("NumberOfPoints=\"3\"") != std::string::npos);
    CHECK(contents.find("NumberOfPolys=\"1\"") != std::string::npos);

    std::filesystem::remove(output_path);
}

TEST_CASE(test_write_vtp_surface_end_to_end_serializes_native_surface_mesh) {
    using namespace isthmus;

    /*
     * Case:
     * Run native 3D surface reconstruction on a centered voxel cube and
     * immediately serialize the produced mesh with the VTP writer.
     *
     * Sketch:
     *   voxel cube -> extracted surface -> .vtp file
     *
     *   This checks the export path after a real mesh has been generated.
     *
     * Expected outcome:
     * The end-to-end export path should write a non-empty VTP file whose point
     * and polygon counts match the reconstructed mesh and whose per-triangle
     * metadata arrays are present in the output.
    */
    // Set up a marching-windows run with a compact voxel cube that should produce a small but nontrivial surface mesh.
    const double marching_grid_length = 4.0e-6;
    const double cube_side_length = std::cbrt(0.75) * (marching_grid_length / 4.0);

    RunOptions options;
    options.dimension = Dimension::D3;
    options.voxel_size = cube_side_length / 2.0;
    options.marching_voxel_ratio = (marching_grid_length / 4.0) / options.voxel_size;
    options.weighting = false;
    options.build_surface = true;
    options.build_flux_association = false;

    MarchingWindows mw;
    const auto result = mw.run(make_voxel_cube(cube_side_length, 2), options);

    // Create a temporary output file and serialize the reconstructed mesh to VTP.
    const std::filesystem::path output_path = std::filesystem::temp_directory_path() / "isthmus_test_native_surface.vtp";
    std::filesystem::remove(output_path);
    io::write_vtp_surface(result.surface_mesh, output_path);

    // check the file exists and has content before trying to read it for substring checks
    CHECK(std::filesystem::exists(output_path));
    CHECK(std::filesystem::file_size(output_path) > 0);

    // check the file contains the expected point and polygon counts and metadata arrays for the reconstructed mesh
    const auto contents = read_file(output_path);
    CHECK(contents.find("NumberOfPoints=\"" + std::to_string(result.surface_mesh.vertices.size()) + "\"") != std::string::npos);
    CHECK(contents.find("NumberOfPolys=\"" + std::to_string(result.surface_mesh.triangles.size()) + "\"") != std::string::npos);
    CHECK(contents.find("Name=\"triangle_id\"") != std::string::npos);

    std::filesystem::remove(output_path);
}

TEST_CASE(test_write_flux_association_serializes_expected_legacy_blocks) {
    using namespace isthmus;

    /*
     * Case:
     * Serialize a compact in-memory triangle ownership result using the
     * legacy flux-association writer.
     *
     * Sketch:
     *   triangle 1 -> voxels {3, 5}
     *   triangle 2 -> voxel  {9}
     *
     * Expected outcome:
     * The file should contain the expected triangle header and the paired
     * start/end blocks with one-based element ids and the stored ownership
     * values.
     */
    FluxAssociation association;
    association.elements = {
        FluxElementOwnership{0, {3, 5}, {0.25, 0.75}},
        FluxElementOwnership{1, {9}, {1.0}}
    };

    const std::filesystem::path output_path =
        std::filesystem::temp_directory_path() / "isthmus_test_flux_association.dat";
    std::filesystem::remove(output_path);

    io::write_flux_association(association, Dimension::D3, output_path);
    CHECK(std::filesystem::exists(output_path));

    const auto contents = read_file(output_path);
    CHECK(contents.find("2 total triangles") != std::string::npos);
    CHECK(contents.find("start id 1") != std::string::npos);
    CHECK(contents.find("    3 0.25") != std::string::npos);
    CHECK(contents.find("    5 0.75") != std::string::npos);
    CHECK(contents.find("end id 1") != std::string::npos);
    CHECK(contents.find("start id 2") != std::string::npos);
    CHECK(contents.find("    9 1") != std::string::npos);
    CHECK(contents.find("end id 2") != std::string::npos);

    std::filesystem::remove(output_path);
}
