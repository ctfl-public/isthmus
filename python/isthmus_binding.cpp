#include <array>
#include <sstream>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "isthmus/exceptions.hpp"
#include "isthmus/marching_windows.hpp"
#include "isthmus/types.hpp"

namespace py = pybind11;
using namespace isthmus;

PYBIND11_MODULE(_isthmus, m) {
    m.doc() = "Python bindings for the isthmus marching-windows library";

    m.attr("MAX_DIMS") = kMaxDims;

    // ---- Exceptions --------------------------------------------------------

    auto base_exc = py::register_exception<IsthmusError>(m, "IsthmusError");
    py::register_exception<InvalidInputError>(m, "InvalidInputError", base_exc.ptr());
    py::register_exception<NotImplementedError>(m, "IsthmusNotImplementedError", base_exc.ptr());

    // ---- Dimension enum ----------------------------------------------------

    py::enum_<Dimension>(m, "Dimension")
        .value("D2", Dimension::D2)
        .value("D3", Dimension::D3)
        .export_values();

    // ---- RunOptions --------------------------------------------------------

    py::class_<RunOptions>(m, "RunOptions",
            "Controls which optional algorithm stages run in addition to corner-fill computation.")
        .def(py::init<>())
        .def_readwrite("build_surface", &RunOptions::build_surface,
            "Run the surface extraction stage (default: False).")
        .def_readwrite("build_flux_association", &RunOptions::build_flux_association,
            "Compute surface-voxel ownership fractions for flux mapping (default: False).")
        .def_readwrite("write_diagnostics", &RunOptions::write_diagnostics,
            "Reserved for future debug output (default: False).")
        .def_readwrite("verbose", &RunOptions::verbose,
            "Print progress messages to stdout during the run (default: False).")
        .def("__repr__", [](const RunOptions& r) {
            std::ostringstream ss;
            ss << "RunOptions(build_surface=" << (r.build_surface ? "True" : "False")
               << ", build_flux_association=" << (r.build_flux_association ? "True" : "False")
               << ", write_diagnostics=" << (r.write_diagnostics ? "True" : "False")
               << ", verbose=" << (r.verbose ? "True" : "False") << ")";
            return ss.str();
        });

    // ---- DomainConfig ------------------------------------------------------

    py::class_<DomainConfig>(m, "DomainConfig",
            "Describes the physical marching-windows domain.")
        .def(py::init<>())
        .def_readwrite("dimension", &DomainConfig::dimension,
            "Grid dimension: Dimension.D2 or Dimension.D3.")
        .def_readwrite("limits", &DomainConfig::limits,
            "[[xmin, ymin, zmin], [xmax, ymax, zmax]] coordinate bounds.")
        .def_readwrite("cell_counts", &DomainConfig::cell_counts,
            "[nx, ny, nz] number of marching cells along each active axis.")
        .def_readwrite("voxel_size", &DomainConfig::voxel_size,
            "Edge length of one voxel in the caller's model (meters).")
        .def_readwrite("weighting", &DomainConfig::weighting,
            "Apply depth-based weighting to the corner fill field (default: True).")
        .def_readwrite("iso_value", &DomainConfig::iso_value,
            "Isosurface threshold for the marching cubes/squares algorithm (default: 0.5).")
        .def("__repr__", [](const DomainConfig& d) {
            std::ostringstream ss;
            ss << "DomainConfig(dimension=D" << static_cast<int>(d.dimension)
               << ", voxel_size=" << d.voxel_size
               << ", iso_value=" << d.iso_value
               << ", weighting=" << (d.weighting ? "True" : "False") << ")";
            return ss.str();
        });

    // ---- VoxelRecord -------------------------------------------------------

    py::class_<VoxelRecord>(m, "VoxelRecord",
            "One occupied voxel supplied to a MarchingWindows run.")
        .def(py::init<>())
        .def_readwrite("centroid", &VoxelRecord::centroid,
            "[x, y, z] physical-space centroid of this voxel.")
        .def_readwrite("original_id", &VoxelRecord::original_id,
            "Caller-assigned identifier for this voxel.")
        .def_readwrite("material_tag", &VoxelRecord::material_tag,
            "Optional material label string (or None).")
        .def("__repr__", [](const VoxelRecord& v) {
            std::ostringstream ss;
            ss << "VoxelRecord(original_id=" << v.original_id
               << ", centroid=[" << v.centroid[0] << ", "
               << v.centroid[1] << ", " << v.centroid[2] << "])";
            return ss.str();
        });

    // ---- VoxelSet ----------------------------------------------------------

    py::class_<VoxelSet>(m, "VoxelSet",
            "Collection of occupied voxels supplied to a MarchingWindows run.")
        .def(py::init<>())
        .def_readwrite("voxels", &VoxelSet::voxels,
            "List of VoxelRecord objects.");

    // ---- SurfaceVoxelInfo --------------------------------------------------

    py::class_<SurfaceVoxelInfo>(m, "SurfaceVoxelInfo",
            "Metadata for a voxel on the detected outer boundary.")
        .def_readonly("original_id", &SurfaceVoxelInfo::original_id)
        .def_readonly("voxel_indices", &SurfaceVoxelInfo::voxel_indices,
            "[i, j, k] integer grid indices of this voxel.")
        .def_readonly("centroid", &SurfaceVoxelInfo::centroid,
            "[x, y, z] physical-space centroid of this voxel.")
        .def_readonly("depth", &SurfaceVoxelInfo::depth,
            "Depth classification value from the boundary pass.")
        .def_readonly("weight", &SurfaceVoxelInfo::weight,
            "Depth-based weight assigned to this voxel.");

    // ---- SurfaceMesh -------------------------------------------------------

    py::class_<SurfaceMesh>(m, "SurfaceMesh",
            "Surface connectivity produced by the reconstruction stage.")
        .def_readonly("vertices", &SurfaceMesh::vertices,
            "List of [x, y, z] vertex coordinates.")
        .def_readonly("triangles", &SurfaceMesh::triangles,
            "List of [i, j, k] vertex-index triplets (3D).")
        .def_readonly("segments", &SurfaceMesh::segments,
            "List of [i, j] vertex-index pairs (2D).");

    // ---- FluxElementOwnership ----------------------------------------------

    py::class_<FluxElementOwnership>(m, "FluxElementOwnership",
            "Ownership fractions for one surface element.")
        .def_readonly("element_id", &FluxElementOwnership::element_id)
        .def_readonly("voxel_ids", &FluxElementOwnership::voxel_ids,
            "original_ids of voxels that own this element.")
        .def_readonly("scalar_fractions", &FluxElementOwnership::scalar_fractions,
            "Ownership fractions parallel to voxel_ids, summing to 1.0.");

    // ---- FluxAssociation ---------------------------------------------------

    py::class_<FluxAssociation>(m, "FluxAssociation",
            "All surface-element ownership records from one run.")
        .def_readonly("elements", &FluxAssociation::elements,
            "List of FluxElementOwnership objects.");

    // ---- MarchingWindowsResult ---------------------------------------------

    py::class_<MarchingWindowsResult>(m, "MarchingWindowsResult",
            "Complete in-memory result of a MarchingWindows run.")
        .def_readonly("domain", &MarchingWindowsResult::domain,
            "Validated DomainConfig used for this run.")
        .def_readonly("corner_fill_fractions", &MarchingWindowsResult::corner_fill_fractions,
            "Flat list of corner fill fractions; length equals product of corner_dims.")
        .def_readonly("corner_dims", &MarchingWindowsResult::corner_dims,
            "[nx+1, ny+1, nz+1] dimensions of the corner fill fraction field.")
        .def_readonly("surface_voxels", &MarchingWindowsResult::surface_voxels,
            "List of SurfaceVoxelInfo for voxels on the outer boundary.")
        .def_readonly("surface_mesh", &MarchingWindowsResult::surface_mesh,
            "Reconstructed SurfaceMesh (populated when build_surface=True).")
        .def_readonly("flux_association", &MarchingWindowsResult::flux_association,
            "FluxAssociation (populated when build_flux_association=True).");

    // ---- MarchingWindows ---------------------------------------------------

    py::class_<MarchingWindows>(m, "MarchingWindows",
            "Public entry point for the isthmus marching-windows library.")
        .def(py::init<>())
        .def("run", &MarchingWindows::run,
            py::arg("domain"),
            py::arg("voxels"),
            py::arg("options") = RunOptions{},
            R"(Execute one marching-windows pass and return all results in memory.

Args:
    domain  (DomainConfig): Physical domain description.
    voxels  (VoxelSet):     Occupied voxel centroids.
    options (RunOptions):   Optional algorithm stages (default: all off).

Returns:
    MarchingWindowsResult

Raises:
    InvalidInputError:         Bad domain, voxel size, or voxel set.
    IsthmusNotImplementedError: Requested stage not yet implemented.
)");
}
