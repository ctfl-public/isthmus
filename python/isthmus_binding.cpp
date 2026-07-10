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
            "Controls physical settings and optional algorithm stages.")
        .def(py::init<>())
        .def_readwrite("dimension", &RunOptions::dimension,
            "Grid dimension: Dimension.D2 or Dimension.D3 (default: Dimension.D3).")
        .def_readwrite("voxel_size", &RunOptions::voxel_size,
            "Edge length of one voxel in the caller's model (meters).")
        .def_readwrite("marching_voxel_ratio", &RunOptions::marching_voxel_ratio,
            "Required marching-cell / voxel-size ratio.")
        .def_readwrite("weighting", &RunOptions::weighting,
            "Apply depth-based weighting to the corner fill field (default: True).")
        .def_readwrite("iso_value", &RunOptions::iso_value,
            "Isosurface threshold for the marching cubes/squares algorithm (default: 0.5).")
        .def_readwrite("edge_clamp", &RunOptions::edge_clamp,
            "Minimum interpolation fraction kept between a surface vertex and the "
            "marching-grid corners of its edge, in [0, 0.5). Bounds the smallest "
            "edge/triangle of the mesh to ~edge_clamp * cell length (default: 0.01).")
        .def_readwrite("build_surface", &RunOptions::build_surface,
            "Run the surface extraction stage (default: True).")
        .def_readwrite("build_flux_association", &RunOptions::build_flux_association,
            "Compute surface-voxel ownership fractions for flux mapping (default: True).")
        .def_readwrite("min_component_volume_voxels", &RunOptions::min_component_volume_voxels,
            "De-noising threshold in voxel volumes: isolated closed components "
            "(cavities and specks alike) with |enclosed volume| below this are "
            "removed as reconstruction noise. 0 keeps everything (default: 0.1).")
        .def_readwrite("remove_sealed_pores", &RunOptions::remove_sealed_pores,
            "Remove ALL sealed cavities regardless of size. Enable for DSMC/SPARTA "
            "use; off by default because enclosed porosity is real information for "
            "other consumers (default: False).")
        .def_readwrite("verbose", &RunOptions::verbose,
            "Print progress messages to stdout during the run (default: False).")
        .def("__repr__", [](const RunOptions& r) {
            std::ostringstream ss;
            ss << "RunOptions(dimension=D" << static_cast<int>(r.dimension)
               << ", voxel_size=" << r.voxel_size
               << ", marching_voxel_ratio=" << r.marching_voxel_ratio
               << ", iso_value=" << r.iso_value
               << ", weighting=" << (r.weighting ? "True" : "False")
               << ", build_surface=" << (r.build_surface ? "True" : "False")
               << ", build_flux_association=" << (r.build_flux_association ? "True" : "False")
               << ", verbose=" << (r.verbose ? "True" : "False") << ")";
            return ss.str();
        });

    // ---- DomainConfig ------------------------------------------------------

    py::class_<DomainConfig>(m, "DomainConfig",
            "Resolved physical marching-windows domain returned by a run.")
        .def(py::init<>())
        .def_readwrite("dimension", &DomainConfig::dimension,
            "Grid dimension: Dimension.D2 or Dimension.D3.")
        .def_readwrite("limits", &DomainConfig::limits,
            "[[xmin, ymin, zmin], [xmax, ymax, zmax]] coordinate bounds derived during run().")
        .def_readwrite("cell_counts", &DomainConfig::cell_counts,
            "[nx, ny, nz] marching cells derived during run().")
        .def_readwrite("voxel_size", &DomainConfig::voxel_size,
            "Edge length of one voxel in the caller's model (meters).")
        .def_readwrite("marching_voxel_ratio", &DomainConfig::marching_voxel_ratio,
            "Required marching-cell / voxel-size ratio. limits and cell_counts are derived from the voxel set.")
        .def_readwrite("weighting", &DomainConfig::weighting,
            "Apply depth-based weighting to the corner fill field (default: True).")
        .def_readwrite("iso_value", &DomainConfig::iso_value,
            "Isosurface threshold for the marching cubes/squares algorithm (default: 0.5).")
        .def_readwrite("edge_clamp", &DomainConfig::edge_clamp,
            "Minimum interpolation fraction kept between a surface vertex and the "
            "marching-grid corners of its edge, in [0, 0.5) (default: 0.01).")
        .def("__repr__", [](const DomainConfig& d) {
            std::ostringstream ss;
            ss << "DomainConfig(dimension=D" << static_cast<int>(d.dimension)
               << ", voxel_size=" << d.voxel_size
               << ", marching_voxel_ratio=" << d.marching_voxel_ratio
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
            "Resolved DomainConfig used for this run.")
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
            py::arg("voxels"),
            py::arg("options"),
            R"(Execute one marching-windows pass and return all results in memory.

Args:
    voxels  (VoxelSet):     Occupied voxel centroids.
    options (RunOptions):   Required physical settings and optional algorithm stages.

Returns:
    MarchingWindowsResult

Raises:
    InvalidInputError:         Bad domain, voxel size, or voxel set.
    IsthmusNotImplementedError: Requested stage not yet implemented.
)");
}
