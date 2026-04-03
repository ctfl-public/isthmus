#include "isthmus/io.hpp"

#include <fstream>
#include <stdexcept>

namespace isthmus::io {

namespace {

/*
 * Write one XML data array payload using the plain-text VTK XML encoding.
 *
 * Keeping this helper small avoids repeating the tag boilerplate across the
 * points, connectivity, offsets, cell types, and triangle id arrays.
 */
template <typename Writer>
void write_data_array(
    std::ofstream& out,
    const char* type,
    const char* name,
    std::size_t components,
    Writer&& writer) {
    out << "        <DataArray type=\"" << type << '"';
    if (name != nullptr) {
        out << " Name=\"" << name << '"';
    }
    if (components > 0) {
        out << " NumberOfComponents=\"" << components << '"';
    }
    out << " format=\"ascii\">\n";
    out << "          ";
    writer(out);
    out << "\n";
    out << "        </DataArray>\n";
}

/*
 * Fail early when the target file cannot be opened.
 *
 * The example and tests both rely on a clear failure mode instead of silently
 * producing an empty or missing file when the path is invalid.
 */
std::ofstream open_output_stream(const std::filesystem::path& output_path) {
    std::ofstream out(output_path);
    if (!out) {
        throw std::runtime_error("Failed to open output file: " + output_path.string());
    }
    return out;
}

}  // namespace

void write_sparta_surface(
    const SurfaceMesh& mesh,
    Dimension dimension,
    const std::filesystem::path& output_path) {
    std::ofstream out = open_output_stream(output_path);
    out << "surf file from isthmus-cpp\n\n";
    if (dimension == Dimension::D3) {
        out << mesh.vertices.size() << " points\n" << mesh.triangles.size() << " triangles\n\nPoints\n\n";
        for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
            const auto& v = mesh.vertices[i];
            out << (i + 1) << ' ' << v[0] << ' ' << v[1] << ' ' << v[2] << '\n';
        }
        out << "\nTriangles\n\n";
        for (std::size_t i = 0; i < mesh.triangles.size(); ++i) {
            const auto& t = mesh.triangles[i];
            out << (i + 1) << ' ' << (t[0] + 1) << ' ' << (t[1] + 1) << ' ' << (t[2] + 1) << '\n';
        }
    } else {
        out << mesh.vertices.size() << " points\n" << mesh.segments.size() << " lines\n\nPoints\n\n";
        for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
            const auto& v = mesh.vertices[i];
            out << (i + 1) << ' ' << v[0] << ' ' << v[1] << '\n';
        }
        out << "\nLines\n\n";
        for (std::size_t i = 0; i < mesh.segments.size(); ++i) {
            const auto& s = mesh.segments[i];
            out << (i + 1) << ' ' << (s[1] + 1) << ' ' << (s[0] + 1) << '\n';
        }
    }
}

/*
 * Export the native triangle mesh in the compact VTK XML PolyData format.
 *
 * ParaView opens `.vtp` files directly, so this writer gives the new example a
 * zero-dependency path from the in-memory mesh to a desktop visualization tool.
 */
void write_vtp_surface(
    const SurfaceMesh& mesh,
    const std::filesystem::path& output_path) {
    std::ofstream out = open_output_stream(output_path);

    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    out << "  <PolyData>\n";
    out << "    <Piece NumberOfPoints=\"" << mesh.vertices.size()
        << "\" NumberOfPolys=\"" << mesh.triangles.size() << "\">\n";

    out << "      <Points>\n";
    write_data_array(
        out,
        "Float64",
        nullptr,
        3,
        [&](std::ofstream& stream) {
            for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
                const auto& vertex = mesh.vertices[i];
                stream << vertex[0] << ' ' << vertex[1] << ' ' << vertex[2];
                if (i + 1 != mesh.vertices.size()) {
                    stream << ' ';
                }
            }
        });
    out << "      </Points>\n";

    out << "      <Polys>\n";
    write_data_array(
        out,
        "Int64",
        "connectivity",
        0,
        [&](std::ofstream& stream) {
            for (std::size_t i = 0; i < mesh.triangles.size(); ++i) {
                const auto& tri = mesh.triangles[i];
                stream << tri[0] << ' ' << tri[1] << ' ' << tri[2];
                if (i + 1 != mesh.triangles.size()) {
                    stream << ' ';
                }
            }
        });
    write_data_array(
        out,
        "Int64",
        "offsets",
        0,
        [&](std::ofstream& stream) {
            for (std::size_t i = 0; i < mesh.triangles.size(); ++i) {
                stream << (3 * (i + 1));
                if (i + 1 != mesh.triangles.size()) {
                    stream << ' ';
                }
            }
        });
    out << "      </Polys>\n";

    /*
     * VTK PolyData stores polygon topology under <Polys>, but many readers also
     * accept a cell-types array. Writing it keeps the file explicit and matches
     * the requested triangle type annotation for ParaView inspection.
     */
    out << "      <CellData>\n";
    write_data_array(
        out,
        "Int64",
        "triangle_id",
        0,
        [&](std::ofstream& stream) {
            for (std::size_t i = 0; i < mesh.triangles.size(); ++i) {
                stream << i;
                if (i + 1 != mesh.triangles.size()) {
                    stream << ' ';
                }
            }
        });
    write_data_array(
        out,
        "UInt8",
        "vtk_cell_type",
        0,
        [&](std::ofstream& stream) {
            for (std::size_t i = 0; i < mesh.triangles.size(); ++i) {
                stream << 5;
                if (i + 1 != mesh.triangles.size()) {
                    stream << ' ';
                }
            }
        });
    out << "      </CellData>\n";

    out << "    </Piece>\n";
    out << "  </PolyData>\n";
    out << "</VTKFile>\n";
}

void write_flux_association(
    const FluxAssociation& association,
    Dimension dimension,
    const std::filesystem::path& output_path) {
    std::ofstream out = open_output_stream(output_path);
    out << association.elements.size()
        << (dimension == Dimension::D3 ? " total triangles\n\n" : " total lines\n\n");
    for (const auto& element : association.elements) {
        out << "start id " << (element.element_id + 1) << '\n';
        for (std::size_t i = 0; i < element.voxel_ids.size(); ++i) {
            out << "    " << element.voxel_ids[i] << ' ' << element.scalar_fractions[i] << '\n';
        }
        out << "end id " << (element.element_id + 1) << '\n';
    }
}

}  // namespace isthmus::io
