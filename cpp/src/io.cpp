#include "isthmus/io.hpp"

#include <fstream>

namespace isthmus::io {

void write_sparta_surface(
    const SurfaceMesh& mesh,
    Dimension dimension,
    const std::filesystem::path& output_path) {
    std::ofstream out(output_path);
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

void write_flux_association(
    const FluxAssociation& association,
    Dimension dimension,
    const std::filesystem::path& output_path) {
    std::ofstream out(output_path);
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
