#include "isthmus/utilities.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace isthmus::utilities {

namespace {

/*
 * This small metadata record captures only the TIFF fields required by the
 * bundled single-phase stack and the narrow TIFF helper tests.
 */
struct TiffPageInfo {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint16_t bits_per_sample = 0;
    std::uint16_t compression = 0;
    std::uint16_t samples_per_pixel = 1;
    std::uint32_t rows_per_strip = 0;
    std::vector<std::uint32_t> strip_offsets;
    std::vector<std::uint32_t> strip_byte_counts;
};

/*
 * This in-memory payload keeps the narrow TIFF data in a regular `(z, y, x)`
 * indexing order for the native volume helpers.
 */
struct BinaryTiffVolume {
    std::array<std::size_t, 3> dims{{0u, 0u, 0u}};
    std::vector<std::uint8_t> values;

    /*
     * Flatten one `(z, y, x)` coordinate triplet into the contiguous backing
     * storage used by the narrow TIFF parser.
     */
    std::size_t flattened_index(
        std::size_t z,
        std::size_t y,
        std::size_t x) const {
        return z * dims[1] * dims[2] + y * dims[2] + x;
    }

    /*
     * Bounds-checked read so the slicing helpers fail clearly on malformed
     * indexing instead of silently wandering off into undefined behavior.
     */
    std::uint8_t at(
        std::size_t z,
        std::size_t y,
        std::size_t x) const {
        if (z >= dims[0] || y >= dims[1] || x >= dims[2]) {
            throw std::out_of_range("TIFF voxel index out of range");
        }
        return values[flattened_index(z, y, x)];
    }
};

/*
 * Read one little-endian 16-bit integer from the in-memory TIFF payload.
 */
std::uint16_t read_u16(
    const std::vector<std::uint8_t>& bytes,
    std::size_t offset) {
    return static_cast<std::uint16_t>(bytes.at(offset)) |
           (static_cast<std::uint16_t>(bytes.at(offset + 1u)) << 8u);
}

/*
 * Read one little-endian 32-bit integer from the in-memory TIFF payload.
 */
std::uint32_t read_u32(
    const std::vector<std::uint8_t>& bytes,
    std::size_t offset) {
    return static_cast<std::uint32_t>(bytes.at(offset)) |
           (static_cast<std::uint32_t>(bytes.at(offset + 1u)) << 8u) |
           (static_cast<std::uint32_t>(bytes.at(offset + 2u)) << 16u) |
           (static_cast<std::uint32_t>(bytes.at(offset + 3u)) << 24u);
}

/*
 * Read a small binary file into memory so the TIFF helper can parse it from a
 * simple byte vector.
 */
std::vector<std::uint8_t> read_binary_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open binary file: " + path.string());
    }

    in.seekg(0, std::ios::end);
    const auto size = static_cast<std::size_t>(in.tellg());
    in.seekg(0, std::ios::beg);

    std::vector<std::uint8_t> bytes(size);
    in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!in) {
        throw std::runtime_error("Failed to read binary file: " + path.string());
    }

    return bytes;
}

/*
 * Read a SHORT/LONG TIFF integer field into a normalized 32-bit vector.
 *
 * The bundled stack and the synthetic helper tests only require integer tag
 * storage, so this helper intentionally keeps the supported TIFF subset small.
 */
std::vector<std::uint32_t> read_tiff_integer_values(
    const std::vector<std::uint8_t>& bytes,
    std::uint16_t type,
    std::uint32_t count,
    std::size_t value_field_offset) {
    const std::size_t item_size = type == 3u ? 2u : 4u;
    if (type != 3u && type != 4u) {
        throw std::runtime_error("Unsupported TIFF integer tag type");
    }

    const std::size_t data_offset =
        (static_cast<std::size_t>(count) * item_size <= 4u)
            ? value_field_offset
            : static_cast<std::size_t>(read_u32(bytes, value_field_offset));

    std::vector<std::uint32_t> values;
    values.reserve(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        const std::size_t item_offset = data_offset + static_cast<std::size_t>(i) * item_size;
        if (type == 3u) {
            values.push_back(read_u16(bytes, item_offset));
        } else {
            values.push_back(read_u32(bytes, item_offset));
        }
    }

    return values;
}

/*
 * Parse one image-file directory from the narrow TIFF subset used by the
 * bundled single-phase workflow and the native utility tests.
 */
TiffPageInfo parse_tiff_page(
    const std::vector<std::uint8_t>& bytes,
    std::size_t ifd_offset) {
    TiffPageInfo page{};
    const auto entry_count = static_cast<std::size_t>(read_u16(bytes, ifd_offset));
    std::size_t entry_offset = ifd_offset + 2u;

    for (std::size_t entry = 0; entry < entry_count; ++entry) {
        const auto tag = read_u16(bytes, entry_offset);
        const auto type = read_u16(bytes, entry_offset + 2u);
        const auto count = read_u32(bytes, entry_offset + 4u);
        const std::size_t value_field_offset = entry_offset + 8u;

        switch (tag) {
        case 256u:
            page.width = read_tiff_integer_values(bytes, type, count, value_field_offset).at(0);
            break;
        case 257u:
            page.height = read_tiff_integer_values(bytes, type, count, value_field_offset).at(0);
            break;
        case 258u:
            page.bits_per_sample = static_cast<std::uint16_t>(
                read_tiff_integer_values(bytes, type, count, value_field_offset).at(0));
            break;
        case 259u:
            page.compression = static_cast<std::uint16_t>(
                read_tiff_integer_values(bytes, type, count, value_field_offset).at(0));
            break;
        case 273u:
            page.strip_offsets = read_tiff_integer_values(bytes, type, count, value_field_offset);
            break;
        case 277u:
            page.samples_per_pixel = static_cast<std::uint16_t>(
                read_tiff_integer_values(bytes, type, count, value_field_offset).at(0));
            break;
        case 278u:
            page.rows_per_strip = read_tiff_integer_values(bytes, type, count, value_field_offset).at(0);
            break;
        case 279u:
            page.strip_byte_counts = read_tiff_integer_values(bytes, type, count, value_field_offset);
            break;
        default:
            break;
        }

        entry_offset += 12u;
    }

    return page;
}

/*
 * Load the full TIFF stack into a compact `(z, y, x)` byte volume so the
 * public helpers can either return all active voxels or cut local slices.
 */
BinaryTiffVolume load_tiff_volume(const std::filesystem::path& tiff_path) {
    const auto bytes = read_binary_file(tiff_path);
    if (bytes.size() < 8u ||
        bytes[0] != 'I' ||
        bytes[1] != 'I' ||
        read_u16(bytes, 2u) != 42u) {
        throw std::runtime_error("Unsupported TIFF header: " + tiff_path.string());
    }

    /*
     * Walk the full page list once so the helper can validate the stack shape
     * before it allocates the contiguous volume payload.
     */
    std::vector<TiffPageInfo> pages;
    std::size_t ifd_offset = read_u32(bytes, 4u);
    while (ifd_offset != 0u) {
        pages.push_back(parse_tiff_page(bytes, ifd_offset));
        const auto entry_count = static_cast<std::size_t>(read_u16(bytes, ifd_offset));
        const std::size_t next_ifd_field = ifd_offset + 2u + entry_count * 12u;
        ifd_offset = read_u32(bytes, next_ifd_field);
    }

    if (pages.empty()) {
        throw std::runtime_error("TIFF stack contains no image pages: " + tiff_path.string());
    }

    const std::uint32_t expected_width = pages.front().width;
    const std::uint32_t expected_height = pages.front().height;

    BinaryTiffVolume volume;
    volume.dims = {pages.size(), expected_height, expected_width};
    volume.values.assign(volume.dims[0] * volume.dims[1] * volume.dims[2], 0u);

    /*
     * Decode each uncompressed strip payload directly into the contiguous
     * volume storage while enforcing a uniform stack shape across pages.
     */
    for (std::size_t page_index = 0; page_index < pages.size(); ++page_index) {
        const auto& page = pages[page_index];
        if (page.width != expected_width ||
            page.height != expected_height ||
            page.bits_per_sample != 8u ||
            page.compression != 1u ||
            page.samples_per_pixel != 1u) {
            throw std::runtime_error("Unsupported TIFF page encoding in: " + tiff_path.string());
        }
        if (page.strip_offsets.size() != page.strip_byte_counts.size()) {
            throw std::runtime_error("Corrupt TIFF strip metadata in: " + tiff_path.string());
        }

        const std::uint32_t rows_per_strip =
            page.rows_per_strip == 0u ? page.height : page.rows_per_strip;
        std::size_t written_rows = 0u;

        for (std::size_t strip = 0; strip < page.strip_offsets.size(); ++strip) {
            const std::size_t strip_offset = page.strip_offsets[strip];
            const std::size_t strip_bytes = page.strip_byte_counts[strip];
            const std::size_t rows_in_strip =
                std::min<std::size_t>(rows_per_strip, page.height - written_rows);
            const std::size_t expected_bytes = rows_in_strip * static_cast<std::size_t>(page.width);
            if (strip_bytes < expected_bytes) {
                throw std::runtime_error("Corrupt TIFF strip payload in: " + tiff_path.string());
            }

            const std::size_t page_offset =
                page_index * volume.dims[1] * volume.dims[2] +
                written_rows * volume.dims[2];
            std::copy_n(
                bytes.begin() + static_cast<std::ptrdiff_t>(strip_offset),
                static_cast<std::ptrdiff_t>(expected_bytes),
                volume.values.begin() + static_cast<std::ptrdiff_t>(page_offset));
            written_rows += rows_in_strip;
        }
    }

    return volume;
}

/*
 * Convert one non-negative floating-point lattice coordinate into a truncated
 * integer lattice index.
 */
std::size_t as_lattice_index(
    double value,
    const char* label) {
    if (value < 0.0) {
        throw std::runtime_error(std::string("Expected non-negative lattice coordinate for ") + label);
    }
    return static_cast<std::size_t>(value);
}

/*
 * Convert a non-negative floating-point lattice length into a truncated
 * integer extent.
 */
std::size_t as_lattice_extent(
    double value,
    const char* label) {
    if (value <= 0.0) {
        throw std::runtime_error(std::string("Expected positive lattice extent for ") + label);
    }
    return static_cast<std::size_t>(value);
}

}  // namespace

VoxelSet load_active_voxels_from_tiff(
    const std::filesystem::path& tiff_path,
    double voxel_size) {
    const auto volume = load_tiff_volume(tiff_path);

    VoxelSet voxels;
    std::size_t voxel_id = 0u;

    /*
     * Store active voxel locations as raw lattice indices multiplied by
     * `voxel_size`, not half-cell-shifted centroids.
     */
    for (std::size_t z = 0; z < volume.dims[0]; ++z) {
        for (std::size_t y = 0; y < volume.dims[1]; ++y) {
            for (std::size_t x = 0; x < volume.dims[2]; ++x) {
                if (volume.at(z, y, x) != 1u) {
                    continue;
                }

                VoxelRecord record;
                record.centroid = {
                    static_cast<double>(z) * voxel_size,
                    static_cast<double>(y) * voxel_size,
                    static_cast<double>(x) * voxel_size
                };
                record.original_id = voxel_id++;
                voxels.voxels.push_back(record);
            }
        }
    }

    return voxels;
}

VoxelSliceResult tiff_slicer(
    const std::filesystem::path& tiff_path,
    double x,
    double y,
    double z,
    double length,
    double voxel_size,
    double lb,
    std::optional<double> height) {
    const auto volume = load_tiff_volume(tiff_path);
    const double effective_height = height.has_value() ? *height : length;

    const std::size_t x_center = as_lattice_index(x, "x");
    const std::size_t y_center = as_lattice_index(y, "y");
    const std::size_t z_start = as_lattice_index(z, "z");
    const std::size_t length_extent = as_lattice_extent(length, "length");
    const std::size_t height_extent = as_lattice_extent(effective_height, "height");
    const std::size_t half_length = length_extent / 2u;

    if (x_center < half_length || y_center < half_length) {
        throw std::runtime_error("Requested TIFF slice would start before the volume origin");
    }

    const std::size_t x_start = x_center - half_length;
    const std::size_t y_start = y_center - half_length;
    const std::size_t x_stop = x_start + length_extent;
    const std::size_t y_stop = y_start + length_extent;
    const std::size_t z_stop = z_start + height_extent;

    if (x_stop > volume.dims[2] ||
        y_stop > volume.dims[1] ||
        z_stop > volume.dims[0]) {
        throw std::runtime_error("Requested TIFF slice exceeds the loaded volume bounds");
    }

    VoxelSliceResult result;
    result.limits = {{
        {-lb * voxel_size, -lb * voxel_size, -lb * voxel_size},
        {(effective_height + lb) * voxel_size,
         (length + lb) * voxel_size,
         (length + lb) * voxel_size}
    }};

    std::size_t voxel_id = 0u;

    /*
     * Emit local coordinates within the cropped block so the result is
     * expressed in the cropped block's own lattice frame.
     */
    for (std::size_t local_z = 0; local_z < height_extent; ++local_z) {
        for (std::size_t local_y = 0; local_y < length_extent; ++local_y) {
            for (std::size_t local_x = 0; local_x < length_extent; ++local_x) {
                if (volume.at(z_start + local_z, y_start + local_y, x_start + local_x) != 1u) {
                    continue;
                }

                VoxelRecord record;
                record.centroid = {
                    static_cast<double>(local_z) * voxel_size,
                    static_cast<double>(local_y) * voxel_size,
                    static_cast<double>(local_x) * voxel_size
                };
                record.original_id = voxel_id++;
                result.voxels.voxels.push_back(record);
            }
        }
    }

    return result;
}

FluxAssociation read_flux_association(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Failed to open flux-association file: " + path.string());
    }

    FluxAssociation association;
    std::string line;
    bool inside_element = false;
    FluxElementOwnership current;

    /*
     * Skip the first header line and then parse the legacy block-based
     * ownership format.
     */
    std::getline(in, line);

    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }

        std::istringstream stream(line);
        std::string keyword;
        stream >> keyword;

        if (keyword == "start") {
            std::string id_label;
            std::size_t one_based_id = 0u;
            if (!(stream >> id_label >> one_based_id) || id_label != "id" || one_based_id == 0u) {
                throw std::runtime_error("Malformed flux-association start block in: " + path.string());
            }
            if (inside_element) {
                throw std::runtime_error("Nested flux-association start block in: " + path.string());
            }

            current = FluxElementOwnership{};
            current.element_id = one_based_id - 1u;
            inside_element = true;
            continue;
        }

        if (keyword == "end") {
            std::string id_label;
            std::size_t one_based_id = 0u;
            if (!(stream >> id_label >> one_based_id) || id_label != "id" || one_based_id == 0u) {
                throw std::runtime_error("Malformed flux-association end block in: " + path.string());
            }
            if (!inside_element || current.element_id + 1u != one_based_id) {
                throw std::runtime_error("Mismatched flux-association end block in: " + path.string());
            }

            const std::set<std::size_t> unique_ids(current.voxel_ids.begin(), current.voxel_ids.end());
            if (unique_ids.size() != current.voxel_ids.size()) {
                throw std::runtime_error("Surface voxel double-assigned within one triangle in: " + path.string());
            }

            association.elements.push_back(current);
            inside_element = false;
            continue;
        }

        if (!inside_element) {
            throw std::runtime_error("Unexpected flux-association payload line in: " + path.string());
        }

        std::istringstream payload(line);
        std::size_t voxel_id = 0u;
        double scalar_fraction = 0.0;
        if (!(payload >> voxel_id >> scalar_fraction)) {
            throw std::runtime_error("Malformed flux-association payload line in: " + path.string());
        }

        current.voxel_ids.push_back(voxel_id);
        current.scalar_fractions.push_back(scalar_fraction);
    }

    if (inside_element) {
        throw std::runtime_error("Unterminated flux-association block in: " + path.string());
    }

    return association;
}

}  // namespace isthmus::utilities
