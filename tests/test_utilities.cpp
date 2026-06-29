#include "test_framework.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <vector>

#include "isthmus/io.hpp"
#include "isthmus/utilities.hpp"

namespace {

/*
 * Write one little-endian 16-bit integer into a byte vector so the tests can
 * build a tiny synthetic TIFF file without depending on any external library.
 */
void append_u16(
    std::vector<std::uint8_t>& bytes,
    std::uint16_t value) {
    bytes.push_back(static_cast<std::uint8_t>(value & 0x00ffu));
    bytes.push_back(static_cast<std::uint8_t>((value >> 8u) & 0x00ffu));
}

/*
 * Write one little-endian 32-bit integer into a byte vector.
 */
void append_u32(
    std::vector<std::uint8_t>& bytes,
    std::uint32_t value) {
    bytes.push_back(static_cast<std::uint8_t>(value & 0x000000ffu));
    bytes.push_back(static_cast<std::uint8_t>((value >> 8u) & 0x000000ffu));
    bytes.push_back(static_cast<std::uint8_t>((value >> 16u) & 0x000000ffu));
    bytes.push_back(static_cast<std::uint8_t>((value >> 24u) & 0x000000ffu));
}

/*
 * Patch one previously reserved 32-bit field after the test has computed the
 * final byte offsets for the synthetic TIFF layout.
 */
void overwrite_u32(
    std::vector<std::uint8_t>& bytes,
    std::size_t offset,
    std::uint32_t value) {
    bytes.at(offset + 0u) = static_cast<std::uint8_t>(value & 0x000000ffu);
    bytes.at(offset + 1u) = static_cast<std::uint8_t>((value >> 8u) & 0x000000ffu);
    bytes.at(offset + 2u) = static_cast<std::uint8_t>((value >> 16u) & 0x000000ffu);
    bytes.at(offset + 3u) = static_cast<std::uint8_t>((value >> 24u) & 0x000000ffu);
}

/*
 * Emit one TIFF IFD entry in the narrow subset that the native utilities
 * support.
 */
void append_ifd_entry(
    std::vector<std::uint8_t>& bytes,
    std::uint16_t tag,
    std::uint16_t type,
    std::uint32_t count,
    std::uint32_t value) {
    append_u16(bytes, tag);
    append_u16(bytes, type);
    append_u32(bytes, count);
    append_u32(bytes, value);
}

/*
 * Write a tiny 2-page uncompressed 8-bit grayscale TIFF stack with one active
 * voxel in each slice.
 */
std::filesystem::path write_test_tiff_stack(std::uint8_t second_active_value = 1u) {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "isthmus_test_stack.tif";

    const std::uint16_t entry_count = 8u;
    const std::size_t ifd_size = 2u + static_cast<std::size_t>(entry_count) * 12u + 4u;
    const std::uint32_t first_ifd_offset = 8u;
    const std::uint32_t second_ifd_offset = first_ifd_offset + static_cast<std::uint32_t>(ifd_size);
    const std::uint32_t first_pixels_offset = second_ifd_offset + static_cast<std::uint32_t>(ifd_size);
    const std::uint32_t second_pixels_offset = first_pixels_offset + 4u;

    std::vector<std::uint8_t> bytes;
    bytes.reserve(second_pixels_offset + 4u);

    /*
     * TIFF header:
     *   byte order = II
     *   magic = 42
     *   first IFD at byte 8
     */
    bytes.push_back('I');
    bytes.push_back('I');
    append_u16(bytes, 42u);
    append_u32(bytes, first_ifd_offset);

    /*
     * First 2x2 image page with one active voxel at local `(0, 0)`.
     */
    append_u16(bytes, entry_count);
    append_ifd_entry(bytes, 256u, 4u, 1u, 2u);
    append_ifd_entry(bytes, 257u, 4u, 1u, 2u);
    append_ifd_entry(bytes, 258u, 3u, 1u, 8u);
    append_ifd_entry(bytes, 259u, 3u, 1u, 1u);
    append_ifd_entry(bytes, 273u, 4u, 1u, first_pixels_offset);
    append_ifd_entry(bytes, 277u, 3u, 1u, 1u);
    append_ifd_entry(bytes, 278u, 4u, 1u, 2u);
    append_ifd_entry(bytes, 279u, 4u, 1u, 4u);
    append_u32(bytes, second_ifd_offset);

    /*
     * Second 2x2 image page with one active voxel at local `(1, 1)`.
     */
    append_u16(bytes, entry_count);
    append_ifd_entry(bytes, 256u, 4u, 1u, 2u);
    append_ifd_entry(bytes, 257u, 4u, 1u, 2u);
    append_ifd_entry(bytes, 258u, 3u, 1u, 8u);
    append_ifd_entry(bytes, 259u, 3u, 1u, 1u);
    append_ifd_entry(bytes, 273u, 4u, 1u, second_pixels_offset);
    append_ifd_entry(bytes, 277u, 3u, 1u, 1u);
    append_ifd_entry(bytes, 278u, 4u, 1u, 2u);
    append_ifd_entry(bytes, 279u, 4u, 1u, 4u);
    append_u32(bytes, 0u);

    /*
     * Page payloads are raw row-major grayscale pixels.
     */
    bytes.push_back(1u);
    bytes.push_back(0u);
    bytes.push_back(0u);
    bytes.push_back(0u);

    bytes.push_back(0u);
    bytes.push_back(0u);
    bytes.push_back(0u);
    bytes.push_back(second_active_value);

    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    out.close();
    return path;
}

}  // namespace

TEST_CASE(test_load_active_voxels_from_tiff_reads_narrow_stack) {
    using namespace isthmus;

    /*
     * Case:
     * Load a tiny hand-written 2-page TIFF stack through the public utilities
     * helper.
     *
     * Sketch:
     *   slice 0: active at (0, 0)
     *   slice 1: active at (1, 1)
     *
     * Expected outcome:
     * The loader should emit exactly two active voxels in the native `(z, y,
     * x)` physical coordinate convention.
     */
    const auto path = write_test_tiff_stack();
    const auto voxels = utilities::load_active_voxels_from_tiff(path, 0.5);

    CHECK(voxels.voxels.size() == 2u);
    CHECK_CLOSE(voxels.voxels[0].centroid[0], 0.0, 1e-12);
    CHECK_CLOSE(voxels.voxels[0].centroid[1], 0.0, 1e-12);
    CHECK_CLOSE(voxels.voxels[0].centroid[2], 0.0, 1e-12);
    CHECK_CLOSE(voxels.voxels[1].centroid[0], 0.5, 1e-12);
    CHECK_CLOSE(voxels.voxels[1].centroid[1], 0.5, 1e-12);
    CHECK_CLOSE(voxels.voxels[1].centroid[2], 0.5, 1e-12);

    std::filesystem::remove(path);
}

TEST_CASE(test_load_labeled_voxels_from_tiff_preserves_nonzero_values) {
    using namespace isthmus;

    /*
     * Case:
     * Load a tiny hand-written TIFF stack where occupied voxels use labels 1
     * and 2 rather than a strictly binary mask.
     *
     * Expected outcome:
     * The label-preserving loader should emit both nonzero voxels and store
     * their grayscale labels as material tags.
     */
    const auto path = write_test_tiff_stack(2u);
    const auto labeled = utilities::load_labeled_voxels_from_tiff(path, 0.5);

    CHECK(labeled.dims[0] == 2u);
    CHECK(labeled.dims[1] == 2u);
    CHECK(labeled.dims[2] == 2u);
    CHECK(labeled.voxels.voxels.size() == 2u);
    CHECK(labeled.voxels.voxels[0].material_tag.has_value());
    CHECK(labeled.voxels.voxels[1].material_tag.has_value());
    CHECK(labeled.voxels.voxels[0].material_tag.value_or("") == "1");
    CHECK(labeled.voxels.voxels[1].material_tag.value_or("") == "2");
    CHECK_CLOSE(labeled.voxels.voxels[1].centroid[0], 0.5, 1e-12);
    CHECK_CLOSE(labeled.voxels.voxels[1].centroid[1], 0.5, 1e-12);
    CHECK_CLOSE(labeled.voxels.voxels[1].centroid[2], 0.5, 1e-12);

    std::filesystem::remove(path);
}

TEST_CASE(test_tiff_slicer_matches_local_slice_convention) {
    using namespace isthmus;

    /*
     * Case:
     * Slice the tiny synthetic TIFF stack around its full 2x2 footprint using
     * the native TIFF slicer helper.
     *
     * Expected outcome:
     * The returned limits should match the intended local-box layout and the
     * active voxels should be reported in local slice coordinates.
     */
    const auto path = write_test_tiff_stack();
    const auto slice = utilities::tiff_slicer(path, 1.0, 1.0, 0.0, 2.0, 0.5, 1.0, 2.0);

    CHECK(slice.voxels.voxels.size() == 2u);
    CHECK_CLOSE(slice.limits[0][0], -0.5, 1e-12);
    CHECK_CLOSE(slice.limits[0][1], -0.5, 1e-12);
    CHECK_CLOSE(slice.limits[0][2], -0.5, 1e-12);
    CHECK_CLOSE(slice.limits[1][0], 1.5, 1e-12);
    CHECK_CLOSE(slice.limits[1][1], 1.5, 1e-12);
    CHECK_CLOSE(slice.limits[1][2], 1.5, 1e-12);

    CHECK_CLOSE(slice.voxels.voxels[0].centroid[0], 0.0, 1e-12);
    CHECK_CLOSE(slice.voxels.voxels[0].centroid[1], 0.0, 1e-12);
    CHECK_CLOSE(slice.voxels.voxels[0].centroid[2], 0.0, 1e-12);
    CHECK_CLOSE(slice.voxels.voxels[1].centroid[0], 0.5, 1e-12);
    CHECK_CLOSE(slice.voxels.voxels[1].centroid[1], 0.5, 1e-12);
    CHECK_CLOSE(slice.voxels.voxels[1].centroid[2], 0.5, 1e-12);

    std::filesystem::remove(path);
}

TEST_CASE(test_read_flux_association_round_trips_legacy_writer_output) {
    using namespace isthmus;

    /*
     * Case:
     * Serialize a compact in-memory association using the legacy writer and
     * then parse it back through the new native utility helper.
     *
     * Expected outcome:
     * The parser should recover both populated and empty ownership records
     * without changing ids, voxel ownership lists, or scalar fractions.
     */
    FluxAssociation association;
    association.elements = {
        FluxElementOwnership{0u, {3u, 5u}, {0.25, 0.75}},
        FluxElementOwnership{1u, {}, {}},
        FluxElementOwnership{2u, {9u}, {1.0}}
    };

    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "isthmus_test_flux_association_roundtrip.dat";
    std::filesystem::remove(path);

    io::write_flux_association(association, Dimension::D3, path);
    const auto parsed = utilities::read_flux_association(path);

    CHECK(parsed.elements.size() == 3u);
    CHECK(parsed.elements[0].element_id == 0u);
    CHECK(parsed.elements[0].voxel_ids.size() == 2u);
    CHECK(parsed.elements[0].voxel_ids[0] == 3u);
    CHECK(parsed.elements[0].voxel_ids[1] == 5u);
    CHECK_CLOSE(parsed.elements[0].scalar_fractions[0], 0.25, 1e-12);
    CHECK_CLOSE(parsed.elements[0].scalar_fractions[1], 0.75, 1e-12);
    CHECK(parsed.elements[1].element_id == 1u);
    CHECK(parsed.elements[1].voxel_ids.empty());
    CHECK(parsed.elements[1].scalar_fractions.empty());
    CHECK(parsed.elements[2].element_id == 2u);
    CHECK(parsed.elements[2].voxel_ids.size() == 1u);
    CHECK(parsed.elements[2].voxel_ids[0] == 9u);
    CHECK_CLOSE(parsed.elements[2].scalar_fractions[0], 1.0, 1e-12);

    std::filesystem::remove(path);
}
