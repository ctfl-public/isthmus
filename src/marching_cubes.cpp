/*
 * Internal Lewiner marching-cubes surface reconstruction for ISTHMUS.
 *
 * This implementation embeds the Lewiner topology-preserving case logic
 * directly while keeping the public API simple: the caller still
 * receives only vertices and triangles in physical space.
 */
#include "marching_cubes.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include "isthmus/exceptions.hpp"
#include "marching_cubes_luts.hpp"

namespace isthmus::marching_cubes {

namespace {

/*
 * The Lewiner reference uses a tiny epsilon when dividing by interpolated
 * scalar magnitudes and when disambiguating nearly singular internal tests.
 */
constexpr double kFloatEpsilon = 2.2204460492503131e-16;

/*
 * Small helpers keep the low-level numeric operations compact and explicit.
 */
double dabs(double value) {
    return value >= 0.0 ? value : -value;
}

/*
 * Decode base64-encoded lookup tables from the bundled generated header.
 *
 * The large Lewiner LUTs are stored in compact base64 form and decoded once
 * at startup.
 */
std::vector<std::uint8_t> decode_base64(std::string_view text) {
    auto decode_char = [](char c) -> int {
        if (c >= 'A' && c <= 'Z') {
            return c - 'A';
        }
        if (c >= 'a' && c <= 'z') {
            return 26 + (c - 'a');
        }
        if (c >= '0' && c <= '9') {
            return 52 + (c - '0');
        }
        if (c == '+') {
            return 62;
        }
        if (c == '/') {
            return 63;
        }
        if (c == '=') {
            return -2;
        }
        return -1;
    };

    std::vector<std::uint8_t> out;
    int buffer = 0;
    int bits_in_buffer = 0;

    for (const char c : text) {
        const int decoded = decode_char(c);
        if (decoded == -1) {
            continue;
        }
        if (decoded == -2) {
            break;
        }

        buffer = (buffer << 6) | decoded;
        bits_in_buffer += 6;
        while (bits_in_buffer >= 8) {
            bits_in_buffer -= 8;
            out.push_back(static_cast<std::uint8_t>((buffer >> bits_in_buffer) & 0xFF));
        }
    }

    return out;
}

/*
 * `Lut` stores signed bytes and supports 1D, 2D, and 3D indexing with the
 * layout expected by the Lewiner tables.
 */
class Lut {
public:
    /*
     * Construct one decoded lookup table from the generated base64 payload.
     */
    Lut(int dim0, int dim1, int dim2, std::string_view encoded)
        : l0_(dim0),
          l1_(dim1),
          l2_(dim2) {
        const auto decoded = decode_base64(encoded);
        values_.reserve(decoded.size());
        for (const auto byte_value : decoded) {
            values_.push_back(static_cast<std::int8_t>(byte_value));
        }
    }

    /*
     * Construct one small inline table from raw signed-byte values.
     */
    Lut(int dim0, int dim1, int dim2, std::initializer_list<std::int8_t> raw_values)
        : l0_(dim0),
          l1_(dim1),
          l2_(dim2),
          values_(raw_values) {}

    /*
     * Direct 1D indexing into the decoded lookup storage.
     */
    int get1(int i0) const {
        return values_[static_cast<std::size_t>(i0)];
    }

    /*
     * Direct 2D indexing into the decoded lookup storage.
     */
    int get2(int i0, int i1) const {
        return values_[static_cast<std::size_t>(i0 * l1_ + i1)];
    }

    /*
     * Direct 3D indexing into the decoded lookup storage.
     */
    int get3(int i0, int i1, int i2) const {
        return values_[static_cast<std::size_t>(i0 * l1_ * l2_ + i1 * l2_ + i2)];
    }

private:
    int l0_ = 1;
    int l1_ = 1;
    int l2_ = 1;
    std::vector<std::int8_t> values_;
};

/*
 * `LutProvider` keeps the grouped-access shape used by the Lewiner backend.
 *
 * The generated LUT header provides a single X-macro list, which lets the
 * implementation define and initialize every Lewiner table without hand-copying 48
 * individual declarations and constructor arguments.
 */
struct LutProvider {
    /*
     * The three edge-relative tables are defined inline because they belong to
     * the compact backend support data rather than the generated LUT bundle.
     */
    Lut edges_rel_x;
    Lut edges_rel_y;
    Lut edges_rel_z;

    /*
     * Every remaining table is bundled in the generated header and decoded at
     * construction time.
     */
#define ISTHMUS_DECLARE_LUT(name, rank, dim0, dim1, dim2, data) Lut name;
    ISTHMUS_LEWINER_LUT_LIST(ISTHMUS_DECLARE_LUT)
#undef ISTHMUS_DECLARE_LUT

    /*
     * The constructor wires all encoded tables into typed `Lut` objects once.
     */
    LutProvider()
        : edges_rel_x(
              12,
              2,
              1,
              {
                  0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1,
                  1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0
              }),
          edges_rel_y(
              12,
              2,
              1,
              {
                  0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1,
                  1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1
              }),
          edges_rel_z(
              12,
              2,
              1,
              {
                  0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                  1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1
              })
#define ISTHMUS_INIT_LUT(name, rank, dim0, dim1, dim2, data) \
        , name(dim0, dim1, dim2, data)
          ISTHMUS_LEWINER_LUT_LIST(ISTHMUS_INIT_LUT)
#undef ISTHMUS_INIT_LUT
    {}
};

/*
 * Lazily create the lookup provider once so repeated calls do not keep
 * re-decoding the full Lewiner table bundle.
 */
const LutProvider& lut_provider() {
    static const LutProvider provider{};
    return provider;
}

/*
 * `Cell` owns the cube-local scalar state, the face-layer vertex cache, and
 * the output arrays.
 */
class Cell {
public:
    /*
     * The dimensions refer to the scalar volume, not the number of marching
     * cubes.
     */
    Cell(const LutProvider& luts, int nx, int ny, int nz)
        : luts_(luts),
          nx_(nx),
          ny_(ny),
          nz_(nz),
          face_layer1_(static_cast<std::size_t>(nx * ny * 4), -1),
          face_layer2_(static_cast<std::size_t>(nx * ny * 4), -1),
          active_face_layer_(&face_layer1_) {}

    /*
     * Swap the active z-layers and clear the one that will receive new vertex
     * ids for the next marching slice.
     */
    void new_z_value() {
        std::swap(face_layer1_, face_layer2_);
        std::fill(face_layer2_.begin(), face_layer2_.end(), -1);
        active_face_layer_ = &face_layer1_;
    }

    /*
     * Load one cube from the scalar volume and compute its 8-bit sign mask.
     */
    void set_cube(
        double isovalue,
        int x,
        int y,
        int z,
        int step,
        double v0,
        double v1,
        double v2,
        double v3,
        double v4,
        double v5,
        double v6,
        double v7) {
        x_ = x;
        y_ = y;
        z_ = z;
        step_ = step;

        v0_ = v0 - isovalue;
        v1_ = v1 - isovalue;
        v2_ = v2 - isovalue;
        v3_ = v3 - isovalue;
        v4_ = v4 - isovalue;
        v5_ = v5 - isovalue;
        v6_ = v6 - isovalue;
        v7_ = v7 - isovalue;

        index_ = 0;
        if (v0_ > 0.0) { index_ += 1; }
        if (v1_ > 0.0) { index_ += 2; }
        if (v2_ > 0.0) { index_ += 4; }
        if (v3_ > 0.0) { index_ += 8; }
        if (v4_ > 0.0) { index_ += 16; }
        if (v5_ > 0.0) { index_ += 32; }
        if (v6_ > 0.0) { index_ += 64; }
        if (v7_ > 0.0) { index_ += 128; }

        v12_calculated_ = false;
    }

    /*
     * Expose the raw cube-sign mask so the traversal loop can look up the
     * Lewiner case and configuration ids.
     */
    int index() const {
        return index_;
    }

    /*
     * Emit triangles from a 2D LUT indexed by `config`.
     */
    void add_triangles(const Lut& lut, int lut_index, int triangle_count) {
        prepare_for_adding_triangles();
        for (int tri = 0; tri < triangle_count; ++tri) {
            for (int vertex = 0; vertex < 3; ++vertex) {
                const int edge_index = lut.get2(lut_index, tri * 3 + vertex);
                add_face_from_edge_index(edge_index);
            }
        }
    }

    /*
     * Emit triangles from a 3D LUT indexed by `config` and `subconfig`.
     */
    void add_triangles2(const Lut& lut, int lut_index, int lut_index2, int triangle_count) {
        prepare_for_adding_triangles();
        for (int tri = 0; tri < triangle_count; ++tri) {
            for (int vertex = 0; vertex < 3; ++vertex) {
                const int edge_index = lut.get3(lut_index, lut_index2, tri * 3 + vertex);
                add_face_from_edge_index(edge_index);
            }
        }
    }

    /*
     * Export the final physical-space vertex array after index-to-domain
     * conversion has been applied.
     */
    std::vector<std::array<double, 3>> physical_vertices(
        const DomainConfig& domain,
        const std::array<double, 3>& cell_lengths) const {
        std::vector<std::array<double, 3>> out;
        out.reserve(vertices_.size());

        for (const auto& vertex : vertices_) {
            out.push_back({
                domain.limits[0][0] + vertex[0] * cell_lengths[0],
                domain.limits[0][1] + vertex[1] * cell_lengths[1],
                domain.limits[0][2] + vertex[2] * cell_lengths[2]
            });
        }

        return out;
    }

    /*
     * Export flattened face indices as triangle triples with the backend's
     * expected descent-oriented winding.
     */
    std::vector<std::array<std::size_t, 3>> triangles() const {
        std::vector<std::array<std::size_t, 3>> out;
        out.reserve(faces_.size() / 3U);

        for (std::size_t i = 0; i + 2 < faces_.size(); i += 3U) {
            out.push_back({
                static_cast<std::size_t>(faces_[i]),
                static_cast<std::size_t>(faces_[i + 2U]),
                static_cast<std::size_t>(faces_[i + 1U])
            });
        }

        return out;
    }

    /*
     * The following accessors are used by the Lewiner ambiguity tests.
     */
    double v0() const { return v0_; }
    double v1() const { return v1_; }
    double v2() const { return v2_; }
    double v3() const { return v3_; }
    double v4() const { return v4_; }
    double v5() const { return v5_; }
    double v6() const { return v6_; }
    double v7() const { return v7_; }

private:
    /*
     * Record one vertex and keep normals/values arrays in sync with it. The
     * current public API does not expose normals or values, but the internal
     * storage preserves the Lewiner backend structure for later expansion.
     *
     * Vertices are stored in double precision. The reference implementation
     * used float here, which quantized nearly-coincident vertices onto the
     * same float32 value and produced zero-area triangles and ~1e-10 m edges
     * in production meshes.
     */
    int add_vertex(double x, double y, double z) {
        vertices_.push_back({x, y, z});
        normals_.push_back({0.0F, 0.0F, 0.0F});
        values_.push_back(0.0F);
        return static_cast<int>(vertices_.size()) - 1;
    }

    /*
     * Accumulate one gradient contribution onto an already-created vertex.
     */
    void add_gradient(int vertex_index, float gx, float gy, float gz) {
        normals_[static_cast<std::size_t>(vertex_index)][0] += gx;
        normals_[static_cast<std::size_t>(vertex_index)][1] += gy;
        normals_[static_cast<std::size_t>(vertex_index)][2] += gz;
    }

    /*
     * Reuse one cube-corner gradient through the weighted edge interpolation
     * used by the Lewiner implementation.
     */
    void add_gradient_from_index(int vertex_index, int corner_index, float strength) {
        add_gradient(
            vertex_index,
            static_cast<float>(vg_[static_cast<std::size_t>(corner_index) * 3U + 0U] * strength),
            static_cast<float>(vg_[static_cast<std::size_t>(corner_index) * 3U + 1U] * strength),
            static_cast<float>(vg_[static_cast<std::size_t>(corner_index) * 3U + 2U] * strength));
    }

    /*
     * Store one face index and update the Lewiner `values` accumulator.
     */
    void add_face(int vertex_index) {
        faces_.push_back(vertex_index);
        values_[static_cast<std::size_t>(vertex_index)] =
            std::max(values_[static_cast<std::size_t>(vertex_index)], static_cast<float>(vmax_));
    }

    /*
     * The face-layer cache stores four reusable edge slots per cell. This is
     * the mechanism the reference implementation uses to avoid duplicate mesh
     * vertices while marching through neighboring cubes.
     */
    int get_index_in_facelayer(int vertex_on_edge) {
        int i = nx_ * y_ + x_;
        int j = 0;
        std::vector<int>* face_layer = nullptr;

        if (vertex_on_edge < 8) {
            if (vertex_on_edge < 4) {
                face_layer = &face_layer1_;
            } else {
                vertex_on_edge -= 4;
                face_layer = &face_layer2_;
            }

            if (vertex_on_edge == 1) {
                i += step_;
                j = 1;
            } else if (vertex_on_edge == 2) {
                i += nx_ * step_;
            } else if (vertex_on_edge == 3) {
                j = 1;
            }
        } else if (vertex_on_edge < 12) {
            face_layer = &face_layer1_;
            j = 2;

            if (vertex_on_edge == 9) {
                i += step_;
            } else if (vertex_on_edge == 10) {
                i += nx_ * step_ + step_;
            } else if (vertex_on_edge == 11) {
                i += nx_ * step_;
            }
        } else {
            face_layer = &face_layer1_;
            j = 3;
        }

        active_face_layer_ = face_layer;
        return 4 * i + j;
    }

    /*
     * Prepare the cube-local scalar and gradient scratch arrays before a case
     * adds triangles. This follows the reference implementation line-for-line.
     */
    void prepare_for_adding_triangles() {
        vv_[0] = v0_;
        vv_[1] = v1_;
        vv_[2] = v3_;
        vv_[3] = v2_;
        vv_[4] = v4_;
        vv_[5] = v5_;
        vv_[6] = v7_;
        vv_[7] = v6_;

        double vmin = 0.0;
        double vmax = 0.0;
        for (double value : vv_) {
            vmax = std::max(vmax, value);
            vmin = std::min(vmin, value);
        }
        vmax_ = vmax - vmin;

        vg_[0] = v0_ - v1_;  vg_[1] = v0_ - v3_;  vg_[2] = v0_ - v4_;
        vg_[3] = v0_ - v1_;  vg_[4] = v1_ - v2_;  vg_[5] = v1_ - v5_;
        vg_[6] = v3_ - v2_;  vg_[7] = v1_ - v2_;  vg_[8] = v2_ - v6_;
        vg_[9] = v3_ - v2_;  vg_[10] = v0_ - v3_; vg_[11] = v3_ - v7_;
        vg_[12] = v4_ - v5_; vg_[13] = v4_ - v7_; vg_[14] = v0_ - v4_;
        vg_[15] = v4_ - v5_; vg_[16] = v5_ - v6_; vg_[17] = v1_ - v5_;
        vg_[18] = v7_ - v6_; vg_[19] = v5_ - v6_; vg_[20] = v2_ - v6_;
        vg_[21] = v7_ - v6_; vg_[22] = v4_ - v7_; vg_[23] = v3_ - v7_;
    }

    /*
     * Some ambiguous Lewiner cases introduce a center vertex. The calculation
     * is a weighted center-of-mass interpolation of all cube corners.
     */
    void calculate_center_vertex() {
        const double w0 = 1.0 / (kFloatEpsilon + dabs(v0_));
        const double w1 = 1.0 / (kFloatEpsilon + dabs(v1_));
        const double w2 = 1.0 / (kFloatEpsilon + dabs(v2_));
        const double w3 = 1.0 / (kFloatEpsilon + dabs(v3_));
        const double w4 = 1.0 / (kFloatEpsilon + dabs(v4_));
        const double w5 = 1.0 / (kFloatEpsilon + dabs(v5_));
        const double w6 = 1.0 / (kFloatEpsilon + dabs(v6_));
        const double w7 = 1.0 / (kFloatEpsilon + dabs(v7_));

        double fx = 0.0;
        double fy = 0.0;
        double fz = 0.0;
        double ff = 0.0;

        fx += 0.0 * w0; fy += 0.0 * w0; fz += 0.0 * w0; ff += w0;
        fx += 1.0 * w1; fy += 0.0 * w1; fz += 0.0 * w1; ff += w1;
        fx += 1.0 * w2; fy += 1.0 * w2; fz += 0.0 * w2; ff += w2;
        fx += 0.0 * w3; fy += 1.0 * w3; fz += 0.0 * w3; ff += w3;
        fx += 0.0 * w4; fy += 0.0 * w4; fz += 1.0 * w4; ff += w4;
        fx += 1.0 * w5; fy += 0.0 * w5; fz += 1.0 * w5; ff += w5;
        fx += 1.0 * w6; fy += 1.0 * w6; fz += 1.0 * w6; ff += w6;
        fx += 0.0 * w7; fy += 1.0 * w7; fz += 1.0 * w7; ff += w7;

        const double step_value = static_cast<double>(step_);
        v12_x_ = static_cast<double>(x_) + step_value * fx / ff;
        v12_y_ = static_cast<double>(y_) + step_value * fy / ff;
        v12_z_ = static_cast<double>(z_) + step_value * fz / ff;

        v12_xg_ =
            w0 * vg_[0] + w1 * vg_[3] + w2 * vg_[6] + w3 * vg_[9] +
            w4 * vg_[12] + w5 * vg_[15] + w6 * vg_[18] + w7 * vg_[21];
        v12_yg_ =
            w0 * vg_[1] + w1 * vg_[4] + w2 * vg_[7] + w3 * vg_[10] +
            w4 * vg_[13] + w5 * vg_[16] + w6 * vg_[19] + w7 * vg_[22];
        v12_zg_ =
            w0 * vg_[2] + w1 * vg_[5] + w2 * vg_[8] + w3 * vg_[11] +
            w4 * vg_[14] + w5 * vg_[17] + w6 * vg_[20] + w7 * vg_[23];

        v12_calculated_ = true;
    }

    /*
     * Add one face vertex, either by reusing a cached edge vertex or by
     * interpolating a new one on the fly.
     */
    void add_face_from_edge_index(int vertex_on_edge) {
        const int face_index = get_index_in_facelayer(vertex_on_edge);
        int vertex_index = (*active_face_layer_)[static_cast<std::size_t>(face_index)];

        if (vertex_on_edge == 12) {
            if (!v12_calculated_) {
                calculate_center_vertex();
            }

            if (vertex_index >= 0) {
                add_face(vertex_index);
                add_gradient(vertex_index, v12_xg_, v12_yg_, v12_zg_);
            } else {
                vertex_index = add_vertex(v12_x_, v12_y_, v12_z_);
                (*active_face_layer_)[static_cast<std::size_t>(face_index)] = vertex_index;
                add_face(vertex_index);
                add_gradient(vertex_index, v12_xg_, v12_yg_, v12_zg_);
            }
            return;
        }

        const int dx1 = luts_.edges_rel_x.get2(vertex_on_edge, 0);
        const int dx2 = luts_.edges_rel_x.get2(vertex_on_edge, 1);
        const int dy1 = luts_.edges_rel_y.get2(vertex_on_edge, 0);
        const int dy2 = luts_.edges_rel_y.get2(vertex_on_edge, 1);
        const int dz1 = luts_.edges_rel_z.get2(vertex_on_edge, 0);
        const int dz2 = luts_.edges_rel_z.get2(vertex_on_edge, 1);

        const int corner_index1 = dz1 * 4 + dy1 * 2 + dx1;
        const int corner_index2 = dz2 * 4 + dy2 * 2 + dx2;

        const double strength1 = 1.0 / (kFloatEpsilon + dabs(vv_[static_cast<std::size_t>(corner_index1)]));
        const double strength2 = 1.0 / (kFloatEpsilon + dabs(vv_[static_cast<std::size_t>(corner_index2)]));

        if (vertex_index >= 0) {
            add_face(vertex_index);
            add_gradient_from_index(vertex_index, corner_index1, static_cast<float>(strength1));
            add_gradient_from_index(vertex_index, corner_index2, static_cast<float>(strength2));
            return;
        }

        double fx = 0.0;
        double fy = 0.0;
        double fz = 0.0;
        double ff = 0.0;

        fx += static_cast<double>(dx1) * strength1;
        fy += static_cast<double>(dy1) * strength1;
        fz += static_cast<double>(dz1) * strength1;
        ff += strength1;

        fx += static_cast<double>(dx2) * strength2;
        fy += static_cast<double>(dy2) * strength2;
        fz += static_cast<double>(dz2) * strength2;
        ff += strength2;

        const double step_value = static_cast<double>(step_);
        vertex_index = add_vertex(
            static_cast<double>(x_) + step_value * fx / ff,
            static_cast<double>(y_) + step_value * fy / ff,
            static_cast<double>(z_) + step_value * fz / ff);

        (*active_face_layer_)[static_cast<std::size_t>(face_index)] = vertex_index;
        add_face(vertex_index);
        add_gradient_from_index(vertex_index, corner_index1, static_cast<float>(strength1));
        add_gradient_from_index(vertex_index, corner_index2, static_cast<float>(strength2));
    }

    const LutProvider& luts_;
    int nx_ = 0;
    int ny_ = 0;
    int nz_ = 0;

    int x_ = 0;
    int y_ = 0;
    int z_ = 0;
    int step_ = 1;

    double v0_ = 0.0;
    double v1_ = 0.0;
    double v2_ = 0.0;
    double v3_ = 0.0;
    double v4_ = 0.0;
    double v5_ = 0.0;
    double v6_ = 0.0;
    double v7_ = 0.0;

    std::array<double, 8> vv_{};
    std::array<double, 24> vg_{};
    double vmax_ = 0.0;

    double v12_x_ = 0.0;
    double v12_y_ = 0.0;
    double v12_z_ = 0.0;
    float v12_xg_ = 0.0F;
    float v12_yg_ = 0.0F;
    float v12_zg_ = 0.0F;
    bool v12_calculated_ = false;

    int index_ = 0;

    std::vector<int> face_layer1_;
    std::vector<int> face_layer2_;
    std::vector<int>* active_face_layer_ = nullptr;

    std::vector<std::array<double, 3>> vertices_;
    std::vector<std::array<float, 3>> normals_;
    std::vector<float> values_;
    std::vector<int> faces_;
};

/*
 * Convert the flattened x-fastest corner field into the [z][y][x]
 * access pattern expected by the Lewiner traversal.
 */
double corner_value(
    const std::vector<double>& corner_fill_fractions,
    const std::array<std::size_t, kMaxDims>& corner_dims,
    std::size_t x,
    std::size_t y,
    std::size_t z) {
    return corner_fill_fractions[x + corner_dims[0] * (y + corner_dims[1] * z)];
}

/*
 * Keep the ambiguity predicates separate from the case switch so the control
 * flow stays readable.
 */
int test_face(const Cell& cell, int face) {
    int abs_face = face;
    if (face < 0) {
        abs_face *= -1;
    }

    double a = 0.0;
    double b = 0.0;
    double c = 0.0;
    double d = 0.0;

    if (abs_face == 1) {
        a = cell.v0(); b = cell.v4(); c = cell.v5(); d = cell.v1();
    } else if (abs_face == 2) {
        a = cell.v1(); b = cell.v5(); c = cell.v6(); d = cell.v2();
    } else if (abs_face == 3) {
        a = cell.v2(); b = cell.v6(); c = cell.v7(); d = cell.v3();
    } else if (abs_face == 4) {
        a = cell.v3(); b = cell.v7(); c = cell.v4(); d = cell.v0();
    } else if (abs_face == 5) {
        a = cell.v0(); b = cell.v3(); c = cell.v2(); d = cell.v1();
    } else if (abs_face == 6) {
        a = cell.v4(); b = cell.v7(); c = cell.v6(); d = cell.v5();
    }

    const double ac_minus_bd = a * c - b * d;
    if (ac_minus_bd > -kFloatEpsilon && ac_minus_bd < kFloatEpsilon) {
        return face >= 0;
    }

    return face * a * ac_minus_bd >= 0.0;
}

/*
 * Internal ambiguity resolution follows the Lewiner topology rules closely
 * because this is where the topology-preserving behavior comes from.
 */
int test_internal(const Cell& cell, const LutProvider& luts, int mc_case, int config, int subconfig, int s) {
    double t = 0.0;
    double at = 0.0;
    double bt = 0.0;
    double ct = 0.0;
    double dt = 0.0;
    double a = 0.0;
    double b = 0.0;
    int test = 0;
    int edge = -1;

    if (mc_case == 4 || mc_case == 10) {
        a = (cell.v4() - cell.v0()) * (cell.v6() - cell.v2()) -
            (cell.v7() - cell.v3()) * (cell.v5() - cell.v1());
        b = cell.v2() * (cell.v4() - cell.v0()) + cell.v0() * (cell.v6() - cell.v2()) -
            cell.v1() * (cell.v7() - cell.v3()) - cell.v3() * (cell.v5() - cell.v1());
        t = -b / (2.0 * a + kFloatEpsilon);
        if (t < 0.0 || t > 1.0) {
            return s > 0;
        }

        at = cell.v0() + (cell.v4() - cell.v0()) * t;
        bt = cell.v3() + (cell.v7() - cell.v3()) * t;
        ct = cell.v2() + (cell.v6() - cell.v2()) * t;
        dt = cell.v1() + (cell.v5() - cell.v1()) * t;
    } else if (mc_case == 6 || mc_case == 7 || mc_case == 12 || mc_case == 13) {
        if (mc_case == 6) {
            edge = luts.TEST6.get2(config, 2);
        } else if (mc_case == 7) {
            edge = luts.TEST7.get2(config, 4);
        } else if (mc_case == 12) {
            edge = luts.TEST12.get2(config, 3);
        } else {
            edge = luts.TILING13_5_1.get3(config, subconfig, 0);
        }

        if (edge == 0) {
            t = cell.v0() / (cell.v0() - cell.v1() + kFloatEpsilon);
            at = 0.0;
            bt = cell.v3() + (cell.v2() - cell.v3()) * t;
            ct = cell.v7() + (cell.v6() - cell.v7()) * t;
            dt = cell.v4() + (cell.v5() - cell.v4()) * t;
        } else if (edge == 1) {
            t = cell.v1() / (cell.v1() - cell.v2() + kFloatEpsilon);
            at = 0.0;
            bt = cell.v0() + (cell.v3() - cell.v0()) * t;
            ct = cell.v4() + (cell.v7() - cell.v4()) * t;
            dt = cell.v5() + (cell.v6() - cell.v5()) * t;
        } else if (edge == 2) {
            t = cell.v2() / (cell.v2() - cell.v3() + kFloatEpsilon);
            at = 0.0;
            bt = cell.v1() + (cell.v0() - cell.v1()) * t;
            ct = cell.v5() + (cell.v4() - cell.v5()) * t;
            dt = cell.v6() + (cell.v7() - cell.v6()) * t;
        } else if (edge == 3) {
            t = cell.v3() / (cell.v3() - cell.v0() + kFloatEpsilon);
            at = 0.0;
            bt = cell.v2() + (cell.v1() - cell.v2()) * t;
            ct = cell.v6() + (cell.v5() - cell.v6()) * t;
            dt = cell.v7() + (cell.v4() - cell.v7()) * t;
        } else if (edge == 4) {
            t = cell.v4() / (cell.v4() - cell.v5() + kFloatEpsilon);
            at = 0.0;
            bt = cell.v7() + (cell.v6() - cell.v7()) * t;
            ct = cell.v3() + (cell.v2() - cell.v3()) * t;
            dt = cell.v0() + (cell.v1() - cell.v0()) * t;
        } else if (edge == 5) {
            t = cell.v5() / (cell.v5() - cell.v6() + kFloatEpsilon);
            at = 0.0;
            bt = cell.v4() + (cell.v7() - cell.v4()) * t;
            ct = cell.v0() + (cell.v3() - cell.v0()) * t;
            dt = cell.v1() + (cell.v2() - cell.v1()) * t;
        } else if (edge == 6) {
            t = cell.v6() / (cell.v6() - cell.v7() + kFloatEpsilon);
            at = 0.0;
            bt = cell.v5() + (cell.v4() - cell.v5()) * t;
            ct = cell.v1() + (cell.v0() - cell.v1()) * t;
            dt = cell.v2() + (cell.v3() - cell.v2()) * t;
        } else if (edge == 7) {
            t = cell.v7() / (cell.v7() - cell.v4() + kFloatEpsilon);
            at = 0.0;
            bt = cell.v6() + (cell.v5() - cell.v6()) * t;
            ct = cell.v2() + (cell.v1() - cell.v2()) * t;
            dt = cell.v3() + (cell.v0() - cell.v3()) * t;
        } else if (edge == 8) {
            t = cell.v0() / (cell.v0() - cell.v4() + kFloatEpsilon);
            at = 0.0;
            bt = cell.v3() + (cell.v7() - cell.v3()) * t;
            ct = cell.v2() + (cell.v6() - cell.v2()) * t;
            dt = cell.v1() + (cell.v5() - cell.v1()) * t;
        } else if (edge == 9) {
            t = cell.v1() / (cell.v1() - cell.v5() + kFloatEpsilon);
            at = 0.0;
            bt = cell.v0() + (cell.v4() - cell.v0()) * t;
            ct = cell.v3() + (cell.v7() - cell.v3()) * t;
            dt = cell.v2() + (cell.v6() - cell.v2()) * t;
        } else if (edge == 10) {
            t = cell.v2() / (cell.v2() - cell.v6() + kFloatEpsilon);
            at = 0.0;
            bt = cell.v1() + (cell.v5() - cell.v1()) * t;
            ct = cell.v0() + (cell.v4() - cell.v0()) * t;
            dt = cell.v3() + (cell.v7() - cell.v3()) * t;
        } else if (edge == 11) {
            t = cell.v3() / (cell.v3() - cell.v7() + kFloatEpsilon);
            at = 0.0;
            bt = cell.v2() + (cell.v6() - cell.v2()) * t;
            ct = cell.v1() + (cell.v5() - cell.v1()) * t;
            dt = cell.v0() + (cell.v4() - cell.v0()) * t;
        } else {
            throw InvalidInputError("Invalid Lewiner internal edge index");
        }
    } else {
        throw InvalidInputError("Invalid Lewiner ambiguous marching-cubes case");
    }

    if (at >= 0.0) { test += 1; }
    if (bt >= 0.0) { test += 2; }
    if (ct >= 0.0) { test += 4; }
    if (dt >= 0.0) { test += 8; }

    if (test == 0) { return s > 0; }
    if (test == 1) { return s > 0; }
    if (test == 2) { return s > 0; }
    if (test == 3) { return s > 0; }
    if (test == 4) { return s > 0; }
    if (test == 5) {
        if (at * ct - bt * dt < kFloatEpsilon) { return s > 0; }
    }
    if (test == 6) { return s > 0; }
    if (test == 7) { return s < 0; }
    if (test == 8) { return s > 0; }
    if (test == 9) { return s > 0; }
    if (test == 10) {
        if (at * ct - bt * dt >= kFloatEpsilon) { return s > 0; }
    }
    if (test == 11) { return s < 0; }
    if (test == 12) { return s > 0; }
    if (test == 13) { return s < 0; }
    if (test == 14) { return s < 0; }
    return s < 0;
}

/*
 * The large case dispatcher is the heart of the Lewiner topology rules. It
 * stays direct rather than overly simplified so ambiguous-case behavior
 * remains stable.
 */
void the_big_switch(const LutProvider& luts, Cell& cell, int mc_case, int config) {
    int subconfig = 0;

    if (mc_case == 1) {
        cell.add_triangles(luts.TILING1, config, 1);
    } else if (mc_case == 2) {
        cell.add_triangles(luts.TILING2, config, 2);
    } else if (mc_case == 3) {
        if (test_face(cell, luts.TEST3.get1(config))) {
            cell.add_triangles(luts.TILING3_2, config, 4);
        } else {
            cell.add_triangles(luts.TILING3_1, config, 2);
        }
    } else if (mc_case == 4) {
        if (test_internal(cell, luts, mc_case, config, subconfig, luts.TEST4.get1(config))) {
            cell.add_triangles(luts.TILING4_1, config, 2);
        } else {
            cell.add_triangles(luts.TILING4_2, config, 6);
        }
    } else if (mc_case == 5) {
        cell.add_triangles(luts.TILING5, config, 3);
    } else if (mc_case == 6) {
        if (test_face(cell, luts.TEST6.get2(config, 0))) {
            cell.add_triangles(luts.TILING6_2, config, 5);
        } else if (test_internal(cell, luts, mc_case, config, subconfig, luts.TEST6.get2(config, 1))) {
            cell.add_triangles(luts.TILING6_1_1, config, 3);
        } else {
            cell.add_triangles(luts.TILING6_1_2, config, 9);
        }
    } else if (mc_case == 7) {
        if (test_face(cell, luts.TEST7.get2(config, 0))) { subconfig += 1; }
        if (test_face(cell, luts.TEST7.get2(config, 1))) { subconfig += 2; }
        if (test_face(cell, luts.TEST7.get2(config, 2))) { subconfig += 4; }

        if (subconfig == 0) {
            cell.add_triangles(luts.TILING7_1, config, 3);
        } else if (subconfig == 1) {
            cell.add_triangles2(luts.TILING7_2, config, 0, 5);
        } else if (subconfig == 2) {
            cell.add_triangles2(luts.TILING7_2, config, 1, 5);
        } else if (subconfig == 3) {
            cell.add_triangles2(luts.TILING7_3, config, 0, 9);
        } else if (subconfig == 4) {
            cell.add_triangles2(luts.TILING7_2, config, 2, 5);
        } else if (subconfig == 5) {
            cell.add_triangles2(luts.TILING7_3, config, 1, 9);
        } else if (subconfig == 6) {
            cell.add_triangles2(luts.TILING7_3, config, 2, 9);
        } else if (test_internal(cell, luts, mc_case, config, subconfig, luts.TEST7.get2(config, 3))) {
            cell.add_triangles(luts.TILING7_4_2, config, 9);
        } else {
            cell.add_triangles(luts.TILING7_4_1, config, 5);
        }
    } else if (mc_case == 8) {
        cell.add_triangles(luts.TILING8, config, 2);
    } else if (mc_case == 9) {
        cell.add_triangles(luts.TILING9, config, 4);
    } else if (mc_case == 10) {
        if (test_face(cell, luts.TEST10.get2(config, 0))) {
            if (test_face(cell, luts.TEST10.get2(config, 1))) {
                cell.add_triangles(luts.TILING10_1_1_, config, 4);
            } else {
                cell.add_triangles(luts.TILING10_2, config, 8);
            }
        } else if (test_face(cell, luts.TEST10.get2(config, 1))) {
            cell.add_triangles(luts.TILING10_2_, config, 8);
        } else if (test_internal(cell, luts, mc_case, config, subconfig, luts.TEST10.get2(config, 2))) {
            cell.add_triangles(luts.TILING10_1_1, config, 4);
        } else {
            cell.add_triangles(luts.TILING10_1_2, config, 8);
        }
    } else if (mc_case == 11) {
        cell.add_triangles(luts.TILING11, config, 4);
    } else if (mc_case == 12) {
        if (test_face(cell, luts.TEST12.get2(config, 0))) {
            if (test_face(cell, luts.TEST12.get2(config, 1))) {
                cell.add_triangles(luts.TILING12_1_1_, config, 4);
            } else {
                cell.add_triangles(luts.TILING12_2, config, 8);
            }
        } else if (test_face(cell, luts.TEST12.get2(config, 1))) {
            cell.add_triangles(luts.TILING12_2_, config, 8);
        } else if (test_internal(cell, luts, mc_case, config, subconfig, luts.TEST12.get2(config, 2))) {
            cell.add_triangles(luts.TILING12_1_1, config, 4);
        } else {
            cell.add_triangles(luts.TILING12_1_2, config, 8);
        }
    } else if (mc_case == 13) {
        if (test_face(cell, luts.TEST13.get2(config, 0))) { subconfig += 1; }
        if (test_face(cell, luts.TEST13.get2(config, 1))) { subconfig += 2; }
        if (test_face(cell, luts.TEST13.get2(config, 2))) { subconfig += 4; }
        if (test_face(cell, luts.TEST13.get2(config, 3))) { subconfig += 8; }
        if (test_face(cell, luts.TEST13.get2(config, 4))) { subconfig += 16; }
        if (test_face(cell, luts.TEST13.get2(config, 5))) { subconfig += 32; }
        subconfig = luts.SUBCONFIG13.get1(subconfig);

        if (subconfig == 0) {
            cell.add_triangles(luts.TILING13_1, config, 4);
        } else if (subconfig == 1) {
            cell.add_triangles2(luts.TILING13_2, config, 0, 6);
        } else if (subconfig == 2) {
            cell.add_triangles2(luts.TILING13_2, config, 1, 6);
        } else if (subconfig == 3) {
            cell.add_triangles2(luts.TILING13_2, config, 2, 6);
        } else if (subconfig == 4) {
            cell.add_triangles2(luts.TILING13_2, config, 3, 6);
        } else if (subconfig == 5) {
            cell.add_triangles2(luts.TILING13_2, config, 4, 6);
        } else if (subconfig == 6) {
            cell.add_triangles2(luts.TILING13_2, config, 5, 6);
        } else if (subconfig == 7) {
            cell.add_triangles2(luts.TILING13_3, config, 0, 10);
        } else if (subconfig == 8) {
            cell.add_triangles2(luts.TILING13_3, config, 1, 10);
        } else if (subconfig == 9) {
            cell.add_triangles2(luts.TILING13_3, config, 2, 10);
        } else if (subconfig == 10) {
            cell.add_triangles2(luts.TILING13_3, config, 3, 10);
        } else if (subconfig == 11) {
            cell.add_triangles2(luts.TILING13_3, config, 4, 10);
        } else if (subconfig == 12) {
            cell.add_triangles2(luts.TILING13_3, config, 5, 10);
        } else if (subconfig == 13) {
            cell.add_triangles2(luts.TILING13_3, config, 6, 10);
        } else if (subconfig == 14) {
            cell.add_triangles2(luts.TILING13_3, config, 7, 10);
        } else if (subconfig == 15) {
            cell.add_triangles2(luts.TILING13_3, config, 8, 10);
        } else if (subconfig == 16) {
            cell.add_triangles2(luts.TILING13_3, config, 9, 10);
        } else if (subconfig == 17) {
            cell.add_triangles2(luts.TILING13_3, config, 10, 10);
        } else if (subconfig == 18) {
            cell.add_triangles2(luts.TILING13_3, config, 11, 10);
        } else if (subconfig == 19) {
            cell.add_triangles2(luts.TILING13_4, config, 0, 12);
        } else if (subconfig == 20) {
            cell.add_triangles2(luts.TILING13_4, config, 1, 12);
        } else if (subconfig == 21) {
            cell.add_triangles2(luts.TILING13_4, config, 2, 12);
        } else if (subconfig == 22) {
            cell.add_triangles2(luts.TILING13_4, config, 3, 12);
        } else if (subconfig == 23) {
            subconfig = 0;
            if (test_internal(cell, luts, mc_case, config, subconfig, luts.TEST13.get2(config, 6))) {
                cell.add_triangles2(luts.TILING13_5_1, config, 0, 6);
            } else {
                cell.add_triangles2(luts.TILING13_5_2, config, 0, 10);
            }
        } else if (subconfig == 24) {
            subconfig = 1;
            if (test_internal(cell, luts, mc_case, config, subconfig, luts.TEST13.get2(config, 6))) {
                cell.add_triangles2(luts.TILING13_5_1, config, 1, 6);
            } else {
                cell.add_triangles2(luts.TILING13_5_2, config, 1, 10);
            }
        } else if (subconfig == 25) {
            subconfig = 2;
            if (test_internal(cell, luts, mc_case, config, subconfig, luts.TEST13.get2(config, 6))) {
                cell.add_triangles2(luts.TILING13_5_1, config, 2, 6);
            } else {
                cell.add_triangles2(luts.TILING13_5_2, config, 2, 10);
            }
        } else if (subconfig == 26) {
            subconfig = 3;
            if (test_internal(cell, luts, mc_case, config, subconfig, luts.TEST13.get2(config, 6))) {
                cell.add_triangles2(luts.TILING13_5_1, config, 3, 6);
            } else {
                cell.add_triangles2(luts.TILING13_5_2, config, 3, 10);
            }
        } else if (subconfig == 27) {
            cell.add_triangles2(luts.TILING13_3_, config, 0, 10);
        } else if (subconfig == 28) {
            cell.add_triangles2(luts.TILING13_3_, config, 1, 10);
        } else if (subconfig == 29) {
            cell.add_triangles2(luts.TILING13_3_, config, 2, 10);
        } else if (subconfig == 30) {
            cell.add_triangles2(luts.TILING13_3_, config, 3, 10);
        } else if (subconfig == 31) {
            cell.add_triangles2(luts.TILING13_3_, config, 4, 10);
        } else if (subconfig == 32) {
            cell.add_triangles2(luts.TILING13_3_, config, 5, 10);
        } else if (subconfig == 33) {
            cell.add_triangles2(luts.TILING13_3_, config, 6, 10);
        } else if (subconfig == 34) {
            cell.add_triangles2(luts.TILING13_3_, config, 7, 10);
        } else if (subconfig == 35) {
            cell.add_triangles2(luts.TILING13_3_, config, 8, 10);
        } else if (subconfig == 36) {
            cell.add_triangles2(luts.TILING13_3_, config, 9, 10);
        } else if (subconfig == 37) {
            cell.add_triangles2(luts.TILING13_3_, config, 10, 10);
        } else if (subconfig == 38) {
            cell.add_triangles2(luts.TILING13_3_, config, 11, 10);
        } else if (subconfig == 39) {
            cell.add_triangles2(luts.TILING13_2_, config, 0, 6);
        } else if (subconfig == 40) {
            cell.add_triangles2(luts.TILING13_2_, config, 1, 6);
        } else if (subconfig == 41) {
            cell.add_triangles2(luts.TILING13_2_, config, 2, 6);
        } else if (subconfig == 42) {
            cell.add_triangles2(luts.TILING13_2_, config, 3, 6);
        } else if (subconfig == 43) {
            cell.add_triangles2(luts.TILING13_2_, config, 4, 6);
        } else if (subconfig == 44) {
            cell.add_triangles2(luts.TILING13_2_, config, 5, 6);
        } else if (subconfig == 45) {
            cell.add_triangles(luts.TILING13_1_, config, 4);
        } else {
            throw InvalidInputError("Impossible Lewiner case-13 subconfiguration");
        }
    } else if (mc_case == 14) {
        cell.add_triangles(luts.TILING14, config, 4);
    }
}

}  // namespace

SurfaceMesh extract_surface_mesh_3d(
    const DomainConfig& domain,
    const std::vector<double>& corner_fill_fractions,
    const std::array<std::size_t, kMaxDims>& corner_dims,
    double iso_value) {
    /*
     * The Lewiner backend is still a strict 3D-only stage, so invalid
     * dimensions are rejected before any LUT or traversal work.
     */
    if (domain.dimension != Dimension::D3) {
        throw InvalidInputError("3D surface extraction requires a 3D domain");
    }

    /*
     * Marching cubes requires at least one actual cube, which means two corner
     * samples in each direction.
     */
    if (corner_dims[0] < 2 || corner_dims[1] < 2 || corner_dims[2] < 2) {
        throw InvalidInputError("Corner grid is too small for 3D surface extraction");
    }

    /*
        * The corner field is stored in a flattened x-fastest layout, so
     * its total size must match the supplied lattice dimensions exactly.
     */
    if (corner_fill_fractions.size() != corner_dims[0] * corner_dims[1] * corner_dims[2]) {
        throw InvalidInputError("Corner field size does not match the supplied corner dimensions");
    }

    /*
     * Convert one marching-grid cell in index space into physical-space lengths
     * so the final mesh can be returned directly in caller coordinates.
     */
    const std::array<double, 3> cell_lengths{{
        (domain.limits[1][0] - domain.limits[0][0]) / static_cast<double>(domain.cell_counts[0]),
        (domain.limits[1][1] - domain.limits[0][1]) / static_cast<double>(domain.cell_counts[1]),
        (domain.limits[1][2] - domain.limits[0][2]) / static_cast<double>(domain.cell_counts[2])
    }};

    const int nx = static_cast<int>(corner_dims[0]);
    const int ny = static_cast<int>(corner_dims[1]);
    const int nz = static_cast<int>(corner_dims[2]);

    /*
     * The Lewiner traversal marches the corner volume in [z][y][x] order and
     * reuses vertices through a per-layer face cache.
     */
    const auto& luts = lut_provider();
    Cell cell(luts, nx, ny, nz);

    for (int z = 0; z < nz - 1; ++z) {
        cell.new_z_value();
        for (int y = 0; y < ny - 1; ++y) {
            for (int x = 0; x < nx - 1; ++x) {
                cell.set_cube(
                    iso_value,
                    x,
                    y,
                    z,
                    1,
                    corner_value(corner_fill_fractions, corner_dims, static_cast<std::size_t>(x), static_cast<std::size_t>(y), static_cast<std::size_t>(z)),
                    corner_value(corner_fill_fractions, corner_dims, static_cast<std::size_t>(x + 1), static_cast<std::size_t>(y), static_cast<std::size_t>(z)),
                    corner_value(corner_fill_fractions, corner_dims, static_cast<std::size_t>(x + 1), static_cast<std::size_t>(y + 1), static_cast<std::size_t>(z)),
                    corner_value(corner_fill_fractions, corner_dims, static_cast<std::size_t>(x), static_cast<std::size_t>(y + 1), static_cast<std::size_t>(z)),
                    corner_value(corner_fill_fractions, corner_dims, static_cast<std::size_t>(x), static_cast<std::size_t>(y), static_cast<std::size_t>(z + 1)),
                    corner_value(corner_fill_fractions, corner_dims, static_cast<std::size_t>(x + 1), static_cast<std::size_t>(y), static_cast<std::size_t>(z + 1)),
                    corner_value(corner_fill_fractions, corner_dims, static_cast<std::size_t>(x + 1), static_cast<std::size_t>(y + 1), static_cast<std::size_t>(z + 1)),
                    corner_value(corner_fill_fractions, corner_dims, static_cast<std::size_t>(x), static_cast<std::size_t>(y + 1), static_cast<std::size_t>(z + 1)));

                /*
                 * `CASES` maps the raw 8-bit cube mask into Lewiner’s reduced
                 * case id plus a configuration index for that case.
                 */
                const int mc_case = luts.CASES.get2(cell.index(), 0);
                if (mc_case > 0) {
                    const int config = luts.CASES.get2(cell.index(), 1);
                    the_big_switch(luts, cell, mc_case, config);
                }
            }
        }
    }

    /*
     * The backend stores vertices in marching-grid index space during the
     * traversal and converts them only once here for the final result object.
     */
    SurfaceMesh mesh;
    mesh.vertices = cell.physical_vertices(domain, cell_lengths);
    mesh.triangles = cell.triangles();
    return mesh;
}

}  // namespace isthmus::marching_cubes
