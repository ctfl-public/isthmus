/*
 * Geometric kernels used by the native ISTHMUS implementation.
 *
 * These functions support two later stages of the overall algorithm:
 * surface reconstruction and flux mapping. The current code already uses them
 * in tests and keeps them independent of the voxel/grid classes so that the
 * geometric reasoning stays easy to verify on its own.
 */
#include "isthmus/geometry.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace isthmus::geometry {

double dot(const Vec2& a, const Vec2& b) {
    return a[0] * b[0] + a[1] * b[1];
}

double dot(const Vec3& a, const Vec3& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

Vec3 cross(const Vec3& a, const Vec3& b) {
    return {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    };
}

double norm(const Vec2& a) {
    return std::sqrt(dot(a, a));
}

double norm(const Vec3& a) {
    return std::sqrt(dot(a, a));
}

Vec2 subtract(const Vec2& a, const Vec2& b) {
    return {a[0] - b[0], a[1] - b[1]};
}

Vec3 subtract(const Vec3& a, const Vec3& b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

Vec2 add(const Vec2& a, const Vec2& b) {
    return {a[0] + b[0], a[1] + b[1]};
}

Vec3 add(const Vec3& a, const Vec3& b) {
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

Vec2 scale(const Vec2& a, double s) {
    return {a[0] * s, a[1] * s};
}

Vec3 scale(const Vec3& a, double s) {
    return {a[0] * s, a[1] * s, a[2] * s};
}

static Vec3 normalize(const Vec3& a) {
    const double len = norm(a);
    if (len == 0.0) {
        throw std::runtime_error("Cannot normalize zero-length vector");
    }
    return scale(a, 1.0 / len);
}

/*
 * Classify the segment endpoints relative to a plane and, when the segment
 * crosses the plane, compute the crossing point by linear interpolation.
 *
 * Signed distance is the key quantity here: a negative or near-zero distance
 * means the point is inside the retained half-space for clipping.
 */
SegmentPlaneIntersection segment_plane_intersection(
    const Vec3& p1,
    const Vec3& p2,
    const Vec3& normal,
    const Vec3& point_on_plane,
    double epsilon) {
    SegmentPlaneIntersection result{};
    const double d1 = dot(subtract(p1, point_on_plane), normal);
    const double d2 = dot(subtract(p2, point_on_plane), normal);

    result.p1_inside = d1 < epsilon;
    result.p2_inside = d2 < epsilon;

    if (static_cast<int>(result.p1_inside) + static_cast<int>(result.p2_inside) == 1) {
        if (std::abs(d1) < epsilon) {
            result.intersection = p1;
        } else if (std::abs(d2) < epsilon) {
            result.intersection = p2;
        } else {
            const double frac = std::abs(d1) / (std::abs(d1) + std::abs(d2));
            result.intersection = add(p1, scale(subtract(p2, p1), frac));
        }
    }

    return result;
}

/*
 * Clip a polygon against the three half-planes that define a triangle.
 *
 * The projected voxel face is the subject polygon. Each pass keeps only the
 * part of that polygon that remains inside one triangle edge half-space. After
 * three passes, any surviving polygon is exactly the overlap region between the
 * projected face and the triangle.
 */
std::vector<Vec3> clip_polygon_sutherland_hodgman(
    const std::vector<Vec3>& subject,
    const std::array<Vec3, 3>& clip_plane_normals,
    const std::array<Vec3, 3>& clip_vertices,
    double epsilon) {
    std::vector<Vec3> in_points = subject;
    for (std::size_t i = 0; i < 3; ++i) {
        std::vector<Vec3> out_points;
        for (std::size_t j = 0; j < in_points.size(); ++j) {
            const Vec3& p1 = in_points[(j + in_points.size() - 1) % in_points.size()];
            const Vec3& p2 = in_points[j];
            const auto intersection =
                segment_plane_intersection(p1, p2, clip_plane_normals[i], clip_vertices[i], epsilon);

            if (intersection.p2_inside) {
                if (!intersection.p1_inside) {
                    out_points.push_back(intersection.intersection);
                }
                out_points.push_back(p2);
            } else if (intersection.p1_inside) {
                out_points.push_back(intersection.intersection);
            }
        }
        in_points = std::move(out_points);
        if (in_points.empty()) {
            break;
        }
    }

    // Clipping can create duplicate vertices when an edge lands exactly on a clipping boundary.
    std::vector<Vec3> unique_points;
    for (const auto& candidate : in_points) {
        const bool duplicate = std::any_of(unique_points.begin(), unique_points.end(), [&](const Vec3& existing) {
            return std::abs(existing[0] - candidate[0]) < epsilon &&
                   std::abs(existing[1] - candidate[1]) < epsilon &&
                   std::abs(existing[2] - candidate[2]) < epsilon;
        });
        if (!duplicate) {
            unique_points.push_back(candidate);
        }
    }
    return unique_points;
}

/*
 * Rotate a 3D polygon into a 2D frame so its area can be measured with the
 * shoelace formula.
 *
 * If the polygon normal is already nearly aligned with the z-axis, the x-y
 * coordinates can be used directly. Otherwise, the polygon is rotated so its
 * normal points along +z and the z coordinate can be discarded.
 */
std::vector<Vec2> orient_polygon_xy(
    const std::vector<Vec3>& vertices,
    const Vec3& normal) {
    const double theta = std::acos(std::clamp(normal[2], -1.0, 1.0));
    constexpr double epsilon = 1e-4;
    std::vector<Vec2> out;
    out.reserve(vertices.size());

    if (theta < epsilon || std::abs(M_PI - theta) < epsilon) {
        for (const auto& v : vertices) {
            out.push_back({v[0], v[1]});
        }
        return out;
    }

    Vec3 axis = cross(normal, {0.0, 0.0, 1.0});
    axis = normalize(axis);

    const double ct = std::cos(theta);
    const double st = std::sin(theta);

    for (const auto& v : vertices) {
        const double x =
            (axis[0] * axis[0] * (1.0 - ct) + ct) * v[0] +
            (axis[0] * axis[1] * (1.0 - ct) - axis[2] * st) * v[1] +
            (axis[0] * axis[2] * (1.0 - ct) + axis[1] * st) * v[2];
        const double y =
            (axis[1] * axis[0] * (1.0 - ct) + axis[2] * st) * v[0] +
            (axis[1] * axis[1] * (1.0 - ct) + ct) * v[1] +
            (axis[1] * axis[2] * (1.0 - ct) - axis[0] * st) * v[2];
        out.push_back({x, y});
    }
    return out;
}

double polygon_area(const std::vector<Vec2>& vertices) {
    double area = 0.0;
    for (std::size_t i = 0; i < vertices.size(); ++i) {
        const auto& p1 = vertices[(i + vertices.size() - 1) % vertices.size()];
        const auto& p2 = vertices[i];
        area += (p1[1] + p2[1]) * (p1[0] - p2[0]);
    }
    return std::abs(area * 0.5);
}

double triangle_area(const std::array<Vec3, 3>& vertices) {
    const double a = norm(subtract(vertices[2], vertices[1]));
    const double b = norm(subtract(vertices[1], vertices[0]));
    const double c = norm(subtract(vertices[0], vertices[2]));
    const double s = (a + b + c) * 0.5;
    return std::sqrt(std::max(0.0, s * (s - a) * (s - b) * (s - c)));
}

std::array<std::size_t, 2> longest_triangle_side(const std::array<Vec3, 3>& vertices) {
    const std::array<double, 3> lengths{
        norm(subtract(vertices[1], vertices[0])),
        norm(subtract(vertices[2], vertices[1])),
        norm(subtract(vertices[0], vertices[2]))
    };
    const auto it = std::max_element(lengths.begin(), lengths.end());
    const std::size_t idx = static_cast<std::size_t>(std::distance(lengths.begin(), it));
    if (idx == 0) {
        return {1, 0};
    }
    if (idx == 1) {
        return {2, 1};
    }
    return {0, 2};
}

/*
 * Measure how much of a projected voxel edge overlaps a base segment.
 *
 * The base segment is parameterized from t = 0 at its first endpoint to t = 1
 * at its second endpoint. The projected segment is clipped into that parameter
 * range and the remaining fraction is converted back into a physical length.
 */
double intersection_length(
    const std::array<Vec2, 2>& projected_segment,
    const std::array<Vec2, 2>& base_segment) {
    const Vec2 diff = subtract(base_segment[1], base_segment[0]);
    const double base_len = norm(diff);
    if (base_len < 1e-20) {
        return 0.0;
    }

    const std::size_t idx = std::abs(diff[0]) > std::abs(diff[1]) ? 0 : 1;
    double t_a = (projected_segment[0][idx] - base_segment[0][idx]) / diff[idx];
    double t_b = (projected_segment[1][idx] - base_segment[0][idx]) / diff[idx];
    if ((t_a < 0.0 && t_b < 0.0) || (t_a > 1.0 && t_b > 1.0)) {
        return 0.0;
    }

    t_a = std::clamp(t_a, 0.0, 1.0);
    t_b = std::clamp(t_b, 0.0, 1.0);
    return std::abs(t_b - t_a) * base_len;
}

}  // namespace isthmus::geometry
