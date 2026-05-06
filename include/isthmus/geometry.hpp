// Standalone geometry helpers reused by motion mapping and future flux mapping.
#pragma once

#include <array>
#include <cstddef>
#include <vector>

namespace isthmus::geometry {

using Vec2 = std::array<double, 2>;
using Vec3 = std::array<double, 3>;

// Result of clipping a segment against a plane-aligned half-space test.
struct SegmentPlaneIntersection {
    bool p1_inside = false;
    bool p2_inside = false;
    Vec3 intersection{{0.0, 0.0, 0.0}};
};

double dot(const Vec2& a, const Vec2& b);
double dot(const Vec3& a, const Vec3& b);
Vec3 cross(const Vec3& a, const Vec3& b);
double norm(const Vec2& a);
double norm(const Vec3& a);
Vec2 subtract(const Vec2& a, const Vec2& b);
Vec3 subtract(const Vec3& a, const Vec3& b);
Vec2 add(const Vec2& a, const Vec2& b);
Vec3 add(const Vec3& a, const Vec3& b);
Vec2 scale(const Vec2& a, double s);
Vec3 scale(const Vec3& a, double s);

SegmentPlaneIntersection segment_plane_intersection(
    const Vec3& p1,
    const Vec3& p2,
    const Vec3& normal,
    const Vec3& point_on_plane,
    double epsilon);

std::vector<Vec3> clip_polygon_sutherland_hodgman(
    const std::vector<Vec3>& subject,
    const std::array<Vec3, 3>& clip_plane_normals,
    const std::array<Vec3, 3>& clip_vertices,
    double epsilon);

std::vector<Vec2> orient_polygon_xy(
    const std::vector<Vec3>& vertices,
    const Vec3& normal);

double polygon_area(const std::vector<Vec2>& vertices);
double triangle_area(const std::array<Vec3, 3>& vertices);
std::array<std::size_t, 2> longest_triangle_side(const std::array<Vec3, 3>& vertices);
double intersection_length(
    const std::array<Vec2, 2>& projected_segment,
    const std::array<Vec2, 2>& base_segment);

}  // namespace isthmus::geometry
