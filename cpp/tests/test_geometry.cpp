#include "test_framework.hpp"

#include <vector>

#include "isthmus/geometry.hpp"

TEST_CASE(test_polygon_area_square) {
    using namespace isthmus::geometry;
    // Protects the shoelace-area helper for the simplest closed polygon case.
    std::vector<Vec2> verts{{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}};
    CHECK_CLOSE(polygon_area(verts), 1.0, 1e-9);
}

TEST_CASE(test_segment_plane_intersection_origin) {
    using namespace isthmus::geometry;
    // Protects the signed-distance crossing logic used during polygon clipping.
    const auto result = segment_plane_intersection(
        {1.0, 1.0, 1.0},
        {-1.0, -1.0, -1.0},
        {0.0, 0.0, 1.0},
        {0.0, 0.0, 0.0},
        1e-6);
    CHECK(!result.p1_inside);
    CHECK(result.p2_inside);
    CHECK_CLOSE(result.intersection[0], 0.0, 1e-9);
    CHECK_CLOSE(result.intersection[1], 0.0, 1e-9);
    CHECK_CLOSE(result.intersection[2], 0.0, 1e-9);
}

TEST_CASE(test_clip_triangle_from_square) {
    using namespace isthmus::geometry;
    // Protects the triangle half-plane clipping loop that later overlap calculations depend on.
    const std::vector<Vec3> subject{{0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {2.0, 2.0, 0.0}, {0.0, 2.0, 0.0}};
    const std::array<Vec3, 3> normals{{
        {-1.0, 0.0, 0.0},
        {0.0, -1.0, 0.0},
        {1.0 / std::sqrt(2.0), 1.0 / std::sqrt(2.0), 0.0}
    }};
    const std::array<Vec3, 3> clip_vertices{{{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}}};
    const auto clipped = clip_polygon_sutherland_hodgman(subject, normals, clip_vertices, 1e-6);
    CHECK(clipped.size() == 3);
}

TEST_CASE(test_orient_polygon_xy) {
    using namespace isthmus::geometry;
    // Protects the rotation step that converts a 3D overlap polygon into a 2D area computation problem.
    const std::vector<Vec3> verts{{0.0, 0.0, 0.0}, {0.0, 2.0, 0.0}, {0.0, 2.0, 4.0}, {0.0, 0.0, 4.0}};
    const auto rotated = orient_polygon_xy(verts, {1.0, 0.0, 0.0});
    CHECK(rotated.size() == 4);
    CHECK_CLOSE(rotated[2][0], -4.0, 1e-9);
    CHECK_CLOSE(rotated[2][1], 2.0, 1e-9);
}
