#include "test_framework.hpp"

#include <vector>

#include "isthmus/geometry.hpp"

TEST_CASE(test_polygon_area_square) {
    using namespace isthmus::geometry;
    /*
     * Case:
     * Measure the area of an axis-aligned unit square represented as a closed
     * 2D polygon.
     *
     * Sketch:
     *   (0,1) o-----o (1,1)
     *         |     |
     *         |     |
     *   (0,0) o-----o (1,0)
     *
     * Expected outcome:
     * The shoelace-area helper should return exactly 1.0 for this simplest
     * baseline polygon.
     */
    std::vector<Vec2> verts{{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}};
    CHECK_CLOSE(polygon_area(verts), 1.0, 1e-9);
}

TEST_CASE(test_segment_plane_intersection_origin) {
    using namespace isthmus::geometry;
    /*
     * Case:
     * Intersect a segment that crosses the z=0 plane diagonally through the
     * origin, with one endpoint on each side of the plane.
     *
     * Sketch:
     *   p1 (+,+,+)
     *       \
     *        \
     *         o  intersection at origin on z = 0
     *        /
     *       /
     *   p2 (-,-,-)
     *
     * Expected outcome:
     * The helper should report which endpoint lies inside the clipping half
     * space and compute the intersection point exactly at the origin.
     */
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
    /*
     * Case:
     * Clip a square polygon against the three half-planes that define a right
     * triangle in the same plane.
     *
     * Sketch:
     *   square input           clipped result
     *   o------o              o
     *   |      |              |\
     *   |      |      ->      | \
     *   o------o              o--o
     *
     * Expected outcome:
     * Sutherland-Hodgman clipping should reduce the square to a triangle with
     * exactly three output vertices.
     */
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
    /*
     * Case:
     * Rotate a rectangle that currently lies in the yz-plane so it becomes a
     * polygon expressed in the xy-plane for downstream area calculations.
     *
     * Sketch:
     *   before: yz plane       after: xy plane
     *      z ^                    y ^
     *        |                      |
     *        o--- y                 o--- x
     *
     *   The rectangle keeps its shape but changes orientation.
     *
     * Expected outcome:
     * The helper should preserve the vertex count and place the rotated points
     * at the expected coordinates in the target plane.
     */
    const std::vector<Vec3> verts{{0.0, 0.0, 0.0}, {0.0, 2.0, 0.0}, {0.0, 2.0, 4.0}, {0.0, 0.0, 4.0}};
    const auto rotated = orient_polygon_xy(verts, {1.0, 0.0, 0.0});
    CHECK(rotated.size() == 4);
    CHECK_CLOSE(rotated[2][0], -4.0, 1e-9);
    CHECK_CLOSE(rotated[2][1], 2.0, 1e-9);
}
