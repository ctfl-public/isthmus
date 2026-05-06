import unittest
import numpy as np
try:
    from numba import cuda
    from geometry_gpu import *
    from gpu_test_wrappers import *
    cuda_available = cuda.is_available()
except ImportError:
    cuda_available = False
from mapping import *
from geometry import polygon_area, orient_polygon_xy, segment_plane_intersection, \
    clip_sh, get_intersection_area


class TestPolygonArea(unittest.TestCase):
    # Test a square with side length of 1 that should return an area of 1.
    def test_polygon_area(self):
        verts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        expected_area = 1.0
        self.assertAlmostEqual(polygon_area(verts), expected_area, places=5)
    
    # GPU Test of n squares with side length of 1 that should return area of 1.
    def test_polygon_area_gpu1(self):
        if not cuda_available:
            self.skipTest("Skipping GPU tests (Numba or CUDA not available)")

        n = int(1e6)
        verts = np.tile(np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32), (n, 1, 1))
        n_verts = 4
        expected_area = np.full((n,), 1.0, dtype=np.float32)

        verts_gpu = cuda.to_device(verts)
        area_gpu = cuda.device_array(verts.shape[0], dtype=np.float32)

        threads_per_block = 256
        blocks_per_grid = (verts.shape[0] + threads_per_block - 1) // threads_per_block

        polygon_area_gpu_kernel[blocks_per_grid, threads_per_block](verts_gpu, n_verts, area_gpu)
        
        computed_area = area_gpu.copy_to_host()
        np.testing.assert_almost_equal(computed_area, expected_area)

    # This test to make sure that summation in kernal is n_verts dependent and not on size of verts (since it could be pointer with different size).
    def test_polygon_area_gpu2(self):
        if not cuda_available:
            self.skipTest("Skipping GPU tests (Numba or CUDA not available)")

        n = int(1e6)
        verts = np.tile(np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0]], dtype=np.float32), (n, 1, 1))
        n_verts = 3
        expected_area = np.full((n,), 0.5, dtype=np.float32)

        verts_gpu = cuda.to_device(verts)
        area_gpu = cuda.device_array(verts.shape[0], dtype=np.float32)

        threads_per_block = 256
        blocks_per_grid = (verts.shape[0] + threads_per_block - 1) // threads_per_block

        polygon_area_gpu_kernel[blocks_per_grid, threads_per_block](verts_gpu, n_verts, area_gpu)
        
        computed_area = area_gpu.copy_to_host()
        np.testing.assert_almost_equal(computed_area, expected_area)


class TestOrientPolygonXY(unittest.TestCase):
    # Test a rectangle lying in the YZ plane, expected to rotate into the XY plane.
    def test_orient_polygon_xy(self):
        # Rectangle in YZ plane: (0,0,0) to (0,2,4)
        verts = np.array([[0, 0, 0], [0, 2, 0], [0, 2, 4], [0, 0, 4]], dtype=np.float32)
        normal = np.array([1, 0, 0], dtype=np.float32)  # Normal along X-axis

        rotated = orient_polygon_xy(verts, normal)

        # Expected: (0,0) to (-4,2)
        expected = np.array([[0, 0], [0, 2], [-4, 2], [-4, 0]], dtype=np.float32)

        np.testing.assert_almost_equal(rotated, expected)

    # Parallel test for orient_polygon_xy_gpu.
    def test_orient_polygon_xy_gpu(self):
        if not cuda_available:
            self.skipTest("Skipping GPU tests (Numba or CUDA not available)")

        n = int(1e6)  # Test with 1,000,000 rectangles
        verts = np.tile(np.array([[0, 0, 0], [0, 2, 0], [0, 2, 4], [0, 0, 4]], dtype=np.float32), (n, 1, 1))
        normals = np.tile(np.array([1, 0, 0], dtype=np.float32), (n, 1))

        expected = np.tile(np.array([[0, 0], [0, 2], [-4, 2], [-4, 0]], dtype=np.float32), (n, 1, 1))

        # Allocate GPU memory
        verts_gpu = cuda.to_device(verts)
        normals_gpu = cuda.to_device(normals)
        rotated_out_gpu = cuda.device_array((n, 4, 2), dtype=np.float32)

        # Define CUDA grid and block sizes
        threads_per_block = 256
        blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

        # Launch the GPU kernel
        orient_polygon_xy_gpu_kernel[blocks_per_grid, threads_per_block](verts_gpu, normals_gpu, rotated_out_gpu)

        # Copy result back to host
        computed_rotated = rotated_out_gpu.copy_to_host()

        # Validate the results
        np.testing.assert_almost_equal(computed_rotated, expected)


class TestSegmentPlaneIntersection(unittest.TestCase):
    # Test when a segment intersects the plane at (0,0,0).
    def test_segment_plane_intersection(self):
        p1 = np.array([1, 1, 1], dtype=np.float32)
        p2 = np.array([-1, -1, -1], dtype=np.float32)
        n = np.array([0, 0, 1], dtype=np.float32)  # Plane normal along Z-axis
        q = np.array([0, 0, 0], dtype=np.float32)  # Plane passes through the origin
        epsilon = 1e-6  # Small tolerance

        p1_in, p2_in, intersect = segment_plane_intersection(p1, p2, n, q, epsilon)

        # Expected values
        expected_p1_in = False
        expected_p2_in = True
        expected_intersect = np.array([0, 0, 0], dtype=np.float32)

        # Assertions
        self.assertEqual(p1_in, expected_p1_in)
        self.assertEqual(p2_in, expected_p2_in)
        np.testing.assert_almost_equal(intersect, expected_intersect)

    # Parallel test when a batch of segments intersects the plane at (0,0,0).
    def test_segment_plane_intersection_gpu(self):
        if not cuda_available:
            self.skipTest("Skipping GPU tests (Numba or CUDA not available)")

        n = int(1e6)  # Test with 1,000,000 segments
        p1 = np.tile(np.array([[1, 1, 1]], dtype=np.float32), (n, 1))
        p2 = np.tile(np.array([[-1, -1, -1]], dtype=np.float32), (n, 1))
        normals = np.tile(np.array([[0, 0, 1]], dtype=np.float32), (n, 1))
        q = np.tile(np.array([[0, 0, 0]], dtype=np.float32), (n, 1))
        epsilon = 1e-6

        expected_p1_in = np.full((n,), False, dtype=np.bool_)
        expected_p2_in = np.full((n,), True, dtype=np.bool_)
        expected_intersections = np.tile(np.array([[0, 0, 0]], dtype=np.float32), (n, 1))

        # Allocate GPU memory
        p1_gpu = cuda.to_device(p1)
        p2_gpu = cuda.to_device(p2)
        normals_gpu = cuda.to_device(normals)
        q_gpu = cuda.to_device(q)
        p1_in_gpu = cuda.device_array(n, dtype=np.bool_)
        p2_in_gpu = cuda.device_array(n, dtype=np.bool_)
        intersections_gpu = cuda.device_array((n, 3), dtype=np.float32)

        # Define CUDA grid and block sizes
        threads_per_block = 256
        blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

        # Launch the GPU kernel
        segment_plane_intersection_gpu_kernel[blocks_per_grid, threads_per_block](
            p1_gpu, p2_gpu, normals_gpu, q_gpu, epsilon, p1_in_gpu, p2_in_gpu, intersections_gpu
        )

        # Copy results back to host
        computed_p1_in = p1_in_gpu.copy_to_host()
        computed_p2_in = p2_in_gpu.copy_to_host()
        computed_intersections = intersections_gpu.copy_to_host()
        
        # Validate the results
        np.testing.assert_equal(computed_p1_in, expected_p1_in)
        np.testing.assert_equal(computed_p2_in, expected_p2_in)
        np.testing.assert_almost_equal(computed_intersections, expected_intersections, decimal=5)



class TestClipSH(unittest.TestCase):
    # Test clipping a square against a plane
    def test_clip_sh(self):
        subjects = np.array([[[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]]], dtype=np.float32)  # Square on XY plane
        tri_plane_normal = np.array([[[-1, 0, 0], [0, -1, 0], [1/np.sqrt(2), 1/np.sqrt(2), 0]]], dtype=np.float32)  # Three clip planes
        tri_vertices = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=np.float32)  # Corresponding plane points
        tri_epsilon = np.array([1e-6], dtype=np.float32)

        clipped_points = clip_sh(subjects, tri_plane_normal, tri_vertices, tri_epsilon)

        # Expected output should match the clipped region
        expected_clipped = [[[0, 1, 0], [0, 0, 0], [1, 0, 0]]]

        # Assertions
        np.testing.assert_almost_equal(clipped_points, expected_clipped)


    # Test batch processing of clipping a square against planes
    def test_clip_sh_gpu(self):
        if not cuda_available:
            self.skipTest("Skipping GPU tests (Numba or CUDA not available)")

        n = int(1e6)  # Test with 1,000,000 polygons

        subjects = np.tile(np.array([[[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]]], dtype=np.float32), (n, 1, 1))
        tri_plane_normal = np.tile(np.array([[[-1, 0, 0], [0, -1, 0], [1/np.sqrt(2), 1/np.sqrt(2), 0]]], dtype=np.float32), (n, 1, 1))
        tri_vertices = np.tile(np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=np.float32), (n, 1, 1))
        tri_epsilon = 1e-6

        expected_clipped = np.tile(np.array([[[0, 1, 0], [0, 0, 0], [1, 0, 0]]], dtype=np.float32), (n, 1, 1))

        # Allocate GPU memory
        subjects_gpu = cuda.to_device(subjects)
        tri_plane_normal_gpu = cuda.to_device(tri_plane_normal)
        tri_vertices_gpu = cuda.to_device(tri_vertices)
        # tri_epsilon_gpu = cuda.to_device(tri_epsilon)
        clipped_pts_gpu = cuda.device_array((n, 4, 3), dtype=np.float32)  # Max 4 clipped vertices
        n_clipped_pts_gpu = cuda.device_array(n, dtype=np.int32)  # Number of clipped vertices per polygon

        # Define CUDA grid and block sizes
        threads_per_block = 256
        blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

        # Launch the GPU kernel
        clip_sh_gpu_kernel[blocks_per_grid, threads_per_block](
            subjects_gpu, tri_plane_normal_gpu, tri_vertices_gpu, tri_epsilon, clipped_pts_gpu, n_clipped_pts_gpu
        )

        # Copy results back to host
        clipped_pts = clipped_pts_gpu.copy_to_host()
        n_clipped_pts = n_clipped_pts_gpu.copy_to_host()

        # Assertions
        np.testing.assert_equal(n_clipped_pts, 3)  # All polygons should have 3 clipped vertices
        np.testing.assert_almost_equal(clipped_pts[:, :3, :], expected_clipped, decimal=5)  # Compare only first 3 vertices


class TestIntersectionArea(unittest.TestCase):
    # Test intersection area computation on a single polygon
    def test_get_intersection_area(self):
        proj_faces = np.array([[[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]]], dtype=np.float32)  # Square on XY plane
        tri_normal = np.array([[0, 0, 1]], dtype=np.float32)  # Triangle normal along Z
        tri_plane_normal = np.array([[[-1, 0, 0], [0, -1, 0], [1/np.sqrt(2), 1/np.sqrt(2), 0]]], dtype=np.float32)  # Three clip planes
        tri_vertices = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=np.float32)  # Plane points
        tri_epsilon = np.array([1e-6], dtype=np.float32)

        computed_area = get_intersection_area(proj_faces, tri_normal, tri_plane_normal, tri_vertices, tri_epsilon)

        # Expected output: The clipped triangle should have an area of 0.5 (right triangle)
        expected_area = np.array([0.5], dtype=np.float32)

        # Assertions
        np.testing.assert_almost_equal(computed_area, expected_area)


    # Test batch processing of intersection area computation
    def test_get_intersection_area_gpu(self):
        if not cuda_available:
            self.skipTest("Skipping GPU tests (Numba or CUDA not available)")
            
        n = int(1e6)  # Test with 1,000,000 polygons

        proj_faces = np.tile(np.array([[[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]]], dtype=np.float32), (n, 1, 1))
        tri_normal = np.tile(np.array([[0, 0, 1]], dtype=np.float32), (n, 1))
        tri_plane_normal = np.tile(np.array([[[-1, 0, 0], [0, -1, 0], [1/np.sqrt(2), 1/np.sqrt(2), 0]]], dtype=np.float32), (n, 1, 1))
        tri_vertices = np.tile(np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=np.float32), (n, 1, 1))
        tri_epsilon = np.full((n,), 1e-6, dtype=np.float32)

        expected_area = np.full((n,), 0.5, dtype=np.float32)

        # Allocate GPU memory
        proj_faces_gpu = cuda.to_device(proj_faces)
        tri_normal_gpu = cuda.to_device(tri_normal)
        tri_plane_normal_gpu = cuda.to_device(tri_plane_normal)
        tri_vertices_gpu = cuda.to_device(tri_vertices)
        tri_epsilon_gpu = cuda.to_device(tri_epsilon)

        clipped_pts = cuda.device_array((n, 4, 3), dtype=np.float32)  # clipped polygon vertices, max 4 vertices
        n_clipped_pts = cuda.device_array((n), dtype=np.int32) # number of vertices for each clipped polygon
        areas_gpu = cuda.device_array(n, dtype=np.float32)

        # Define CUDA grid and block sizes
        threads_per_block = 256
        blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

        # Launch the GPU kernel
        get_intersection_area_kernel[blocks_per_grid, threads_per_block](
            proj_faces_gpu, tri_normal_gpu, tri_plane_normal_gpu, tri_vertices_gpu, tri_epsilon_gpu, 
            clipped_pts, n_clipped_pts, areas_gpu
        )

        # Copy results back to host
        computed_areas = areas_gpu.copy_to_host()

        # Assertions
        np.testing.assert_almost_equal(computed_areas, expected_area)


if __name__ == '__main__':
    unittest.main(verbosity=2)
