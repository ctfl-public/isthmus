from numba import cuda
from isthmus_gpu import *

@cuda.jit
def polygon_area_gpu_kernel(verts, n_verts, area):
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)

    for i in range(idx, verts.shape[0], stride):
        area[i] = polygon_area_gpu(verts[i], n_verts)


@cuda.jit
def orient_polygon_xy_gpu_kernel(verts, normals, rotated_out):
    """CUDA kernel to apply orient_polygon_xy_gpu in parallel."""
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)

    for i in range(idx, verts.shape[0], stride):
        orient_polygon_xy_gpu(verts[i], normals[i], rotated_out[i])


@cuda.jit
def segment_plane_intersection_gpu_kernel(p1, p2, n, q, epsilon, p1_in_out, p2_in_out, intersections_out):
    """CUDA kernel for parallel execution of segment_plane_intersection_gpu."""
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)

    for i in range(idx, p1.shape[0], stride):
        p1_in, p2_in, intersect = segment_plane_intersection_gpu(p1[i], p2[i], n[i], q[i], epsilon)
        p1_in_out[i] = p1_in
        p2_in_out[i] = p2_in
        intersections_out[i, 0] = intersect[0]
        intersections_out[i, 1] = intersect[1]
        intersections_out[i, 2] = intersect[2]


@cuda.jit
def clip_sh_gpu_kernel(subjects, tri_plane_normal, tri_vertices, tri_epsilon, clipped_pts_out, n_clipped_pts_out):
    """CUDA kernel for parallel execution of clip_sh_gpu."""
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)

    for i in range(idx, subjects.shape[0], stride):
        clip_sh_gpu(subjects[i], tri_plane_normal[i], tri_vertices[i], tri_epsilon, clipped_pts_out, n_clipped_pts_out, i)
