import numpy as np
from numba import cuda, float32
import math
import time

def get_intersection_area_gpu_profiler(
    proj_faces, tri_normal, tri_plane_normal, tri_vertices, tri_epsilon,
    assume_device_inputs: bool = False,
    ):
    """
    If assume_device_inputs=False: inputs are NumPy host arrays (will time H2D + kernel + D2H).
    If assume_device_inputs=True: inputs are device arrays (will time kernel + D2H only).
    """
    prof = {}

    M = proj_faces.shape[0]

    # --- H -> D transfers (optional) ---
    if not assume_device_inputs:
        cuda.synchronize()
        t0 = time.perf_counter()
        proj_faces_gpu       = cuda.to_device(proj_faces)
        tri_normal_gpu       = cuda.to_device(tri_normal)
        tri_plane_normal_gpu = cuda.to_device(tri_plane_normal)
        tri_vertices_gpu     = cuda.to_device(tri_vertices)
        tri_epsilon_gpu      = cuda.to_device(tri_epsilon)
        cuda.synchronize()
        prof["h2d_s"] = time.perf_counter() - t0
    else:
        proj_faces_gpu       = proj_faces
        tri_normal_gpu       = tri_normal
        tri_plane_normal_gpu = tri_plane_normal
        tri_vertices_gpu     = tri_vertices
        tri_epsilon_gpu      = tri_epsilon
        prof["h2d_s"] = 0.0

    threads_per_block = 256
    blocks_per_grid = min((M + threads_per_block - 1) // threads_per_block, 1024)

    # --- GPU allocations ---
    cuda.synchronize()
    t_alloc = time.perf_counter()
    clipped_pts = cuda.device_array((M, 7, 3), dtype=np.float32)
    n_clipped_pts = cuda.device_array(M, dtype=np.int32)
    areas_gpu = cuda.device_array(M, dtype=np.float32)
    cuda.synchronize()
    prof["alloc_s"] = time.perf_counter() - t_alloc

    # --- Kernel launch + execution ---
    # Use CUDA events for accurate device timing
    start_evt = cuda.event()
    stop_evt = cuda.event()
    start_evt.record()

    t_kernel_submit0 = time.perf_counter()

    get_intersection_area_kernel[blocks_per_grid, threads_per_block](
        proj_faces_gpu, tri_normal_gpu, tri_plane_normal_gpu, tri_vertices_gpu, tri_epsilon_gpu,
        clipped_pts, n_clipped_pts, areas_gpu
    )

    t_kernel_submit1 = time.perf_counter()

    stop_evt.record()
    stop_evt.synchronize()
    t_kernel_done = time.perf_counter()
    prof["kernel_ms"] = cuda.event_elapsed_time(start_evt, stop_evt)

    # Host-side timings: include Python/driver/kernel-launch overhead.
    # - kernel_submit_s: time to enqueue the kernel from Python (no sync)
    # - kernel_call_s: enqueue + wait-for-completion time (sync via stop_evt.synchronize)
    # - kernel_launch_overhead_s: rough estimate of overhead beyond device execution time
    prof["kernel_submit_s"] = t_kernel_submit1 - t_kernel_submit0
    prof["kernel_call_s"] = t_kernel_done - t_kernel_submit0
    prof["kernel_launch_overhead_s"] = max(prof["kernel_call_s"] - (prof["kernel_ms"] * 1e-3), 0.0)

    # --- D -> H transfer ---
    cuda.synchronize()
    t1 = time.perf_counter()
    out_host = areas_gpu.copy_to_host()
    cuda.synchronize()
    prof["d2h_s"] = time.perf_counter() - t1

    # --- Total (for this function) ---
    prof["total_s"] = prof["h2d_s"] + prof["alloc_s"] + (prof["kernel_ms"] * 1e-3) + prof["d2h_s"]
    prof["total_incl_launch_s"] = prof["h2d_s"] + prof["alloc_s"] + prof["kernel_call_s"] + prof["d2h_s"]

    return out_host, prof

def get_intersection_area_gpu(proj_faces, tri_normal, tri_plane_normal, tri_vertices, tri_epsilon):
    M = proj_faces.shape[0] # number of projected faces
    
    proj_faces_gpu       = cuda.to_device(proj_faces)
    tri_normal_gpu       = cuda.to_device(tri_normal)
    tri_plane_normal_gpu = cuda.to_device(tri_plane_normal)
    tri_vertices_gpu     = cuda.to_device(tri_vertices)
    tri_epsilon_gpu      = cuda.to_device(tri_epsilon)

    threads_per_block = 256  # Tuned for GPU efficiency
    blocks_per_grid = min((M + threads_per_block - 1) // threads_per_block, 1024)

    clipped_pts = cuda.device_array((M, 7, 3), dtype=np.float32)  # clipped polygon vertices, max 6 vertices
    n_clipped_pts = cuda.device_array(M, dtype=np.int32) # number of vertices for each clipped polygon
    areas_gpu = cuda.device_array(M, dtype=np.float32)

    get_intersection_area_kernel[blocks_per_grid, threads_per_block](
        proj_faces_gpu, tri_normal_gpu, tri_plane_normal_gpu, tri_vertices_gpu, tri_epsilon_gpu, 
        clipped_pts, n_clipped_pts, areas_gpu
    )

    return areas_gpu.copy_to_host()

@cuda.jit
def get_intersection_area_kernel(proj_faces, tri_normal, tri_plane_normal, tri_vertices, tri_epsilon, clipped_pts, n_clipped_pts, areas):
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    M = proj_faces.shape[0]

    for idx in range(i, M, stride):
        clip_sh_gpu(proj_faces[idx], tri_plane_normal[idx], tri_vertices[idx], tri_epsilon[idx], clipped_pts, n_clipped_pts, idx)

        has_points = n_clipped_pts[idx] > 2
        if not has_points:
            areas[idx] = 0
            continue

        rotated_points = cuda.local.array((7, 2), dtype=float32)
        orient_polygon_xy_gpu(clipped_pts[idx], tri_normal[idx], rotated_points)

        areas[idx] = polygon_area_gpu(rotated_points, n_clipped_pts[idx])


@cuda.jit
def clip_sh_gpu(subject, tri_plane_normal, tri_vertices, tri_epsilon, clipped_pts, n_clipped_pts, idx):
    clip_normals = tri_plane_normal
    clip_verts = tri_vertices
    epsilon = tri_epsilon

    in_pts = cuda.local.array((7, 3), dtype=float32)
    out_pts = cuda.local.array((7, 3), dtype=float32)

    # Copy subject to in_pts
    for k in range(4): # each point of voxel face
        for j in range(3):  # x, y, z
            in_pts[k, j] = subject[k, j]

    n_edges = 4
    for i in range(3): # each clip plane
        num_out_pts = 0
        for j in range(n_edges): # each edge of voxel face
            p1 = in_pts[(j-1)%n_edges]
            p2 = in_pts[j]

            p1_in, p2_in, intersect = segment_plane_intersection_gpu(
                p1, p2, clip_normals[i], clip_verts[i], epsilon
            )

            if p2_in:
                if not p1_in:
                    for j in range(3):
                        out_pts[num_out_pts, j] = intersect[j]
                    num_out_pts += 1
                for j in range(3):
                    out_pts[num_out_pts, j] = p2[j]
                num_out_pts += 1
            elif p1_in:
                for j in range(3):
                    out_pts[num_out_pts, j] = intersect[j]
                num_out_pts += 1
        if num_out_pts > 7:
            print("Warning:: In clip_sh_gpu, num_out_pts > 7: ", num_out_pts)
        n_edges = num_out_pts

        for k in range(num_out_pts):
            for j in range(3):
                in_pts[k, j] = out_pts[k, j]

    n_clipped_pts[idx] = n_edges
    for k in range(n_edges):
        for j in range(3):
            clipped_pts[idx, k, j] = in_pts[k, j]


@cuda.jit(device=True)
def segment_plane_intersection_gpu(p1, p2, n, q, epsilon):
    intersect = cuda.local.array(3, dtype=float32)
    p1_minus_q = cuda.local.array(3, dtype=float32)
    for i in range(3):
        p1_minus_q[i] = p1[i] - q[i]
    p2_minus_q = cuda.local.array(3, dtype=float32)
    for i in range(3):
        p2_minus_q[i] = p2[i] - q[i]
    p1_dist = p1_minus_q[0] * n[0] + p1_minus_q[1] * n[1] + p1_minus_q[2] * n[2]
    p2_dist = p2_minus_q[0] * n[0] + p2_minus_q[1] * n[1] + p2_minus_q[2] * n[2]

    p1_in = p1_dist < epsilon
    p2_in = p2_dist < epsilon

    if p1_in + p2_in == 1:
        frac = abs(p1_dist) / (abs(p2_dist) + abs(p1_dist))
        for i in range(3):
            intersect[i] = p1[i] + frac * (p2[i] - p1[i])
    else:
        for i in range(3):
            intersect[i] = 0

    return p1_in, p2_in, intersect


@cuda.jit(device=True)
def orient_polygon_xy_gpu(verts, normal, rotated_points):
    normal_z = float(normal[2])
    theta = math.acos(normal_z)
    # epsilon = 1e-6

    # Compute rotation axis (cross product of normal and [0,0,1])
    axis = cuda.local.array(3, dtype=float32)
    axis[0] = -normal[1]
    axis[1] = normal[0]
    axis[2] = 0 
    
    # Normalize the axis
    ax_len = math.sqrt(axis[0]**2 + axis[1]**2 + axis[2]**2)
    if ax_len > 0:
        axis[0] /= ax_len
        axis[1] /= ax_len

    # Compute rotation matrix using Rodrigues' formula
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    one_minus_cos = 1.0 - cos_theta
    
    R = cuda.local.array((3, 3), dtype=float32)
    R[0, 0] = cos_theta + axis[0] * axis[0] * one_minus_cos
    R[0, 1] = axis[0] * axis[1] * one_minus_cos
    R[0, 2] = -axis[1] * sin_theta
    
    R[1, 0] = axis[1] * axis[0] * one_minus_cos
    R[1, 1] = cos_theta + axis[1] * axis[1] * one_minus_cos
    R[1, 2] = axis[0] * sin_theta
    
    # R[2, 0] = axis[1] * sin_theta
    # R[2, 1] = -axis[0] * sin_theta
    # R[2, 2] = cos_theta
    
    # Apply rotation matrix
    for i in range(len(verts)):
        x = verts[i, 0]
        y = verts[i, 1]
        z = verts[i, 2]
        
        rotated_points[i, 0] = R[0, 0] * x + R[0, 1] * y + R[0, 2] * z
        rotated_points[i, 1] = R[1, 0] * x + R[1, 1] * y + R[1, 2] * z


@cuda.jit(device=True)
def polygon_area_gpu(verts, n_verts):
    area = 0.0
    for i in range(n_verts):
        # use modulo operatore ensures that (i-1)%n_verts returns the last index either n_verts = 3 or 4
        # since verts is always a pointer of size (4,2)
        p1_x, p1_y = verts[(i-1)%n_verts, 0], verts[(i-1)%n_verts, 1]
        p2_x, p2_y = verts[i, 0], verts[i, 1]
        area += (p1_y + p2_y) * (p1_x - p2_x)
    return abs(area * 0.5)
