import numpy as np
from geometry import get_intersection_area
from geometry_gpu import get_intersection_area_gpu
import argparse

def test(n=1, gpu=False):
    proj_faces = np.tile(np.array([[[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]]], dtype=np.float32), (n, 1, 1))
    tri_normal = np.tile(np.array([[0, 0, 1]], dtype=np.float32), (n, 1))
    tri_plane_normal = np.tile(np.array([[[-1, 0, 0], [0, -1, 0], [1/np.sqrt(2), 1/np.sqrt(2), 0]]], dtype=np.float32), (n, 1, 1))
    tri_vertices = np.tile(np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=np.float32), (n, 1, 1))
    tri_epsilon = np.full((n,), 1e-6, dtype=np.float32)

    if gpu:
        computed_area = get_intersection_area_gpu(proj_faces, tri_normal, tri_plane_normal, tri_vertices, tri_epsilon)
    else:
        computed_area = get_intersection_area(proj_faces, tri_normal, tri_plane_normal, tri_vertices, tri_epsilon)

    # print(f'Computed intersection areas: {computed_area}')

    return computed_area


if __name__ == "__main__":
    # To use gpu option, run the script as:
    # python intersection_test.py --gpu
    # To specify number of elements, run the script as:
    # python intersection_test.py --n 5000
    parser = argparse.ArgumentParser(description='Run with GPU option')

    parser.add_argument('--gpu', action='store_true', help='Enable GPU acceleration')
    parser.add_argument('--n', type=int, default=1, help='Number of elements to process')
    args = parser.parse_args()

    gpu = args.gpu
    n = args.n
    print(f'GPU acceleration is {"enabled" if gpu else "disabled"}.')
    print(f'Number of elements to process: {n}')

    proj_faces = np.tile(np.array([[[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]]], dtype=np.float32), (n, 1, 1))
    tri_normal = np.tile(np.array([[0, 0, 1]], dtype=np.float32), (n, 1))
    tri_plane_normal = np.tile(np.array([[[-1, 0, 0], [0, -1, 0], [1/np.sqrt(2), 1/np.sqrt(2), 0]]], dtype=np.float32), (n, 1, 1))
    tri_vertices = np.tile(np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=np.float32), (n, 1, 1))
    tri_epsilon = np.full((n,), 1e-6, dtype=np.float32)

    if gpu:
        computed_area = get_intersection_area_gpu(proj_faces, tri_normal, tri_plane_normal, tri_vertices, tri_epsilon)
    else:
        computed_area = get_intersection_area(proj_faces, tri_normal, tri_plane_normal, tri_vertices, tri_epsilon)

    print(f'Computed intersection areas: {computed_area}')