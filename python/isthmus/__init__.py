"""
isthmus — Python bindings for the ISTHMUS marching-windows library.

Quick start
-----------
>>> from isthmus import MarchingWindows, DomainConfig, VoxelSet, VoxelRecord, RunOptions, Dimension
>>> mw = MarchingWindows()
>>> domain = DomainConfig()
>>> domain.dimension = Dimension.D3
>>> domain.limits = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
>>> domain.cell_counts = [10, 10, 10]
>>> domain.voxel_size = 0.1
>>> voxels = voxel_set_from_centroids([[0.5, 0.5, 0.5], [0.4, 0.5, 0.5]])
>>> result = mw.run(domain, voxels)
"""

from ._isthmus import (  # noqa: F401
    MAX_DIMS,
    Dimension,
    RunOptions,
    DomainConfig,
    VoxelRecord,
    VoxelSet,
    SurfaceVoxelInfo,
    SurfaceMesh,
    FluxElementOwnership,
    FluxAssociation,
    MarchingWindowsResult,
    MarchingWindows,
    IsthmusError,
    InvalidInputError,
    IsthmusNotImplementedError,
)

__version__ = "0.1.0"
__all__ = [
    "MAX_DIMS",
    "Dimension",
    "RunOptions",
    "DomainConfig",
    "VoxelRecord",
    "VoxelSet",
    "SurfaceVoxelInfo",
    "SurfaceMesh",
    "FluxElementOwnership",
    "FluxAssociation",
    "MarchingWindowsResult",
    "MarchingWindows",
    "IsthmusError",
    "InvalidInputError",
    "IsthmusNotImplementedError",
    "voxel_set_from_centroids",
    "corner_fill_as_array",
]


def voxel_set_from_centroids(centroids, original_ids=None, material_tags=None):
    """Build a VoxelSet from a sequence of centroid coordinates.

    Parameters
    ----------
    centroids : sequence of shape (N, 3)
        Each row is the [x, y, z] centroid of one occupied voxel.
        For 2D domains the third coordinate is ignored by the library.
    original_ids : sequence of int, optional
        Caller-assigned IDs, one per voxel.  Defaults to 0 … N-1.
    material_tags : sequence of str or None, optional
        Material label per voxel.  Pass None (or omit) for unlabelled voxels.

    Returns
    -------
    VoxelSet
    """
    centroids = list(centroids)
    n = len(centroids)

    if original_ids is None:
        original_ids = range(n)

    if material_tags is None:
        material_tags = [None] * n

    records = []
    for i, (c, oid, tag) in enumerate(zip(centroids, original_ids, material_tags)):
        vr = VoxelRecord()
        c = list(c)
        if len(c) == 2:
            c = [c[0], c[1], 0.0]
        elif len(c) != 3:
            raise ValueError(f"centroid at index {i} must have 2 or 3 elements, got {len(c)}")
        vr.centroid = c
        vr.original_id = int(oid)
        vr.material_tag = tag
        records.append(vr)

    # Assign all at once: pybind11 returns a copy on read, so per-element
    # .append() on vs.voxels would modify a temporary, not the C++ vector.
    vs = VoxelSet()
    vs.voxels = records
    return vs


def corner_fill_as_array(result):
    """Reshape the flat corner_fill_fractions into a 3-D numpy array.

    Parameters
    ----------
    result : MarchingWindowsResult

    Returns
    -------
    numpy.ndarray of shape (nx+1, ny+1, nz+1), dtype float64.

    Raises
    ------
    ImportError if numpy is not installed.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("corner_fill_as_array requires numpy") from exc

    dims = list(result.corner_dims)
    data = np.array(result.corner_fill_fractions, dtype=np.float64)
    return data.reshape(dims)
