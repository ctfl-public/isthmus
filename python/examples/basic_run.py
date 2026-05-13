"""
basic_run.py — minimal end-to-end example of the isthmus Python wrapper.

Demonstrates:
  - Building a DomainConfig and VoxelSet from scratch
  - Running MarchingWindows with surface extraction enabled
  - Inspecting the result (corner-fill fractions, surface voxels, mesh)
  - Using the helper utilities (voxel_set_from_centroids, corner_fill_as_array)
"""

from pathlib import Path
import sys

# Make the C++ isthmus package importable when running directly from the
# source tree (manual cmake build, no `pip install`).
# The cmake POST_BUILD step copies _isthmus*.so into python/isthmus/ so only
# the python/ directory needs to be on sys.path.
# Inserting at index 0 overrides any legacy `isthmus` module that may be
# present on PYTHONPATH.
_repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo_root / "python"))

from isthmus import (
    MarchingWindows,
    DomainConfig,
    RunOptions,
    Dimension,
    voxel_set_from_centroids,
    corner_fill_as_array,
)

# ---------------------------------------------------------------------------
# 1. Build domain configuration
# ---------------------------------------------------------------------------

voxel_size = 1.0  # arbitrary units; set to meters in real use

domain = DomainConfig()
domain.dimension  = Dimension.D3
domain.voxel_size = voxel_size
domain.iso_value  = 0.5
domain.weighting  = True

# Voxels span 0..2; add 5-voxel padding on each side (required by the library).
# Grid: -5 to +7 inclusive → 12 cells per axis, each 1 voxel_size wide.
buf = 5
n   = 3
domain.limits      = [[-buf * voxel_size] * 3, [(n + buf) * voxel_size] * 3]
domain.cell_counts = [n + 2 * buf] * 3

# ---------------------------------------------------------------------------
# 2. Build a simple 3-D voxel block (a 3x3x3 solid cube)
# ---------------------------------------------------------------------------

centroids = [
    [x, y, z]
    for x in range(3)
    for y in range(3)
    for z in range(3)
]

voxels = voxel_set_from_centroids(centroids)
print(f"Voxel count  : {len(voxels.voxels)}")

# ---------------------------------------------------------------------------
# 3. Configure run options
# ---------------------------------------------------------------------------

options = RunOptions()
options.build_surface          = True
options.build_flux_association = True

# ---------------------------------------------------------------------------
# 4. Run marching windows
# ---------------------------------------------------------------------------

mw = MarchingWindows()
result = mw.run(domain, voxels, options)

# ---------------------------------------------------------------------------
# 5. Inspect results
# ---------------------------------------------------------------------------

print(f"Corner dims  : {list(result.corner_dims)}")
print(f"Corner fracs : {len(result.corner_fill_fractions)} values")
print(f"Surface vox  : {len(result.surface_voxels)}")
print(f"Triangles    : {len(result.surface_mesh.triangles)}")
print(f"Vertices     : {len(result.surface_mesh.vertices)}")
print(f"Flux entries : {len(result.flux_association.elements)}")

# ---------------------------------------------------------------------------
# 6. Optional: reshape corner fill into a numpy array
# ---------------------------------------------------------------------------
try:
    import numpy as np
    fill = corner_fill_as_array(result)
    print(f"Corner fill shape : {fill.shape}")
    print(f"Volume fraction   : {fill.mean():.4f}")
except ImportError:
    print("(install numpy to use corner_fill_as_array)")

# ---------------------------------------------------------------------------
# 7. Access individual surface voxel metadata
# ---------------------------------------------------------------------------
if result.surface_voxels:
    sv = result.surface_voxels[0]
    print(f"\nFirst surface voxel:")
    print(f"  original_id    : {sv.original_id}")
    print(f"  centroid       : {list(sv.centroid)}")
    print(f"  voxel_indices  : {list(sv.voxel_indices)}")
    print(f"  depth          : {sv.depth}")
    print(f"  weight         : {sv.weight:.4f}")

# ---------------------------------------------------------------------------
# 8. Access flux association entries
# ---------------------------------------------------------------------------
if result.flux_association.elements:
    fe = result.flux_association.elements[0]
    print(f"\nFirst flux element:")
    print(f"  element_id       : {fe.element_id}")
    print(f"  voxel_ids        : {list(fe.voxel_ids)}")
    print(f"  scalar_fractions : {[round(f, 4) for f in fe.scalar_fractions]}")

print("\nDone.")
