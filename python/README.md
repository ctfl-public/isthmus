# ISTHMUS Python Wrapper

Python bindings for the [ISTHMUS](https://github.com/ctfl-public/isthmus) marching-windows C++ library, built with [pybind11](https://pybind11.readthedocs.io/) and [scikit-build-core](https://scikit-build-core.readthedocs.io/).

The wrapper exposes `MarchingWindows::run` and all associated data types directly to Python while preserving the full result structure (corner-fill fractions, surface mesh, flux association) so downstream code can consume them without intermediate files.

If you are browsing from the repository root, the native C++ project is documented in [../README.md](../README.md), and the Python binding setup is documented here.

---

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation](#installation)
    - [Which path should I use?](#which-path-should-i-use)
    - [Quick Install](#quick-install)
    - [Manual Installation](#manual-installation)
    - [Development — Option 1: pip editable install](#development--option-1-pip-editable-install)
    - [Development — Option 2: manual CMake build](#development--option-2-manual-cmake-build)
    - [Coexisting with the legacy isthmus module](#coexisting-with-the-legacy-isthmus-module)
- [Usage](#usage)
  - [Domain configuration](#domain-configuration)
  - [Voxel input](#voxel-input)
  - [Running the algorithm](#running-the-algorithm)
  - [Reading results](#reading-results)
  - [NumPy helpers](#numpy-helpers)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)

---

## System Requirements

| Requirement | Minimum version |
|---|---|
| Python | 3.8 |
| CMake | 3.22 |
| C++ compiler | GCC 11, Clang 14, or MSVC 2022 (C++20 support required) |
| git | any recent version (used to fetch pybind11 if not pre-installed) |

Optional but recommended:

- **ninja** — faster incremental builds (`pip install ninja`)
- **numpy** — required only for the `corner_fill_as_array` helper

---

## Installation

### Which path should I use?

| Path | Best for |
|---|---|
| [Quick Install](#quick-install) | First-time setup; shortest path from zero to working |
| [Manual Installation](#manual-installation) | Reproducible installs, CI, custom environments |
| Development | Active development on Python or C++ code | Keeps the install linked to the source tree and rebuilds the extension in place with `pip install -e . --no-build-isolation`. |
| [- Development Option 1 — pip editable](#development--option-1-pip-editable-install) | Iterating on the Python layer (`__init__.py`, etc.) |
| [- Development Option 2 — manual CMake](#development--option-2-manual-cmake-build) | Iterating on C++ sources, full control over build flags |

---

### Quick Install

The provided script handles everything (build dependencies, compilation, install, smoke test):

Use this path when you want the shortest route from a clean working Python install. It is the most convenient option for first-time setup because it installs the build dependencies it needs and then builds the wrapper for you.

```bash
cd /path/to/isthmus-cpp
chmod +x install.sh
./install.sh
```

Options:

```bash
./install.sh --user        # install to ~/.local (no root/sudo required)
./install.sh --editable    # development install (re-compiles on source change)
```

---

### Manual Installation

Manual installation is the right choice when you want to control the environment yourself.

#### Step 1 — Python build dependencies

```bash
pip install "pybind11>=2.11" "scikit-build-core>=0.5" ninja
```

If pybind11 is already installed as a Python package, CMake uses its bundled
headers automatically.  Otherwise CMake fetches pybind11 v2.13.6 via
`FetchContent` at configure time (requires network access and git).

#### Step 2 — Build and install

```bash
cd /path/to/isthmus-cpp
pip install .
```

This invokes CMake with `-DISTHMUS_BUILD_PYTHON=ON -DISTHMUS_BUILD_TESTS=OFF`,
compiles the `_isthmus` extension module, and installs the `isthmus` package
into the active environment.

To pass extra CMake variables (e.g. a custom compiler):

```bash
CMAKE_ARGS="-DCMAKE_CXX_COMPILER=clang++" pip install .
```

#### Step 3 — Verify

```bash
python3 -c "import isthmus; print(isthmus.__version__)"
# 0.1.0

python3 python/examples/basic_run.py
```

---

### Development — Option 1: pip editable install

Keeps the install linked to the source tree; re-running the command recompiles
without reinstalling.

```bash
pip install "pybind11>=2.11" "scikit-build-core>=0.5" ninja
pip install -e . --no-build-isolation
```

**What each flag does:**

- **`pip install .`** — reads `pyproject.toml`, invokes scikit-build-core as
  the build backend, which runs CMake to compile `_isthmus*.so`, then registers
  the `isthmus` package in the active Python environment.

- **`-e` (editable)** — instead of copying Python source files into
  `site-packages`, pip writes a small pointer (a `.pth` file) that redirects
  `import isthmus` back to the `python/isthmus/` directory in the repository.
  Edits to `__init__.py` or any other pure-Python file are reflected
  immediately with no reinstall.  The compiled `_isthmus*.so` is placed inside
  `python/isthmus/` so the relative import `from ._isthmus import ...` finds
  it there.

- **`--no-build-isolation`** — by default pip creates a temporary, empty
  virtual environment and re-installs the build requirements into it before
  compiling.  This flag skips that step and uses the packages already present
  in the current environment (the `pybind11` and `scikit-build-core` installed
  above), saving a redundant download on every rebuild.

After any C++ source change, re-run `pip install -e . --no-build-isolation` to
recompile.  Python-only changes (`__init__.py`, etc.) are reflected immediately
with no rebuild.

---

### Development — Option 2: manual CMake build

Full control over CMake flags and the build directory, with no pip involved.

#### Configure and build

```bash
cmake -S . -B build-py \
      -DISTHMUS_BUILD_PYTHON=ON \
      -DISTHMUS_BUILD_TESTS=OFF \
      -DCMAKE_BUILD_TYPE=Release
cmake --build build-py --parallel
```

After each build the compiled extension (`_isthmus*.so`) is **automatically
copied into `python/isthmus/`** by a CMake `POST_BUILD` step.  No manual copy
or `PYTHONPATH` change is needed.

#### Running scripts without installing

Add `python/` to `sys.path` at the top of your script so it takes priority
over any same-named module on `PYTHONPATH`:

```python
from pathlib import Path
import sys

_repo_root = Path(__file__).resolve().parents[N]  # adjust N to reach repo root
sys.path.insert(0, str(_repo_root / "python"))

import isthmus  # resolves to python/isthmus/__init__.py + _isthmus*.so
```

`python/examples/basic_run.py` already does this, so you can run it directly:

```bash
python3 python/examples/basic_run.py
```

#### After changing C++ sources

```bash
cmake --build build-py --parallel   # recompiles and re-copies the .so
```

---

### Coexisting with the legacy isthmus module

Both development options install or import a package named `isthmus`.  If your
`PYTHONPATH` already contains the legacy Python implementation (e.g.
`/home/ahmed/isthmus/src/isthmus.py`), Python's search order determines which
one wins:

```
sys.path order (highest priority first)
─────────────────────────────────────────
1. PYTHONPATH entries   ← legacy isthmus.py lives here
2. site-packages        ← Option 1 (pip editable) installs here
3. stdlib
```

**Option 1 (pip editable):** because `site-packages` sits below `PYTHONPATH`,
the legacy module still wins unless you isolate the environment.  Use a virtual
environment, which starts with an empty `sys.path` and no PYTHONPATH bleed-in:

```bash
python3 -m venv ~/.venvs/isthmus-cpp
source ~/.venvs/isthmus-cpp/bin/activate
pip install "pybind11>=2.11" "scikit-build-core>=0.5" ninja
pip install -e . --no-build-isolation
```

**Option 2 (manual CMake):** `sys.path.insert(0, ...)` inserts at index 0,
ahead of all PYTHONPATH entries, so the C++ wrapper always takes priority
regardless of what is on PYTHONPATH.  No virtual environment is required.

---

## Usage

A complete working example is available in [python/examples/basic_run.py](examples/basic_run.py).

### Domain configuration

```python
from isthmus import DomainConfig, Dimension

domain = DomainConfig()
domain.dimension  = Dimension.D3          # or Dimension.D2
domain.voxel_size = 3.376e-6              # meters per voxel edge
domain.limits     = [                     # [[xmin, ymin, zmin], [xmax, ymax, zmax]]
    [-5 * domain.voxel_size] * 3,
    [105 * domain.voxel_size] * 3,
]
domain.cell_counts = [100, 100, 100]      # marching cells per axis
domain.weighting   = True                 # depth-based corner weights
domain.iso_value   = 0.5                  # isosurface threshold
```

**Buffer requirement:** the domain must include at least 5 voxels of padding on
every side of the voxel cloud, or the library raises
`InvalidInputError: Insufficient buffer`.

```python
buf = 5                                        # voxels of margin on each side
n   = <extent of voxel cloud in cells>
domain.limits      = [[-buf * voxel_size] * 3, [(n + buf) * voxel_size] * 3]
domain.cell_counts = [n + 2 * buf] * 3
```

For 2D domains only the first two elements of `limits`, `cell_counts`, and
voxel centroids are used; the third is ignored by the library.

---

### Voxel input

**Option A — helper function (recommended)**

```python
from isthmus import voxel_set_from_centroids

centroids = [
    [x * 1e-6, y * 1e-6, z * 1e-6]
    for x in range(50)
    for y in range(50)
    for z in range(10)
]
voxels = voxel_set_from_centroids(centroids)

# With explicit IDs and material labels:
voxels = voxel_set_from_centroids(
    centroids,
    original_ids  = range(len(centroids)),
    material_tags = ["carbon"] * len(centroids),
)
```

**Option B — direct construction**

```python
from isthmus import VoxelSet, VoxelRecord

records = []
for i, (x, y, z) in enumerate(my_centroid_list):
    vr = VoxelRecord()
    vr.centroid     = [x, y, z]
    vr.original_id  = i
    vr.material_tag = "carbon"   # or None
    records.append(vr)

voxels = VoxelSet()
voxels.voxels = records   # assign the full list at once
```

> **Note:** do NOT call `voxels.voxels.append()` one by one.  pybind11 returns
> a copy of the underlying C++ vector on read, so appending to it does not
> modify the actual `VoxelSet`.  Always build a Python list first, then assign
> it.

**Option C — from a NumPy array**

```python
import numpy as np
from isthmus import voxel_set_from_centroids

coords = np.load("voxels.npy")          # shape (N, 3), dtype float64
voxels = voxel_set_from_centroids(coords.tolist())
```

---

### Running the algorithm

```python
from isthmus import MarchingWindows, RunOptions

options = RunOptions()
options.build_surface          = True   # reconstruct the isosurface
options.build_flux_association = True   # compute triangle→voxel ownership

mw     = MarchingWindows()
result = mw.run(domain, voxels, options)
```

`MarchingWindows` is stateless; the same instance can be called repeatedly with
different domains or voxel sets.

---

### Reading results

```python
# Corner-fill fractions (always populated)
fracs = result.corner_fill_fractions   # flat list, length = product of corner_dims
dims  = result.corner_dims             # [nx+1, ny+1, nz+1]

# Surface voxels (always populated)
for sv in result.surface_voxels:
    print(sv.original_id, sv.centroid, sv.depth, sv.weight)

# Surface mesh (populated when build_surface=True)
mesh = result.surface_mesh
for tri in mesh.triangles:
    v0, v1, v2 = mesh.vertices[tri[0]], mesh.vertices[tri[1]], mesh.vertices[tri[2]]

# Flux association (populated when build_flux_association=True)
for elem in result.flux_association.elements:
    print(elem.element_id, elem.voxel_ids, elem.scalar_fractions)
```

---

### NumPy helpers

```python
import numpy as np
from isthmus import corner_fill_as_array

fill = corner_fill_as_array(result)    # ndarray, shape = corner_dims, dtype float64
volume_fraction = fill.mean()

vertices  = np.array(result.surface_mesh.vertices)   # (N, 3) float64
triangles = np.array(result.surface_mesh.triangles)  # (M, 3) int
```

---

## API Reference

### `MarchingWindows`

```
MarchingWindows()
    .run(domain, voxels, options=RunOptions()) -> MarchingWindowsResult
```

### `DomainConfig`

| Attribute | Type | Default | Description |
|---|---|---|---|
| `dimension` | `Dimension` | `D3` | Spatial dimension |
| `limits` | `[[float×3], [float×3]]` | `[[0,0,0],[1,1,1]]` | Domain bounds |
| `cell_counts` | `[int×3]` | `[1,1,1]` | Cells per axis |
| `voxel_size` | `float` | `1.0` | Voxel edge length (m) |
| `weighting` | `bool` | `True` | Depth weighting |
| `iso_value` | `float` | `0.5` | Isosurface threshold |

### `RunOptions`

| Attribute | Type | Default | Description |
|---|---|---|---|
| `build_surface` | `bool` | `False` | Run surface extraction |
| `build_flux_association` | `bool` | `False` | Compute flux ownership |
| `write_diagnostics` | `bool` | `False` | Reserved for future use |

### `VoxelRecord`

| Attribute | Type | Description |
|---|---|---|
| `centroid` | `[float×3]` | Physical-space centroid `[x, y, z]` |
| `original_id` | `int` | Caller-assigned voxel ID |
| `material_tag` | `str \| None` | Optional material label |

### `SurfaceVoxelInfo` (read-only)

| Attribute | Type | Description |
|---|---|---|
| `original_id` | `int` | Caller-assigned ID |
| `voxel_indices` | `[int×3]` | Integer grid indices `[i, j, k]` |
| `centroid` | `[float×3]` | Physical-space centroid |
| `depth` | `int` | Depth classification from boundary pass |
| `weight` | `float` | Depth-based weight |

### `SurfaceMesh` (read-only)

| Attribute | Type | Description |
|---|---|---|
| `vertices` | `list[[float×3]]` | Vertex coordinates |
| `triangles` | `list[[int×3]]` | Vertex-index triplets (3D) |
| `segments` | `list[[int×2]]` | Vertex-index pairs (2D) |

### `FluxElementOwnership` (read-only)

| Attribute | Type | Description |
|---|---|---|
| `element_id` | `int` | Surface element index |
| `voxel_ids` | `list[int]` | Owning voxel `original_id`s |
| `scalar_fractions` | `list[float]` | Ownership fractions, sum = 1.0 |

### `MarchingWindowsResult` (read-only)

| Attribute | Type | Description |
|---|---|---|
| `domain` | `DomainConfig` | Validated domain used for this run |
| `corner_fill_fractions` | `list[float]` | Flat corner-fill field |
| `corner_dims` | `[int×3]` | Dimensions of the corner-fill field |
| `surface_voxels` | `list[SurfaceVoxelInfo]` | Boundary voxel metadata |
| `surface_mesh` | `SurfaceMesh` | Reconstructed surface |
| `flux_association` | `FluxAssociation` | Triangle→voxel ownership map |

### Helper functions

| Function | Description |
|---|---|
| `voxel_set_from_centroids(centroids, original_ids=None, material_tags=None)` | Build `VoxelSet` from a sequence of `[x, y, z]` rows |
| `corner_fill_as_array(result)` | Reshape `corner_fill_fractions` into a NumPy ndarray of shape `corner_dims` |

### Exceptions

| Exception | Inherits from | Raised when |
|---|---|---|
| `IsthmusError` | `RuntimeError` | Base for all library errors |
| `InvalidInputError` | `IsthmusError` | Bad domain, voxel size, or voxel set |
| `IsthmusNotImplementedError` | `IsthmusError` | Requested stage not yet implemented |

---

## Troubleshooting

**`ImportError: cannot import name 'MarchingWindows' from 'isthmus'`**
→ A different `isthmus` module (e.g. the legacy Python implementation at
`/home/ahmed/isthmus/src/isthmus.py`) is shadowing the C++ package on
`PYTHONPATH`.  Fix this in your script by inserting the correct path first:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[N] / "python"))
import isthmus   # now resolves to the C++ wrapper
```
`python/examples/basic_run.py` already does this automatically.

**`ModuleNotFoundError: No module named 'isthmus'`**
→ Run `pip install .` from the repo root (where `pyproject.toml` lives), or
use Option 2 (manual CMake build) and add the `sys.path.insert` shown above.

**`InvalidInputError: Insufficient buffer`**
→ The domain limits do not leave enough margin around the voxel cloud.
Add at least 5 voxels of padding on every side (see [Domain configuration](#domain-configuration)).

**`CMake Error: pybind11 not found`**
→ pybind11 is fetched automatically via git at configure time.  Ensure git is
installed and has network access, or pre-install it: `pip install pybind11`.

**`relocation … can not be used when making a shared object; recompile with -fPIC`**
→ The static `libisthmus_cpp.a` was built without `-fPIC`.  This is fixed in
the current `CMakeLists.txt` (`POSITION_INDEPENDENT_CODE ON`).  Delete your
build directory and reconfigure from scratch.

**`ImportError: undefined symbol`**
→ The extension was compiled against a different `libstdc++`.  Build in a
clean environment or set `-DCMAKE_BUILD_TYPE=Release`.

**Slow compile**
→ `pip install ninja` — scikit-build-core and CMake detect it automatically.

**C++ changes not reflected after `cmake --build`**
→ The `POST_BUILD` step copies the updated `.so` into `python/isthmus/`
automatically.  If you still see old behaviour, check that you are importing
from `python/isthmus/` and not a stale installed copy in site-packages.