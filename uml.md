# Native C++ UML Diagram

This diagram reflects the current native C++ implementation at the repository root.
It focuses on the public entry point, the internal motion-mapping pipeline,
the grid/result model, and the geometry and IO helpers that support the native
backend.

```mermaid
---
config:
  layout: elk
---
classDiagram
direction LR

class MarchingWindows {
  +run(domain, voxels, options) MarchingWindowsResult
  -motion_mapper_ : MotionMapper
}

class MotionMapper {
  +run(domain, voxels, options) MarchingWindowsResult
  -validate_inputs(...)
  -build_voxel_grid(...)
  -classify_voxels(...)
  -assign_exposed_faces(...)
  -build_corner_grid(...)
  -collect_surface_voxels(...)
}

class Dimension {
  <<enumeration>>
  D2
  D3
}

class RunOptions {
  +build_surface : bool
  +build_flux_association : bool
  +write_diagnostics : bool
}

class DomainConfig {
  +dimension : Dimension
  +limits : 2 bounds x 3 coordinates
  +cell_counts : 3 axis counts
  +voxel_size : double
  +weighting : bool
}

class VoxelRecord {
  +centroid : 3D point
  +original_id : size_t
  +material_tag : optional string
}

class VoxelSet {
  +voxels : list of VoxelRecord
}

class SurfaceMesh {
  +vertices : list of 3D points
  +triangles : list of triangle index triplets
  +segments : list of segment index pairs
}

class FluxElementOwnership {
  +element_id : size_t
  +voxel_ids : list of size_t
  +scalar_fractions : list of double
}

class FluxAssociation {
  +elements : list of FluxElementOwnership
}

class SurfaceVoxelInfo {
  +original_id : size_t
  +voxel_indices : lattice index
  +centroid : 3D point
  +depth : int
  +weight : double
}

class MarchingWindowsResult {
  +domain : DomainConfig
  +corner_fill_fractions : list of double
  +corner_dims : 3 axis counts
  +surface_voxels : list of SurfaceVoxelInfo
  +surface_mesh : SurfaceMesh
  +flux_association : FluxAssociation
}

class VoxelFace3D {
  +corners : 4 points
  +normal : 3D vector
  +exposed : bool
}

class VoxelFace2D {
  +corners : 2 points
  +normal : 2D vector
  +exposed : bool
}

class VoxelCell {
  +centroid : 3D point
  +indices : lattice index
  +flattened_index : size_t
  +original_id : size_t
  +type : int
  +finalized : bool
  +surface : bool
  +weight : double
  +faces3d : list of VoxelFace3D
  +faces2d : list of VoxelFace2D
}

class CornerData {
  +position : 3D point
  +indices : lattice index
  +volume : double
  +inside : int
  +owned_voxel_indices : list of size_t
}

class RegularGrid2D {
  <<template alias>>
}

class RegularGrid3D {
  <<template alias>>
}

class Geometry {
  <<utility>>
  +dot(...)
  +cross(...)
  +norm(...)
  +segment_plane_intersection(...)
  +clip_polygon_sutherland_hodgman(...)
  +orient_polygon_xy(...)
  +polygon_area(...)
  +triangle_area(...)
  +intersection_length(...)
}

class IO {
  <<utility>>
  +write_sparta_surface(...)
  +write_vtp_surface(...)
  +write_flux_association(...)
}

class MarchingCubesBackend {
  <<internal>>
  +extract_surface_mesh_3d(...)
}

MarchingWindows *-- MotionMapper : owns
MarchingWindows --> DomainConfig
MarchingWindows --> VoxelSet
MarchingWindows --> RunOptions
MarchingWindows --> MarchingWindowsResult

MotionMapper --> DomainConfig
MotionMapper --> VoxelSet
MotionMapper --> MarchingWindowsResult
MotionMapper ..> VoxelCell
MotionMapper ..> CornerData
MotionMapper ..> SurfaceVoxelInfo
MotionMapper ..> MarchingCubesBackend : build_surface

DomainConfig --> Dimension
VoxelSet *-- VoxelRecord
MarchingWindowsResult *-- SurfaceMesh
MarchingWindowsResult *-- FluxAssociation
MarchingWindowsResult *-- SurfaceVoxelInfo
FluxAssociation *-- FluxElementOwnership
VoxelCell *-- VoxelFace3D
VoxelCell *-- VoxelFace2D
VoxelCell ..> RegularGrid2D
VoxelCell ..> RegularGrid3D
IO ..> SurfaceMesh
IO ..> FluxAssociation
IO ..> Dimension
```

The diagram intentionally treats `isthmus::geometry`, `isthmus::io`, and
`isthmus::marching_cubes` as module-level utility areas rather than class
hierarchies. The native code is still centered on `MarchingWindows`, which
delegates to `MotionMapper` and optionally fills `surface_mesh` when the
surface backend is requested.

## Legend

- `A --> B`: Association. A has a named or direct relationship to B.
- `A *-- B`: Composition. A owns B as part of its data; B usually lives and dies with A.
- `A o-- B`: Aggregation. A has B, but B can exist independently of A.
- `A ..> B`: Dependency. A depends on B in a lighter, non-owning way, such as calling a helper or using a type temporarily.
- `A <|-- B`: Inheritance. B is a specialized kind of A.
- `A <|.. B`: Realization. B implements an interface or abstract contract from A.
- `A -- B`: Simple association line with no arrow direction shown.
- `+member`: Public field or method that callers can use.
- `-member`: Private field or method used only inside the class.
- `<<enumeration>>`: A fixed set of named values, like `Dimension::D2` and `Dimension::D3`.
- `<<utility>>`: A helper namespace or module with functions rather than a normal object with state.
- `<<internal>>`: A type that is part of the implementation but not intended as the main public API.
