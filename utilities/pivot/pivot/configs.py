"""
Typed configuration objects for Paraview Tools.

These classes represent *validated solver-specific configuration*.
They contain no file I/O and no TOML parsing logic.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# -------------------------
# Solver configs
# -------------------------

@dataclass(frozen=True)
class FlowConfig:
    flow_dir: Path
    flow_dt: float
    field_map: dict[str, str] = field(default_factory=dict)
    step: int = field(default=1)

    @classmethod
    def from_dict(cls, cfg: dict) -> "FlowConfig":
        try:
            flow_dir = Path(cfg["flow_dir"])
            flow_dt = float(cfg["flow_dt"])
            field_map = cfg.get("field_map", {})
            step = int(cfg.get("step", 1))  # optional, defaults to 1
        except KeyError as e:
            raise ValueError(f"[flow] missing required key: {e.args[0]}")

        if flow_dt < 0:
            raise ValueError("[flow] flow_dt must be >= 0")
        if step <= 0:
            raise ValueError("[flow] step must be > 0")

        return cls(flow_dir=flow_dir, flow_dt=flow_dt, field_map=field_map, step=step)


@dataclass(frozen=True)
class VisualizationConfig:
    enabled: bool = False
    field: str = ""
    output: Path = Path("visualization_output/flow_contour.png")
    timestep: Optional[int] = None
    levels: int = 0
    value_range: Optional[tuple[float, float]] = None
    resolution: tuple[int, int] = (1200, 800)
    cmap: str = "turbo"
    chunk_size: int = 500_000

    @classmethod
    def from_dict(cls, cfg: dict) -> "VisualizationConfig":
        enabled = bool(cfg.get("enabled", False))
        if not enabled:
            return cls()

        field = cfg.get("field")
        if not field:
            raise ValueError("[visualization] 'field' is required when enabled = true")

        value_range_raw = cfg.get("value_range")
        if value_range_raw is not None:
            if len(value_range_raw) != 2:
                raise ValueError("[visualization] value_range must contain exactly two numbers")
            value_range = (float(value_range_raw[0]), float(value_range_raw[1]))
        else:
            value_range = None

        resolution = cfg.get("resolution", [1200, 800])
        if len(resolution) != 2:
            raise ValueError("[visualization] resolution must contain exactly two integers")

        levels = int(cfg.get("levels", 0))
        chunk_size = int(cfg.get("chunk_size", 500_000))
        if levels < 0:
            raise ValueError("[visualization] levels must be >= 0")
        if chunk_size <= 0:
            raise ValueError("[visualization] chunk_size must be > 0")

        timestep = cfg.get("timestep")
        if timestep is not None:
            timestep = int(timestep)

        return cls(
            enabled=enabled,
            field=str(field),
            output=Path(cfg.get("output", "visualization_output/flow_contour.png")),
            timestep=timestep,
            levels=levels,
            value_range=value_range,
            resolution=(int(resolution[0]), int(resolution[1])),
            cmap=str(cfg.get("cmap", "turbo")),
            chunk_size=chunk_size,
        )


@dataclass(frozen=True)
class SurfaceConfig:
    surf_data_dir: Path
    surf_geom_dir: Path
    surface_dt: float
    step: int = field(default=1)
    surface_static: bool = False

    @classmethod
    def from_dict(cls, cfg: dict) -> "SurfaceConfig":
        try:
            data_dir = Path(cfg["surf_data_dir"])
            geom_dir = Path(cfg["surf_geom_dir"])
            dt = float(cfg["surface_dt"])
            step = int(cfg.get("step", 1))  # optional, defaults to 1
            surface_static = bool(cfg.get("surface_static", False))
        except KeyError as e:
            raise ValueError(f"[surface] missing required key: {e.args[0]}")

        if dt < 0:
            raise ValueError("[surface] surface_dt must be >= 0")
        if step <= 0:
            raise ValueError("[surface] step must be > 0")

        return cls(
            surf_data_dir=data_dir,
            surf_geom_dir=geom_dir,
            surface_dt=dt,
            step=step,
            surface_static=surface_static,
        )


@dataclass(frozen=True)
class SolidConfig:
    solid_dir: Path
    solid_dt: float
    voxel_size: float
    step: int = field(default=1)

    @classmethod
    def from_dict(cls, cfg: dict) -> "SolidConfig":
        try:
            solid_dir = Path(cfg["solid_dir"])
            solid_dt = float(cfg["solid_dt"])
            voxel_size = float(cfg["voxel_size"])
            step = int(cfg.get("step", 1))  # optional, defaults to 1
        except KeyError as e:
            raise ValueError(f"[solid] missing required key: {e.args[0]}")

        if solid_dt < 0:
            raise ValueError("[solid] solid_dt must be >= 0")
        if voxel_size <= 0:
            raise ValueError("[solid] voxel_size must be > 0")
        if step <= 0:
            raise ValueError("[solid] step must be > 0")

        return cls(
            solid_dir=solid_dir,
            solid_dt=solid_dt,
            voxel_size=voxel_size,
            step=step,
        )


# -------------------------
# Filter config
# -------------------------

@dataclass(frozen=True)
class FilterConfig:
    enabled: bool = False
    slice_axis: Optional[str] = None
    slice_value: Optional[float] = None

    @classmethod
    def from_dict(cls, cfg: dict) -> "FilterConfig":
        enabled = bool(cfg.get("enabled", False))

        if not enabled:
            return cls()

        slice_axis = cfg.get("slice_axis", None)
        slice_value = cfg.get("slice_value", None)

        if slice_axis is None:
            raise ValueError("[filters] 'slice_axis' is required when enabled = true")
        if slice_axis not in {"x", "y", "z"}:
            raise ValueError(
                f"[filters] slice_axis must be 'x', 'y', or 'z', got '{slice_axis}'"
            )
        if slice_value is None:
            raise ValueError("[filters] 'slice_value' is required when enabled = true")

        return cls(
            enabled=enabled,
            slice_axis=slice_axis,
            slice_value=float(slice_value),
        )


# -------------------------
# Postprocess config
# -------------------------

@dataclass(frozen=True)
class PostprocessConfig:
    archive: bool = False
    sync_enabled: bool = False

    @classmethod
    def from_dict(cls, cfg: dict) -> "PostprocessConfig":
        return cls(
            archive=bool(cfg.get("archive", False)),
            sync_enabled=bool(cfg.get("sync_enabled", False)),
        )
