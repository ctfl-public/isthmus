"""
Typed configuration objects for Paraview Tools.

These classes represent *validated solver-specific configuration*.
They contain no file I/O and no TOML parsing logic.
"""

from dataclasses import dataclass, field
from pathlib import Path

# -------------------------
# Solver configs
# -------------------------

@dataclass(frozen=True)
class FlowConfig:
    flow_dir: Path
    flow_dt: float
    step: int = field(default=1)

    @classmethod
    def from_dict(cls, cfg: dict) -> "FlowConfig":
        try:
            flow_dir = Path(cfg["flow_dir"])
            flow_dt = float(cfg["flow_dt"])
            step = int(cfg.get("step", 1))  # optional, defaults to 1
        except KeyError as e:
            raise ValueError(f"[flow] missing required key: {e.args[0]}")

        if flow_dt < 0:
            raise ValueError("[flow] flow_dt must be >= 0")
        if step <= 0:
            raise ValueError("[flow] step must be > 0")

        return cls(flow_dir=flow_dir, flow_dt=flow_dt, step=step)


@dataclass(frozen=True)
class SurfaceConfig:
    surf_data_dir: Path
    surf_geom_dir: Path
    surface_dt: float
    step: int = field(default=1)

    @classmethod
    def from_dict(cls, cfg: dict) -> "SurfaceConfig":
        try:
            data_dir = Path(cfg["surf_data_dir"])
            geom_dir = Path(cfg["surf_geom_dir"])
            dt = float(cfg["surface_dt"])
            step = int(cfg.get("step", 1))  # optional, defaults to 1
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
# Postprocess config
# -------------------------

@dataclass(frozen=True)
class PostprocessConfig:
    archive: bool = False
    sync_enabled: bool = False
    vis_enabled: bool = False

    @classmethod
    def from_dict(cls, cfg: dict) -> "PostprocessConfig":
        return cls(
            archive=bool(cfg.get("archive", False)),
            sync_enabled=bool(cfg.get("sync_enabled", False)),
            vis_enabled=bool(cfg.get("vis_enabled", False)),
        )
