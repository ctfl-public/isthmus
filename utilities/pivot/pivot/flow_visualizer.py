import logging
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from pivot.config_manager import ConfigManager
from pivot.sparta_items import SpartaItems

log = logging.getLogger(__name__)


def render_flow_contour(config: ConfigManager) -> Path:
    """Render a single flow-field contour PNG without building a VTK mesh."""
    vis = config.visualization
    flow = config.flow
    if flow is None:
        raise ValueError("[visualization] flow visualization requires a [flow] config section")

    filepath = _select_flow_file(flow.flow_dir, vis.timestep)
    header = _read_flow_header(filepath)
    field_name = _resolve_field_name(vis.field, flow.field_map, header["field_items"])

    data = _rasterize_flow_field(
        filepath=filepath,
        header=header,
        field_name=field_name,
        resolution=vis.resolution,
        chunk_size=vis.chunk_size,
        filters=config.filters,
    )

    # Auto-scale to the actual data range when value_range is not configured.
    if vis.value_range is None:
        finite_vals = data["raster"][np.isfinite(data["raster"])]
        vmin, vmax = float(finite_vals.min()), float(finite_vals.max())
        if vmin >= vmax:
            vmax = vmin + 1.0
        value_range = (vmin, vmax)
    else:
        value_range = vis.value_range

    output = vis.output
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_contour_png(
        output=output,
        raster=data["raster"],
        x_edges=data["x_edges"],
        y_edges=data["y_edges"],
        field_label=vis.field,
        timestep=header["timestep"],
        value_range=value_range,
        levels=vis.levels,
        cmap=vis.cmap,
        axis_a=data["axis_a"],
        axis_b=data["axis_b"],
        slice_axis=data["slice_axis"],
        slice_value=data["slice_value"],
    )

    print(f"Flow contour written: {output}")
    return output


def _select_flow_file(flow_dir: Path, timestep: Optional[int]) -> Path:
    files = sorted(Path(flow_dir).iterdir())
    if not files:
        raise FileNotFoundError(f"No flow files found in {flow_dir}")

    if timestep is None:
        return files[0]

    matches = [path for path in files if path.name.endswith(str(timestep)) or f".{timestep}" in path.name]
    if not matches:
        raise FileNotFoundError(f"No flow file matching timestep {timestep} found in {flow_dir}")
    return matches[0]


def _read_flow_header(filepath: Path) -> dict:
    header = {
        "timestep": None,
        "num_cells": None,
        "box_bounds": None,
        "field_items": None,
    }

    with open(filepath, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            stripped = line.strip()

            if stripped.startswith(SpartaItems.TIMESTEP):
                header["timestep"] = int(f.readline().strip())
                continue

            if stripped.startswith(SpartaItems.NUM_CELLS):
                header["num_cells"] = int(f.readline().strip())
                continue

            if stripped.startswith(SpartaItems.BOX_BOUNDS):
                xbounds = [float(x) for x in f.readline().strip().split()]
                ybounds = [float(x) for x in f.readline().strip().split()]
                zbounds = [float(x) for x in f.readline().strip().split()]
                header["box_bounds"] = (xbounds, ybounds, zbounds)
                continue

            if stripped.startswith(SpartaItems.CELLS):
                header["field_items"] = stripped.split()[2:]
                break

    if header["num_cells"] is None or header["field_items"] is None or header["box_bounds"] is None:
        raise ValueError(f"Could not parse a complete SPARTA flow header from {filepath}")
    return header


def _resolve_field_name(requested: str, field_map: dict[str, str], field_items: list[str]) -> str:
    if requested in field_items:
        return requested

    reverse_map = {mapped: raw for raw, mapped in field_map.items()}
    if requested in reverse_map and reverse_map[requested] in field_items:
        return reverse_map[requested]

    raise ValueError(
        f"Field '{requested}' was not found in the flow file or field_map. "
        f"Available mapped names: {sorted(field_map.values())}"
    )


def _rasterize_flow_field(
    filepath: Path,
    header: dict,
    field_name: str,
    resolution: tuple[int, int],
    chunk_size: int,
    filters,
) -> dict:
    field_items = header["field_items"]
    is_3d = "zlo" in field_items
    plane = _get_visualization_plane(field_items, filters)
    axis_a, axis_b = plane["plot_axes"]
    required = [
        f"{axis_a}lo",
        f"{axis_b}lo",
        f"{axis_a}hi",
        f"{axis_b}hi",
        field_name,
    ]
    if plane["slice_axis"] is not None:
        required.extend([f"{plane['slice_axis']}lo", f"{plane['slice_axis']}hi"])

    missing = [name for name in required if name not in field_items]
    if missing:
        raise ValueError(f"Cannot render flow contour; missing field(s): {missing}")

    usecols = [field_items.index(name) for name in required]
    bounds_by_axis = {
        "x": header["box_bounds"][0],
        "y": header["box_bounds"][1],
        "z": header["box_bounds"][2],
    }
    if is_3d and plane["slice_axis"] is None:
        log.warning(
            "Rendering 3D flow data without an enabled slice filter. "
            "Values will be projected/averaged over the full %s direction.",
            plane["collapsed_axis"],
        )

    axis_a_bounds = bounds_by_axis[axis_a]
    axis_b_bounds = bounds_by_axis[axis_b]
    nx, ny = resolution
    x_edges = np.linspace(axis_a_bounds[0], axis_a_bounds[1], nx + 1, dtype=np.float64)
    y_edges = np.linspace(axis_b_bounds[0], axis_b_bounds[1], ny + 1, dtype=np.float64)
    sums = np.zeros((nx, ny), dtype=np.float64)
    counts = np.zeros((nx, ny), dtype=np.float64)
    kept_cells = 0

    with open(filepath, "r") as f:
        for line in f:
            if line.strip().startswith(SpartaItems.CELLS):
                break

        remaining = header["num_cells"]
        with tqdm(total=remaining, desc="Rendering flow contour", unit="cells") as pbar:
            while remaining > 0:
                rows = min(chunk_size, remaining)
                chunk = np.loadtxt(
                    f,
                    dtype=np.float32,
                    max_rows=rows,
                    usecols=usecols,
                )
                if chunk.ndim == 1:
                    chunk = chunk.reshape(1, -1)

                xlo = chunk[:, 0]
                ylo = chunk[:, 1]
                xhi = chunk[:, 2]
                yhi = chunk[:, 3]
                values = chunk[:, 4]
                valid = np.isfinite(values)
                if plane["slice_axis"] is not None:
                    lo = chunk[:, 5]
                    hi = chunk[:, 6]
                    valid &= (lo <= plane["slice_value"]) & (hi >= plane["slice_value"])

                if np.any(valid):
                    kept_cells += int(valid.sum())
                    _accumulate_cell_footprints(
                        sums=sums,
                        counts=counts,
                        x_edges=x_edges,
                        y_edges=y_edges,
                        xlo=xlo[valid],
                        xhi=xhi[valid],
                        ylo=ylo[valid],
                        yhi=yhi[valid],
                        values=values[valid],
                    )

                remaining -= len(chunk)
                pbar.update(len(chunk))

    if plane["slice_axis"] is not None and kept_cells == 0:
        raise ValueError(
            f"Visualization slice kept no cells at {plane['slice_axis']}={plane['slice_value']}. "
            "Check [filters] slice_axis and slice_value."
        )

    raster = np.full_like(sums, np.nan)
    np.divide(sums, counts, out=raster, where=counts > 0)
    log.info(
        "Visualization raster: plotted %s-%s plane%s, kept %d finite source cells.",
        axis_a,
        axis_b,
        (
            f" at {plane['slice_axis']}={plane['slice_value']}"
            if plane["slice_axis"] is not None
            else ""
        ),
        kept_cells,
    )

    return {
        "raster": raster,
        "x_edges": x_edges,
        "y_edges": y_edges,
        "axis_a": axis_a,
        "axis_b": axis_b,
        "slice_axis": plane["slice_axis"],
        "slice_value": plane["slice_value"],
    }


def _accumulate_cell_footprints(
    sums: np.ndarray,
    counts: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    xlo: np.ndarray,
    xhi: np.ndarray,
    ylo: np.ndarray,
    yhi: np.ndarray,
    values: np.ndarray,
):
    """Accumulate cell-centered values over every raster bin touched by each cell footprint."""
    nx, ny = sums.shape
    x_start = np.searchsorted(x_edges, xlo, side="right") - 1
    x_stop = np.searchsorted(x_edges, xhi, side="left")
    y_start = np.searchsorted(y_edges, ylo, side="right") - 1
    y_stop = np.searchsorted(y_edges, yhi, side="left")

    x_start = np.clip(x_start, 0, nx - 1)
    x_stop = np.clip(np.maximum(x_stop, x_start + 1), 1, nx)
    y_start = np.clip(y_start, 0, ny - 1)
    y_stop = np.clip(np.maximum(y_stop, y_start + 1), 1, ny)

    for ix0, ix1, iy0, iy1, value in zip(x_start, x_stop, y_start, y_stop, values):
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        sums[ix0:ix1, iy0:iy1] += value
        counts[ix0:ix1, iy0:iy1] += 1.0


def _get_visualization_plane(field_items: list[str], filters) -> dict:
    is_3d = "zlo" in field_items
    if filters.enabled:
        slice_axis = filters.slice_axis
        slice_value = filters.slice_value
    else:
        slice_axis = None
        slice_value = None

    if not is_3d:
        if slice_axis == "z":
            raise ValueError(
                "Visualization cannot apply a z slice to 2D flow data because zlo/zhi are not present."
            )
        return {
            "plot_axes": ("x", "y"),
            "slice_axis": slice_axis,
            "slice_value": np.float32(slice_value) if slice_value is not None else None,
            "collapsed_axis": slice_axis,
        }

    if slice_axis is None:
        return {
            "plot_axes": ("x", "y"),
            "slice_axis": None,
            "slice_value": None,
            "collapsed_axis": "z",
        }

    plot_axes_by_slice = {
        "x": ("y", "z"),
        "y": ("x", "z"),
        "z": ("x", "y"),
    }
    return {
        "plot_axes": plot_axes_by_slice[slice_axis],
        "slice_axis": slice_axis,
        "slice_value": np.float32(slice_value),
        "collapsed_axis": slice_axis,
    }


def _write_contour_png(
    output: Path,
    raster: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    field_label: str,
    timestep: int,
    value_range: tuple[float, float],
    levels: int,
    cmap: str,
    axis_a: str,
    axis_b: str,
    slice_axis: Optional[str],
    slice_value: Optional[float],
):
    import os
    import tempfile
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "pivot_matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vmin, vmax = value_range
    finite = raster[np.isfinite(raster)]
    if finite.size == 0:
        raise ValueError(f"Cannot render {field_label}: all raster bins are NaN/no-data")

    color_map = plt.get_cmap(cmap).copy()
    color_map.set_bad("lightgray")
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    masked = np.ma.masked_invalid(raster)

    slice_label = f", {slice_axis}={slice_value:.6g}" if slice_axis is not None else ""
    title = f"{field_label}  |  timestep {timestep}{slice_label}"

    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)

    # bilinear interpolation gives a smooth colour gradient (no discrete contour bands)
    im = ax.imshow(
        masked.T,
        origin="lower",
        extent=extent,
        cmap=color_map,
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
        aspect="equal",
    )

    # optional iso-lines (levels=0 means clean gradient, no lines)
    if levels > 0:
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
        contour_lines = ax.contour(
            x_centers,
            y_centers,
            masked.T,
            levels=np.linspace(vmin, vmax, levels),
            colors="black",
            linewidths=0.35,
            alpha=0.4,
        )
        ax.clabel(contour_lines, inline=True, fontsize=7, fmt="%.2g")

    colorbar = fig.colorbar(im, ax=ax, label=field_label)
    colorbar.ax.text(
        0.5, -0.06, "gray = no data", ha="center", va="top",
        transform=colorbar.ax.transAxes, fontsize=8,
    )
    ax.set_title(title)
    ax.set_xlabel(axis_a)
    ax.set_ylabel(axis_b)
    fig.savefig(output, dpi=220)
    plt.close(fig)

    log.info(
        "Rendered %s timestep %s with %.2f%% finite bins. NaN/no-data bins are light gray.",
        field_label,
        timestep,
        100.0 * finite.size / raster.size,
    )
