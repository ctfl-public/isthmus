import numpy as np
import logging

log = logging.getLogger(__name__)

VALID_AXES = {"x", "y", "z"}


class SliceFilter:
    """Keep only cells that intersect a plane at a fixed coordinate value.

    Parameters
    ----------
    axis : str
        'x', 'y', or 'z'
    value : float
        Coordinate of the slice plane.
    """

    def __init__(self, axis: str, value: float):
        if axis not in VALID_AXES:
            raise ValueError(f"SliceFilter: axis must be one of {VALID_AXES}, got '{axis}'")
        self.axis = axis
        self.value = value

    # ------------------------------------------------------------------
    # Flow
    # ------------------------------------------------------------------

    def apply_flow(self, timestep_data: dict) -> dict:
        """Return a copy of timestep_data with cell_array filtered to cells
        that straddle the slice plane (lo <= value <= hi)."""
        field_items = timestep_data["field_items"]
        data = timestep_data["cell_array"]

        lo_key = f"{self.axis}lo"
        hi_key = f"{self.axis}hi"

        if lo_key not in field_items:
            raise KeyError(
                f"SliceFilter (flow): '{lo_key}' not found in field_items {field_items}. "
                "Make sure the flow dump includes cell bounds."
            )
        if hi_key not in field_items:
            raise KeyError(
                f"SliceFilter (flow): '{hi_key}' not found in field_items {field_items}. "
                "Make sure the flow dump includes cell bounds."
            )

        i_lo = field_items.index(lo_key)
        i_hi = field_items.index(hi_key)

        domain_lo = float(data[:, i_lo].min())
        domain_hi = float(data[:, i_hi].max())
        if not (domain_lo <= self.value <= domain_hi):
            raise ValueError(
                f"SliceFilter (flow): slice_value={self.value} is outside the domain "
                f"[{domain_lo}, {domain_hi}] in the '{self.axis}' direction."
            )

        mask = (data[:, i_lo] <= self.value) & (data[:, i_hi] >= self.value)
        n_before = len(data)
        filtered = data[mask]
        n_after = len(filtered)

        if n_after == 0:
            raise ValueError(
                f"SliceFilter (flow): No cells found at {self.axis}={self.value}. "
                "The domain bounds are correct but no cell spans this plane — "
                "check for a gap in the grid."
            )

        log.debug(
            "SliceFilter (flow): kept %d / %d cells at %s=%.6g (domain=[%.4g, %.4g])",
            n_after, n_before, self.axis, self.value, domain_lo, domain_hi,
        )

        result = dict(timestep_data)
        result["cell_array"] = filtered
        result["num_cells"] = n_after
        return result

    # ------------------------------------------------------------------
    # Solid
    # ------------------------------------------------------------------

    def apply_solid(self, timestep_data: dict, voxel_size: float) -> dict:
        """Return a copy of timestep_data with data filtered to voxels whose
        extent spans the slice plane (center ± half_voxel_size)."""
        data = timestep_data["data"]

        axis_col = {"x": 1, "y": 2, "z": 3}
        col = axis_col[self.axis]

        if data.shape[1] <= col:
            raise ValueError(
                f"SliceFilter (solid): expected at least {col + 1} columns (id, x, y, z, ...) "
                f"but data has {data.shape[1]}."
            )

        half = voxel_size / 2.0
        centers = data[:, col]

        domain_lo = float(centers.min()) - half
        domain_hi = float(centers.max()) + half
        if not (domain_lo <= self.value <= domain_hi):
            raise ValueError(
                f"SliceFilter (solid): slice_value={self.value} is outside the domain "
                f"[{domain_lo:.6g}, {domain_hi:.6g}] in the '{self.axis}' direction."
            )

        mask = (centers - half <= self.value) & (centers + half >= self.value)
        n_before = len(data)
        filtered = data[mask]
        n_after = len(filtered)

        if n_after == 0:
            raise ValueError(
                f"SliceFilter (solid): No voxels found at {self.axis}={self.value} "
                f"(voxel_size={voxel_size}). Check slice_value and voxel_size."
            )

        log.debug(
            "SliceFilter (solid): kept %d / %d voxels at %s=%.6g",
            n_after, n_before, self.axis, self.value,
        )

        result = dict(timestep_data)
        result["data"] = filtered
        result["count"] = n_after
        return result
