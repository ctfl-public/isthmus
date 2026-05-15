import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pyvista as pv
from tqdm import tqdm

from pivot.base_converter import BaseConverter
from pivot.config_manager import ConfigManager
from pivot.simulation_data import SimulationData
from pivot.sparta_items import SpartaItems

log = logging.getLogger(__name__)

class FlowConverter(BaseConverter):
    """Converts SPARTA flow data to VTK."""
    def __init__(self, config : ConfigManager, sim_data : SimulationData):
        super().__init__(config, sim_data, solver_name="flow")
        self.flow_output_ext = ".vtr"
        
    @property
    def flow_dir(self) -> Path:
        """Return directory where flow data files are located"""
        return self.dir
    
    @property
    def flow_dt(self) -> float:
        """Return flow timestep"""
        return self.settings.flow_dt
    
    @property
    def field_map(self) -> dict[str, str]:
        """Return the field mapping dictionary from config."""
        return self.settings.field_map
        
    def processFlowDirectory(self):
        """Processes flow files in a specified directory"""
        flow_files = self.getFiles(self.flow_dir, step=self.step)
        
        with tqdm(flow_files, desc="Processing flow files") as pbar:
            for filepath in pbar:
                pbar.set_postfix_str(f"reading {filepath.name}")
                timestep_data = self.processFlowFile(filepath)
                if self.slice_filter:
                    timestep_data = self.slice_filter.apply_flow(timestep_data)
                pbar.set_postfix_str(f"building grid")
                is_3d = 'zlo' in timestep_data['field_items']
                if self.slice_filter and is_3d:
                    grid = self.createFlowSlice(timestep_data)
                else:
                    grid = self.createFlowGrid(timestep_data)
                pbar.set_postfix_str(f"writing VTK")
                self.writeFlowVTK(grid, timestep_data)
            pbar.set_postfix_str("done")

    def processFlowFile(self, filename):
        """Processes a single flow file and returns the data for the given timestep."""
        timestep_data = {
            'filepath': filename,
            'timestep': None,
            'num_cells': None,
            'box_bounds': None,
            "field_items": None}

        with open(filename, "r") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                line = line.strip()

                if line.startswith(SpartaItems.TIMESTEP):
                    timestep_data["timestep"] = int(f.readline().strip())
                    continue

                if line.startswith(SpartaItems.NUM_CELLS):
                    timestep_data['num_cells'] = int(f.readline().strip())
                    continue

                if line.startswith(SpartaItems.BOX_BOUNDS):
                    xbounds = [float(x) for x in f.readline().strip().split()]
                    ybounds = [float(x) for x in f.readline().strip().split()]
                    zbounds = [float(x) for x in f.readline().strip().split()]
                    timestep_data["box_bounds"] = (xbounds, ybounds, zbounds)
                    continue

                if line.startswith(SpartaItems.CELLS):
                    field_items = line.split()[2:]
                    timestep_data['field_items'] = field_items
                    num_cells = timestep_data['num_cells']
                    num_cols = len(field_items)

                    # Pre-allocate once and fill line-by-line to keep peak memory at
                    # exactly (num_cells * num_cols * 4) bytes. np.loadtxt accumulates
                    # Python lists internally and can use several times that on large dumps.
                    cell_array = np.empty((num_cells, num_cols), dtype=np.float32)
                    for i in tqdm(range(num_cells), desc=f"  Reading {filename.name}", leave=False, unit="cells"):
                        cell_array[i] = f.readline().split()
                    timestep_data['cell_array'] = cell_array

                    for trailing_line in f:
                        if trailing_line.strip().startswith("ITEM:"):
                            log.warning(
                                "Unexpected additional ITEM block in flow file '%s' after cell data. "
                                "Trailing data will be ignored.",
                                filename,
                            )
                            break
                    break

        return timestep_data

    def isFieldItemsCorrect(self, timestep_data, is_2d):
        """Checks if the required flow field items are in the file and returns a list of missing items for user to be aware of."""
        field_items = set(timestep_data['field_items'])
        required_items_3d = SpartaItems.REQUIRED_FLOW_FIELDS
        required_items_2d = SpartaItems.REQUIRED_FLOW_FIELDS_2D

        if is_2d:
            required_items = required_items_2d
        else:
            required_items = required_items_3d

        missing_items = sorted(required_items - field_items)
        has_correct_items = len(missing_items) == 0

        return has_correct_items, missing_items
    

    def getGeomIndexes(self, timestep_data):
        """Gets the index of the geometry columns in the SPARTA dump file"""
        field_items = timestep_data['field_items']
        
        if 'zlo' not in field_items:
            return {
                    f'i_{item}': idx for idx, item in enumerate(field_items)
                    if item in {'xlo', 'xhi', 'ylo', 'yhi'}
                }

        return {
            f"i_{item}": idx
            for idx, item in enumerate(field_items)
            if item in {'xlo', 'xhi', 'ylo', 'yhi', 'zlo', 'zhi'}
        }
                    

    def _attach_rectilinear_cell_data(
        self,
        grid: pv.RectilinearGrid,
        field_names: list[str],
        field_data: np.ndarray,
        cell_shape: tuple[int, int, int],
        cell_bounds: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ):
        """Attach SPARTA cell data to a rectilinear lattice."""
        log.debug("Attaching %d field(s) to rectilinear grid.", len(field_names))

        unmapped_keys = set(self.field_map.keys()) - set(field_names)
        if unmapped_keys:
            log.warning("field_map contains key(s) not found in field_names: %s", sorted(unmapped_keys))

        x_start, x_stop, y_start, y_stop, z_start, z_stop = cell_bounds
        covered = np.zeros(cell_shape, dtype=bool)
        for ix0, ix1, iy0, iy1, iz0, iz1 in zip(x_start, x_stop, y_start, y_stop, z_start, z_stop):
            covered[ix0:ix1, iy0:iy1, iz0:iz1] = True

        for idx, name in enumerate(field_names):
            mapped_name = self.field_map.get(name, name)
            values = np.full(cell_shape, np.nan, dtype=np.float32)
            for cell_idx, (ix0, ix1, iy0, iy1, iz0, iz1) in enumerate(
                zip(x_start, x_stop, y_start, y_stop, z_start, z_stop)
            ):
                values[ix0:ix1, iy0:iy1, iz0:iz1] = field_data[cell_idx, idx]
            grid.cell_data[mapped_name] = values.ravel(order="F")

        if not np.all(covered):
            hidden_cell = np.uint8(32)
            grid.cell_data["vtkGhostType"] = np.where(
                covered.ravel(order="F"),
                np.uint8(0),
                hidden_cell,
            )

        log.debug("Grid cell_data arrays: %s", list(grid.cell_data.keys()))

    def _create_rectilinear_grid(
        self,
        field_names: list[str],
        field_data: np.ndarray,
        xlo: np.ndarray,
        xhi: np.ndarray,
        ylo: np.ndarray,
        yhi: np.ndarray,
        zlo: np.ndarray,
        zhi: np.ndarray,
        collapsed_axis: Optional[str] = None,
        collapsed_value: Optional[np.float32] = None,
    ):
        """Create a VTK rectilinear grid from axis-aligned SPARTA cell bounds."""
        if collapsed_axis == "x":
            x = np.array([collapsed_value], dtype=np.float32)
        else:
            x = np.unique(np.concatenate((xlo, xhi))).astype(np.float32)

        if collapsed_axis == "y":
            y = np.array([collapsed_value], dtype=np.float32)
        else:
            y = np.unique(np.concatenate((ylo, yhi))).astype(np.float32)

        if collapsed_axis == "z":
            z = np.array([collapsed_value], dtype=np.float32)
        else:
            z = np.unique(np.concatenate((zlo, zhi))).astype(np.float32)

        cell_shape = (
            max(len(x) - 1, 1),
            max(len(y) - 1, 1),
            max(len(z) - 1, 1),
        )
        rect_cells = int(np.prod(cell_shape))
        expansion = rect_cells / max(len(field_data), 1)
        log.info(
            "Flow rectilinear lattice: points=(%d, %d, %d), cells=%d, source_cells=%d, expansion=%.3g",
            len(x),
            len(y),
            len(z),
            rect_cells,
            len(field_data),
            expansion,
        )

        x_start = np.zeros(len(field_data), dtype=np.int64) if collapsed_axis == "x" else np.searchsorted(x, xlo)
        x_stop = np.ones(len(field_data), dtype=np.int64) if collapsed_axis == "x" else np.searchsorted(x, xhi)
        y_start = np.zeros(len(field_data), dtype=np.int64) if collapsed_axis == "y" else np.searchsorted(y, ylo)
        y_stop = np.ones(len(field_data), dtype=np.int64) if collapsed_axis == "y" else np.searchsorted(y, yhi)
        z_start = np.zeros(len(field_data), dtype=np.int64) if collapsed_axis == "z" else np.searchsorted(z, zlo)
        z_stop = np.ones(len(field_data), dtype=np.int64) if collapsed_axis == "z" else np.searchsorted(z, zhi)

        grid = pv.RectilinearGrid(x, y, z)
        if grid.n_cells != rect_cells:
            log.debug("PyVista rectilinear grid reports %d cells for logical shape %s.", grid.n_cells, cell_shape)

        cell_bounds = (x_start, x_stop, y_start, y_stop, z_start, z_stop)
        self._attach_rectilinear_cell_data(grid, field_names, field_data, cell_shape, cell_bounds)
        return grid

    def createFlowGrid(self, timestep_data):
        """Create VTK rectilinear grid from SPARTA flow data."""
        
        field_items = timestep_data['field_items']
        data = timestep_data['cell_array']
        n_cells = len(data)
        
        # 2D case requires a different approach
        is_2d = 'zlo' not in field_items
        
        # check if user has required items for flow grid
        hasCorrectFieldItems, missingFieldItems = self.isFieldItemsCorrect(timestep_data, is_2d)
        if not hasCorrectFieldItems:
            log.error("Missing required flow field item(s): %s", missingFieldItems)
            raise ValueError(f"Missing required flow field item(s): {missingFieldItems}")
        
        if is_2d:
            # 2D case: id, xlo, ylo, xhi, yhi, [fields...]
            geom_indexes = self.getGeomIndexes(timestep_data)
            i_xlo, i_ylo, i_xhi, i_yhi = geom_indexes['i_xlo'], geom_indexes['i_ylo'], geom_indexes['i_xhi'], geom_indexes['i_yhi']
            # .copy() so we can free data before building vertex arrays
            xlo, ylo = data[:, i_xlo].copy(), data[:, i_ylo].copy()
            xhi, yhi = data[:, i_xhi].copy(), data[:, i_yhi].copy()

            dx = np.mean(xhi - xlo)
            dy = np.mean(yhi - ylo)
            z_thickness = max(dx, dy)
            zlo = np.full(n_cells, -z_thickness/2, dtype=np.float32)
            zhi = np.full(n_cells,  z_thickness/2, dtype=np.float32)

            field_start_idx = 5
        else:
            geom_indexes = self.getGeomIndexes(timestep_data)
            i_xlo, i_ylo, i_zlo = geom_indexes['i_xlo'], geom_indexes['i_ylo'], geom_indexes['i_zlo']
            i_xhi, i_yhi, i_zhi = geom_indexes['i_xhi'], geom_indexes['i_yhi'], geom_indexes['i_zhi']
            # .copy() so we can free data before building vertex arrays
            xlo, ylo, zlo = data[:, i_xlo].copy(), data[:, i_ylo].copy(), data[:, i_zlo].copy()
            xhi, yhi, zhi = data[:, i_xhi].copy(), data[:, i_yhi].copy(), data[:, i_zhi].copy()

            field_start_idx = 7

        # Extract field data as a copy, then free the full cell_array.
        field_data = data[:, field_start_idx:].copy()
        field_names = field_items[field_start_idx:]
        del data
        timestep_data['cell_array'] = None
        grid = self._create_rectilinear_grid(field_names, field_data, xlo, xhi, ylo, yhi, zlo, zhi)
        del field_data
        return grid

    def createFlowSlice(self, timestep_data):
        """Create a flat rectilinear VTK grid from a sliced 3D flow dataset."""
        field_items = timestep_data['field_items']
        data = timestep_data['cell_array']
        n_cells = len(data)
        axis = self.slice_filter.axis
        value = np.float32(self.slice_filter.value)

        is_2d = 'zlo' not in field_items
        hasCorrectFieldItems, missingFieldItems = self.isFieldItemsCorrect(timestep_data, is_2d)
        if not hasCorrectFieldItems:
            log.error("Missing required flow field item(s): %s", missingFieldItems)
            raise ValueError(f"Missing required flow field item(s): {missingFieldItems}")

        gi = self.getGeomIndexes(timestep_data)
        xlo = data[:, gi['i_xlo']].copy()
        xhi = data[:, gi['i_xhi']].copy()
        ylo = data[:, gi['i_ylo']].copy()
        yhi = data[:, gi['i_yhi']].copy()
        zlo = data[:, gi['i_zlo']].copy()
        zhi = data[:, gi['i_zhi']].copy()
        field_start_idx = 7

        field_data = data[:, field_start_idx:].copy()
        field_names = field_items[field_start_idx:]
        del data
        timestep_data['cell_array'] = None

        grid = self._create_rectilinear_grid(
            field_names,
            field_data,
            xlo,
            xhi,
            ylo,
            yhi,
            zlo,
            zhi,
            collapsed_axis=axis,
            collapsed_value=value,
        )
        del field_data
        return grid

    def writeFlowVTK(self, grid: pv.RectilinearGrid, timestep_data):
        """Save a flow grid to VTK and track timestep."""
        self.flow_output_ext = ".vtr" if isinstance(grid, pv.RectilinearGrid) else ".vtu"
        self.writeVTK(
            data_obj=grid,
            timestep=timestep_data['timestep'],
            solver_name="flow",
            folder="flow_output",
            ext=self.flow_output_ext
        )
          
    def writeFlowPVD(self):
        self.writePVD("flow", ext=self.flow_output_ext)

def runFlow():
    """Run FlowConverter standalone against ./config.toml."""
    print(
        "WARNING: Running FlowConverter as a standalone module.\n"
        "Only flow-related settings from config.toml will be applied.\n"
        "Syncing with solid or surface data will NOT occur."
    )
    start = time.perf_counter()
    config = ConfigManager("config.toml")
    sim_data = SimulationData()
    fc = FlowConverter(config, sim_data)
    fc.processFlowDirectory()
    end = time.perf_counter()
    elapsed_time = end - start
    print(f"Total time to process flow steps: {elapsed_time:.2f} s")
    fc.writeFlowPVD()
    
if __name__ == "__main__":
    runFlow()
