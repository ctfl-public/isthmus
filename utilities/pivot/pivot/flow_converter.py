# standard modules
import numpy as np
import pyvista as pv
from pathlib import Path
from typing import *
from tqdm import tqdm
import time
import sys

import logging
log = logging.getLogger(__name__)

# custom modules
from pivot.simulation_data import SimulationData
from pivot.config_manager import ConfigManager
from pivot.base_converter import BaseConverter
from pivot.sparta_items import SpartaItems

class FlowConverter(BaseConverter):
    """Converts SPARTA flow data to VTK.
    
    Attributes
    ----------
    config : ConfigManager
        ConfigManager object to get flow settings
    sim_data : SimulationData
        SimulationData object to pass timestep information during runtime
    solver_name = "flow"
        Used to init parent class BaseConverter
    required_keys = ["flow_dir", "flow_dt"]
        Keys that should be set in config.toml before the FlowConverter can initialize
    
    """
    def __init__(self, config : ConfigManager, sim_data : SimulationData):
        super().__init__(config, sim_data, solver_name="flow")
        
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
                    items = line.split()
                    field_items = items[2:]
                    timestep_data['field_items'] = field_items
                    num_cells = timestep_data['num_cells']
                    num_cols = len(field_items)

                    cell_array = np.empty((num_cells, num_cols), dtype=np.float32)
                    for i in tqdm(range(num_cells), desc=f"  Reading {Path(filename).name}", leave=False, unit="cells"):
                        cell_array[i] = f.readline().split()
                    timestep_data['cell_array'] = cell_array

                    # check if user output another timestep
                    for trailing_line in f:
                        if trailing_line.strip().startswith("ITEM:"):
                            log.warning(
                                f"Unexpected additional ITEM block detected in flow file '{filename}' after cell data. "
                                "This tool expects exactly one timestep per file. Trailing data will be ignored."
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
                    

    def createFlowGrid(self, timestep_data):
        """Create VTK unstructured grid from SPARTA flow data."""
        
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
        # This drops ~2 GB before allocating the vertex/connectivity arrays.
        field_data = data[:, field_start_idx:].copy()
        field_names = field_items[field_start_idx:]
        del data
        timestep_data['cell_array'] = None

        # Create vertices — float32 halves the memory vs float64 (8 verts × 3 coords per cell)
        verts = np.empty((n_cells, 8, 3), dtype=np.float32)
        verts[:, 0] = np.column_stack([xlo, ylo, zlo])
        verts[:, 1] = np.column_stack([xhi, ylo, zlo])
        verts[:, 2] = np.column_stack([xlo, yhi, zlo])
        verts[:, 3] = np.column_stack([xhi, yhi, zlo])
        verts[:, 4] = np.column_stack([xlo, ylo, zhi])
        verts[:, 5] = np.column_stack([xhi, ylo, zhi])
        verts[:, 6] = np.column_stack([xlo, yhi, zhi])
        verts[:, 7] = np.column_stack([xhi, yhi, zhi])
        vertices = verts.reshape(n_cells * 8, 3)
        del xlo, ylo, zlo, xhi, yhi, zhi

        # Create cell connectivity — int32 is sufficient for < ~268M total vertices
        base = np.arange(n_cells, dtype=np.int32) * 8
        cells = np.empty((n_cells, 9), dtype=np.int32)
        cells[:, 0] = 8
        cells[:, 1:] = base[:, np.newaxis] + np.arange(8, dtype=np.int32)
        del base

        # Cell types (VTK_VOXEL = 11)
        cell_types = np.full(n_cells, 11, dtype=np.uint8)

        grid = pv.UnstructuredGrid(cells.ravel(), cell_types, vertices)
        del cells, vertices
        
        # Attach field data to grid
        log.debug("Attaching %d field(s) to grid. field_map has %d entry/entries.", len(field_names), len(self.field_map))

        unmapped_keys = set(self.field_map.keys()) - set(field_names)
        if unmapped_keys:
            log.warning("field_map contains key(s) not found in field_names: %s", sorted(unmapped_keys))

        for idx, name in enumerate(field_names):
            mapped_name = self.field_map.get(name, name)
            if mapped_name != name:
                log.debug("  Renaming field '%s' -> '%s'", name, mapped_name)
            else:
                log.debug("  Field '%s' (no mapping)", name)
            grid.cell_data[mapped_name] = field_data[:, idx]

        log.debug("Grid cell_data arrays: %s", list(grid.cell_data.keys()))

        return grid

    def createFlowSlice(self, timestep_data):
        """Create a flat 2D VTK surface from a sliced flow dataset.

        Each filtered cell is projected onto the slice plane as a VTK_QUAD,
        producing a true flat surface that spans the full domain cross-section
        with no artificial thickness in the slice direction.
        """
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

        # Project each cell onto the slice plane as a quad.
        # Vertex winding is CCW when viewed from the positive axis direction.
        verts = np.empty((n_cells, 4, 3), dtype=np.float32)
        fill = np.full(n_cells, value, dtype=np.float32)

        if axis == 'z':
            # XY plane at z=value: corners span (xlo..xhi) x (ylo..yhi)
            verts[:, 0] = np.column_stack([xlo, ylo, fill])
            verts[:, 1] = np.column_stack([xhi, ylo, fill])
            verts[:, 2] = np.column_stack([xhi, yhi, fill])
            verts[:, 3] = np.column_stack([xlo, yhi, fill])
        elif axis == 'y':
            # XZ plane at y=value: corners span (xlo..xhi) x (zlo..zhi)
            verts[:, 0] = np.column_stack([xlo, fill, zlo])
            verts[:, 1] = np.column_stack([xhi, fill, zlo])
            verts[:, 2] = np.column_stack([xhi, fill, zhi])
            verts[:, 3] = np.column_stack([xlo, fill, zhi])
        else:  # axis == 'x'
            # YZ plane at x=value: corners span (ylo..yhi) x (zlo..zhi)
            verts[:, 0] = np.column_stack([fill, ylo, zlo])
            verts[:, 1] = np.column_stack([fill, yhi, zlo])
            verts[:, 2] = np.column_stack([fill, yhi, zhi])
            verts[:, 3] = np.column_stack([fill, ylo, zhi])

        del xlo, xhi, ylo, yhi, zlo, zhi, fill
        vertices = verts.reshape(n_cells * 4, 3)
        del verts

        # VTK_QUAD (type 9) connectivity
        base = np.arange(n_cells, dtype=np.int32) * 4
        cells_conn = np.empty((n_cells, 5), dtype=np.int32)
        cells_conn[:, 0] = 4
        cells_conn[:, 1:] = base[:, np.newaxis] + np.arange(4, dtype=np.int32)
        del base

        cell_types = np.full(n_cells, 9, dtype=np.uint8)
        grid = pv.UnstructuredGrid(cells_conn.ravel(), cell_types, vertices)
        del cells_conn, vertices

        log.debug("Attaching %d field(s) to slice grid.", len(field_names))
        unmapped_keys = set(self.field_map.keys()) - set(field_names)
        if unmapped_keys:
            log.warning("field_map contains key(s) not found in field_names: %s", sorted(unmapped_keys))

        for idx, name in enumerate(field_names):
            mapped_name = self.field_map.get(name, name)
            grid.cell_data[mapped_name] = field_data[:, idx]

        del field_data
        return grid

    def writeFlowVTK(self, grid : pv.UnstructuredGrid, timestep_data):
        """
        Save a flow grid to VTK and track timestep.

        Parameters
        ----------
        grid: pv.UnstructuredGrid
            PyVista grid with flow quantities
        timestep_data: 
            Data from a single timestep of a flow dump
        """
        self.writeVTK(
            data_obj=grid,
            timestep=timestep_data['timestep'],
            solver_name="flow",
            folder="flow_output",
            ext=".vtu"
        )
          
    def writeFlowPVD(self):
        self.writePVD("flow", ext=".vtu")

def runFlow():
    """completes the loop for flow data"""
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