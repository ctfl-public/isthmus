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
        
    def processFlowDirectory(self):
        """Processes flow files in a specified directory"""
        flow_files = self.getFiles(self.flow_dir, step=self.step)
        
        for filepath in tqdm(flow_files, desc="Processing flow files"):
            # process each file in serial for now
            timestep_data = self.processFlowFile(filepath)
            grid = self.createFlowGrid(timestep_data)
            self.writeFlowVTK(grid, timestep_data)

    def processFlowFile(self, filename):
        """Processes a single flow file and returns the data for the given timestep."""
        with open(filename, "r") as f:
            lines = f.readlines()
            
        cell_array = None
            
        timestep_data = {
            'filepath': filename,
            'timestep': None,
            'num_cells': None,
            'box_bounds': None,
            "field_items": None}
        
        i = 0 # index 
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith(SpartaItems.TIMESTEP):
                timestep_data["timestep"] = int(lines[i + 1].strip())
                i += 2
                continue
            
            if line.startswith(SpartaItems.NUM_CELLS):
                timestep_data['num_cells'] = int(lines[i + 1].strip())
                i += 2
                continue
            
            if line.startswith(SpartaItems.BOX_BOUNDS):
                xbounds = [float(x) for x in lines[i + 1].strip().split()]
                ybounds = [float(x) for x in lines[i + 2].strip().split()]
                zbounds = [float(x) for x in lines[i + 3].strip().split()]
                timestep_data["box_bounds"] = (xbounds, ybounds, zbounds)
                i += 4
                continue
            
            if line.startswith(SpartaItems.CELLS):
                items = line.split()
                timestep_data['field_items'] = items[2:len(items)]
                num_cells = timestep_data['num_cells']
                cell_data = []
                i += 1 
                for _ in range(num_cells):
                    cell_data.append([float(x) for x in lines[i].split()])
                    i += 1
                timestep_data['cell_array'] = np.array(cell_data, dtype=np.float32)
                
                # check if user output another timestep
                for j in range(i, len(lines)):
                    trailing = lines[j].strip()
                    if trailing.startswith("ITEM:"):
                        log.warning(
                            f"Unexpected additional ITEM block detected in flow file '{filename} after cell data. This tool expects exactly one timestep per file. Trailing data will be ignored."
                            )
                        break

        return timestep_data

    def isFieldItemsCorrect(self, timestep_data):
        """Checks if the required flow field items are in the file and returns a list of missing items for user to be aware of."""
        field_items = set(timestep_data['field_items'])
        required_items = SpartaItems.REQUIRED_FLOW_FIELDS

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
        
        # check if user has required items for flow grid
        hasCorrectFieldItems, missingFieldItems = self.isFieldItemsCorrect(timestep_data)
        if not hasCorrectFieldItems:
            log.error("Missing required flow field item(s): %s", missingFieldItems)
            raise ValueError(f"Missing required flow field item(s): {missingFieldItems}")
        
        # 2D case requires a different approach
        is_2d = 'zlo' not in field_items
        
        if is_2d:
            # 2D case: id, xlo, ylo, xhi, yhi, [fields...]
            geom_indexes = self.getGeomIndexes(timestep_data)
            i_xlo, i_ylo, i_xhi, i_yhi = geom_indexes['i_xlo'], geom_indexes['i_ylo'], geom_indexes['i_xhi'], geom_indexes['i_yhi']
            xlo, ylo, xhi, yhi = data[:, i_xlo], data[:, i_ylo], data[:, i_xhi], data[:, i_yhi]
            
            # Calculate typical cell size in X-Y plane
            dx = np.mean(xhi - xlo)
            dy = np.mean(yhi - ylo)
            z_thickness = max(dx, dy)  # Use cell size as Z thickness
            
            zlo = np.full(n_cells, -z_thickness/2)
            zhi = np.full(n_cells, z_thickness/2)
            
            # Field data starts after: id, xlo, ylo, xhi, yhi (5 columns)
            field_start_idx = 5
        else:
            # get geometry index using string comparison
            geom_indexes = self.getGeomIndexes(timestep_data)
            i_xlo = geom_indexes['i_xlo']
            i_ylo = geom_indexes['i_ylo']
            i_zlo = geom_indexes['i_zlo']
            i_xhi = geom_indexes['i_xhi']
            i_yhi = geom_indexes['i_yhi']
            i_zhi = geom_indexes['i_zhi']
            
            xlo, ylo, zlo = data[:, i_xlo], data[:, i_ylo], data[:, i_zlo]
            xhi, yhi, zhi = data[:, i_xhi], data[:, i_yhi], data[:, i_zhi]
        
            field_start_idx = 7
        
        # Create vertices for each cell (8 vertices per cell)
        vertices = np.zeros((n_cells * 8, 3))
        for i in range(n_cells):
            base = i * 8
            vertices[base + 0] = [xlo[i], ylo[i], zlo[i]]
            vertices[base + 1] = [xhi[i], ylo[i], zlo[i]]
            vertices[base + 2] = [xlo[i], yhi[i], zlo[i]]
            vertices[base + 3] = [xhi[i], yhi[i], zlo[i]]
            vertices[base + 4] = [xlo[i], ylo[i], zhi[i]]
            vertices[base + 5] = [xhi[i], ylo[i], zhi[i]]
            vertices[base + 6] = [xlo[i], yhi[i], zhi[i]]
            vertices[base + 7] = [xhi[i], yhi[i], zhi[i]]
        
        # Create cells (connectivity)
        cells = np.zeros((n_cells, 9), dtype=np.int64)
        for i in range(n_cells):
            base = i * 8
            cells[i] = [8, base, base+1, base+2, base+3, base+4, base+5, base+6, base+7]
        
        # Cell types (VTK_VOXEL = 11)
        cell_types = np.full(n_cells, 11, dtype=np.uint8)
        
        grid = pv.UnstructuredGrid(cells.ravel(), cell_types, vertices)
        
        # Extract only the field data columns (skip id and geometry)
        field_data = data[:, field_start_idx:]
        
        # Get field names (skip id and geometry column names)
        field_names = field_items[field_start_idx:]
        
        # Attach field data to grid
        for idx, name in enumerate(field_names):
            grid.cell_data[name] = field_data[:, idx]
        
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