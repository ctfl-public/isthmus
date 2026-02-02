# standard python modules
import time
from pathlib import Path
import numpy as np
import pyvista as pv
from tqdm import tqdm
import pandas as pd
import sys

import logging
log = logging.getLogger(__name__)

# custom modules
from pivot.config_manager import ConfigManager
from pivot.simulation_data import SimulationData
from pivot.base_converter import BaseConverter

class SolidConverter(BaseConverter):
    """Converts solid simulation data (CSV with cell centers) to VTK."""
    
    def __init__(self, config : ConfigManager, sim_data : SimulationData):
        super().__init__(config, sim_data, solver_name="solid")
        self.voxel_size = self.settings.voxel_size
            
    @property
    def solid_dir(self):
        """Returns path of directory of solid data files input by user in config.toml"""
        return self.dir
    
    @property
    def solid_dt(self):
        """Returns timestep at which the solid solver is running at."""
        return self.settings.solid_dt
            
    def processSolidDirectory(self):
        """Process the solid files in a specified directory."""
        solid_files = self.getFiles(self.solid_dir, self.step)

        if not solid_files:
            raise FileNotFoundError(f"No files found in {self.solid_dir}")

        log.info("Processing %d solid file(s) (step=%d)", len(solid_files), self.step)

        for file in tqdm(solid_files, desc="Processing solid files"):
            try:
                log.debug("Processing solid file: %s", file)
                timestep_data = self.processSolidFile(file)
                log.debug("Parsed solid file %s: timestep = %d, n_cells= %d",
                          file,
                          timestep_data['timestep'],
                          timestep_data['count'])
                grid = self.createSolidGrid(timestep_data)
                self.writeSolidVTK(grid, timestep_data)
            except Exception as e:
                log.error("Failed processing solid file %s: %s", file, e)
                raise

        log.info("Solid data processing complete. \nSolid .vtu files are stored in solid_output directory")
            

    def processSolidFile(self, filepath):
        """Parse solid data CSV file for a single timestep."""
        
        # new method to get timesteps for solid
        with open(filepath, "r") as f:
            lines = f.readlines()
        
        accumulated_timestep = None
        csv_start_idx = None

        for i, line in enumerate(lines):
            line = line.strip()
            if line == "ITEM: SOLID TIMESTEP":
                # this timestep doesn't reset / is global 
                try:
                    accumulated_timestep = int(lines[i + 1].strip())
                except (IndexError, ValueError):
                    raise ValueError(
                    f"{filepath}: 'ITEM: SOLID TIMESTEP' "
                    "must be followed by an integer value.")
                csv_start_idx = i + 2
                break
        
        if accumulated_timestep is None:
            raise ValueError(
                f"{filepath}: Missing required header "
                "'ITEM: SOLID TIMESTEP'."
            )
        
        if csv_start_idx is None:
            raise ValueError(f"Could not find CSV header in {filepath}")
        
        df = pd.read_csv(filepath, skiprows=csv_start_idx, dtype=np.float32)
        
        if df.empty:
            raise ValueError(f"{filepath}: CSV contains no data rows")
        
        if df.shape[1] < 4:
            raise ValueError(f"{filepath}: Expected at least 4 columns (id, x, y, z); got {df.shape[1]}")
        
        log.debug("%s: CSV parsed with %d rows, %d columns", filepath, df.shape[0], df.shape[1])
        log.debug("%s: Columns = %s", filepath, df.columns.tolist())

        return {
                'header': df.columns.tolist(),
                'data': df.values,
                'count': len(df),
                'voxel_size': self.voxel_size,
                'timestep': accumulated_timestep,
            }


    def createSolidGrid(self, timestep_data):
        """Create the grid of solid data for a single timestep."""
        
        field_items = timestep_data['header']
        data = timestep_data['data']
        n_cells = len(data)
        
        if n_cells == 0:
            raise ValueError("No solid cells found; cannot create grid")
        
        if data.shape[1] < 4:
            raise ValueError("Solid data must contain at least id, x, y, z, columns")
        
        # construct each grid cell as a voxel centered at x,y,z with given voxel size 
        voxel_size = self.voxel_size
        half_size = voxel_size / 2.0 
        
        # extract x,y,z coordinates
        x_c = data[:, 1]
        y_c = data[:, 2]
        z_c = data[:, 3]
        xlo, xhi = x_c - half_size, x_c + half_size
        ylo, yhi = y_c - half_size, y_c + half_size
        zlo, zhi = z_c - half_size, z_c + half_size
        
        vertices = np.zeros((n_cells * 8, 3))
        for i in range(n_cells):
            base = i * 8
            vertices[base + 0] = [xlo[i], ylo[i], zlo[i]]
            vertices[base + 1] = [xhi[i], ylo[i], zlo[i]]
            vertices[base + 2] = [xhi[i], yhi[i], zlo[i]]
            vertices[base + 3] = [xlo[i], yhi[i], zlo[i]]
            vertices[base + 4] = [xlo[i], ylo[i], zhi[i]]
            vertices[base + 5] = [xhi[i], ylo[i], zhi[i]]
            vertices[base + 6] = [xhi[i], yhi[i], zhi[i]]
            vertices[base + 7] = [xlo[i], yhi[i], zhi[i]]
            
        cells = np.zeros((n_cells, 9), dtype=np.int64)
        for i in range(n_cells):
            base = i * 8
            cells[i] = [8, base, base+1, base+2, base+3, base+4, base+5, base+6, base+7]
            
        cell_types = np.full(n_cells, 12, dtype=np.uint8)
        
        grid = pv.UnstructuredGrid(cells.ravel(), cell_types, vertices)
        
        # Columns that are NOT geometric
        geom_cols = {"id", "x", "y", "z"}

        for col_idx, name in enumerate(field_items):
            if name in geom_cols:
                continue

            # Attach as cell data (1 value per voxel)
            grid.cell_data[name] = data[:, col_idx].astype(np.float32)
        
        return grid


    def writeSolidVTK(self, grid : pv.UnstructuredGrid, timestep_data):
        """
        Write solid grid to VTK and tracks timestep.
        
        Parameters
        ----------
        grid: pv.UnstructuredGrid
            Solid grid created using voxel data
        timestep_data: 
            Data from each timestep
        """
        try:
            self.writeVTK(data_obj=grid, 
                        timestep=timestep_data['timestep'], 
                        solver_name="solid",
                        folder="solid_output", 
                        ext=".vtu")
        except Exception as e:
            log.error("Failed to write solid VTK (timestep=%d): %s", timestep_data['timestep'], e)
            raise
        
    def writeSolidPVD(self):
        self.writePVD("solid", ext=".vtu")

def runSolid():
    """completes the loop for solid data"""
    print(
        "WARNING: Running SolidConverter as a standalone module.\n"
        "Only solid-related settings from config.toml will be applied.\n"
        "Syncing with flow or surface data will NOT occur."
    )
    start = time.perf_counter()
    config = ConfigManager("config.toml")
    sim_data = SimulationData()
    sc = SolidConverter(config, sim_data)
    sc.processSolidDirectory()
    sc.writeSolidPVD()
    end = time.perf_counter()
    elapsed_time = end - start
    print(f"Total time to process solid steps: {elapsed_time:.2f} s")
    
if __name__ == "__main__":
    runSolid()