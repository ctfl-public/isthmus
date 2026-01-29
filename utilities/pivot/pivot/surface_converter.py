# "standard" modules
import numpy as np
import pyvista as pv
from pathlib import Path
from tqdm import tqdm
import time
import logging
log = logging.getLogger(__name__)

# custom modules
from pivot.config_manager import ConfigManager
from pivot.simulation_data import SimulationData
from pivot.base_converter import BaseConverter
from pivot.sparta_items import SpartaItems

class SurfaceConverter(BaseConverter):
    """Converts SPARTA surface data to VTK PolyData."""
    def __init__(self, config : ConfigManager, sim_data : SimulationData):
        super().__init__(config, sim_data, "surface")
    
    @property
    def surf_data_dir(self) -> Path:
        """Returns path to SPARTA surface data directory given by config.toml"""
        return self.settings.surf_data_dir
    
    @property
    def surf_geom_dir(self) -> Path:
        """Returns path to ISTHMUS geometry file directory given by config.toml"""
        return self.settings.surf_geom_dir
    
    @property
    def surf_dt(self):
        """Returns the surface timesteps"""
        return self.settings.surface_dt
    
    def processSurfaceDirectory(self):
        """Processes surface files in a specified directory"""
        step = self.step
        surf_data_files = self.getFiles(self.surf_data_dir, step)
        surf_geom_files = self.getFiles(self.surf_geom_dir, step)
        
        num_data_files = len(surf_data_files)
        num_geom_files = len(surf_geom_files)

        log.info("Found %d surface data files", num_data_files)
        log.info("Found %d surface geometry files", num_geom_files)


        if num_data_files != num_geom_files:
            log.warning(
                "Data/geometry file count mismatch (%d vs %d). "
                "Some timesteps may be skipped.",
                num_data_files, num_geom_files
            )

        for surf_data_file, surf_geom_file in tqdm(
            zip(surf_data_files, surf_geom_files),
            desc="Processing surface files", total=len(surf_data_files)):
            geom_info = self.processSurfGeometryFile(str(surf_geom_file))
            timestep_data = self.processSurfDataFile(str(surf_data_file))
            poly_data = self.attachDataToSurfs(geom_info, timestep_data)
            
            if log.isEnabledFor(logging.DEBUG):
                for name, arr in poly_data.cell_data.items():
                    log.debug(
                        "Processing surf timestep: data=%s geom=%s",
                        surf_data_file.name,
                        surf_geom_file.name
                        )
                    log.debug(
                        "%s non-NaN cells: %d / %d",
                        name,
                        np.count_nonzero(~np.isnan(arr)),
                        arr.size
                        )

            self.writeSurfVTK(timestep_data, poly_data)
        
        log.info("Surface data processing complete. Surface .vtk files are stored in surface_output directory.")
    
    # surface data is associated with surface geometry
    def processSurfGeometryFile(self, filename):
        """Processes a single surface geometry file produced by ISTHMUS at a single timestep."""
        try:
            with open(filename, "r") as f:
                lines = f.readlines()
        except OSError as e:
            log.error("Failed to read surface geometry file: %s", filename)
            raise
        
        if filename.split(".")[-1] == "surf":
            # if no step after the . assume its the initial surf geom
            timestep = 0
        else:
            timestep = filename.split(".")[2]
        timestep_data = {
            'filepath' : filename,
            'timestep' : timestep,
            'num_points': None,
            "num_tris" : None,
            'points': None,
            'triangles': None,
            "triangle_ids": None
        }
        
        i = 0 # index
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith("surf file from isthmus"):
                timestep_data['num_points'] = int(lines[i + 2].strip().split()[0])
                timestep_data['num_tris'] = int(lines[i + 3].strip().split()[0])
                
                if timestep_data['num_points'] is None or timestep_data['num_tris'] is None:
                    raise ValueError(f"Invalid surface geometry header in {filename}")
                
            if line.startswith("Points"):
                points = []
                point_ids = []
                i += 2  # Skip header
                while len(points) < timestep_data['num_points']:
                    point_split = lines[i].strip().split()
                    point_id = int(point_split[0])  # store ID
                    coords = [float(x) for x in point_split[1:]]  # x, y, z
                    point_ids.append(point_id)
                    points.append(coords)
                    i += 1
                timestep_data['points'] = np.array(points, dtype=np.float32)
                timestep_data['point_ids'] = np.array(point_ids, dtype=np.int32)
                

            if line.startswith("Triangles"):
                tris = []
                tri_ids = []
                i += 2
                while len(tris) < timestep_data['num_tris']:
                    tri_split = lines[i].strip().split()
                    tri_ids.append(int(tri_split[0]))
                    tris.append([int(x) - 1 for x in tri_split[1:]])
                    i += 1
                timestep_data['triangles'] = np.array(tris, dtype=np.int32)
                timestep_data['triangle_ids'] = np.array(tri_ids, dtype=np.int32)

            i += 1     
        return timestep_data
    
    def processSurfDataFile(self, filename):
        """Process SPARTA surface data file at a single timestep."""
        with open(filename, "r") as f:
            lines = f.readlines()
            
        surf_data = None
            
        timestep_data = {
            'filepath': filename,
            'timestep': None,
            'num_surfs': None,
            'box_bounds': None,
            "field_items": None}
        
        i = 0 # index 
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith(SpartaItems.TIMESTEP):
                timestep_data["timestep"] = int(lines[i + 1].strip())
                i += 2
                continue
            
            if line.startswith(SpartaItems.NUM_SURFS):
                timestep_data['num_surfs'] = int(lines[i + 1].strip())
                i += 2
                continue
            
            if line.startswith(SpartaItems.BOX_BOUNDS):
                xbounds = [float(x) for x in lines[i + 1].strip().split()]
                ybounds = [float(x) for x in lines[i + 2].strip().split()]
                zbounds = [float(x) for x in lines[i + 3].strip().split()]
                timestep_data["box_bounds"] = (xbounds, ybounds, zbounds)
                i += 4
                continue
            
            if line.startswith(SpartaItems.SURFS):
                items = line.split()
                timestep_data['field_items'] = items[3:]
                surf_data = []
                i += 1 
                while i < len(lines):
                    if lines[i].strip().startswith("ITEM:"):
                        log.warning(f"Unexpected additional ITEM block detected in surface file '{filename}' after first SURFS block. This tool expects exactly one timestep per file. Trailing data will be ignored")
                        break
                    surf_data.append([float(x) for x in lines[i].split()])
                    i += 1
                timestep_data['surf_data'] = np.array(surf_data, dtype=np.float32)
                break

        return timestep_data
                
    def attachDataToSurfs(self, geom_info, timestep_data):
        """Attach data to surfaces"""
        points = geom_info['points']
        triangles = geom_info['triangles']
        triangle_ids = geom_info['triangle_ids']
        
        # build faces 
        faces = np.hstack([np.insert(tri, 0, 3) for tri in triangles])
        
        poly = pv.PolyData(points, faces)
        
        tri_id_to_cell = {tri_id: idx for idx, tri_id in enumerate(triangle_ids)}
        
        n_cells = poly.n_cells
        field_names = timestep_data['field_items']
        data_array = timestep_data['surf_data']
        id_index = 0 
        
        for name in field_names:
            poly.cell_data[name] = np.full(n_cells, np.nan, dtype=np.float32)
            
        missing = 0
        for row in data_array:
            surf_id = int(row[id_index])
            cell_idx = tri_id_to_cell.get(surf_id)
            if cell_idx is None:
                missing += 1
                continue
            for field_idx , value in enumerate(row[1:]):
                name = timestep_data['field_items'][field_idx]
                poly.cell_data[name][cell_idx] = value
        
        if missing and log.isEnabledFor(logging.DEBUG):
            log.debug("%d surface data entries had no matching triangle (timestep %s)", missing, timestep_data['timestep'])
                
        return poly
        
    def writeSurfVTK(self, timestep_data, poly_data):
        self.writeVTK(
            data_obj=poly_data,
            timestep=timestep_data['timestep'],
            solver_name='surface',
            folder="surface_output",
            ext='.vtp'
            )
        
    def writeSurfPVD(self):
        """Writes the PVD for surf files. Must have vtu data in directory first."""
        self.writePVD("surface", ".vtp")


def runSurf():
    """completes the loop for surface data"""
    print(
        "WARNING: Running SurfaceConverter as a standalone module.\n"
        "Only surface-related settings from config.toml will be applied.\n"
        "Syncing with flow or solid data will NOT occur."
    )
    start = time.perf_counter()
    config = ConfigManager("config.toml")
    sim_data = SimulationData()
    sc = SurfaceConverter(config, sim_data)
    sc.processSurfaceDirectory()
    sc.writeSurfPVD()
    end = time.perf_counter()
    elapsed_time = end - start
    print(f"Total time to process surface steps: {elapsed_time:.2f} s")
    

if __name__ == "__main__":
    runSurf()