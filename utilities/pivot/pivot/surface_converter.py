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
        
        if self.settings.surface_static:
            surf_geom_file = sorted(self.surf_geom_dir.glob("*"))[0]
            geom_info = self.processSurfGeometryFile(str(surf_geom_file))
            iterator = [(f, surf_geom_file) for f in surf_data_files]
        else:
            iterator = zip(surf_data_files, surf_geom_files)
        
        num_data_files = len(surf_data_files)
        num_geom_files = len(surf_geom_files)

        log.debug("Found %d surface data files", num_data_files)
        log.debug("Found %d surface geometry files", num_geom_files)


        if num_data_files != num_geom_files:
            log.warning(
                "Data/geometry file count mismatch (%d vs %d). "
                "Some timesteps may be skipped.",
                num_data_files, num_geom_files
            )

        for surf_data_file, surf_geom_file in tqdm(
            iterator,
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
        
        print("Surface data processing complete. Surface .vtk files are stored in surface_output directory.")
    
    # surface data is associated with surface geometry
    def processSurfGeometryFile(self, filename):
        """Processes a single surface geometry file produced by ISTHMUS at a single timestep."""
        if filename.split(".")[-1] == "surf":
            timestep = 0
        else:
            timestep = filename.split(".")[2]

        timestep_data = {
            'filepath': filename,
            'timestep': timestep,
            'num_points': None,
            'num_tris': None,
            'points': None,
            'triangles': None,
            'triangle_ids': None
        }

        try:
            with open(filename, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue

                    # Parse "N points" / "N triangles" / "N surfs" count lines
                    # Works for both standard SPARTA surf files and ISTHMUS-generated files.
                    parts = line.split()
                    if len(parts) == 2 and parts[0].isdigit():
                        if parts[1] == "points":
                            timestep_data['num_points'] = int(parts[0])
                            continue
                        if parts[1] in ("triangles", "surfs"):
                            timestep_data['num_tris'] = int(parts[0])
                            continue

                    if line.startswith("Points"):
                        next(f)  # blank line after section header
                        num_points = timestep_data['num_points']
                        if num_points is None:
                            raise ValueError(
                                f"'Points' section reached but point count was not found in header of {filename}"
                            )
                        points = np.empty((num_points, 3), dtype=np.float32)
                        point_ids = np.empty(num_points, dtype=np.int32)
                        for i in range(num_points):
                            parts = next(f).strip().split()
                            point_ids[i] = int(parts[0])
                            points[i] = parts[1:]
                        timestep_data['points'] = points
                        timestep_data['point_ids'] = point_ids
                        continue

                    if line.startswith("Triangles") or line.startswith("Surfs"):
                        next(f)  # blank line after section header
                        num_tris = timestep_data['num_tris']
                        if num_tris is None:
                            raise ValueError(
                                f"'Triangles' section reached but triangle count was not found in header of {filename}"
                            )
                        tris = np.empty((num_tris, 3), dtype=np.int32)
                        tri_ids = np.empty(num_tris, dtype=np.int32)
                        for i in range(num_tris):
                            parts = next(f).strip().split()
                            tri_ids[i] = int(parts[0])
                            tris[i] = [int(x) - 1 for x in parts[1:]]
                        timestep_data['triangles'] = tris
                        timestep_data['triangle_ids'] = tri_ids
                        continue
        except OSError:
            log.error("Failed to read surface geometry file: %s", filename)
            raise

        return timestep_data
    
    def processSurfDataFile(self, filename):
        """Process SPARTA surface data file at a single timestep."""
        timestep_data = {
            'filepath': filename,
            'timestep': None,
            'num_surfs': None,
            'box_bounds': None,
            'field_items': None}

        with open(filename, "r") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                line = line.strip()

                if line.startswith(SpartaItems.TIMESTEP):
                    timestep_data["timestep"] = int(f.readline().strip())
                    continue

                if line.startswith(SpartaItems.NUM_SURFS):
                    timestep_data['num_surfs'] = int(f.readline().strip())
                    continue

                if line.startswith(SpartaItems.BOX_BOUNDS):
                    xbounds = [float(x) for x in f.readline().strip().split()]
                    ybounds = [float(x) for x in f.readline().strip().split()]
                    zbounds = [float(x) for x in f.readline().strip().split()]
                    timestep_data["box_bounds"] = (xbounds, ybounds, zbounds)
                    continue

                if line.startswith(SpartaItems.SURFS):
                    items = line.split()
                    field_items = items[3:]
                    timestep_data['field_items'] = field_items
                    num_surfs = timestep_data['num_surfs']
                    num_cols = len(field_items) + 1  # +1 for id column not in field_items

                    surf_array = np.empty((num_surfs, num_cols), dtype=np.float32)
                    for i in range(num_surfs):
                        surf_array[i] = f.readline().split()
                    timestep_data['surf_data'] = surf_array

                    for trailing_line in f:
                        if trailing_line.strip().startswith("ITEM:"):
                            log.warning(
                                f"Unexpected additional ITEM block detected in surface file '{filename}' after first SURFS block. "
                                "This tool expects exactly one timestep per file. Trailing data will be ignored"
                            )
                            break
                    break

        return timestep_data
                
    def attachDataToSurfs(self, geom_info, timestep_data):
        """Attach data to surfaces"""
        points = geom_info['points']
        triangles = geom_info['triangles']
        triangle_ids = geom_info['triangle_ids']

        # build faces — vectorized, avoids one np.insert allocation per triangle
        n_tris = len(triangles)
        face_buf = np.empty((n_tris, 4), dtype=np.int32)
        face_buf[:, 0] = 3
        face_buf[:, 1:] = triangles
        faces = face_buf.ravel()

        poly = pv.PolyData(points, faces)

        n_cells = poly.n_cells
        field_names = timestep_data['field_items']
        data_array = timestep_data['surf_data']

        # Build a direct lookup array: surf_id -> cell index (-1 if absent)
        max_tri_id = int(triangle_ids.max()) if len(triangle_ids) > 0 else 0
        id_to_cell = np.full(max_tri_id + 1, -1, dtype=np.int64)
        id_to_cell[triangle_ids] = np.arange(len(triangle_ids), dtype=np.int64)

        surf_ids = data_array[:, 0].astype(np.int32)
        in_range = surf_ids <= max_tri_id
        cell_indices = np.where(in_range, id_to_cell[np.clip(surf_ids, 0, max_tri_id)], -1)
        valid = cell_indices >= 0
        missing = int((~valid).sum())

        for field_idx, name in enumerate(field_names):
            arr = np.full(n_cells, np.nan, dtype=np.float32)
            arr[cell_indices[valid]] = data_array[valid, field_idx + 1]
            poly.cell_data[name] = arr

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