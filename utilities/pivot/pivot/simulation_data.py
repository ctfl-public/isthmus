from dataclasses import dataclass, field
from typing import Optional, List, Dict, Union
from pathlib import Path
import pyvista as pv
import numpy as np
import os
import logging
log = logging.getLogger(__name__)

from pivot.config_manager import ConfigManager

@dataclass
class SimulationData:
    """Container for parsed SPARTA simulation and solid data shared across converters.
    
    Attributes
    ----------
    timesteps : Dict[str, List[int]]
        Dictionary for storing the available timesteps for each of the solvers.
    output_dirs : Dict[str, Optional[Path]]
        Dictionary of output directory paths for each solver. 
    data : Dict[str, Union[np.ndarray, float, None]]
        Dictionary associating the data for each solver for future filtering. 
        
    Methods
    -------
    - scanSolverOutputs() : Looks for files in output_dirs to get matching timesteps. Only used if only postprocessing set.
    - getFileFromTimestep() : Self explanatory
    - getSyncedTimesteps() : Self explanatory 
    - writeSyncedPVD() : Self explanatory 
    """
    
    timesteps: Dict[str, List[int]] = field(default_factory=lambda: {
        "flow" : [],
        "surface" : [],
        "solid" : []
        })
    
    output_dirs: Dict[str, Optional[Path]] = field(default_factory=lambda : {
        "flow" : None,
        "surface": None,
        "solid": None
        })
    
    data: Dict[str, Union[np.ndarray, float, None]] = field(default_factory=lambda : {
        "flow" : None,
        "surface": None,
        "solid": None
        })
        
    def scanSolverOutputs(self):
        """
        Populate self.timesteps by scanning existing solver output directories.
        Works for any file extension as long as filename is <solver>_<timestep>.<ext>
        """
        for solver, out_dir in self.output_dirs.items():
            if out_dir is None or not out_dir.exists():
                log.warning("No output directory set for solver '%s'", solver)
                self.timesteps[solver] = []
                continue
            
            if not out_dir.exists():
                log.warning("Output directory does not exist for solver '%s' : %s", solver, out_dir)
                self.timesteps[solver] = []
                continue

            ts_list = []
            for f in out_dir.glob(f"{solver}_*.*"):
                try:
                    t = int(f.stem.split("_")[1])
                    ts_list.append(t)
                except (IndexError, ValueError):
                    # skip files that don't match pattern
                    log.debug("Skipping non-matching file: %s", f)
                    continue

            self.timesteps[solver] = sorted(ts_list)
            log.info("Found %d timesteps for solver %s", len(ts_list), solver)
    
    def getFileFromTimestep(self, solver: str, timestep: int):
        """Returns filepath based on a given timestep extension"""
        directory = self.output_dirs[solver]
        if directory is None:
            raise ValueError(f"No output directory set for solver '{solver}'")

        # look for any file starting with <solver>_<timestep> and any extension
        matches = list(directory.glob(f"{solver}_{timestep}.*"))
        if not matches:
            raise FileNotFoundError(f"No file for solver '{solver}' at timestep {timestep}")
        if len(matches) > 1:
            log.warning("Multiple files found for %s at timestep %d; using %s", solver, timestep, matches[0])
        
        return matches[0]
    
    def getSyncedTimesteps(self, solvers : List[str] = None):
        """Gets the synced timesteps for the solvers with matching timestep values"""
        if solvers is None:
            solvers = list(self.timesteps.keys())
            
        if not solvers:
            log.error("No solvers specified for syncing")
            return []
        
        sets = []
        for solver in solvers:
            # effectively what this loop does: sets = [set(self.timesteps[solver]) for solver in solvers]
            ts = self.timesteps.get(solver, [])
            if not ts:
                log.warning("Solver '%s' has no timesteps; sync will be empty", solver)
            sets.append(set(ts))
            
        if not sets:
            return []
        
        synced_ts = set.intersection(*sets)
        synced_list = sorted(synced_ts)
        
        log.info("Computed %d synced timesteps for solvers %s", len(synced_list), solvers)

        return synced_list
    
    def writeSyncedPVD(self, solvers: Optional[List[str]] = None, output_dir: Optional[Path] = None, filename: str = "synced.pvd"):
        """
        Write a PVD file that references only the timesteps present for all specified solvers.

        Parameters
        ----------
        solvers : List[str], optional
            Solvers to include in the synced PVD. Defaults to all available solvers.
        output_dir : Path, optional
            Directory to write the synced PVD. Defaults to current working directory.
        filename : str, optional
            Name of the PVD file to create.
        """
        if solvers is None:
            solvers = list(self.timesteps.keys())

        # Determine synced timesteps
        synced_ts = self.getSyncedTimesteps(solvers)
        if not synced_ts:
            log.error("No common timesteps found among solvers: %s", solvers)
            raise RuntimeError("Cannot write synced PVD: no commond timesteps")

        # Determine output directory
        if output_dir is None:
            output_dir = Path.cwd()
        output_dir.mkdir(parents=True, exist_ok=True)

        pvd_path = output_dir / filename
        with open(pvd_path, 'w') as f:
            f.write('<?xml version="1.0"?>\n')
            f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
            f.write('  <Collection>\n')

            # Loop over timesteps and solvers
            for t in synced_ts:
                for solver in solvers:
                    try:
                        file_path = self.getFileFromTimestep(solver, t)
                    except FileNotFoundError:
                        # skip if file missing; shouldn't happen if synced properly
                        continue
                    # write relative path from output_dir
                    rel_path = os.path.relpath(file_path.resolve(), start=output_dir.resolve())
                    f.write(f'    <DataSet timestep="{t}" group="{solver}" part="0" file="{rel_path}"/>\n')

            f.write('  </Collection>\n')
            f.write('</VTKFile>\n')

        log.info("Synced PVD written to %s", pvd_path)