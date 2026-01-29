from paraview_tools.config_manager import ConfigManager
from paraview_tools.simulation_data import SimulationData
import sys
from pathlib import Path

import logging
log = logging.getLogger(__name__)

class BaseConverter:
    """
    Class for BaseConverter that the Flow, Surface, and Solid converters inherit from. 
    
    Attributes
    ----------
    config : ConfigManager
        ConfigManager object for handling user configuration
    sim_data : SimulationData
        SimulationData object used for passing data for syncing solvers
    solver_name : str
        The name of the solver that is being initialized
    
    Methods
    -------
    createOutputDir(subfolder : str)
        Creates output directory for a given solver's vtk files with name subfolder
    getFiles(directory : Path, step : int = 1)
        Gets the data files from a directory, optionally skipping steps for speed-up
    writeVTK(data_obj, timestep, solver_name, folder, ext)
        Writes VTK depending on the type of data and solver
    """
    def __init__(self,
                config : ConfigManager, 
                sim_data : SimulationData,
                solver_name : str):
        
        self.config = config
        self.sim_data = sim_data
        self.solver_name = solver_name
        self.settings = getattr(config, solver_name) # returns config.solver instance 
        self.output = config.output
        self.root_dir = Path(f"{self.__class__.__name__}.py").parent.parent
        
    @property
    def step(self):
        return getattr(self.settings, "step", 1)
            
    @property
    def dir(self):
        """Return directory for this solver"""
        if self.solver_name == "flow":
            return self.settings.flow_dir
        elif self.solver_name == "solid":
            return self.settings.solid_dir
    
    def createOutputDir(self, subfolder : str):
        """Create output folder for this solver"""
        out_dir = self.root_dir / subfolder
        out_dir.mkdir(parents=True, exist_ok=True)
        log.debug("Created output directory: %s", out_dir)
        return out_dir
    
    def getFiles(self, directory: Path, step: int = 1) -> list[Path]:
        """
        Returns a sorted list of files from a directory, optionally skipping every `step` files.
        """
        if not directory.exists() or not directory.is_dir():
            log.error(f"Output directory {directory} not found")
            raise FileNotFoundError(f"{directory} does not exist or is not a directory")

        files = sorted(Path(directory).iterdir())
        if not files:
            log.error("No files in output directory")
            return []
            
        if step <= 0:
            raise ValueError("Step must be a positive integer")

        return files[::step]
    
    def writeVTK(self, data_obj, timestep: int, solver_name: str, folder: str, ext: str):
        """
        Write VTK for a specific solver (grid or polydata) and update simulation data.

        Parameters
        ----------
        data_obj: 
            PyVista object (grid for flow/solid or polydata for surface)
        timestep: int 
            timestep of this data
        solver_name: str
            'flow', 'surface', or 'solid'
        folder: str
            subfolder under root_dir to save output
        ext:  str
            file extension ('.vtu' or '.vtp')
        """
        
        out_dir = self.createOutputDir(folder)
        
        # Only set the output directory once
        if self.sim_data.output_dirs[solver_name] is None:
            self.sim_data.output_dirs[solver_name] = out_dir
            log.info("Registered output directory for %s: %s", solver_name, out_dir)
        
        # Save the file
        outfile_name = out_dir / f"{solver_name}_{timestep}{ext}"
        data_obj.save(str(outfile_name))
        
        log.debug(
            "Wrote %s VTK at timestep %d -> %s",
            solver_name,
            timestep,
            outfile_name
            )
        
        # Append timestep to sim_data timesteps
        if timestep not in self.sim_data.timesteps[solver_name]:
            self.sim_data.timesteps[solver_name].append(timestep)
        
        self.sim_data.timesteps[solver_name].sort() # go ahead sort ts
        

    def writePVD(self, solver_name : str, ext : str):
        """
        Writes the PVD using timesteps stored in sim_data.
        
        Parameters
        ----------
        solver_name : str
            Name of the solver for PVD writing
        ext : str
            File extension (.vtu for solid/flow, .vtp for surface)
        """
        
        timesteps = self.sim_data.timesteps.get(solver_name)
        out_dir = self.sim_data.output_dirs.get(solver_name)
        
        if not timesteps or not out_dir:
            raise FileNotFoundError(f"No {solver_name} data available for PVD")
        
        pvd_path = f"{solver_name}.pvd"
        
        with open(pvd_path, 'w') as f:
            f.write('<?xml version="1.0"?>\n')
            f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
            f.write('  <Collection>\n')
            for t in timesteps:
                filename = f"{solver_name}_{t}{ext}"
                f.write(f'    <DataSet timestep="{t}" file="{out_dir / filename}"/>\n')
            f.write('  </Collection>\n')
            f.write('</VTKFile>\n')
            
        log.info("%s written successfully", pvd_path)
        log.info("Use: paraview %s", pvd_path)