from pivot.config_manager import ConfigManager
from pivot.flow_converter import FlowConverter
from pivot.surface_converter import SurfaceConverter
from pivot.solid_converter import SolidConverter
from pivot.simulation_data import SimulationData
from pivot.archive import archiveOutputs
from pivot.flow_visualizer import render_flow_contour

import time
from typing import Optional, List
from pathlib import Path
import logging
import sys
import argparse

def parseArgs() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Paraview data conversion tool")
    
    parser.add_argument(
        "--logging",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level"
        )
    
    parser.add_argument(
        "--postprocessing",
        default="OFF",
        choices=["ON"],
        help="Run postprocessing tools"
        )
    
    parser.add_argument(
        "config",
        nargs="?",
        default="config.toml",
        help="Path to config.toml file (default = ./config.toml)"
    )
    
    return parser.parse_args()

def setupLogging(level=logging.INFO):
    # File handler - everything
    file_handler = logging.FileHandler("pivot.log")
    file_handler.setLevel(level)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", 
                         datefmt="%H:%M:%S")
    )
    
    # Console handler - clean, simple messages
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(
        logging.Formatter("%(message)s")  # Just the message, no timestamp/level
    )
    
    logging.basicConfig(
        level=level,
        handlers=[console_handler, file_handler]
    )

def runFlow(fc : FlowConverter):
    fc.processFlowDirectory()
    fc.writeFlowPVD()

def runSurf(surf_c : SurfaceConverter):
    surf_c.processSurfaceDirectory()
    surf_c.writeSurfPVD()

def runSolid(solid_c : SolidConverter):
    solid_c.processSolidDirectory()
    solid_c.writeSolidPVD()
    
def runSynced(sim_data: SimulationData, solvers: Optional[List[str]] = None):
    """Write a synced PVD for timesteps present across all specified solvers."""
    if solvers is None:
        solvers = [s for s in sim_data.timesteps.keys() if sim_data.timesteps[s]]

    synced_timesteps = sim_data.getSyncedTimesteps(solvers)
    print(f"Found {len(synced_timesteps)} synced timesteps")
    sim_data.writeSyncedPVD(solvers)

def postProcess(config : ConfigManager, sim_data : SimulationData):
    """Post processing that is done after the VTK output is created.
    
    Parameters
    ----------
        config : ConfigManager 
            object that holds user configuration information
        sim_data : SimulationData
            object that holds information shared across the three solvers
    """
    opts = config.postprocess
    if not opts:
        return

    if opts.archive:
        try:
            archiveOutputs()
        except Exception as e:
            print(f"Archiving failed: {e}")

    if opts.sync_enabled:
        sim_data.output_dirs = {
            "flow": Path("flow_output"),
            "surface": Path("surface_output"),
            "solid": Path("solid_output"),
        }
        sim_data.scanSolverOutputs()
        for solver, ts in sim_data.timesteps.items():
            print(f"{solver} timesteps : {ts}")
        runSynced(sim_data)

def runVisualization(config: ConfigManager):
    """Render a flow contour PNG from the raw dump (skips VTK conversion)."""
    render_flow_contour(config)
        
def runOnlyPostProcess(log : logging.Logger, config : ConfigManager, sim_data : SimulationData):
    """Run with only the post processing features"""
    
    log.debug("Running postprocessing only")
    sim_data.output_dirs = {
            "flow": Path("flow_output"),
            "surface": Path("surface_output"),
            "solid": Path("solid_output"),
            }
    sim_data.scanSolverOutputs()
    postProcess(config, sim_data)
        
def run(config : ConfigManager, sim_data : SimulationData):
    flow_dir = None
    surf_data_dir = None
    surf_geom_dir = None
    solid_dir = None

    flow = config.flow
    if flow:
        flow_dir = flow.flow_dir

    surf = config.surface
    if surf:
        surf_data_dir = surf.surf_data_dir
        surf_geom_dir = surf.surf_geom_dir

    solid = config.solid
    if solid:
        solid_dir = solid.solid_dir

    if flow_dir:
        fc = FlowConverter(config, sim_data)
        runFlow(fc)

    if solid_dir:
        solid_c = SolidConverter(config, sim_data)
        runSolid(solid_c)

    # surface needs BOTH data and geometry
    if surf_data_dir or surf_geom_dir:
        if not (surf_data_dir and surf_geom_dir):
            raise RuntimeError(
                "Surface conversion requires BOTH "
                "surface.surf_data_dir and surface.surf_geom_dir."
            )
        surf_c = SurfaceConverter(config, sim_data)
        runSurf(surf_c)

def main():
    """Main entry point for the tool.
    
    Command Line Arguments
    -----------------------
    - --logging
        - options = [DEBUG, INFO, WARNING, ERROR, CRITICAL]
    - --postprocessing
        - options = [ON]
        - if on will only run the postprocessing tools
    """
    args = parseArgs()
    log_level = getattr(logging, args.logging)
    setupLogging(log_level)
    
    log = logging.getLogger(__name__)
    
    print("=" * 60)
    print("Paraview INterface for Voxel and Surface OuTput (PIVOT)")
    print(
        "If you encounter any errors you don’t understand, "
        "please contact Savio Poovathingal or Robbie Harper for assistance."
    )
    print("=" * 60)

    start = time.perf_counter()
    
    try:
        config = ConfigManager(args.config)
        sim_data = SimulationData()
    except FileNotFoundError as e:
        log.error("No config file found: %s", e)
        return
    
    if args.postprocessing == "ON":
        log.debug("Running with only postprocessing")
        runOnlyPostProcess(log, config, sim_data)
    elif config.visualization.enabled:
        log.debug("Running visualization only")
        runVisualization(config)
    else:
        log.debug("Starting conversion")
        run(config, sim_data)
        log.debug("Starting post-processing")
        postProcess(config, sim_data)
            

    end = time.perf_counter()
    log.debug("Total time: %.2f s", end - start)

if __name__ == "__main__":
    main()
