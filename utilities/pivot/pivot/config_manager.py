import sys
from pivot.configs import FlowConfig, SurfaceConfig, SolidConfig, PostprocessConfig
import logging
log = logging.getLogger(__name__)

try:
    import tomllib as toml #type: ignore
except ImportError:
   try:
       import tomli as toml
   except ImportError:
       raise ImportError("ConfigManager requires either 'tomllib' (Python > 3.11) or 'tomli'")

from pathlib import Path

class ConfigManager:
    """Manages the input config.toml"""
    def __init__(self, config_file):
        with open(config_file, "rb") as f:
            raw = toml.load(f)
        
        log.info("Loaded configuration from %s", config_file)
        
        self.flow = (
            FlowConfig.from_dict(raw['flow']) 
            if "flow" in raw else None)
        
        self.surface = (
            SurfaceConfig.from_dict(raw["surface"])
            if "surface" in raw else None
            )
        
        self.solid = (
            SolidConfig.from_dict(raw["solid"])
            if "solid" in raw
            else None
            )
        
        self.postprocess = PostprocessConfig.from_dict(raw.get("postprocess", {}))
        self.output = raw.get("output", {})
        
        self._log_construction()
        
    @property
    def sync_enabled(self) -> bool:
        return self.postprocess.sync_enabled
    
    @property
    def vis_enabled(self) -> bool:
        return self.postprocess.vis_enabled
    
    def _log_construction(self):
        log.info("Constructed configuration objects:")
        
        if self.flow:
            log.info("  FlowConfig: %s", self.flow)
        else:
            log.info("  FlowConfig: not present")
            
        if self.surface:
            log.info("  SurfaceConfig: %s", self.surface)
        else:
            log.info("  SurfaceConfig: not present")
            
        if self.solid:
            log.info("  SolidConfig: %s", self.solid)
        else:
            log.info("  SolidConfig: not present")
            
        log.info("  PostprocessConfig: %s", self.postprocess)
        
    def _validate(self):
        log.debug("Starting config validation")
        
        if self.postprocess.sync_enabled:
            if not any([self.flow, self.surface, self.solid]):
                log.warning("sync_enabled=true but no solvers defined in the config. Postprocessing may not find any solver outputs.")
                
        log.debug("ConfigManager validation passed.")