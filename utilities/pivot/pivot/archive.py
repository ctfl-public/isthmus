from pathlib import Path
import zipfile as zip 
from tqdm import tqdm
import os

COMPRESSION_TYPE = zip.ZIP_STORED # standard compression
ROOT_DIR = Path(__file__).parent.parent.resolve()

def getPathsToArchive():
    paths = {
        "flow_dir": None,
        "surf_dir": None,
        "solid_dir": None,
        "flow_pvd_path": None,
        "solid_pvd_path": None,
        "surf_pvd_path": None
    }

    for subdir in ROOT_DIR.iterdir():
        if not subdir.is_dir():
            continue

        # Use subdir.name instead of string splitting
        if subdir.name == "flow_output":
            paths["flow_dir"] = subdir
            paths['flow_pvd_path'] = ROOT_DIR / "flow.pvd"
        elif subdir.name == "solid_output":
            paths["solid_dir"] = subdir
            paths['solid_pvd_path'] = ROOT_DIR / "solid.pvd"
        elif subdir.name == "surface_output":
            paths["surf_dir"] = subdir
            paths['surf_pvd_path'] = ROOT_DIR / "surface.pvd"

    return paths

def archiveDirectory(output_dir, pvd_path, description="Archiving files"):
    zip_name = output_dir.name + ".zip"
    with zip.ZipFile(zip_name, "w", COMPRESSION_TYPE) as zf:
        if pvd_path.exists():
            # write pvd at top level
            zf.write(pvd_path, arcname=pvd_path.name)
        else:
            print(f"Warning: PVD file not found: {pvd_path}")
        
        files = [f for f in output_dir.rglob("*") if f.is_file()]
        for file in tqdm(files, desc=description):
            zf.write(file, arcname=file.relative_to(ROOT_DIR)) 


def archiveOutputs():
    
    paths = getPathsToArchive()
    
    if paths['flow_dir']:
        archiveDirectory(paths['flow_dir'], paths['flow_pvd_path'], description="Archiving flow files")
    if paths['solid_dir']:
        archiveDirectory(paths['solid_dir'], paths['solid_pvd_path'], description="Archiving solid files")
    if paths['surf_dir']:
        archiveDirectory(paths['surf_dir'], paths['surf_pvd_path'], description="Archiving surface files")
    

if __name__ == "__main__":
    archiveOutputs()
            

    