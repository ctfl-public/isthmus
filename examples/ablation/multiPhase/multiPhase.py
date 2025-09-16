# If not added to the PYTHONPATH, uncomment bellow to add the ISTHMUS source directory.
# import sys
# sys.path.append('/path/to/isthmus/src/')
from isthmus import marchingWindows
print('ISTHMUS module loaded')
#
# Import custom case functions for this script
from utils import *
#
# Case initialization
case = multiPhaseCase()
gpu = False
weight = False
#
# Initial step: Generate initial mesh
step = 0
print(f'Step {step:d}/7')
#
# Run ISTHMUS on loaded voxels and parse volumes, faces, and vertices
resultsMC = marchingWindows(case.lims, case.nCells, case.voxelSize, case.voxs, 'vox2surf.surf', step,
                            weight=weight, gpu=gpu)
cornerVolumes = resultsMC.corner_volumes
faces = resultsMC.faces
vertices = resultsMC.verts
#
# Generate surface .stl file and volume fraction file
case.postProcess(cornerVolumes,vertices,faces,step)
#
# During remaining steps, ablate the material and update the grid
for step in range(1,7):
    print('Step {:d}/7', step)
    #
    # Run DSMC
    case.runDSMC(step)
    #
    # Here the ablation is done and material ablates
    case.ablate(step)
    #
    # Run ISTHMUS on updated voxels for next iteration
    resultsMC = marchingWindows(case.lims, case.nCells, case.voxelSize, case.voxs, 'vox2surf.surf', step, 
                                weight=weight, gpu=gpu)
    cornerVolumes = resultsMC.corner_volumes
    faces = resultsMC.faces
    vertices = resultsMC.verts
    #
    # Post process results
    case.postProcess(cornerVolumes,vertices,faces,step)
#
# Remove temporary files
case.clean()
