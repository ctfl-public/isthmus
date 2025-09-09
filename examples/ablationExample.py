#!/usr/bin/env python3
#
# Add the ISTHMUS source directory and load the marching cubes library
import sys
sys.path.append('/path/to/isthmus/src/')
from isthmus_prototype import MC_System
print('ISTHMUS marching cubes module loaded')
#
# Import other required modules
import os
import numpy as np
import pandas as pd
import trimesh
import imageio
import warnings
import json
#
# Import custom functions for this script
import utils
myfuncs = utils.utils()
#
# Create required directories if they don't already exist
dirs = ['grids','voxelData','voxelTri']
for d in dirs:
    os.makedirs(d,exist_ok=True)
print('Directories created')
#
# Load the geometry of the sample in microns
voxelSize = 3.3757*10**-6
width = 200
height = 100
#
# Sample was cropped with a 5 micron thick boundary of unused voxels
lo = [-5, -5, -5]
hi = [height + 5, (width + 5), (width + 5)]
lims = voxelSize*np.array([lo, hi])
nCells = np.array([int(height),int(width),int(width)])
#
# Load timescale and some quantities for DSMC
timescale = 2
timestepDSMC = 7.5e-9
fnum = 14866.591116363918
avog = 6.022*10**23
#
# Load voxels from sample and rearrange from :volread: format to match ISTHMUS
fileName = 'sample.tif'
voxelMatrix = imageio.volread(fileName)
voxs = []
for i in range(int(width)):
    for j in range(int(width)):
        for k in range(int(height)):
            if voxelMatrix[k,j,i] == 1:
                voxs.append([k,j,i])
voxs = np.array(voxs)*voxelSize
print(f'{len(voxs):d} voxels loaded from sample')
#
# Initial step: Generate initial mesh
step = 0
print(f'Step {step:d}/7')
#
# Run marching cubes on loaded voxels and parse volumes, faces, and vertices
resultsMC = MC_System(lims, nCells, voxelSize, voxs, 'vox2surf.surf', step, os.getcwd())
cornerVolumes, faces, vertices = myfuncs.parseResultsMC(resultsMC, step)
#
# Write coordinate voxel data
cRemovedVox = np.zeros((len(voxs),1))
voxs = np.column_stack((voxs,cRemovedVox))
cVolFrac = np.sum(cornerVolumes)/(nCells[0]*nCells[1]*nCells[2])
#
# Write the file containing volume fraction of the material
f = open('volFrac.dat','w+')
f.write(str(cVolFrac)+'\n')
f.close()
#
# Remaining steps: Ablate the material and update the grid
for step in range(1,7):
    print('Step {:d}/7', step)

    with open('volFrac.dat') as f:
        cVolFrac = f.readline().strip('\n')
    # 
    # Read voxel data 
    with open('voxelData/voxelData_'+str(step-1)+'.dat') as f: 
        lines = (line for line in f if not line.startswith('#')) 
        voxsTemp = np.loadtxt(lines, delimiter=',', skiprows=0) 
    # 
    # Associate voxels to tirangles
    tri_voxs,tri_sfracs = myfuncs.readVoxelTri('voxelTri/triangle_voxels_'+str(step-1)+'.dat')
    # 
    # Read surface reactions
    COFormed = myfuncs.readReactionSPARTA('reactionFiles/surf_react_sparta_'+str(step)+'.out',timescale,timestepDSMC)
    COFormed = COFormed[COFormed[:, 0].argsort()]
    # 
    # Calculate mass of carbon associated with each voxel
    volFracC = float(cVolFrac)
    volC = volFracC*((voxelSize)**3)*(lims[1,0]-lims[0,0])*(lims[1,1]-lims[0,1])*(lims[1,2]-lims[0,2])
    massC = volC*1800
    massCVox = massC/len(voxsTemp)
    # 
    # Calculate the mass of carbon removed from each voxel
    cRemovedVox = np.zeros((len(voxsTemp)))
    for i in range(len(COFormed)):
        vox_no = np.array((tri_voxs[(i+1)]),dtype = int)
        sfracs = np.array((tri_sfracs[(i+1)]),dtype = float)
        for k in range(len(vox_no)):
            cRemovedVox[vox_no[k]] = cRemovedVox[vox_no[k]] + sfracs[k] * COFormed[i,1]
    cRemovedVox[:] = cRemovedVox[:] + voxsTemp[:,3]
    # 
    # Remove voxels
    voxsTemp = np.column_stack((voxsTemp[:,0:3],cRemovedVox))
    for i in range(len(cRemovedVox)):
        if cRemovedVox[i] > massCVox:
            voxsTemp[i,:] = 0
    voxsTemp = voxsTemp[~np.all(voxsTemp == 0, axis=1)]
    voxs_isthmus = voxsTemp[:,0:3]
    # 
    # Create triangle mesh, assign voxels to triangles and save mesh
    resultsMC = MC_System(lims*voxelSize, nCells, voxelSize, voxs_isthmus, 'vox2surf.surf', step, os.getcwd(), weight=True, ndims=3)
    cornerVolumes, faces, vertices = myfuncs.parseResultsMC(resultsMC, step)
    #
    # Write coordinate voxel data
    writeCoordinateVoxelData(voxsTemp)
    #
    # Write the file containing volume fraction of the material
    cVolFrac = np.sum(cornerVolumes)/(nCells[0]*nCells[1]*nCells[2])
    f = open('volFrac.dat','w+')
    f.write(str(cVolFrac)+'\n')
    f.close()
    print(f'{len(voxsTemp):d} voxels remaining')
    #
    with open('voxelData/types'+str(step)+'.dat', 'w+') as file:
        json.dump(voxs_types, file, indent=4)
#
# Remove temporary files
os.remove('voxelData')
os.remove('voxelTri')
os.remove('volFrac.dat')
print('Temporary directories removed')
print('Done!')
