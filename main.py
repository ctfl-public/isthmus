# -*- coding: utf-8 -*-
"""
Main skeleton for isthmus
"""

import numpy as np
import matplotlib.pyplot as plt
from isthmus_prototype import MC_System
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_results(verts, faces, lo, hi):
    tris = Poly3DCollection(verts[faces])
    tris.set_edgecolor('k')
    tris.set_alpha(0.9)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.add_collection3d(tris)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2]) 
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.savefig('triangle_concept', dpi=400)
    plt.show()

def make_ellipsoid(v_size, lims):
    diff = (lims[1] - lims[0])
    nvox_1d = (diff/v_size).astype(int)
    for i in range(3):
        if (nvox_1d[i] % 2):
            nvox_1d[i] += 1
    nvox_1d = (nvox_1d/2 + 0.1).astype(int)
    voxs = []
    a = 1 
    b = 1
    c = 1
    for i in range(nvox_1d[0]*2):
        x = -nvox_1d[0]*v_size + 0.5*v_size + i*v_size + 2.5
        for j in range(nvox_1d[1]*2):
            y = -nvox_1d[1]*v_size + 0.5*v_size + j*v_size + 2.5
            for k in range(nvox_1d[2]*2):
                z = -nvox_1d[2]*v_size + 0.5*v_size + k*v_size + 2.5
                if (((x - 2.5)/a)**2 + ((y - 2.5)/b)**2 + ((z - 2.5)/c)**2 < 1):
                    voxs.append([x,y,z])
    return np.array(voxs)

# input
lo = [0]*3                          # x,y,z limits at low
hi = [5]*3                          # and high positions
lims = np.array([lo, hi])
ncells = np.array([40,40,40])       # x,y,z numbers of cells
v_size = 0.08                       # side length of voxel
name = 'vox2surf.surf'              # name of outputted surface file
voxs = make_ellipsoid(v_size, lims) # for n voxels, make nx3 array of [[x1,y1,z1],[x2,y2,z2],...,[xn,yn,zn]]
# end input

# create triangle mesh and assign voxels to triangles
mc_system = MC_System(lims, ncells, v_size, voxs, name,cell_tri=True)
plot_results(mc_system.verts, mc_system.faces, lo, hi)


"""
import imageio
import trimesh

def loadData(surf):
    if surf.endswith('.tif'):
        # Load the TIFF file
        image_volume = imageio.volread(surf)

    elif surf.endswith('.txt') or surf.endswith('.dat'):
        tempdata = np.loadtxt(surf, skiprows=2)
        xmax = int(max(tempdata[:, 0]))
        ymax = int(max(tempdata[:, 1]))
        zmax = int(max(tempdata[:, 2]))
        image_volume = np.zeros((xmax, ymax, zmax), dtype='int')
        for val in tempdata:
            image_volume[int(val[0]) - 1, int(val[1]) - 1, int(val[2]) - 1] = int(val[3])

    return image_volume

voxs_alt = []
fileName = '/Users/vijaybmohan/Desktop/Cam/isthmus/50.tif'
voxelMTX = loadData(fileName) # integers
for i in range(len(voxelMTX)):
    for j in range(len(voxelMTX)):
        for k in range(len(voxelMTX)):
            if voxelMTX[k,j,i] == 1:
                voxs_alt.append([k,j,i]) # my format

voxs_alt = np.array(voxs_alt)*10**-6 # reduce scale to micrometers


# create triangle mesh and assign voxels to triangles
mc_system = MC_System(lims, ncells, v_size, voxs_alt, name)
voxel_triangle_ids = mc_system.voxel_triangle_ids.astype(int)
faces = mc_system.faces
vertices = mc_system.verts
corner_volumes = mc_system.corner_volumes
combined_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
stlFileName = "originalJoined.stl"

combined_mesh.export(stlFileName, file_type='stl_ascii')   
"""
