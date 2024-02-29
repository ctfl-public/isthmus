# -*- coding: utf-8 -*-
"""
Main skeleton for isthmus
"""

import numpy as np
import matplotlib.pyplot as plt
from isthmus_prototype import MC_System

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
        x = -nvox_1d[0]*v_size + 0.5*v_size + i*v_size
        for j in range(nvox_1d[1]*2):
            y = -nvox_1d[1]*v_size + 0.5*v_size + j*v_size
            for k in range(nvox_1d[2]*2):
                z = -nvox_1d[2]*v_size + 0.5*v_size + k*v_size
                if ((x/a)**2 + (y/b)**2 + (z/c)**2 < 1):
                    voxs.append([x,y,z])
    return np.array(voxs)

# input
lo = [-2.5]*3
hi = [2.5]*3
lims = np.array([lo, hi])
ncells = np.array([40,40,40])
v_size = 0.045
name = 'vox2surf.surf' # name of outputted surface file
# end inputs

voxs = make_ellipsoid(v_size, lims)

plot_voxs = np.transpose(voxs)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.view_init(elev=0, azim=0, roll=0)
ax.scatter(plot_voxs[0], plot_voxs[1], plot_voxs[2])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.view_init(elev=90, azim=0, roll=0)
ax.scatter(plot_voxs[0], plot_voxs[1], plot_voxs[2])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.tight_layout()

# create triangle mesh and assign voxels to triangles
mc_system = MC_System(lims, ncells, v_size, voxs, name)
voxel_triangle_ids = mc_system.voxel_triangle_ids.astype(int)
