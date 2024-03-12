# -*- coding: utf-8 -*-
"""
Testing suite for isthmus
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from isthmus_prototype import MC_System
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh
import copy
 

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
    plt.savefig('test_geometry', dpi=400)
    plt.show()

def make_ellipsoid(v_size, lims):
    diff = (lims[1] - lims[0])
    nvox_1d = (diff/v_size).astype(int)
    for i in range(3):
        if (nvox_1d[i] % 2):
            nvox_1d[i] += 1
    nvox_1d = (nvox_1d/2 + 0.1).astype(int)
    voxs = []
    a = 0.25
    b = 1
    c = 2
    
    valid_voxels = np.zeros((nvox_1d[0]*2,nvox_1d[1]*2,nvox_1d[2]*2))
    for i in range(nvox_1d[0]*2):
        x = -nvox_1d[0]*v_size + 0.5*v_size + i*v_size + 2.5
        for j in range(nvox_1d[1]*2):
            y = -nvox_1d[1]*v_size + 0.5*v_size + j*v_size + 2.5
            for k in range(nvox_1d[2]*2):
                z = -nvox_1d[2]*v_size + 0.5*v_size + k*v_size + 2.5
                if (((x - 2.5)/a)**2 + ((y - 2.5)/b)**2 + ((z - 2.5)/c)**2 < 1):
                    voxs.append([x,y,z])
                
    big_sum = (a*b)**1.6 + (a*c)**1.6 + (b*c)**1.6
    asa = 4*np.pi*(big_sum/3)**(1/1.6) # surface area of ellipsoid
    return np.array(voxs), asa






# generate random shape? random walk shape?

print('Running isthmus')
print('------------------')
print('Generating input...', end='')
# input
lo = [0]*3
hi = [5]*3
lims = np.array([lo, hi])
ncells = np.array([40,40,40])
v_size = 0.08
name = 'vox2surf.surf'
voxs, analytical_sa = make_ellipsoid(v_size, lims)
ct = True
print(' {:d} voxels created'.format(len(voxs)))
# end inputs

print('Executing marching cubes...')
# create triangle mesh and assign voxels to triangles
mc_system = MC_System(lims, ncells, v_size, voxs, name, cell_tri=ct)

print('\nReading generated data')
print('---------------------------')
print('Reading in output files...')
# read in created data
f = open('vox2surf.surf')
surf_lines = f.readlines()
npoints = int(surf_lines[2].split()[0]) # number of vertices
ntris =  int(surf_lines[3].split()[0])
f.close()

points = surf_lines[7:7+npoints]
points = np.array([x.split() for x in points])
points = pd.DataFrame(data=points, columns=['id', 'x', 'y', 'z'], dtype=np.double).set_index(['id'], verify_integrity=True)
points = points.set_index(points.index.astype(int))

tris = surf_lines[11+npoints:]
tris = np.array([x.split() for x in tris])
tris = pd.DataFrame(data=tris, columns=['id', 'p1', 'p2', 'p3'], dtype=np.double).set_index(['id'], verify_integrity=True).astype(int)
tris = tris.set_index(tris.index.astype(int))


vox_tris = pd.read_csv('voxel_triangles.dat').set_index('vox_idx', verify_integrity=True)
if (ct):
    tri_cells = pd.read_csv('triangle_cells.dat').set_index('tri', verify_integrity=True)

#print('Checking triangle-cell assignment...')
# check triangle assignment to cells, verify centroid is inside cell


# check geometric properties
print('\nChecking geometry validity')
print('--------------------------------')
tmesh = trimesh.Trimesh(vertices=np.array(points), faces=np.array(tris) - 1)
if (tmesh.is_watertight):
    print('Mesh is watertight...')
else:
    raise Exception('ERROR: mesh not watertight')

if (tmesh.is_winding_consistent):
    print('Mesh winding is consistent...')
else:
    raise Exception('ERROR: mesh winding not consistent')

if (tmesh.is_volume):
    print('Mesh is a manifold...')
else:
    raise Exception('ERROR: mesh is not a manifold')


print('\nChecking surface quality')
print('----------------------------')

# check voxels are inside surface
vol_voxels = mc_system.voxels
result = tmesh.contains(vol_voxels).astype(int)
print(str(len(vol_voxels)) + ' voxels')
print(str(round(100*sum(result)/len(vol_voxels), 1)) + '% inside')


# check actual surface area vs expected surface area vs analytical
mesh_sa = mc_system.get_surface_area()
print('Mesh surface area: {:.2f}'.format(mesh_sa))
print('Analytical surface area: {:.2f}'.format(analytical_sa))


#print('Checking voxel-triangle assignment...')
# check voxel assignment to triangles, voxel distance to centroid as fraction of cell length

# spatial distribution of scalar data? progressively smaller discretization?
# conservation of scalar data
# no-voxel triangles?
# generate placeholder data
# normal distribution of data
# can do testing with just this to determine reasonable convergence
# ahead of time
# aside: do lpm crush/sphere tests ahead of time to test convergence?

# plot results for sanity check
plot_results(mc_system.verts, mc_system.faces, lo, hi)

