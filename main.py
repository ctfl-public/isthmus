# -*- coding: utf-8 -*-
"""
Main skeleton for isthmus
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from time import time
from isthmus_prototype import MC_System

def generate_test_voxels(v_size, ns, lims):
    rs = 1
    xc = ((np.random.rand(ns))*(lims[1][0] - lims[0][0] - 2*rs)) + lims[0][0] + rs
    yc = ((np.random.rand(ns))*(lims[1][1] - lims[0][1] - 2*rs)) + lims[0][1] + rs
    zc = ((np.random.rand(ns))*(lims[1][2] - lims[0][2] - 2*rs)) + lims[0][2] + rs
    r = np.array([rs]*ns)

    xs = []
    ys = []
    zs = []
    for i in range(ns):
        sphere_x, sphere_y, sphere_z = make_sphere(xc, yc, zc, r, v_size)
        xs += sphere_x
        ys += sphere_y
        zs += sphere_z
        
    return xs, ys, zs

def make_sphere(xc, yc, zc, r, v_size):
    nvox_1d = int(2*r/v_size)
    if (nvox_1d % 2):
        nvox_1d += 1
    nvox_1d = int(nvox_1d/2 + 0.1)
    
    xs = []
    ys = []
    zs = []
    for i in range(nvox_1d*2):
        x = -nvox_1d*v_size + 0.5*v_size + i*v_size
        for j in range(nvox_1d*2):
            y = -nvox_1d*v_size + 0.5*v_size + j*v_size
            for k in range(nvox_1d*2):
                z = -nvox_1d*v_size + 0.5*v_size + k*v_size
                if (np.sqrt(x**2 + y**2 + z**2) < r):
                    xs.append(x + xc)
                    ys.append(y + yc)
                    zs.append(z + zc)
    return xs, ys, zs   

def plot_results(verts, faces, lo, hi, proj=False):
    tris = Poly3DCollection(verts[faces])
    tris.set_edgecolor('k')
    tris.set_alpha(0.7)
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')
    ax.add_collection3d(tris)
    #for i in range(len(xs)):
    #    ax.quiver(xs[i][0], xs[i][1], xs[i][2], ns[i][0], ns[i][1], ns[i][2], color='red',length=1)

    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2]) 
    if (proj):
        ax.view_init(elev=90, azim=90, roll=0)
    plt.tight_layout()
    #plt.savefig('isthmus_test', dpi=400)
    plt.show()


np.random.seed(199)
ns = 1


lo = [-2.5, -2.5, -2.5]
hi = [2.5, 2.5, 2.5]
lims = np.array([lo, hi])
ncells = np.array([10, 10, 10])


mc_time = []
nvox = []

v_size = [0.1,0.05,0.04]

for v in v_size:
    xs, ys, zs = make_sphere(0,0,0,1, v)
    nvox.append(len(xs))
    
    voxels = np.transpose(np.array([xs, ys, zs]))
    
    # create triangle mesh and assign voxels to triangles
    t1 = time()
    mc_system = MC_System(lims, ncells, v, voxels)
    voxel_triangle_ids = mc_system.voxel_triangle_ids.astype(int)
    mc_time.append(time() - t1)
    print('done...')
    
    #plot_results(mc_system.mesh.verts, mc_system.mesh.faces, lo, hi)

nvox = np.array(nvox)
mc_time = np.array(mc_time)
plt.figure()
a, b = np.polyfit(nvox, mc_time, 1)
plt.plot(nvox, a*nvox + b, color='blue', label='Best Fit Line')
plt.scatter(nvox, mc_time, color='red', label='Isthmus Output')
plt.xlabel('Number of voxels')
plt.ylabel('Time for Marching Cubes System')
plt.title('10x10x10 MC Grid')
plt.legend()
plt.grid()

"""
# create array of triangle ids (add 1 to match sparta indices)
voxs = mc_system.voxels
triangles = []
for i in range(len(voxs)):
    triangles.append(voxs[i].triangle + 1)

vx_x = []
vx_y = []
vx_z = []
vx_tri = []
for i in range(len(voxs)):
    vx_x.append(voxs[i].position[0])
    vx_y.append(voxs[i].position[1])
    vx_z.append(voxs[i].position[2])
    vx_tri.append(voxs[i].triangle)
    
f = open('solid.tris', 'w')
f.write('x,y,z,t\n')
for i in range(len(vx_x)):
    f.write('{x},{y},{z},{t}\n'.format(x = vx_x[i], y = vx_y[i], z = vx_z[i], t = vx_tri[i]))
f.close()
"""



"""
nvox = np.array(nvox)
mc_time = np.array(mc_time)
plt.figure()
a, b = np.polyfit(nvox, mc_time, 1)
plt.plot(nvox, a*nvox + b, color='blue', label='Best Fit Line')
plt.scatter(nvox, mc_time, color='red', label='Isthmus Output')
plt.xlabel('Number of voxels')
plt.ylabel('Time for Marching Cubes System')
plt.title('50x50x50 MC Grid')
plt.legend()
plt.grid()


mc_time = []
mc_cells = []
ncells = [[50, 50, 50], [60,60,60]]
v_size = 0.04
xs, ys, zs = generate_test_voxels(v_size, ns, lims)
total_voxels = len(xs)
voxels = np.transpose(np.array([xs, ys, zs]))
for i in range(len(ncells)):
    
    mc_cells.append(np.prod(np.array(ncells[i])))
    
    t1 = time()
    mc_grid = MC_Grid(lims, ncells[i], v_size, voxels)
    mc_time.append(time() - t1)

    plot_results(mc_grid.mesh.verts, mc_grid.mesh.faces, lo, hi)
    
mc_cells = np.array(mc_cells)
mc_time = np.array(mc_time)
plt.figure()
a, b = np.polyfit(mc_cells, mc_time, 1)
plt.plot(mc_cells, a*mc_cells + b, color='blue', label='Best Fit Line')
plt.scatter(mc_cells, mc_time, color='red', label='Isthmus Output')
plt.xlabel('Number of MC Cells')
plt.ylabel('Time for Marching Cubes System')
plt.title(str(total_voxels) + ' voxels')
plt.legend()
plt.grid()

"""
