# -*- coding: utf-8 -*-
"""
Main skeleton for isthmus
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
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
noc = [4,10,15,20,30,40,50,60,70]
ncells = []
for i in range(len(noc)):
    ncells.append([noc[i]]*3)
ncells = np.array(ncells)

mc_time = []
nf = [] # number of faces
systems = []

v_size = 0.045

direct = False
for i in range(2):
    mc_time.append([])
    nf.append([])
    systems.append([])
    for c in ncells:
        xs, ys, zs = make_sphere(0,0,0,1, v_size)
        
        voxels = np.transpose(np.array([xs, ys, zs]))
        name = 'vox2surf.' + str(len(voxels)) + '.surf' # name of outputted surface file
        
        # create triangle mesh and assign voxels to triangles
        mc_system = MC_System(lims, c, v_size, voxels, name, direct)
        voxel_triangle_ids = mc_system.voxel_triangle_ids.astype(int)
        mc_time[-1].append(mc_system.cell_tri_time)
        nf[-1].append(len(mc_system.faces))
        systems[-1].append(mc_system)
        print('done...')
        #plot_results(mc_system.verts, mc_system.faces, lo, hi)
    print()
    direct = True

systems = np.transpose(np.array(systems))

for i in range(len(systems)):
    sys0 = systems[i][0].cell_grid.cells
    sys1 = systems[i][1].cell_grid.cells
    for j in range(len(sys0)):
        tri0 = np.array(sys0[j].triangles)
        tri1 = np.array(sys1[j].triangles)
        if ((len(tri0) == 0 or len(tri1) == 0) and len(tri1) != len(tri0)):
            raise Exception('algorithms not equivalent')
        elif (len(tri0) == 0 and len(tri1) == 0):
            continue
        elif (not (tri0 == tri1).all()):
            raise Exception('algorithms not equivalent')

nf = np.array(nf)
mc_time = np.array(mc_time)
plt.figure()
cl = ['red', 'blue']
lab = ['Binary', 'Direct']
for i in range(2):
    plt.scatter(nf[i], mc_time[i], color=cl[i], label=lab[i])
plt.xlabel('Number of triangles')
plt.ylabel('Time for MC System [s]')
plt.title(str(len(voxels)) + ' voxel system')
plt.legend()
plt.grid()
plt.savefig('alg_compare_tri',dpi=400)
