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
import subprocess
from time import sleep
import sys
 
# roughly, size is a 'half-length' for shapes

class Ellipsoid():
    def __init__(self, size):
        self.a = 1*np.sqrt(size) # x dimension should be very thin .25
        self.b = 1*np.sqrt(size)
        self.c = 1*np.sqrt(size) # z dimension should be very long 1.5
    
    def is_voxel_valid(self,x,y,z):
        return (x/self.a)**2 + (y/self.b)**2 + (z/self.c)**2 < 1
    
    def get_shape_area(self):
        big_sum = (self.a*self.b)**1.6 + (self.a*self.c)**1.6 + (self.b*self.c)**1.6
        return 4*np.pi*(big_sum/3)**(1/1.6) # surface area of ellipsoid
    
    def get_shape_volume(self):
        return (4/3)*np.pi*self.a*self.b*self.c

class Cylinder():
    def __init__(self, size):
        self.length = size*2
        self.radius = self.length/6
        
    def is_voxel_valid(self,x,y,z):
        return (x**2 + y**2 < self.radius**2 and abs(z) < self.length/2)
    
    def get_shape_area(self):
        circle_sum = 2*np.pi*self.radius**2
        return circle_sum + np.pi*2*self.radius*self.length # surface area of cylinder
    
    def get_shape_volume(self):
        return self.length*np.pi*self.radius**2
    
class Cube():
    def __init__(self, size):
        self.hlength = size
        
    def is_voxel_valid(self,x,y,z):
        return (abs(x) < self.hlength and abs(y) < self.hlength and abs(z) < self.hlength)
        
    def get_shape_area(self):
        return 6*(self.hlength*2)**2
    
    def get_shape_volume(self):
        return (self.size*2)**3
    

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

def make_shape(v_size, lims, size, shape):
    voxs = []
    diff = (lims[1] - lims[0])
    nvox_1d = (diff/v_size).astype(int)
    for i in range(3):
        if (nvox_1d[i] % 2):
            nvox_1d[i] += 1
    nvox_1d = (nvox_1d/2 + 0.1).astype(int)
    
    if shape == 'ellipsoid':
        s_obj = Ellipsoid(size)
    elif shape == 'cylinder':
        s_obj = Cylinder(size)
    elif shape == 'cube':
        s_obj = Cube(size)
    else:
        raise Exception('Invalid shape type given')
    
    for i in range(nvox_1d[0]*2):
        x = -nvox_1d[0]*v_size + 0.5*v_size + i*v_size
        for j in range(nvox_1d[1]*2):
            y = -nvox_1d[1]*v_size + 0.5*v_size + j*v_size
            for k in range(nvox_1d[2]*2):
                z = -nvox_1d[2]*v_size + 0.5*v_size + k*v_size
                if (s_obj.is_voxel_valid(x,y,z)):
                    voxs.append([x,y,z])
        
    return np.array(voxs), s_obj.get_shape_area(), s_obj.get_shape_volume()


# read in output
def read_output():
    # read in created data
    f = open('vox2surf.surf')
    surf_lines = f.readlines()
    npoints = int(surf_lines[2].split()[0]) # number of vertices
    f.close()
    
    points = surf_lines[7:7+npoints]
    points = np.array([x.split() for x in points])
    points = pd.DataFrame(data=points, columns=['id', 'x', 'y', 'z'], dtype=np.double).set_index(['id'], verify_integrity=True)
    points = points.set_index(points.index.astype(int))
    
    tris = surf_lines[10+npoints:]
    tris = np.array([x.split() for x in tris])
    tris = pd.DataFrame(data=tris, columns=['id', 'p1', 'p2', 'p3'], dtype=np.double).set_index(['id'], verify_integrity=True).astype(int)
    tris = tris.set_index(tris.index.astype(int))
    
    
    vox_tris = pd.read_csv('voxel_triangles.dat').set_index('vox_idx', verify_integrity=True)
    if (ct):
        tri_cells = pd.read_csv('triangle_cells.dat').set_index('tri', verify_integrity=True)
        
    return points, tris, vox_tris, tri_cells

def validate_geometry(tmesh):
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
   
    sp_out = subprocess.run(["powershell", "bash", "./run.sh"], capture_output=True)
    sp_out = sp_out.stdout.decode('utf-8')
    sp_out = sp_out.split(sep='\n')
    for st in sp_out:
        print(st)



############## IMPORTANT INPUT #############
ncells = np.array([20,20,20])
v_size = 0.12
shapes = ['ellipsoid']

###########################################

print('Running isthmus')
print('------------------')
domain = 2
lo = [-domain]*3
hi = [domain]*3
lims = np.array([lo, hi])
name = 'vox2surf.surf'
ct = True


for s in shapes:    
    print('Creating ' + s)
    print('............................')
    print('Generating volume of voxels...', end='')
    voxs, analytical_sa, analytical_vol = make_shape(v_size, lims, 1, s)
    print(' {:d} voxels created'.format(len(voxs)))
    
    print('Executing marching cubes...')
    # create triangle mesh and assign voxels to triangles
    mc_system = MC_System(lims, ncells, v_size, voxs, name, cell_tri=ct)
    
    print('\nReading generated data')
    print('---------------------------')
    print('Reading in output files...')
    points, tris, vox_tris, tri_cells = read_output()
    
    #print('Checking triangle-cell assignment...')
    # check triangle assignment to cells, verify centroid is inside cell
    
    
    # check geometric properties
    print('\nChecking geometry validity')
    print('--------------------------------')
    tmesh = trimesh.Trimesh(vertices=np.array(points), faces=np.array(tris) - 1)
    validate_geometry(tmesh)
    
    print('\nChecking surface quality')
    print('----------------------------')
    
    # check voxels are inside surface
    vol_voxels = mc_system.voxels
    result = tmesh.contains(vol_voxels).astype(int)
    print(str(len(vol_voxels)) + ' voxels')
    print(str(round(100*sum(result)/len(vol_voxels), 1)) + '% inside')
    
    
    # check actual surface area vs expected surface area vs analytical
    print('Mesh volume: {:.2f}'.format(tmesh.volume))
    print('Voxel volume: {:.2f}'.format(len(voxs)*(v_size**3)))
    print('Analytical volume: {:.2f}'.format(analytical_vol))
    
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

    print('\n\n')
    
    """
    total = 21
    for i in range(total):
        sys.stdout.write('\r')
        finished = int(10*(i+1)/total)
        sys.stdout.write('[' + '='*finished + ' '*(10 - finished) + ']')
        sys.stdout.flush()
        sleep(0.25)
    sys.stdout.write('\n')
    """
