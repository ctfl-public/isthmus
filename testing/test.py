# -*- coding: utf-8 -*-
"""
Testing suite for isthmus
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../src/')
from isthmus_prototype import MC_System
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh
import subprocess
from time import sleep
import os
 
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
        
    return points, tris, vox_tris

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
   
    if os.path.exists('forces.dat.0'):
        os.remove('forces.dat.0')
    if os.path.exists('forces.dat.10'):
        os.remove('forces.dat.10')
    
    print('Testing mesh in SPARTA...')
    sp_out = subprocess.run(["powershell", "bash", "./run.sh"], capture_output=True)
    sp_out = sp_out.stdout.decode('utf-8')
    sp_out = sp_out.split(sep='\n')
    print('\n==============================================')
    for st in sp_out:
        print(st)
    print('\n==============================================\n')
    if not os.path.exists('forces.dat.0'):
        raise Exception('ERROR: SPARTA run failed')



############## IMPORTANT INPUT #############
ncells = np.array([20,20,20])
iterations = 1
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


for s in shapes:    
    # generate voxels to be used by marching cubes
    print('Generating '+ s + ' volume of voxels...', end='')
    voxs, analytical_sa, analytical_vol = make_shape(v_size, lims, 1, s)
    print(' {:d} voxels created'.format(len(voxs)))
    
    for i in range(iterations):
        # create triangle mesh and assign voxels to triangles; read in mesh data
        print('Executing marching cubes...')
        mc_system = MC_System(lims, ncells, v_size, voxs, name)
        print('Reading in generated mesh files...')
        points, tris, vox_tris = read_output()
        print('Output successfully generated\n')
        
        
        # check geometric properties
        print('\nChecking geometry validity')
        print('--------------------------------')
        tmesh = trimesh.Trimesh(vertices=np.array(points), faces=np.array(tris) - 1)
        validate_geometry(tmesh)
        print('Geometry is valid\n')
        
        print('\nChecking surface quality')
        print('----------------------------')
        # check voxels are inside surface
        vol_voxels = voxs #vol_voxels = mc_system.voxels
        result = tmesh.contains(vol_voxels).astype(int)
        print(str(round(100*sum(result)/len(vol_voxels), 1)) + '% of voxels inside mesh out of ' + str(len(vol_voxels)) + ' total voxels')
        
        
        # check actual surface area vs expected surface area vs analytical
        print('{:.1f}% of voxel volume taken up by mesh ({:.2f} of {:.2f})'.format( \
              100*tmesh.volume/(len(voxs)*(v_size**3)), tmesh.volume, (len(voxs)*(v_size**3))))
        if (i == 0):
            print('{:.1f}% of analytical volume taken up by mesh ({:.2f} of {:.2f})'.format( \
                  100*tmesh.volume/analytical_vol, tmesh.volume, analytical_vol))
        
        print('\nChecking mesh quality')
        print('----------------------------')
        # area = sqrt(s(s - a)(s - b)(s - c)), Heron's formula, for triangle of sides a,b,c, s = 1/2*(a + b + c)
        # AR = abc/(8*(s-a)*(s-b)*(s-c)) 
        tri_area = []
        tri_AR = []
        tris['area'] = 0.0
        for i in range(len(tris)):
            ind = tris.iloc[i].name
            p1 = np.array(points.loc[tris.iloc[i].loc['p1']])
            p2 = np.array(points.loc[tris.iloc[i].loc['p2']])
            p3 = np.array(points.loc[tris.iloc[i].loc['p3']])
            
            a = np.linalg.norm(p2 - p1)
            b = np.linalg.norm(p3 - p2)
            c = np.linalg.norm(p1 - p3)
            s = 0.5*(a + b + c)
            c_area = np.sqrt(s*(s - a)*(s - b)*(s - c))
            tri_area.append(c_area)
            tris.loc[ind, 'area'] = c_area
            c_AR = a*b*c/(8*(s - a)*(s - b)*(s - c))
            tri_AR.append(c_AR)        
            
        print('Area ratio of triangle to voxel face: Min {:.2f} Avg {:.2f} Max {:.2f}'.format(min(tri_area)/(v_size)**2, (sum(tri_area)/len(tri_area))/(v_size)**2, max(tri_area)/(v_size**2)))
        print('Aspect ratio of triangles:            Min {:.2f} Avg {:.2f} Max {:.2f}'.format(min(tri_AR), sum(tri_AR)/len(tri_AR), max(tri_AR)))
        
        print('\n\nChecking voxel-triangle assignment...')
        print('------------------------------------------')
        # check voxel assignment to triangles, voxel distance to centroid as fraction of cell length
        
        # first check that triangles haven't been changed by sparta
        # tris is from vox2surf file written by isthmus
        tris['v1x'] = 0.0
        tris['v1y'] = 0.0
        tris['v1z'] = 0.0
        tris['v2x'] = 0.0
        tris['v2y'] = 0.0
        tris['v2z'] = 0.0
        tris['v3x'] = 0.0
        tris['v3y'] = 0.0
        tris['v3z'] = 0.0
        for i in range(len(tris)):
            ind = tris.iloc[i].name
            tris.loc[ind, 'v1x'] = points['x'].loc[tris['p1'].iloc[i]]
            tris.loc[ind, 'v1y'] = points['y'].loc[tris['p1'].iloc[i]]
            tris.loc[ind, 'v1z'] = points['z'].loc[tris['p1'].iloc[i]]
            tris.loc[ind, 'v2x'] = points['x'].loc[tris['p2'].iloc[i]]
            tris.loc[ind, 'v2y'] = points['y'].loc[tris['p2'].iloc[i]]
            tris.loc[ind, 'v2z'] = points['z'].loc[tris['p2'].iloc[i]]
            tris.loc[ind, 'v3x'] = points['x'].loc[tris['p3'].iloc[i]]
            tris.loc[ind, 'v3y'] = points['y'].loc[tris['p3'].iloc[i]]
            tris.loc[ind, 'v3z'] = points['z'].loc[tris['p3'].iloc[i]]
        
        
        # read in SPARTA data
        f = open('forces.dat.10')
        surf_lines = f.readlines()
        f.close()
        # stris for 'sparta-triangles'
        n_stris = int(surf_lines[3])
        if n_stris != len(tris):
            raise Exception('ERROR: SPARTA has different number of faces than isthmus output')
        stris = surf_lines[9:9+n_stris]
        stris = np.array([x.split() for x in stris])
        cols = ['id', 'v1x', 'v1y', 'v1z', 'v2x', 'v2y', 'v2z', 'v3x', 'v3y', 'v3z', 'f1', 'f2']
        stris = pd.DataFrame(data=stris, columns=cols, dtype=np.double).set_index(['id'], verify_integrity=True)
        stris = stris.set_index(stris.index.astype(int))
        
        epsilon_sq = (1e-6)**2
        pc = 0
        print('Are triangles of data equivalent to isthmus? ', end='')
        sq_norm1 = (stris['v1x'] - tris['v1x'])**2 + (stris['v1y'] - tris['v1y'])**2 + (stris['v1z'] - tris['v1z'])**2
        sq_norm2 = (stris['v2x'] - tris['v2x'])**2 + (stris['v2y'] - tris['v2y'])**2 + (stris['v2z'] - tris['v2z'])**2
        sq_norm3 = (stris['v3x'] - tris['v3x'])**2 + (stris['v3y'] - tris['v3y'])**2 + (stris['v3z'] - tris['v3z'])**2
    
        for i in range(len(sq_norm1)):
            if (sq_norm1.iloc[i] < epsilon_sq and sq_norm2.iloc[i] < epsilon_sq and sq_norm3.iloc[i] < epsilon_sq):
                pc += 1
        pc *= 100/len(stris)
    
        print('{:.2f}% yes'.format(pc))
        tris['f1'] = stris['f1']
        tris['f2'] = stris['f2']
        
        # now check number of voxels assigned to each triangle
        tris['nvox'] = 0
        tris['centx'] = (tris['v1x'] + tris['v2x'] + tris['v3x'])/3
        tris['centy'] = (tris['v1y'] + tris['v2y'] + tris['v3y'])/3
        tris['centz'] = (tris['v1z'] + tris['v2z'] + tris['v3z'])/3
        min_dist = 3*domain
        max_dist = 0
        total_dists = 0

        
        vox_tris['vx'] = 0.0
        vox_tris['vy'] = 0.0
        vox_tris['vz'] = 0.0
        vox_tris['tx'] = 0.0
        vox_tris['ty'] = 0.0
        vox_tris['tz'] = 0.0
        # populate voxel-triangle data
        for i in range(len(vox_tris)):
            ind = vox_tris.iloc[i].name
            vox_tris.loc[ind, 'vx'] = voxs[ind][0]
            vox_tris.loc[ind, 'vy'] = voxs[ind][1]
            vox_tris.loc[ind, 'vz'] = voxs[ind][2]
            vox_tris.loc[ind, 'tx'] = tris.loc[vox_tris.loc[ind, 'tri_id'], 'centx']
            vox_tris.loc[ind, 'ty'] = tris.loc[vox_tris.loc[ind, 'tri_id'], 'centy']
            vox_tris.loc[ind, 'tz'] = tris.loc[vox_tris.loc[ind, 'tri_id'], 'centz']
            
            tris.loc[vox_tris.loc[ind, 'tri_id'], 'nvox'] += 1
    
        vox_tris['dist'] = np.sqrt((vox_tris['vx'] - vox_tris['tx'])**2 + (vox_tris['vy'] - vox_tris['ty'])**2 + (vox_tris['vz'] - vox_tris['tz'])**2)


        print('{:.1f}% of {:d} triangles have voxel(s) assigned'.format(100*sum(tris.loc[:, 'nvox'].astype(bool))/len(tris), len(tris)))
        print('Voxels assigned per-triangle:                     Min {:.2f} Avg {:.2f} Max {:.2f}'.format(tris.loc[:, 'nvox'].min(), tris.loc[:, 'nvox'].mean(), tris.loc[:, 'nvox'].max()))
        print('Distance from voxel center to triangle centroid:  Min {:.2f} Avg {:.2f} Max {:.2f}'.format(vox_tris.loc[:, 'dist'].min(), vox_tris.loc[:, 'dist'].mean(), vox_tris.loc[:, 'dist'].max()))
        
        
        # now dummy algorithm for applying ablation/force data to voxels
        # cosine designed to have some difference over the surface
        tris['r'] = 0.0
        tris['angle'] = 0.0
        for t in tris.index:
            cent = np.array(tris.loc[t].loc[['centx', 'centy', 'centz']])
            r = np.linalg.norm(cent)
            tris.loc[t, 'r'] = r
            tris.loc[t, 'angle'] = np.arccos(np.dot(cent/r, np.array([1,0,0])))

        tris['scalar'] = (5*np.cos(tris['angle']) + 5)*tris['area']
        scalar_total_t = tris.loc[:, 'scalar'].sum()
        
        vox_tris['scalar'] = 0.0
        vox_tris['r'] = 0.0
        vox_tris['angle'] = 0.0
        for i in range(len(vox_tris)):
            ind = vox_tris.iloc[i].name
            triangle_data = tris.loc[vox_tris.loc[ind, 'tri_id']]
            vox_tris.loc[ind, 'scalar'] = triangle_data['scalar']/triangle_data['nvox']
            
            # get geometric data for voxels
            cent = np.array(vox_tris.loc[ind].loc[['vx', 'vy', 'vz']])
            r = np.linalg.norm(cent)
            vox_tris.loc[ind, 'r'] = r
            vox_tris.loc[ind, 'angle'] = np.arccos(np.dot(cent/r, np.array([1,0,0])))
           
        scalar_total_v = vox_tris.loc[:, 'scalar'].sum()
        print('Voxel scalar total is {:.1f}% of triangle scalar total ({:.2f})'.format(100*scalar_total_v/scalar_total_t, scalar_total_t))
        
        xa = np.linspace(0, np.pi)
        ya = 5*np.cos(xa) + 5
        plt.figure()
        plt.plot(xa*180/np.pi, ya, color='green', label='Analytical')

        tris = tris.sort_values(by=['angle'])
        vox_tris = vox_tris.sort_values(by=['angle'])
        divisions = [4, 8]
        cl = ['blue', 'red', 'purple', 'orange']
        sphere_rad = 1.0
        j = 0
        for d in divisions:
            borders = np.linspace(0, np.pi, d+1)
            angles = []
            t_pressures = []
            v_pressures = []
            for i in range(1, len(borders)):
                b = borders[i]
                a = borders[i - 1]
                angles.append((a + b)/2)
                section_area = -2*np.pi*(sphere_rad**2)*(np.cos(b) - np.cos(a))
                
                c_tris = tris[tris['angle'] < b]
                c_tris = c_tris[c_tris['angle'] >= a]
                c_voxs = vox_tris[vox_tris['angle'] < b]
                c_voxs = c_voxs[c_voxs['angle'] >= a]

                t_pressures.append(c_tris.loc[:,'scalar'].sum()/section_area)
                v_pressures.append(c_voxs.loc[:,'scalar'].sum()/section_area)
            angles = np.array(angles)
            plt.scatter(angles*180/np.pi, v_pressures, color=cl[j], marker='s', label='Voxels ' + str(d))
            plt.scatter(angles*180/np.pi, t_pressures, color=cl[j], marker='^', label='Triangles ' + str(d))
            j += 1
        plt.ylabel('Scalar Pressure Value')
        plt.xlabel('Angle from Stagnation Point')
        plt.legend()
        plt.show()

                    
        # spatial distribution of scalar data? progressively smaller discretization?
        # generate placeholder distribution data
        # can do testing with just this to determine reasonable convergence ahead of time
        
        
        
        # plot results for sanity check
        #plot_results(mc_system.verts, mc_system.faces, lo, hi)
        print()
        
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
