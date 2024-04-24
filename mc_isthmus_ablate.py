#!/Users/vijaybmohan/opt/anaconda3/envs/pymesh/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 08:50:33 2024

@author: vijaybmohan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/vijaybmohan/Desktop/Cam/isthmus/src/')  #change the path to folder containing isthmus_prototype
from isthmus_prototype import MC_System
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh
import subprocess
from time import sleep
import os

#%% classes and functions to create geometrical figures for ablation

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

#%%functions to read surface reaction data and create paraview visualization files

def read_voxel_tri(name):
    f = open(name)
    tv_lines = f.readlines()
    f.close()
        
    
    tv_split = [tv.split() for tv in tv_lines]
    triangle_ids = []
    owned_voxels = []
    owned_sfracs = []
    c_voxels = []
    c_scalar_fracs = []
    tri_flag = 0
    for i in range(1, len(tv_split)):
        if (tv_split[i]):
            if (tv_split[i][0] == 'start'):
                triangle_ids.append(int(tv_split[i][-1]))
                c_voxels = []
                c_scalar_fracs = []
                tri_flag = 1
            elif (tv_split[i][0] == 'end'):
                owned_voxels.append(c_voxels)
                owned_sfracs.append(c_scalar_fracs)
                tri_flag = 0
            elif (tri_flag == 1):
                c_voxels.append(int(tv_split[i][0]))
                c_scalar_fracs.append(float(tv_split[i][1]))
            else:
                raise Exception("ERROR: unable to read triangle_voxels.dat")
    tri_voxs = {triangle_ids[i] : owned_voxels[i] for i in range(len(triangle_ids))}
    tri_sfracs = {triangle_ids[i] : owned_sfracs[i] for i in range(len(triangle_ids)) }
    
    for tv in tri_voxs.values():
        for v in range(len(tv)):
            for v2 in range(v+1, len(tv)):
                if tv[v] == tv[v2]:
                    raise Exception("ERROR: surface voxel double-assigned to a triangle")
    return tri_voxs,tri_sfracs

def read_sparta_reaction(name,timescale):
    
    time_flag=0
    ind = 0
    co_formed = []
    
    #Read reation file from sparta
    f=open(name,'r')
    for num, line in enumerate(f, 1):
        if 'ITEM: TIMESTEP' in line:
            first_line=num
            time_flag+=1
        if time_flag == 2:
            ind+=1
        if time_flag == 2 and ind > 9:
            s=tuple(line.split())
            co_formed.append([float(s[0]),float(s[1])*(12*10**-3)*fnum*timescale/(avog*9.5*10**-10)])
    f.close()
    
    return np.array(co_formed)

def write_csv(voxs_plot,c_removed_vox,co_formed,file_no,mass_c_vox):
    
    co_in_cell_plot = np.zeros((len(voxs_plot),4))
    
    for i in range(len(c_removed_vox)):
        if c_removed_vox[i] > mass_c_vox:
            co_in_cell_plot[i,3] = 1
        else:
            co_in_cell_plot[i,3] = 0
    
    f = open('csv/test_del_'+str(file_no)+'.csv','w+')
    for i in range(len(voxs_plot)):
        if i == 0:
            f.write('X,Y,Z,voxel\n')
        f.write(str(voxs_plot[i,0])+','+str(voxs_plot[i,1])+','+str(voxs_plot[i,2])+','+str(co_in_cell_plot[i,3])+'\n')
    f.close()

#%% System initialization

if __name__ == '__main__':
    ############## Create directories for storing data #############

    pathg='/Users/vijaybmohan/Desktop/Cam/isthmus/'      #path of the simulation folder
    
    try:
        if not os.path.exists(pathg+'Grids/'):
            os.mkdir(pathg)
    except OSError as err:
            print(err)
    
    try:
        if not os.path.exists(pathg+'voxel_data/'):
            os.mkdir(pathg+'voxel_data/')
    except OSError as err:
            print(err)
    
    try:
        if not os.path.exists(pathg+'voxel_tri/'):
            os.mkdir(pathg+'voxel_tri/')
    except OSError as err:
            print(err)
    
    try:
        if not os.path.exists(pathg+'csv/'):
            os.mkdir(pathg+'csv/')
    except OSError as err:
            print(err)
    
    try:
        if not os.path.exists(pathg+'ReactionFiles/'):
            os.mkdir(pathg+'ReactionFiles/')
    except OSError as err:
            print(err)
    
    try:
        if not os.path.exists(pathg+'Flowfiles/'):
            os.mkdir(pathg+'Flowfiles/')
    except OSError as err:
            print(err)
    
    
    ############## Choose test case #############
    ncells = np.array([50,50,50])
    v_size = 1*10**-6
    shapes = ['ellipsoid']
    
    ############## Important input #############
    
    print('Running isthmus')
    print('------------------')
    domain = 25*10**-6
    lo = [-25*10**-6]*3
    hi = [domain]*3
    lims = np.array([lo, hi])
    name = 'vox2surf.surf'
    
    s = 'ellipsoid'
    
    #############################################
    
    ############## Sparta variables #############
    
    file_no = int(sys.argv[3])
    fnum = float(sys.argv[2])
    nrho = 6.0386473e+22
    avog = 6.022*10**23
    timescale = 1                    # geometry updates performed in real time
    
    #############################################

#%% Intialization loop for setting up Sparta 
    if file_no == 0:
        # generate voxels to be used by marching cubes
        print('Generating '+ s + ' volume of voxels...', end='')
        voxs, analytical_sa, analytical_vol = make_shape(v_size, lims, 625*10**-12, s)
        print(' {:d} voxels created'.format(len(voxs)))
        
        # create triangle mesh and assign voxels to triangles; read in mesh data
        print('Executing marching cubes...')
        mc_system = MC_System(lims, ncells, v_size, voxs, name, file_no)
        corner_volumes = mc_system.corner_volumes
        faces = mc_system.faces
        vertices = mc_system.verts
        combined_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        stlFileName = 'Grids/grid_'+str(file_no)+'.stl'
        combined_mesh.export(stlFileName, file_type='stl_ascii') 
        
        #write co-rdinate voxel data
        f = open('voxel_data/voxel_data_'+str(file_no)+'.dat','w+')
        for i in range(len(voxs)):
            f.write(str(voxs[i,0])+','+str(voxs[i,1])+','+str(voxs[i,2])+'\n')
        f.close()
        
        vol_frac_c = np.sum(corner_volumes)/(ncells[0]*ncells[1]*ncells[2])
        
        #Write the file containing volume fraction of the material
        f = open('vol_frac.dat','w+')
        f.write(str(vol_frac_c)+'\n')
        f.close()
        
        print('All files written')
        print('Surface ready for sparta run')
    
#%% Coupling loop executed after at regular intervals to alter the geometry of the material depending on surface reactions    
    else:
    ############## Read volume fraction, voxel and surface reaction data ###################
        
        with open('vol_frac.dat') as f:
            vol_frac_c = f.readline().strip('\n')
        
        #Read voxel data
        with open('voxel_data/voxel_data_'+str(file_no-1)+'.dat') as f:
            lines = (line for line in f if not line.startswith('#'))
            voxs_alt = np.loadtxt(lines, delimiter=',', skiprows=0)
           
        #Read voxel data for preparing csv file
        with open('voxel_data/voxel_data_'+str(file_no-1)+'.dat') as f:
            lines = (line for line in f if not line.startswith('#'))
            voxs_plot = np.loadtxt(lines, delimiter=',', skiprows=0)
        
        #Read voxel_triangles
        tri_voxs,tri_sfracs = read_voxel_tri('voxel_tri/triangle_voxels_'+str(file_no-1)+'.dat')
    
        #Read surface reactions
        co_formed = read_sparta_reaction('/Users/vijaybmohan/Desktop/Cam/Grids/low_r/ReactionFiles/surf_react_sparta_'+str(file_no)+'.out',timescale)
        
    #############################################################################################    
        co_formed = co_formed[co_formed[:, 0].argsort()]
       
        #Calculate mass of Carbon associated with each voxel
        volfrac_c = float(vol_frac_c)
        vol_C = volfrac_c*(lims[1,0]-lims[0,0])**3
        mass_c = vol_C*1800
        mass_c_vox = mass_c/len(voxs_alt)
        
        #Triangles check between sparta and isthmus
        if len(co_formed) != len(tri_voxs):
            raise Exception("No of triangles in sparta is not equal to isthmus, debug!!!")
        
        #Calculate the mass of carbon removed from each voxel
        c_removed_vox = np.zeros((len(voxs_alt)))
        for i in range(len(tri_voxs)):
            vox_no = np.array((tri_voxs[(i+1)]),dtype = int)
            sfracs = np.array((tri_sfracs[(i+1)]),dtype = int)
            for k in range(len(vox_no)):
                if c_removed_vox[vox_no[k]] == 0:
                    c_removed_vox[vox_no[k]] = sfracs[k] * co_formed[i,1]
                else:
                    c_removed_vox[vox_no[k]] = c_removed_vox[vox_no[k]] + sfracs[k] * co_formed[i,1]
        
        #Remove voxels
        for i in range(len(c_removed_vox)):
            if c_removed_vox[i] > mass_c_vox:
                voxs_alt[i,:] = 0
        voxs_alt = voxs_alt[~np.all(voxs_alt == 0, axis=1)]   
        
        #write csv files for parview visualization
        write_csv(voxs_plot,c_removed_vox,co_formed,file_no,mass_c_vox)
        
        #Print stats for debugging sparta simulation
        print('\nPrint stats for debugging sparta simulation')
        print('----------------------------------------------\n')
        print('Mass of carbon associated with each voxel --', mass_c_vox)
        print('Maximum of co_formed on each surface --', max(co_formed[:,1]))
        print('Number of voxels deleted this iteration --', len(voxs_plot)-len(voxs_alt))
        print('\n')
        
        # create triangle mesh, assign voxels to triangles and save mesh
        mc_system = MC_System(lims, ncells, v_size, voxs_alt, name, file_no)
        corner_volumes = mc_system.corner_volumes
        faces = mc_system.faces
        vertices = mc_system.verts
        combined_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        stlFileName = 'Grids/grid_'+str(file_no)+'.stl'
        combined_mesh.export(stlFileName, file_type='stl_ascii') 
        
        f = open('voxel_data/voxel_data_'+str(file_no)+'.dat','w+')
        for i in range(len(voxs_alt)):
            f.write(str(voxs_alt[i,0])+','+str(voxs_alt[i,1])+','+str(voxs_alt[i,2])+'\n')
        f.close()
        
        #Write the file containing volume fraction of the material
        vol_frac_c = np.sum(corner_volumes)/(ncells[0]*ncells[1]*ncells[2])
        f = open('vol_frac.dat','w+')
        f.write(str(vol_frac_c)+'\n')
        f.close()
        
        print('All files written')
        print('Surface ready for next iteration of sparta run')
    