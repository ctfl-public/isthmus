#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 08:31:26 2024

@author: vijaybmohan

testing suite for isthmus ablation
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/vijaybmohan/Desktop/git/isthmus/src')
from isthmus_prototype import MC_System
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh
import subprocess
import os
import platform
from shape_types import Ellipsoid, Cylinder, Cube, make_shape
import pickle
import random
    
#%% classes and functions to create geometrical figures for ablation
# distance to nearest point on triangle, by combining normal and planar components
def distance_to_point(p, tri):
    n_dist = np.dot(tri.normal, p - tri.vertices[0])
    v_proj = (p - tri.normal*n_dist)  # project voxel onto plane
    trans_vox = np.matmul(tri.trans_matrix, v_proj - tri.vertices[0]) # projected voxel in triangle basis
    
    if (trans_vox[0] >= 0 and trans_vox[1] >= 0 and trans_vox[1] <= 1 - trans_vox[0]):  
        return abs(n_dist)
    else:
        if (trans_vox[0] <= 0 or trans_vox[1] > trans_vox[0] + 1):
            pn = np.array([0, np.clip(trans_vox[1], 0, 1), 0]) # on v-axis
        elif (trans_vox[1] <= 0 or trans_vox[1] < trans_vox[0] - 1):
            pn = np.array([np.clip(trans_vox[0], 0, 1), 0, 0]) # on u-axis
        else:
            n_h = np.array([1/np.sqrt(2), 1/np.sqrt(2), 0])    # on hypotenuse
            pn = np.dot((trans_vox - [0, 1, 0]), n_h)
            pn = trans_vox - pn*n_h
        # return to normal Cartesian coordinates with original origin
        contact_point = np.matmul(tri.revert_matrix, pn) + tri.vertices[0]
        plane_dist = np.linalg.norm(v_proj - contact_point)
        
        return np.sqrt(plane_dist**2 + n_dist**2) 

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

# read in output
def read_output(file_no):
    print('\nChecking geometry validity')
    print('--------------------------------')
    print('Reading in generated mesh files...', end=' ')
    
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
    
    
    f = open('voxel_tri/triangle_voxels_'+str(file_no)+'.dat')
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
    
    print('output successfully generated')
    
    return points, tris, tri_voxs, tri_sfracs

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
    
    print('Geometry is valid\n')
            

def surface_quality_check(all_voxs, surf_voxs, tmesh, it, analytical_vol):
    print('\nChecking surface quality')
    print('----------------------------')
    # check voxels are inside surface
    result = tmesh.contains(surf_voxs).astype(int)
    print(str(round(100*sum(result)/len(surf_voxs), 1)) + '% of surface voxels inside mesh (out of ' + str(len(surf_voxs)) + ')')
    
    
    # check actual surface area vs expected surface area vs analytical
    print('{:.1f}% of all voxels\' volume taken up by mesh ({:.2f} of {:.2f})'.format( \
          100*tmesh.volume/(len(all_voxs)*(v_size**3)), tmesh.volume, (len(all_voxs)*(v_size**3))))
    if (it == 0):
        print('{:.1f}% of analytical volume taken up by mesh ({:.2f} of {:.2f})'.format( \
              100*tmesh.volume/analytical_vol, tmesh.volume, analytical_vol))

def mesh_quality_check(tris, points):
    print('\nChecking mesh quality')
    print('----------------------------')
    
    # area = sqrt(s(s - a)(s - b)(s - c)), Heron's formula, for triangle of sides a,b,c, s = 1/2*(a + b + c)
    # AR = ratio of max to min length 
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
        c_AR = max([a,b,c])/min([a,b,c])
        tri_AR.append(c_AR)        
        
    print('Area ratio of triangle to voxel face: Min {:.2f} Avg {:.2f} Max {:.2f}'.format(min(tri_area)/(v_size)**2, (sum(tri_area)/len(tri_area))/(v_size)**2, max(tri_area)/(v_size**2)))
    print('Aspect ratio of triangles:            Min {:.2f} Avg {:.2f} Max {:.2f}'.format(min(tri_AR), sum(tri_AR)/len(tri_AR), max(tri_AR)))
    

def voxel_triangle_assignment_check(tris, tri_voxs, tri_sfracs, mc_system):
    print('\n\nChecking voxel-triangle assignment...')
    print('------------------------------------------')
    # check voxel assignment to triangles, voxel distance to centroid as fraction of cell length
    
    nvoxs_per_triangle = np.zeros(len(tris))
    for tv in tri_voxs.keys():
        if nvoxs_per_triangle[tv - 1] != 0:
            print('WARNING: duplicated triangle in triangle_voxels association')
        nvoxs_per_triangle[tv - 1] = len(tri_voxs[tv])
    print('{:.1f}% of {:d} triangles have voxel(s) assigned'.format( \
            100*np.count_nonzero(nvoxs_per_triangle)/len(tris), len(tris)))
        
    used_voxels = sum(tri_voxs.values(), [])
    print('{:.1f}% of {:d} surface voxels have triangle(s) assigned\n'.format( \
           100*len(set(used_voxels))/len(mc_system.surface_voxels), len(mc_system.surface_voxels)))
    
    surface_flag = np.zeros(len(mc_system.voxels))
    for sv in mc_system.surface_voxels:
        surface_flag[sv.id] = 1
    
    ntris_per_vox = np.zeros(len(mc_system.voxels))
    for uv in used_voxels:
        ntris_per_vox[uv] += 1
    temp_pd = pd.DataFrame(data=np.transpose(np.array([surface_flag, ntris_per_vox])), columns=['surf_flag', 'ntris'])
    temp_pd = temp_pd[temp_pd['surf_flag'] == 1]
    
    print('Triangles assigned per-surface voxel: Min {:.2f} Avg {:.2f} Max {:.2f}'.format( \
           temp_pd.loc[:, 'ntris'].min(), temp_pd.loc[:, 'ntris'].mean(), temp_pd.loc[:, 'ntris'].max()))
    print('Surface voxels assigned per-triangle: Min {:.2f} Avg {:.2f} Max {:.2f}'.format( \
           min(nvoxs_per_triangle), sum(nvoxs_per_triangle)/len(nvoxs_per_triangle), max(nvoxs_per_triangle)))
    print()
        
    max_dist = 0
    min_dist = 1e12
    avg_dist = 0
    nbonds = 0
    for t in mc_system.cell_grid.triangles:
        for v in t.voxel_ids:
            c_dist = distance_to_point(mc_system.voxels[v], t)
            avg_dist += c_dist
            nbonds += 1
            if c_dist > max_dist:
                max_dist = c_dist
            if c_dist < min_dist:
                min_dist = c_dist
    avg_dist /= nbonds
    vs = v_size
    print('Triangle to voxel centroid (multiple of voxel size): Min {:.2f} Avg {:.2f} Max {:.2f}'.format( \
           min_dist/vs, avg_dist/vs, max_dist/vs))
    print('Triangle to voxel centroid (length):                 Min {:.2f} Avg {:.2f} Max {:.2f}\n'.format( \
           min_dist, avg_dist, max_dist))
            
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

def write_csv_all(voxs_plot,c_removed_vox,co_formed,file_no,mass_c_vox):

    co_in_cell_plot = np.zeros((len(voxs_plot),4))

    for i in range(len(c_removed_vox)):
        if c_removed_vox[i] > 0:
            co_in_cell_plot[i,3] = 1
        else:
            co_in_cell_plot[i,3] = 0

    f = open('csv/test_'+str(file_no)+'.csv','w+')
    for i in range(len(voxs_plot)):
        if i == 0:
            f.write('X,Y,Z,voxel\n')
        f.write(str(voxs_plot[i,0])+','+str(voxs_plot[i,1])+','+str(voxs_plot[i,2])+','+str(co_in_cell_plot[i,3])+'\n')
    f.close()

def quadrants(voxs_plot,c_removed_vox,co_formed,file_no,mass_c_vox,lims,quad_no):
    
    co_in_cell_plot = np.zeros((len(voxs_plot),4))

    for i in range(len(c_removed_vox)):
        if c_removed_vox[i] > 0:
            co_in_cell_plot[i,3] = 1
        else:
            co_in_cell_plot[i,3] = 0
    
    f = open('csv/quadrants_'+str(quad_no)+'_'+str(file_no)+'.csv','w+')
    for i in range(len(voxs_plot)):
        if i == 0:
            f.write('X,Y,Z,voxel\n')
        if (voxs_plot[i,0] >= lims[0,0] and voxs_plot[i,0] <= lims[1,0]) and (voxs_plot[i,1] >= lims[0,1] and voxs_plot[i,1] <= lims[1,1]) and (voxs_plot[i,2] >= lims[0,2] and voxs_plot[i,2] <= lims[1,2]):
            f.write(str(voxs_plot[i,0])+','+str(voxs_plot[i,1])+','+str(voxs_plot[i,2])+','+str(co_in_cell_plot[i,3])+'\n')
    f.close()

def find_normals(file_no):
    normals = []
    f=open('Grids/grid_'+str(file_no-1)+'.stl','r')
    for num, line in enumerate(f, 1):
        if 'facet normal' in line:
            s=tuple(line.split())
            normals.append([float(s[2]),float(s[3]),float(s[4])])
    f.close()
     
    return np.array(normals,dtype='float')

def quadrant_stl(file_no,normals,vertices,faces,co_formed_quadrants,quad_no):
    f = open('Grids/grid_quadrant_'+str(quad_no)+'_'+str(file_no)+'.stl','w+')
    for i in range(len(co_formed_quadrants)):
        if i == 0:
            f.write('solid\n')
        f.write('facet normal '+str(normals[co_formed_quadrants[i],0])+' '+str(normals[co_formed_quadrants[i],1])+' '+str(normals[co_formed_quadrants[i],2])+'\n')
        f.write('outer loop\n')
        f.write('vertex '+str(vertices[faces[co_formed_quadrants[i],0],0])+' '+str(vertices[faces[co_formed_quadrants[i],0],1])+' '+str(vertices[faces[co_formed_quadrants[i],0],2])+'\n')
        f.write('vertex '+str(vertices[faces[co_formed_quadrants[i],1],0])+' '+str(vertices[faces[co_formed_quadrants[i],1],1])+' '+str(vertices[faces[co_formed_quadrants[i],1],2])+'\n')
        f.write('vertex '+str(vertices[faces[co_formed_quadrants[i],2],0])+' '+str(vertices[faces[co_formed_quadrants[i],2],1])+' '+str(vertices[faces[co_formed_quadrants[i],2],2])+'\n')
        f.write('endloop\n')
        f.write('endfacet\n')
    f.write('endsolid')
    f.close()

def quadrant_check(file_no,q_lims,quad_no):
    #load the full geometry to isolate the triangles belonging to the quadrant
    vertices = trimesh.load('Grids/grid_'+str(file_no-1)+'.stl').vertices
    faces = trimesh.load('Grids/grid_'+str(file_no-1)+'.stl').faces
    
    co_formed_quadrants_ind = []
    
    for i in range(len(faces)):
        first_vert_check = (vertices[faces[i,0],0] >= q_lims[0,0] and vertices[faces[i,0],0] <= q_lims[1,0]) and (vertices[faces[i,0],1] >= q_lims[0,1] and vertices[faces[i,0],1] <= q_lims[1,1]) and (vertices[faces[i,0],2] >= q_lims[0,2] and vertices[faces[i,0],2] <= q_lims[1,2])
        second_vert_check = (vertices[faces[i,1],0] >= q_lims[0,0] and vertices[faces[i,1],0] <= q_lims[1,0]) and (vertices[faces[i,1],1] >= q_lims[0,1] and vertices[faces[i,1],1] <= q_lims[1,1]) and (vertices[faces[i,1],2] >= q_lims[0,2] and vertices[faces[i,1],2] <= q_lims[1,2])
        third_vert_check = (vertices[faces[i,2],0] >= q_lims[0,0] and vertices[faces[i,2],0] <= q_lims[1,0]) and (vertices[faces[i,2],1] >= q_lims[0,1] and vertices[faces[i,2],1] <= q_lims[1,1]) and (vertices[faces[i,2],2] >= q_lims[0,2] and vertices[faces[i,2],2] <= q_lims[1,2])
        if first_vert_check and second_vert_check and third_vert_check:
            co_formed_quadrants_ind.append(i)
    
    co_formed_quadrants_ind = np.array(co_formed_quadrants_ind,dtype = 'int')
    normals = find_normals(file_no)
    quadrant_stl(file_no,normals,vertices,faces,co_formed_quadrants_ind,quad_no)
    
    #Calculate surface reactions on defined quadrant
    co_formed_quadrant = 0
    c_removed_quadrant = 0
    
    for i in range(len(co_formed_quadrants_ind)):
        
        co_formed_quadrant = co_formed_quadrant + co_formed[co_formed_quadrants_ind[i],1]
        vox_no = np.array((tri_voxs[(co_formed_quadrants_ind[i]+1)]),dtype = int)
        sfracs = np.array((tri_sfracs[(co_formed_quadrants_ind[i]+1)]),dtype = float)
        
        for k in range(len(vox_no)):
            
            c_removed_quadrant = c_removed_quadrant + sfracs[k] * co_formed[co_formed_quadrants_ind[i],1]
    
    quadrants(voxs_plot,c_removed_vox,co_formed,file_no,mass_c_vox,q_lims,quad_no)
    print('x: '+str(q_lims[0,0])+' '+str(q_lims[1,0])+' y: '+str(q_lims[0,1])+' '+str(q_lims[1,1])+' z: '+str(q_lims[0,2])+' '+str(q_lims[1,2]))
    print(co_formed_quadrant)
    print(c_removed_quadrant)           


#%% System initialization  


############## Create directories for storing data #############

pathg='/Users/vijaybmohan/Desktop/git/isthmus/testing/'      #path of the simulation folder

try:
    if not os.path.exists(pathg+'Grids/'):
        os.mkdir(pathg+'Grids/')
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

############## IMPORTANT INPUT #############
ncells = np.array([50,50,50])
iterations = 1
v_size = 1*10**-6
s = ['ellipsoid']
lo = [-25*10**-6]*3
hi = [25*10**-6]*3
lims = np.array([lo, hi])
name = 'vox2surf.surf'
radius = 25*10**-6
rng = np.random.default_rng(999)
#############################################

file_no = 0

#%% Intialization loop for setting up Sparta 
if file_no == 0:
        
    ################ geometrical shapes ########################
    
    #generate voxels to be used by marching cubes
    
    voxs, analytical_sa, analytical_vol = make_shape(v_size, lims, radius, s[0])
    
    ##############################################################
    ############ test quadrant creation ##########################
    no_of_quadrants = 5
    quadrant_lims = np.zeros((no_of_quadrants*2,3))
    for i in range(len(quadrant_lims)):
        if i % 2 == 0:
            quadrant_lims[i,0] = random.randint(-25, 0)
            quadrant_lims[i,1] = random.randint(-25, 0)
            quadrant_lims[i,2] = random.randint(-25, 0)
        if i % 2 == 1:
            quadrant_lims[i,0] = random.randint(quadrant_lims[i-1,0], 25)
            quadrant_lims[i,1] = random.randint(quadrant_lims[i-1,1], 25)
            quadrant_lims[i,2] = random.randint(quadrant_lims[i-1,2], 25)
    
    quadrant_lims = quadrant_lims*10**-6
    pickle.dump(quadrant_lims, open( "quad_lims.pkl", "wb+" ) )
    print('Quadrants created for debugging\n')
    print(quadrant_lims)
    ############################################################# 
        
    # create triangle mesh and assign voxels to triangles; read in mesh data
    print('Executing marching cubes...')
    mc_system = MC_System(lims, ncells, v_size, voxs, name, file_no)
    corner_volumes = mc_system.corner_volumes
    faces = mc_system.faces
    vertices = mc_system.verts
    combined_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    stlFileName = 'Grids/grid_'+str(file_no)+'.stl'
    combined_mesh.export(stlFileName, file_type='stl_ascii') 
    
    # check surface and mesh quality
    points, tris, tri_voxs, tri_sfracs = read_output(file_no)
            
    tmesh = trimesh.Trimesh(vertices=np.array(points), faces=np.array(tris) - 1)
    validate_geometry(tmesh)

    surf_vox_ps = []
    for sv in mc_system.surface_voxels:
        surf_vox_ps.append(sv.position)
    surface_quality_check(voxs, surf_vox_ps, tmesh, file_no, analytical_vol)

    mesh_quality_check(tris, points)
    
    #write co-rdinate voxel data
    c_removed_vox = np.zeros((len(voxs),1))     ######place holder for carry over mass
    voxs = np.column_stack((voxs,c_removed_vox))
    f = open('voxel_data/voxel_data_'+str(file_no)+'.dat','w+')
    for i in range(len(voxs)):
        f.write(str(voxs[i,0])+','+str(voxs[i,1])+','+str(voxs[i,2])+','+str(voxs[i,3])+'\n')
    f.close()
    
    vol_frac_c = np.sum(corner_volumes)/(ncells[0]*ncells[1]*ncells[2])
    
    #Write the file containing volume fraction of the material
    f = open('vol_frac.dat','w+')
    f.write(str(vol_frac_c)+'\n')
    f.close()
    
    print('All files written')
    print('Surface ready for sparta run')
    
#%% Coupling loop executed after regular intervals to alter the geometry of the material depending on surface reactions    
else:
    
    if os.path.exists('vox2surf.surf'):
        os.remove('vox2surf.surf')                # removing existing surface files
    
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
    co_formed = read_sparta_reaction(pathg+'ReactionFiles/surf_react_sparta_'+str(file_no)+'.out',timescale)
    
#############################################################################################    
    co_formed = co_formed[co_formed[:, 0].argsort()]
   
    #Calculate mass of Carbon associated with each voxel
    volfrac_c = float(vol_frac_c)
    vol_C = volfrac_c*(lims[1,0]-lims[0,0])*(lims[1,1]-lims[0,1])*(lims[1,2]-lims[0,2])
    mass_c = vol_C*1800
    mass_c_vox = mass_c/len(voxs_alt)
    
    #Triangles check between sparta and isthmus
    if len(co_formed) != len(tri_voxs):
        raise Exception("No of triangles in sparta is not equal to isthmus, debug!!!")
    
    #Calculate the mass of carbon removed from each voxel
    c_removed_vox = np.zeros((len(voxs_alt)))
    for i in range(len(tri_voxs)):
        vox_no = np.array((tri_voxs[(i+1)]),dtype = int)
        sfracs = np.array((tri_sfracs[(i+1)]),dtype = float)
        for k in range(len(vox_no)):
            c_removed_vox[vox_no[k]] = c_removed_vox[vox_no[k]] + sfracs[k] * co_formed[i,1]
    
    
    #Remove voxels
    voxs_alt = np.column_stack((voxs_alt,c_removed_vox))
    for i in range(len(c_removed_vox)):
        if c_removed_vox[i] > mass_c_vox:
            voxs_alt[i,:] = 0
    voxs_alt = voxs_alt[~np.all(voxs_alt == 0, axis=1)]

    voxs_isthmus = voxs_alt[:,0:3]           ###assign for marching cubes   
    
    
    #write csv files for parview visualization
    write_csv(voxs_plot,c_removed_vox,co_formed,file_no,mass_c_vox)
    write_csv_all(voxs_plot,c_removed_vox,co_formed,file_no,mass_c_vox)
    
    #Print stats for debugging sparta simulation
    print('\nPrint stats for debugging sparta simulation')
    print('----------------------------------------------\n')
    print('Mass of carbon associated with each voxel --', mass_c_vox)
    print('Maximum of co_formed on each surface --', max(co_formed[:,1]))
    print('Number of voxels deleted this iteration --', len(voxs_plot)-len(voxs_alt))
    print('Mass of carbon loss obtained from DSMC --', np.sum(co_formed[:,1]))
    print('Mass of carbon loss assigned to all voxels --', np.sum(c_removed_vox[:]))
    print('\n')
    
    #test qudrant
    quad_lims = pickle.load(open("quad_lims.pkl", "rb+" ))
    
    for i in range(len(quad_lims)):
        if i % 2 == 0:
            q_lims = np.zeros((2,3))
            q_lims[0,:] = quad_lims[i,:]
            q_lims[1,:] = quad_lims[i+1,:]
            # q_lims[0,0] = lims[0,0]  #xlow
            # q_lims[0,1] = lims[0,1]  #ylow
            # q_lims[0,2] = lims[0,2]  #zlow
            # q_lims[1,0] = 5*10**-6          #xhi
            # q_lims[1,1] = 5*10**-6        #yhi
            # q_lims[1,2] = 5*10**-6          #zhi
            
            
            for j in range(3):
                if q_lims[0,j] < lims[0,j]:
                    raise Exception("Defined lower limit of the quadrant is not bounded by the main geometry, debug!!!")
            for j in range(3):
                if q_lims[1,j] > lims[1,j]:
                    raise Exception("Defined higher limit of the quadrant is not bounded by  the main geometry, debug!!!")        
            
            quadrant_check(file_no,q_lims,int(i/2))
    
    # create triangle mesh, assign voxels to triangles and save mesh
    mc_system = MC_System(lims, ncells, v_size, voxs_isthmus, name, file_no)
    corner_volumes = mc_system.corner_volumes
    faces = mc_system.faces
    vertices = mc_system.verts
    combined_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    stlFileName = 'Grids/grid_'+str(file_no)+'.stl'
    combined_mesh.export(stlFileName, file_type='stl_ascii') 
    
    # check surface and mesh quality
    points, tris, tri_voxs, tri_sfracs = read_output(file_no)
            
    tmesh = trimesh.Trimesh(vertices=np.array(points), faces=np.array(tris) - 1)
    validate_geometry(tmesh)

    surf_vox_ps = []
    for sv in mc_system.surface_voxels:
        surf_vox_ps.append(sv.position)
    surface_quality_check(voxs, surf_vox_ps, tmesh, file_no, analytical_vol)

    mesh_quality_check(tris, points)
    
    #write co-rdinate voxel data
    f = open('voxel_data/voxel_data_'+str(file_no)+'.dat','w+')
    for i in range(len(voxs_alt)):
        f.write(str(voxs_alt[i,0])+','+str(voxs_alt[i,1])+','+str(voxs_alt[i,2])+','+str(voxs_alt[i,3])+'\n')
    f.close()
    
    #Write the file containing volume fraction of the material
    vol_frac_c = np.sum(corner_volumes)/(ncells[0]*ncells[1]*ncells[2])
    f = open('vol_frac.dat','w+')
    f.write(str(vol_frac_c)+'\n')
    f.close()
    
    print('All files written')
    print('Surface ready for next iteration of sparta run')
    
