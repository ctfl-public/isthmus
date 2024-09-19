# -*- coding: utf-8 -*-
"""
Testing suite for isthmus
Vijay and Ethan
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
import os
import platform
from shape_types import Ellipsoid, Cylinder, Cube, make_shape
    

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

# read in output
def read_output():
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
    
    
    f = open('triangle_voxels_0.dat')
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
   
    
    if os.path.exists('forces.dat.0'):
        os.remove('forces.dat.0')
    if os.path.exists('forces.dat.10'):
        os.remove('forces.dat.10')
    
    print('Testing mesh in SPARTA...')
    cmdargs = []
    if (platform.system() == 'Windows'):
        cmdargs = ["powershell", "bash", "./run.sh"]
    elif (platform.system() == 'Linux'):
        cmdargs = ["./run.sh"]
    else:
        raise Exception('ERROR: Mac not supported for testing')
    sp_out = subprocess.run(cmdargs, capture_output=True)
    sp_out = sp_out.stdout.decode('utf-8')
    sp_out = sp_out.split(sep='\n')
    print('\n==============================================')
    for st in sp_out:
        print(st)
    print('\n==============================================\n')
    if not os.path.exists('forces.dat.0'):
        raise Exception('ERROR: SPARTA run failed')
    
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
    
def voxel_removal(voxs, surface_voxels):
    # remove voxels to test robustness of algorithm
    print('.......................................')
    print('..........REMOVING VOXELS..............')
    print('.......................................')
    voxel_types = np.zeros(len(voxs)).astype(int)
    for sv in surface_voxels:
        voxel_types[sv.id] = 1
        
    tvoxs = np.transpose(voxs)
    pd_voxel = pd.DataFrame(np.transpose(np.array([tvoxs[0], tvoxs[1], tvoxs[2], voxel_types])), columns=['x','y','z','type'])
    remove_pool = pd_voxel[pd_voxel['type'] != 0]
    remove_pool = remove_pool.reset_index(drop=True)
    nremove = int(len(remove_pool)/3) # remove 1/3 of the surface voxels
    print(pd_voxel)
    removed = rng.integers(0, len(remove_pool), size=nremove)
    remove_pool = remove_pool.drop(index=removed)
    zero_pool = pd_voxel[pd_voxel['type'] == 0]
    new_voxels = pd.concat([zero_pool,remove_pool])
    voxs = new_voxels[['x','y','z']].to_numpy()
    print(pd_voxel)
    
    return voxs

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

pathg='/Users/vijaybmohan/Desktop/Cam/isthmus_latest/isthmus/testing/'      #path of the simulation folder

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
        sfracs = np.array((tri_sfracs[(i+1)]),dtype = float)
        for k in range(len(vox_no)):
            if c_removed_vox[vox_no[k]] == 0:
                c_removed_vox[vox_no[k]] = sfracs[k] * co_formed[i,1]
            else:
                c_removed_vox[vox_no[k]] = c_removed_vox[vox_no[k]] + sfracs[k] * co_formed[i,1]
    
    #Remove voxels
    hanging_voxs = np.zeros((len(voxs_plot)),)
    for i in range(len(c_removed_vox)):
        hanging_voxs[i,] = voxs_plot[i,0]**2+voxs_plot[i,1]**2+voxs_plot[i,2]**2
        if c_removed_vox[i] > mass_c_vox:
            voxs_alt[i,:] = 0
    voxs_alt = voxs_alt[~np.all(voxs_alt == 0, axis=1)]   
    
    # max_dist = max(hanging_voxs[:,])
    # ind_max = np.where(hanging_voxs == max_dist)
    
    # ind_max = np.array(ind_max,dtype = 'int')
    # for k in range(len(ind_max[0])):
    #     if c_removed_vox[ind_max[0][k]] == 0:
    #         print('hanging voxel index --', ind_max[0][k])
    #         print('x y z',voxs_plot[ind_max[0][k],0],voxs_plot[ind_max[0][k],1],voxs_plot[ind_max[0][k],2])
    
    
    #write csv files for parview visualization
    write_csv(voxs_plot,c_removed_vox,co_formed,file_no,mass_c_vox)
    write_csv_all(voxs_plot,c_removed_vox,co_formed,file_no,mass_c_vox)
    
    print(np.sum(co_formed[:,1]))
    print(np.sum(c_removed_vox[:]))
    
    #test qudrant
    quad_lims = pickle.load( open( "quad_lims.pkl", "rb+" ) )
    #define boundaries of the quadrant
    
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
        f.write(str(voxs_alt[i,0])+','+str(voxs_alt[i,1])+','+str(voxs_alt[i,2])+'\n')
    f.close()
    
    #Write the file containing volume fraction of the material
    vol_frac_c = np.sum(corner_volumes)/(ncells[0]*ncells[1]*ncells[2])
    f = open('vol_frac.dat','w+')
    f.write(str(vol_frac_c)+'\n')
    f.close()
    
    print('All files written')
    print('Surface ready for next iteration of sparta run')
    


def test_isthmus(lims, ncells, size, shapes, iterations, v_size, name, rng):
    print('Running isthmus')
    print('------------------')
    mc_system = [] # placeholder for marching cubes object

    
    for u in range(len(shapes)):    
        # generate voxels to be used by marching cubes
        voxs, analytical_sa, analytical_vol = make_shape(v_size, lims, size, shapes[u])
        
        for it in range(iterations):
            if (it != 0):
                voxs = voxel_removal(voxs, mc_system.surface_voxels)
            
            # create triangle mesh and assign voxels to triangles; read in mesh data
            mc_system = MC_System(lims, ncells, v_size, voxs, name, 0)

            points, tris, tri_voxs, tri_sfracs = read_output()
            
            tmesh = trimesh.Trimesh(vertices=np.array(points), faces=np.array(tris) - 1)
            validate_geometry(tmesh)

            surf_vox_ps = []
            for sv in mc_system.surface_voxels:
                surf_vox_ps.append(sv.position)
            surface_quality_check(voxs, surf_vox_ps, tmesh, it, analytical_vol)

            mesh_quality_check(tris, points)
            
            # first check that triangles haven't been changed by sparta
            # tris is from vox2surf file written by isthmus
            for i in range(3):
                si = str(i+1)
                tris['v' + si + 'x'] = points['x'].loc[tris['p' + si]].values
                tris['v' + si + 'y'] = points['y'].loc[tris['p' + si]].values
                tris['v' + si + 'z'] = points['z'].loc[tris['p' + si]].values
            tris['centx'] = (tris['v1x'] + tris['v2x'] + tris['v3x'])/3
            tris['centy'] = (tris['v1y'] + tris['v2y'] + tris['v3y'])/3
            tris['centz'] = (tris['v1z'] + tris['v2z'] + tris['v3z'])/3
            
            voxel_triangle_assignment_check(tris, tri_voxs, tri_sfracs, mc_system)

            
            tvoxs = np.transpose(voxs)
            pd_voxel = pd.DataFrame(np.transpose(np.array([tvoxs[0], tvoxs[1], tvoxs[2]])), columns=['x','y','z'])
            pd_voxel['r'] = np.sqrt(pd_voxel['x']**2 + pd_voxel['y']**2 + pd_voxel['z']**2)
            


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
            
            epsilon_sq = (max(abs(tris['centx']).max(), abs(tris['centy']).max(), abs(tris['centz']).max())*1e-4)**2
            pc = 0
            print('Are triangles of data equivalent to isthmus triangles? ', end='')
            sq_norm1 = (stris['v1x'] - tris['v1x'])**2 + (stris['v1y'] - tris['v1y'])**2 + (stris['v1z'] - tris['v1z'])**2
            sq_norm2 = (stris['v2x'] - tris['v2x'])**2 + (stris['v2y'] - tris['v2y'])**2 + (stris['v2z'] - tris['v2z'])**2
            sq_norm3 = (stris['v3x'] - tris['v3x'])**2 + (stris['v3y'] - tris['v3y'])**2 + (stris['v3z'] - tris['v3z'])**2
        
            for i in range(len(sq_norm1)):
                if (sq_norm1.iloc[i] < epsilon_sq and sq_norm2.iloc[i] < epsilon_sq and sq_norm3.iloc[i] < epsilon_sq):
                    pc += 1
            pc *= 100/len(stris)
        
            print('{:.2f}% yes'.format(pc))
    
            
            # now dummy algorithm for applying ablation/force data to voxels
            # cosine designed to have some difference over the surface
            
            # apply to faces
            tris['r'] = 0.0
            tris['angle'] = 0.0
            for t in tris.index:
                cent = np.array(tris.loc[t].loc[['centx', 'centy', 'centz']])
                r = np.linalg.norm(cent)
                tris.loc[t, 'r'] = r
                tris.loc[t, 'angle'] = np.arccos(np.dot(cent/r, np.array([1,0,0])))
    
            scalar_func = lambda x : 5*np.cos(x) + 3
            tris['scalar'] = (scalar_func(tris['angle']))*tris['area']
            scalar_total_t = tris.loc[:, 'scalar'].sum()
            
            # translate to voxels
            pd_voxel = pd.DataFrame(data=voxs, columns=['x','y','z'])
            pd_voxel['scalar'] = 0.0
            pd_voxel['r'] = np.sqrt(pd_voxel['x']**2 + pd_voxel['y']**2 + pd_voxel['z']**2)
            pd_voxel['angle'] = np.arccos(pd_voxel['x']/pd_voxel['r']) # simplified dot product of unit position vector with x axis, [x,y,z]/r . [1,0,0]
            for ind in tris.index:
                ct_scalar = tris.loc[ind]['scalar']
                ct_vids = tri_voxs[ind]     # voxel ids for triangle
                ct_vfracs = tri_sfracs[ind] # and corresponding scalar fractions
                for v in range(len(ct_vids)):
                    pd_voxel.loc[ct_vids[v], 'scalar'] += ct_scalar*ct_vfracs[v]
            
            scalar_total_v = pd_voxel.loc[:, 'scalar'].sum()
            print('Total scalar values are {:.2f}% conserved from triangle ({:.2e}) to voxel ({:.2e})'.format(\
                   100*scalar_total_v/scalar_total_t, scalar_total_t, scalar_total_v))
        
            xa = np.linspace(0, np.pi, 1000)
            ya = scalar_func(xa)
            plt.figure()
            plt.plot(xa*180/np.pi, ya, color='green', label='Analytical')
    
            tris = tris.sort_values(by=['angle'])
            pd_voxel = pd_voxel.sort_values(by=['angle'])
            divisions = [8, 16, 32]
            cl = ['blue', 'red', 'purple', 'orange']
            
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
                    section_area = -2*np.pi*(size**2)*(np.cos(b) - np.cos(a))
                    
                    c_tris = tris[tris['angle'] < b]
                    c_tris = c_tris[c_tris['angle'] >= a]
                    c_voxs = pd_voxel[pd_voxel['angle'] < b]
                    c_voxs = c_voxs[c_voxs['angle'] >= a]
    
                    t_pressures.append(c_tris.loc[:,'scalar'].sum()/section_area)
                    v_pressures.append(c_voxs.loc[:,'scalar'].sum()/section_area)
                angles = np.array(angles)
                plt.scatter(angles*180/np.pi, v_pressures, color=cl[j], marker='s', label='Voxels ' + str(d))
                #plt.scatter(angles*180/np.pi, t_pressures, color=cl[j], marker='^', label='Triangles ' + str(d))
                j += 1
            plt.ylabel('Scalar Pressure Value')
            plt.xlabel('Angle from Stagnation Point (degrees)')
            plt.xlim(0, 180)
            plt.ylim(min(ya) - 0.2*abs(min(ya)), max(ya) + 0.2*abs(max(ya)))
            plt.grid()
            plt.legend()
            plt.show()
                            
        
            xa = np.linspace(0, np.pi,1000)
            ya = xa*0 + size
            plt.figure()
            plt.plot(xa*180/np.pi, ya, color='green', label='Analytical')
            
            j = 0
            for d in divisions:
                borders = np.linspace(0, np.pi, d+1)
                angles = []
                t_wrad = [] # avg radius weighted by scalars
                v_wrad = []
                for i in range(1, len(borders)):
                    b = borders[i]
                    a = borders[i - 1]
                    angles.append((a + b)/2)
                    section_area = -2*np.pi*(size**2)*(np.cos(b) - np.cos(a))
                    
                    c_tris = tris[tris['angle'] < b]
                    c_tris = c_tris[c_tris['angle'] >= a]
                    c_voxs = pd_voxel[pd_voxel['angle'] < b]
                    c_voxs = c_voxs[c_voxs['angle'] >= a]
                    
                    c_tris['wsum'] = c_tris['scalar']*c_tris['r']
                    c_voxs['wsum'] = c_voxs['scalar']*c_voxs['r']
    
                    t_wrad.append(c_tris.loc[:,'wsum'].sum()/c_tris.loc[:,'scalar'].sum())
                    v_wrad.append(c_voxs.loc[:,'wsum'].sum()/c_voxs.loc[:,'scalar'].sum())
                angles = np.array(angles)
                plt.scatter(angles*180/np.pi, v_wrad, color=cl[j], marker='s', label='Voxels ' + str(d))
                plt.scatter(angles*180/np.pi, t_wrad, color=cl[j], marker='^', label='Triangles ' + str(d))
                j += 1
            plt.ylabel('Average Radius Weighted by Scalar')
            plt.xlabel('Angle from Stagnation Point')
            plt.ylim(0, size*1.2)
            plt.xlim(0, 180)
            plt.grid()
            plt.legend()
            plt.show()
            
            
            # plot results for sanity check
            plot_results(mc_system.verts, mc_system.faces, lo, hi)
                 
            print()
            

############## IMPORTANT INPUT #############
ncells = np.array([60,60,60])
iterations = 3
v_size = 0.04
shapes = ['ellipsoid']

###########################################

domain = 2
lo = [-domain]*3
hi = [domain]*3
lims = np.array([lo, hi])
name = 'vox2surf.surf'
radius = 1
rng = np.random.default_rng(999)

tris = test_isthmus(lims, ncells, radius, shapes, iterations, v_size, name, rng)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('/project/sjpo228_uksr/EthanHuff/isthmus/src/')
from isthmus_prototype import MC_System
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh
import subprocess
import os
import platform
from shape_types import Ellipsoid, Cylinder, Cube, make_shape
import trimesh


def get_tri_area(faces,vertices):
    areas = []
    for i in range(len(faces)):
        
        # herons formula = sqrt(s(s - a)(s - b)(s - c)) for triangle lengths a,b,c, s= half-perimeter
        a = np.linalg.norm(vertices[faces[i,2]] - vertices[faces[i,1]])
        b = np.linalg.norm(vertices[faces[i,1]] - vertices[faces[i,0]])
        c = np.linalg.norm(vertices[faces[i,0]] - vertices[faces[i,2]])
    
        s = (a + b + c)/2
        area = np.sqrt(s*(s - a)*(s - b)*(s - c))
        areas.append([area])

    return np.array(areas,dtype='float')

def find_normals(file_no):
    normals = []
    f=open('../Grids/grid_'+str(file_no-1)+'.stl','r')
    for num, line in enumerate(f, 1):
        if 'facet normal' in line:
            s=tuple(line.split())
            normals.append([float(s[2]),float(s[3]),float(s[4])])
    f.close()
     
    return np.array(normals,dtype='float')

def co_formation(vertices,normals,areas,mass_flux,timescale,file_no,v_size):
    co_formed  = np.zeros((len(faces),2))
    min_x = min(vertices[:,1])
    areas_deleted = []
    f = open('tri_data.csv', 'w')
    f.write('x,y,z,abl\n')
    for i in range(len(faces)):
        cverts = vertices[faces[i]]
        centroid = (cverts[0] + cverts[1] + cverts[2])/3
        co_formed[i,0] = int(i)
        if abs(vertices[faces[i,0],0] - min_x) < file_no*v_size or \
           abs(vertices[faces[i,1],0] - min_x) < file_no*v_size or \
           abs(vertices[faces[i,2],0] - min_x) < file_no*v_size or \
           all(normals[i,0] - [-1,0,0] < 1e-6):
            co_formed[i,1] = timescale*mass_flux*areas[i]
            areas_deleted.append([areas[i]])
        f.write('{:.2e},{:.2e},{:.2e},{:.2e}\n'.format(centroid[0], centroid[1], centroid[2], co_formed[i,1]))
    f.close()
    
    return co_formed, np.array(areas_deleted,dtype='float')

def write_csv_all(voxs_plot,c_removed_vox,co_formed,file_no,mass_c_vox,ntris):

    co_in_cell_plot = np.zeros((len(voxs_plot),4))

    for i in range(len(c_removed_vox)):
        if c_removed_vox[i] > 0.5*mass_c_vox: # 0.5 or 1 as factor
            co_in_cell_plot[i,3] = 1
        else:
            co_in_cell_plot[i,3] = 0

    f = open('../csv/test_'+str(file_no)+'.csv','w+')
    for i in range(len(voxs_plot)):
        if i == 0:
            f.write('X,Y,Z,deleted,abl,ntris\n')
        f.write(str(voxs_plot[i,0])+','+str(voxs_plot[i,1])+','+str(voxs_plot[i,2])+','+str(co_in_cell_plot[i,3])+','+str(c_removed_vox[i])+','+str(ntris[i])+'\n')
    f.close()

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


############## IMPORTANT INPUT #############
ncells = np.array([50,50,50])
iterations = 1
v_size = 1*10**-6
shapes = ['cube']

###########################################

domain = 2
lo = [-30*10**-6]*3
hi = [30*10**-6]*3
lims = np.array([lo, hi])
name = 'vox2surf.surf'
length = 25*10**-6
rng = np.random.default_rng(999)
mass_flux = 10 #kgm-2s-1
timescale = 0.000181

for i in range(2):
    file_no = i
    print(i)

    if file_no == 0:
        print('Running isthmus')
        print('------------------')
        mc_system = [] # placeholder for marching cubes object
        voxs, analytical_sa, analytical_vol = make_shape(v_size, lims, length, shapes[0])
        
        # create triangle mesh and assign voxels to triangles; read in mesh data
        print('Executing marching cubes...')
        mc_system = MC_System(lims, ncells, v_size, voxs, name, file_no)
        corner_volumes = mc_system.corner_volumes
        faces = mc_system.faces
        vertices = mc_system.verts
        combined_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        stlFileName = '../Grids/grid_'+str(file_no)+'.stl'
        combined_mesh.export(stlFileName, file_type='stl_ascii') 
        
        #write co-rdinate voxel data
        f = open('../voxel_data/voxel_data_'+str(file_no)+'.dat','w+')
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
        
    else:
        if os.path.exists('vox2surf.surf'):
            os.remove('vox2surf.surf')                # removing existing surface files
        
        ############## Read volume fraction, voxel and surface reaction data ###################
        
        with open('vol_frac.dat') as f:
            vol_frac_c = f.readline().strip('\n')
        
        #Read voxel_triangles
        tri_voxs,tri_sfracs = read_voxel_tri('../voxel_tri/triangle_voxels_'+str(file_no-1)+'.dat')
        
        #Construct dsmc reaction data
        vertices = trimesh.load('../Grids/grid_'+str(file_no-1)+'.stl').vertices
        faces = trimesh.load('../Grids/grid_'+str(file_no-1)+'.stl').faces
        #area = trimesh.load('Grids/grid_'+str(file_no-1)+'.stl').area
        
        normals = find_normals(file_no)
        areas = get_tri_area(faces,vertices)
        
        
        co_formed, areas_deleted = co_formation(vertices,normals,areas,mass_flux,timescale,file_no,v_size)
        print('Sum of reacted C mass for all triangles: {:.2e}'.format(sum(co_formed[:,1])))
        #Read voxel data
        with open('../voxel_data/voxel_data_'+str(file_no-1)+'.dat') as f:
            lines = (line for line in f if not line.startswith('#'))
            voxs_alt = np.loadtxt(lines, delimiter=',', skiprows=0)
           
        #Read voxel data for preparing csv file
        with open('../voxel_data/voxel_data_'+str(file_no-1)+'.dat') as f:
            lines = (line for line in f if not line.startswith('#'))
            voxs_plot = np.loadtxt(lines, delimiter=',', skiprows=0)
        
        #Calculate mass of Carbon associated with each voxel
        volfrac_c = float(vol_frac_c)
        vol_C = volfrac_c*(lims[1,0]-lims[0,0])**3
        mass_c = vol_C*1800
        mass_c_vox = mass_c/len(voxs_alt)
        
        
        #Calculate the mass of carbon removed from each voxel
        c_removed_vox = np.zeros((len(voxs_alt)))
        ntris = np.zeros((len(voxs_alt))).astype(int)
        for i in range(len(tri_voxs)):
            vox_no = np.array((tri_voxs[(i+1)]),dtype = int)
            sfracs = np.array((tri_sfracs[(i+1)]),dtype = float)
            for k in range(len(vox_no)):
                ntris[vox_no[k]] += 1
                c_removed_vox[vox_no[k]] += sfracs[k] * co_formed[i,1]

        
        # #Mass of C removed
        # mass_rate = timescale*mass_flux*(lims[1,0]-lims[0,0])**2
        # mass_rate_c = mass_rate*12/28
        # no_of_voxs_removed = int(mass_rate_c/mass_c_vox)
        
        # #Remove voxels
        # voxs_alt = voxs_alt[no_of_voxs_removed:len(voxs_alt),:]
        #Remove voxels
        vox_mass_sum = 0
        for i in range(len(c_removed_vox)):
            vox_mass_sum += c_removed_vox[i]
            if c_removed_vox[i] > 0.5*mass_c_vox: # 0.5 or 1
                voxs_alt[i,:] = 0
                #print(c_removed_vox[i])
        voxs_alt = voxs_alt[~np.all(voxs_alt == 0, axis=1)]  
        
        print('Sum of reacted C mass for all voxels: {:.2e}'.format(vox_mass_sum))

        #write csv files for parview visualization
        
        write_csv_all(voxs_plot,c_removed_vox,co_formed,file_no,mass_c_vox,ntris)
        
        ##printing
        
        
        #Print stats for debugging sparta simulation
        print('\nPrint stats for debugging sparta simulation')
        print('----------------------------------------------\n')
        print('Critical mass of carbon associated with each voxel --', mass_c_vox)
        print('Maximum of carbon removed associated with each voxel --', max(c_removed_vox[:]))
        print('Number of voxels deleted this iteration --', len(voxs_plot)-len(voxs_alt))
        print('Total mass flux of carbon removed --', np.sum(c_removed_vox)/(np.sum(areas_deleted)*timescale))
        print('\n')
        
        # create triangle mesh, assign voxels to triangles and save mesh
        mc_system = MC_System(lims, ncells, v_size, voxs_alt, name, file_no)
        corner_volumes = mc_system.corner_volumes
        faces = mc_system.faces
        vertices = mc_system.verts
        combined_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        stlFileName = '../Grids/grid_'+str(file_no)+'.stl'
        combined_mesh.export(stlFileName, file_type='stl_ascii') 
        
        #write co-rdinate voxel data
        f = open('../voxel_data/voxel_data_'+str(file_no)+'.dat','w+')
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
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
Created on Sun Feb 18 20:59:38 2024

@author: lch285

import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
import trimesh
from Marching_Cubes import marching_cubes, mesh_surface_area
import multiprocessing as mp
import time
import imageio

def plotVoxels(label_image):
    # Create a figure
    fig = plt.figure()
    
    # Plot the original visualization
    ax1 = fig.add_subplot(111, projection='3d')
    # Get indices of non-zero values in label_image
    nonzero_indices = np.nonzero(label_image)
    
    # Scatter plot using non-zero indices and color each point based on its label
    ax1.scatter(nonzero_indices[0], nonzero_indices[1], nonzero_indices[2], c=label_image[nonzero_indices], cmap='viridis', edgecolor='black')

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Fiber Detection and centerline')
    plt.show()

def generate_stl(voxelMTX, filename='output.stl', translation = np.array([0,0,0])):
    # Use marching cubes to obtain the mesh
    verts, faces, normals, values, Dict = marching_cubes(voxelMTX)
                                                        
    # Apply translation
    verts += translation
    
    # Create a mesh object
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    
    # Ensure the normals are pointing outward
    mesh.faces = np.flip(mesh.faces, axis=1) 
    
    return mesh

def combine_meshes(meshes):
    # Initialize lists to store vertices and faces
    vertices_combined = []
    faces_combined = []
    vertex_offset = 0
    
    # Concatenate vertices and faces of each mesh
    for mesh in meshes:
        vertices_combined.append(mesh.vertices)
        faces_combined.append(mesh.faces + vertex_offset)
        vertex_offset += len(mesh.vertices)
    
    # Concatenate all vertices and faces arrays
    vertices_combined = np.concatenate(vertices_combined)
    faces_combined = np.concatenate(faces_combined)
    
    # Adjust vertices given voxel size
    vertices_combined *= voxelSize
    
    # Create a new mesh
    combined_mesh = trimesh.Trimesh(vertices=vertices_combined, faces=faces_combined)
    
    return combined_mesh

def generate_stl_parallel(args):
    data, translation, filename = args
    mesh = generate_stl(data, filename, translation)
    return mesh

def createPadding(image_volume):
    # Set the desired padding size (in pixels) for each dimension
    padding_size = 1  # Adjust this value based on your needs
    
    # Create a padded volume with the same data type as the original image
    padded_volume = np.zeros(
        (image_volume.shape[0] + 2 * padding_size,
         image_volume.shape[1] + 2 * padding_size,
         image_volume.shape[2] + 2 * padding_size),
        dtype=image_volume.dtype,
    )
    
    # Calculate padding ranges for each dimension
    x_range = slice(padding_size, padding_size + image_volume.shape[0])
    y_range = slice(padding_size, padding_size + image_volume.shape[1])
    z_range = slice(padding_size, padding_size + image_volume.shape[2])
    
    # Copy the original image into the center of the padded volume
    padded_volume[x_range, y_range, z_range] = image_volume
    
    
    #Load padded volume
    binary_volume = np.squeeze(np.array(padded_volume))
    
    return binary_volume

def createCircle(r=50,domainSize=150):

    voxelMTX = np.zeros((domainSize,domainSize,domainSize))
    
    # Create grid coordinates
    x, y, z = np.ogrid[:domainSize, :domainSize, :domainSize]
    
    # Calculate distances to the center for all points
    distances = np.sqrt((x )**2 + (y )**2 + (z )**2)
    
    # Set voxel values based on distance from the center
    voxelMTX[distances <= r] = 1

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
    
if __name__ == '__main__':
    fileName = '/Users/vijaybmohan/Desktop/Cam/isthmus/50.tif'
    voxelSize = 1
    
    # Load in Tif
    voxelMTX = loadData(fileName)
    
    # Add padding to the volume
    voxelMTX = createPadding(voxelMTX)
    
    # Get dimesnion with padding
    xdim, ydim, zdim = voxelMTX.shape
    
    print('Starting to create surface in Parallel!')
    # Number of voxels per split
    divisionVoxels = 30

    
    splitListX = np.arange(0,xdim+1,divisionVoxels)
    splitListY = np.arange(0,ydim+1,divisionVoxels)
    splitListZ = np.arange(0,zdim+1,divisionVoxels)
    
    if splitListX[-1] != xdim:
        splitListX = np.append(splitListX, xdim)
    if splitListY[-1] != ydim:
        splitListY = np.append(splitListY, ydim)
    if splitListZ[-1] != zdim:
        splitListZ = np.append(splitListZ, zdim)
    
    splitMTX = []
    
    for i in range(len(splitListX)-1):
        for j in range(len(splitListY)-1):
            for k in range(len(splitListY)-1):
                tempMTX = voxelMTX[splitListX[i]:splitListX[i+1]+1,splitListY[j]:splitListY[j+1]+1,splitListZ[k]:splitListZ[k+1]+1]
                tempXdim, tempYdim, tempZim = tempMTX.shape
                if np.sum(tempMTX) != 0 and np.sum(tempMTX) != tempXdim*tempYdim*tempZim:
                    # plotVoxels(tempMTX)
                    splitMTX.append([tempMTX,[splitListX[i],splitListY[j],splitListZ[k]]])
    
    startTime = time.time()
    with mp.Pool(processes=mp.cpu_count()) as pool: # Adjust the number of processes as needed
        args = [(data, translation, str(num) + '.stl') for num, (data, translation) in enumerate(splitMTX)]
        meshesList = pool.map(generate_stl_parallel, args)
    
    combinedMesh = combine_meshes(meshesList)
    endTime = time.time()
    print('It took %i sec to genereate the surface in parallel with %i divisions!'%((endTime-startTime),divisionVoxels))
    
    stlFileName = "originalJoined.stl"
    
    print('Saving stl as',stlFileName)
    
    # Save the joined mesh
    combinedMesh.export(stlFileName, file_type='stl_ascii')
    
    print(combinedMesh.is_watertight)
    print(combinedMesh.is_volume)
    


import pandas as pd
import numpy as np
import copy

# read in output
def read_output(name):
    # read in created data
    f = open(name)
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
        tris['v1x'].iloc[i] = points['x'].loc[tris['p1'].iloc[i]]
        tris['v1y'].iloc[i] = points['y'].loc[tris['p1'].iloc[i]]
        tris['v1z'].iloc[i] = points['z'].loc[tris['p1'].iloc[i]]
        tris['v2x'].iloc[i] = points['x'].loc[tris['p2'].iloc[i]]
        tris['v2y'].iloc[i] = points['y'].loc[tris['p2'].iloc[i]]
        tris['v2z'].iloc[i] = points['z'].loc[tris['p2'].iloc[i]]
        tris['v3x'].iloc[i] = points['x'].loc[tris['p3'].iloc[i]]
        tris['v3y'].iloc[i] = points['y'].loc[tris['p3'].iloc[i]]
        tris['v3z'].iloc[i] = points['z'].loc[tris['p3'].iloc[i]]
           
    return points, tris

def read_data(name):
    # read in created data
    f = open(name)
    surf_lines = f.readlines()
    f.close()
    
    ntris = int(surf_lines[3])
    tris = surf_lines[9:9+ntris]
    tris = np.array([x.split() for x in tris])
    cols = ['id', 'v1x', 'v1y', 'v1z', 'v2x', 'v2y', 'v2z', 'v3x', 'v3y', 'v3z', 'f1', 'f2']
    tris = pd.DataFrame(data=tris, columns=cols, dtype=np.double).set_index(['id'], verify_integrity=True)
    tris = tris.set_index(tris.index.astype(int))    
    
    return tris

# 1 is output from sparta, 2 is output from isthmus 

# raw input and output geometry read in, points and triangles
print('Reading data... ', end='')
p_in, t_in = read_output('vox2surf.surf')
p_out, t_out = read_output('final.surf')
t_data = read_data('forces.dat')
print('done')

epsilon_sq = (1e-6)**2
pc = 0
print('Are triangles of data equivalent to isthmus? ', end='')
sq_norm1 = (t_data['v1x'] - t_in['v1x'])**2 + (t_data['v1y'] - t_in['v1y'])**2 + (t_data['v1z'] - t_in['v1z'])**2
sq_norm2 = (t_data['v2x'] - t_in['v2x'])**2 + (t_data['v2y'] - t_in['v2y'])**2 + (t_data['v2z'] - t_in['v2z'])**2
sq_norm3 = (t_data['v3x'] - t_in['v3x'])**2 + (t_data['v3y'] - t_in['v3y'])**2 + (t_data['v3z'] - t_in['v3z'])**2

for i in range(len(sq_norm1)):
    if (sq_norm1.iloc[i] and sq_norm2.iloc[i] and sq_norm3.iloc[i]):
        pc += 1
pc *= 100/len(t_data)

print('{:.2f}% yes'.format(pc))

pc = 0
print('Are triangles of data equivalent to sparta-outputted mesh? ', end='')
sq_norm1 = (t_data['v1x'] - t_out['v1x'])**2 + (t_data['v1y'] - t_out['v1y'])**2 + (t_data['v1z'] - t_out['v1z'])**2
sq_norm2 = (t_data['v2x'] - t_out['v2x'])**2 + (t_data['v2y'] - t_out['v2y'])**2 + (t_data['v2z'] - t_out['v2z'])**2
sq_norm3 = (t_data['v3x'] - t_out['v3x'])**2 + (t_data['v3y'] - t_out['v3y'])**2 + (t_data['v3z'] - t_out['v3z'])**2

for i in range(len(sq_norm1)):
    if (sq_norm1.iloc[i] and sq_norm2.iloc[i] and sq_norm3.iloc[i]):
        pc += 1
pc *= 100/len(t_data)

print('{:.2f}% yes'.format(pc))

p_out_dup = p_out[p_out.duplicated()]
p_out_red = p_out[~p_out.duplicated()]

p_out_red = p_out_red.sort_values(by=['x','y','z'])
p_in = p_in.sort_values(by=['x','y','z'])

p_diff_array = np.array(p_out_red) - np.array(p_in)
for i in range(len(p_diff_array)):
    if (np.linalg.norm(p_diff_array[i]) > epsilon):
        raise Exception('Point index {d} is not equal'.format(d=i))
print('Points are equivalent...')

"""
"""
dupes = [[] for x in range(len(p_out_red))]
for i in range(len(p_out_dup)):
    cp = p_out_dup.iloc[i]
    print(len(p_out_dup) - i)
    for i in range(len(p_out_red)):
        if (all(abs(p_out_red.iloc[i] - cp) < epsilon)):
            dupes[i].append(cp.name)
            break
"""
"""
dupes = []
uniques = []
for i in range(len(p_out_red)):
    reduced_point = np.array(p_out_red.iloc[i])
    p_out_dup['norm'] = (p_out_dup['x'] - reduced_point[0])**2 + (p_out_dup['y'] - reduced_point[1])**2 + (p_out_dup['z'] - reduced_point[2])**2
    c_dupes = p_out_dup[p_out_dup['norm'] < epsilon]
    p_out_dup = p_out_dup[p_out_dup['norm'] >= epsilon]
    uniques.append(p_out_red.iloc[i].name)
    dupes.append([p_out_red.iloc[i].name] + list(c_dupes.index))

test = pd.DataFrame(np.array([dupes[i][j] for i in range(len(dupes)) for j in range(len(dupes[i]))]))
if test.duplicated().sum():
    raise Exception('Oopsies, duplicate point filtering didn\'t work')
for i in range(len(uniques)):
    unique = p_out.loc[uniques[i]]
    for j in range(0, len(dupes[i])):
        dupe = p_out.loc[dupes[i][j]]
        if np.linalg.norm(unique - dupe) > epsilon:
            raise Exception('Oopsies, duplicate point filtering didn\'t work')

sparta_to_isthmus = {dupes[i][j] : uniques[i] for i in range(len(uniques)) for j in range(len(dupes[i]))}

t_out_translated = copy.deepcopy(t_out)
for i in range(len(t_out)):
    for j in range(3):
        t_out_translated.iloc[i,j] = sparta_to_isthmus[t_out.iloc[i,j]]

"""
"""
p_out.insert(len(p_in.columns), 'dupes', [[] for x in p_out.index])
for i in range(len(p_out)):
    for j in range(i + 1, len(p_out)):
        if (all(p_out.iloc[i] == p_out.iloc[j])):
            p_out.iloc[i].loc['dupes'].append(j)
"""
"""

output_points = []
for i in range(len(t_out)):
    output_points.append(t_out.iloc[i][0])
    output_points.append(t_out.iloc[i][1])
    output_points.append(t_out.iloc[i][2])
    
output_points = set(output_points)

t_in_inds = t_in.index
t_out_inds = t_out.index
"""

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
    