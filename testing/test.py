# -*- coding: utf-8 -*-
"""
Testing suite for isthmus
Vijay and Ethan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/project/sjpo228_uksr/EthanHuff/isthmus/src')
from isthmus_prototype import MC_System
import trimesh
import subprocess
import os
import copy
import platform
import random
import pickle
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
    
    
    f = open('./voxel_tri/triangle_voxels_0.dat')
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

def test_isthmus(lims, ncells, size, shapes, iterations, v_size, name, rng):
    print('Running isthmus')
    print('------------------')
    mc_system = [] # placeholder for marching cubes object

    
    for u in range(len(shapes)):    
        # generate voxels to be used by marching cubes, with surface area and volume of ideal geometry
        voxs, analytical_sa, analytical_vol = make_shape(v_size, lims, size, shapes[u])
        
        for it in range(iterations):
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
            
            print()

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
    f=open('./Grids/grid_'+str(file_no-1)+'.stl','r')
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

    f = open('./csv/test_'+str(file_no)+'.csv','w+')
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

def safe_mkdir(path, name):
    full_path = path + name + '/'
    try:
        if not os.path.exists(full_path):
            os.mkdir(full_path)
    except OSError as err:
            print(err)

#%% System initialization
# ntris not right, nans in abl result from 0 area for these triangles, what's up?
# give each cell access to any voxels whose corner(s) or center are inside
if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    pathg= os.getcwd() + '/'

    safe_mkdir(pathg, 'Grids')
    safe_mkdir(pathg, 'voxel_data')
    safe_mkdir(pathg, 'voxel_tri')
    safe_mkdir(pathg, 'csv')
    safe_mkdir(pathg, 'ReactionFiles')
    safe_mkdir(pathg, 'Flowfiles')

    ############## IMPORTANT INPUT #############
    ncells = np.array([30, 30, 30])
    iterations = 5
    v_size = 2*10**-6
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

    voxs = []
    analytical_vol = 0

    for i in range(iterations):
        file_no = i
        print(i)

        if file_no == 0:
            print('Running isthmus')
            print('------------------')
            mc_system = [] # placeholder for marching cubes object
            voxs, analytical_sa, analytical_vol = make_shape(v_size, lims, length, shapes[0])
            
            # create triangle mesh and assign voxels to triangles; read in mesh data
            mc_system = MC_System(lims, ncells, v_size, voxs, name, file_no)
            corner_volumes = mc_system.corner_volumes
            faces = mc_system.faces
            vertices = mc_system.verts
            combined_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            stlFileName = './Grids/grid_{:d}.stl'.format(file_no)
            combined_mesh.export(stlFileName, file_type='stl_ascii') 
            
            #write co-rdinate voxel data
            np.savetxt('./voxel_data/voxel_data_{:d}.dat'.format(file_no), voxs, delimiter=',')
            
            vol_frac_c = np.sum(corner_volumes)/(ncells[0]*ncells[1]*ncells[2])
            
            #Write the file containing volume fraction of the material
            f = open('./vol_frac.dat','w+')
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
            tri_voxs,tri_sfracs = read_voxel_tri('./voxel_tri/triangle_voxels_'+str(file_no-1)+'.dat')
            
            #Construct dsmc reaction data
            prev_grid = trimesh.load('./Grids/grid_'+str(file_no-1)+'.stl')
            vertices = prev_grid.vertices
            faces = prev_grid.faces
            #area = trimesh.load('Grids/grid_'+str(file_no-1)+'.stl').area
            
            normals = find_normals(file_no)
            areas = get_tri_area(faces,vertices)
            
            co_formed, areas_deleted = co_formation(vertices,normals,areas,mass_flux,timescale,file_no,v_size)
            print('Sum of reacted C mass for all triangles: {:.2e}'.format(sum(co_formed[:,1])))
            #Read voxel data
            with open('./voxel_data/voxel_data_'+str(file_no-1)+'.dat') as f:
                lines = (line for line in f if not line.startswith('#'))
                voxs_alt = np.loadtxt(lines, delimiter=',', skiprows=0)
                voxs_plot = copy.deepcopy(voxs_alt)
            
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

            #write csv files for paraview visualization
            
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
            combined_mesh = trimesh.Trimesh(vertices=mc_system.verts, faces=mc_system.faces)
            stlFileName = './Grids/grid_{:d}.stl'.format(file_no)
            combined_mesh.export(stlFileName, file_type='stl_ascii') 
            
            #write co-rdinate voxel data
            f = open('./voxel_data/voxel_data_'+str(file_no)+'.dat','w+')
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
