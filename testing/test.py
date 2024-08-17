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
