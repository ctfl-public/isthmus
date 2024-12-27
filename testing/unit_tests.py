# %%
import numpy as np
import trimesh
import sys
import pandas as pd
import os
sys.path.insert(0, r'C:\Users\ethan\lab_codes\isthmus\src')
from isthmus_prototype import *

# read in output
def read_output():

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
    
    return tri_voxs, tri_sfracs


# %%
"""
TEST VOXEL VOLUME DIVISION BEFORE MARCHING CUBES
Test a 10x10x10 voxel cube centered in a 6x6x6 cell grid
6x6x6 cell grid makes a 7x7x7 corner grid
Cell side length: 1
Voxel side length: 0.35

Grid should have following volume fractions:
    Corners at edge of grid: 0.0
    3x3x3 central corner grid: 1.0
    Just outside of cube face: 1/4
    Just outside of cube edge: 1/16
    Just outside of cube corner: 1/64
"""
def voxel_division_test():

    # voxel cube initialization
    v_size = 0.35
    cube_lo = np.asarray([-5*v_size]*3)
    nvoxs = np.asarray([10,10,10])

    voxels = []
    for k in range(nvoxs[2]):
        z = cube_lo[2] + (0.5 + k)*v_size
        for j in range(nvoxs[1]):
            y = cube_lo[1] + (0.5 + j)*v_size
            for i in range(nvoxs[0]):
                x = cube_lo[0] + (0.5 + i)*v_size
                voxels.append([x,y,z])
    voxels = np.asarray(voxels)

    # grid initialization
    lims = np.asarray([[-3,-3,-3], [3,3,3]])
    ncells = np.asarray([7,7,7])

    # divide volumes
    corner_grid = Corner_Grid(lims, ncells, voxels, v_size)

    # test resulting grid
    inner = []
    mid_face = []
    mid_edge = []
    mid_corner = []
    outer = []
    for c in corner_grid.corners:
        if (any(abs(c.position) > 2.5)):
            outer.append(c)
        elif (any(abs(c.position) > 1.5)):
            ext = sum(abs(c.position) > 1.5)
            if (ext == 3):
                mid_corner.append(c)
            elif (ext == 2):
                mid_edge.append(c)
            else:
                mid_face.append(c)
        else:
            inner.append(c)

    epsilon = 1e-6
    for c in outer:
        if (abs(c.volume - 0) > epsilon):
            return False

    for c in mid_corner:
        if (abs(c.volume - 1/64) > epsilon):
            return False

    for c in mid_edge:
        if (abs(c.volume - 1/16) > epsilon):
            return False

    for c in mid_face:
        if (abs(c.volume - 1/4) > epsilon):
            return False

    for c in inner:
        if (abs(c.volume - 1.0) > epsilon):
            return False

    return True


"""
TEST MARCHING CUBES SURFACE CREATION
marching cubes 2x2x2 grid of side length 2e-6
voxel cube such that grid corners are all 0s except 0.75 in center
should create diamond-shape surface if marching cubes is working
the centroid of each triangle is in a different octo-quadrant thing
 x  y  z : id
-1 -1 -1 : 0
-1 -1  1 : 1
-1  1 -1 : 2
-1  1  1 : 3
 1 -1 -1 : 4
 1 -1  1 : 5
 1  1 -1 : 6
 1  1  1 : 7
"""
class SurfTriSC:
    def __init__(self, vs=[]):
        self.verts = np.asarray(vs)
        if (vs == []):
            self.init = False
        else:
            self.init = True

    # compare triangle vertices (order does not matter)
    # assumes triangles not degenerate
    def compare_tris(self, stri, epsilon):
        # do self's 3 verts each have a matching vert in stri?
        matching = [False, False, False]

        for i in range(3):
            for j in range(3):
                if (all(abs(self.verts[i] - stri.verts[j]) < epsilon)):
                    if (matching[i] == True):
                        return False
                    matching[i] = True

        return all(matching)

class SurfTri:
    def __init__(self, vs=[]):
        self.verts = np.asarray(vs)
        self.plane = -1 # -1 is invalid plane

        # basis vectors created from triangle
        u = self.verts[1] - self.verts[0]
        v = self.verts[2] - self.verts[0]
        n = np.cross(u, v)
        self.normal = n/np.linalg.norm(n) # outward normal

        plane_normals = np.asarray([[-1.,  0.,  0.],
                                    [ 1.,  0.,  0.],
                                    [ 0., -1.,  0.],
                                    [ 0.,  1.,  0.],
                                    [ 0.,  0., -1.],
                                    [ 0.,  0.,  1.],])
        for i in range(6):
            pn = plane_normals[i]
            if (all(abs(self.normal - pn) < 1e-6)):
                self.plane = i
                break

def surface_creation_test():
    # grid length and cube length
    gl = 2e-6
    # this cube length gives 0.75 volume fraction at corner at origin, 0 everywhere else
    vl = (0.75**(1/3))*(gl/2)

    # voxel cube initialization (2x2x2 also)
    v_size = vl/2
    cube_lo = np.asarray([-v_size]*3)
    nvoxs = np.asarray([2,2,2])

    voxels = []
    for k in range(nvoxs[2]):
        z = cube_lo[2] + (0.5 + k)*v_size
        for j in range(nvoxs[1]):
            y = cube_lo[1] + (0.5 + j)*v_size
            for i in range(nvoxs[0]):
                x = cube_lo[0] + (0.5 + i)*v_size
                voxels.append([x,y,z])
    voxels = np.asarray(voxels)

    # grid initialization
    lims = np.asarray([[-gl/2]*3, [gl/2]*3])
    ncells = np.asarray([2,2,2])

    # create surface
    mc_system = MC_System(lims, ncells, v_size, voxels, 'vox2surf.surf', 0)
    new_tris = mc_system.verts[mc_system.faces]

    # should be 8 triangles
    if (len(new_tris) != 8):
        return False
    
    # checking which octo-quadrant triangle centroid is in,
    # put triangle vertices in surf_tris
    centroid_quad = np.asarray([(sum(v)/3 > 0) for v in new_tris])

    surf_tris = [SurfTriSC() for i in range(8)]

    quad_indices = [[0, 0, 0],
                    [0, 0, 1],
                    [0, 1, 0],
                    [0, 1, 1],
                    [1, 0, 0],
                    [1, 0, 1],
                    [1, 1, 0],
                    [1, 1, 1]]
    for i in range(len(centroid_quad)):
        cq = centroid_quad[i]
        for j in range(len(quad_indices)):
            if (all(cq == quad_indices[j])):
                if (surf_tris[j].init == True):
                    return False
                surf_tris[j].verts = new_tris[i]
                surf_tris[j].init = True
                break

    """
    test surf_tris against analytical marching cubes solution
    lc is set because threshold in marching cubes is 0.5, so
    the vertices will be set where the estimated 0.5 contour is;
    since central corner is 0.75 and all others are 0, the triangle
    vertices will be set 1/3 of the way from the origin to the other
    corners, and one marching cubes cell is 1e-6 long, so distance
    of vertices from origin is 0.3333e-6
    """

    lc = 1/3*10**-6
    correct_triangles = [
        SurfTriSC(vs=[[-lc,   0,   0],
                      [  0, -lc,   0],
                      [  0,   0, -lc]]),
        SurfTriSC(vs=[[-lc,   0,   0],
                      [  0, -lc,   0],
                      [  0,   0,  lc]]),
        SurfTriSC(vs=[[-lc,   0,   0],
                      [  0,  lc,   0],
                      [  0,   0, -lc]]),
        SurfTriSC(vs=[[-lc,   0,   0],
                      [  0,  lc,   0],
                      [  0,   0,  lc]]),
        SurfTriSC(vs=[[ lc,   0,   0],
                      [  0, -lc,   0],
                      [  0,   0, -lc]]),
        SurfTriSC(vs=[[ lc,   0,   0],
                      [  0, -lc,   0],
                      [  0,   0,  lc]]),
        SurfTriSC(vs=[[ lc,   0,   0],
                      [  0,  lc,   0],
                      [  0,   0, -lc]]),
        SurfTriSC(vs=[[ lc,   0,   0],
                      [  0,  lc,   0],
                      [  0,   0,  lc]]),
    ]

    epsilon = 1e-6*(2e-6) # 1 micro-grid size
    for i in range(8):
        if (correct_triangles[i].compare_tris(surf_tris[i], epsilon) == False):
            return False

    return True



"""
This tests correct association and fractionizing of triangles to voxels

Ok, counteroffer: make a cube with voxels slightly misaligned to triangles,
so all interior face voxels should have a nice value
"""
def voxel_association_test():
    # grid length and number of mc cells in each dir
    # cell length = 1e-7
    gl = 2e-6
    ref = 20
    clength = gl/ref

    # get a cube roughly 2/3 the length of grid (1.32e-6)
    nline = 26
    vl = 1.32e-6
    # voxel length
    v_size = vl/nline


    # voxel cube initialization (10x10x10)
    cube_lo = np.asarray([-vl/2]*3)
    nvoxs = np.asarray([nline,nline,nline])

    voxels = []
    for k in range(nvoxs[2]):
        z = cube_lo[2] + (0.5 + k)*v_size
        for j in range(nvoxs[1]):
            y = cube_lo[1] + (0.5 + j)*v_size
            for i in range(nvoxs[0]):
                x = cube_lo[0] + (0.5 + i)*v_size
                voxels.append([x,y,z])
    voxels = np.asarray(voxels)

    # grid initialization
    lims = np.asarray([[-gl/2]*3, [gl/2]*3])
    ncells = np.asarray([ref, ref, ref])

    # create surface
    mc_system = MC_System(lims, ncells, v_size, voxels, 'vox2surf.surf', 0)
    new_tris = mc_system.verts[mc_system.faces]

    combined_mesh = trimesh.Trimesh(vertices=mc_system.verts, faces=mc_system.faces)
    stlFileName = 'voxel_association_test.stl'
    combined_mesh.export(stlFileName, file_type='stl_ascii') 

    surf_tris = [SurfTri(t) for t in new_tris]

    # read triangle-voxel assignment
    # two matching dictionaries:
    # tri_voxs   is dict of tri_id -> list of voxel ids
    # tri_sfracs is dict of tri_id -> list of voxel fracs
    tri_voxs, tri_sfracs = read_output()

    # assign values to triangles
    # faces are xlo (0), xhi (1), ylo (2), yhi (3),
    # zlo (4), zhi (5)
    kv = 1e-2 # xlo scalar value for triangles
    tri_vals = np.zeros(len(new_tris))
    for i in range(len(surf_tris)):
        st = surf_tris[i]
        if (st.plane != -1):
            tri_vals[i] = (st.plane + 1)*kv


    # use tri-vox association fractions to assign values to voxels
    vox_vals = np.zeros(len(voxels))
    for tri in tri_voxs.keys():
        for idn in range(len(tri_voxs[tri])):
            vid = tri_voxs[tri][idn]
            vfrac = tri_sfracs[tri][idn]
            vox_vals[vid] += vfrac*tri_vals[tri - 1]

    vox_file = open('voxel_test.csv', 'w')
    vox_file.write('x,y,z,v\n')
    for i in range(len(voxels)):
        vox_file.write('{},{},{},{}\n'.format(voxels[i][0],voxels[i][1],voxels[i][2],vox_vals[i]))
    vox_file.close()

    # check all 'internal face' surface voxels, i.e. not on edges or corners
    pl_vox_lim = (nline/2 - 2)*v_size
    norm_vox_lim = (nline/2 - 1)*v_size

    # collect voxels of each 'internal face', using face code above
    face_voxels = [[],[],[],[],[],[]]
    for i in range(len(voxels)):
        # xlo or xhi
        if (abs(voxels[i][1]) < pl_vox_lim and abs(voxels[i][2]) < pl_vox_lim):
            if (voxels[i][0] < -norm_vox_lim):
                face_voxels[0].append(i)
            elif (voxels[i][0] > norm_vox_lim):
                face_voxels[1].append(i)
        # ylo or yhi
        elif (abs(voxels[i][0]) < pl_vox_lim and abs(voxels[i][2]) < pl_vox_lim):
            if (voxels[i][1] < -norm_vox_lim):
                face_voxels[2].append(i)
            elif (voxels[i][1] > norm_vox_lim):
                face_voxels[3].append(i)
        # zlo or zhi
        elif (abs(voxels[i][1]) < pl_vox_lim and abs(voxels[i][0]) < pl_vox_lim):
            if (voxels[i][2] < -norm_vox_lim):
                face_voxels[4].append(i)
            elif (voxels[i][2] > norm_vox_lim):
                face_voxels[5].append(i)

    vox_face_area = v_size*v_size
    tri_area = clength*clength/2
    for i in range(6):
        kvt = (i + 1)*kv
        epsilon = 1e-6*kvt
        for fv in range(len(face_voxels[i])):
            cv = face_voxels[i][fv]
            if (abs(vox_vals[cv] - kvt*(vox_face_area/tri_area)) > epsilon):
                print('face {}: {}, {}'.format(i + 1, vox_vals[cv], kvt*(vox_face_area/tri_area)))
                return False

    return True


def safe_mkdir(path, name):
    full_path = path + name + '/'
    try:
        if not os.path.exists(full_path):
            os.mkdir(full_path)
    except OSError as err:
            print(err)



# %%
os.chdir(os.path.dirname(os.path.abspath(__file__)))
pathg= os.getcwd() + '/'
safe_mkdir(pathg, 'test_results')

os.chdir('./test_results')
v2c_pass = voxel_division_test()
print('Voxel division test passed:    ' + str(v2c_pass))
mc_pass = surface_creation_test()
print('Surface creation test passed:  ' + str(mc_pass))
vass_pass = voxel_association_test()
print('Voxel association test passed: ' + str(vass_pass))
