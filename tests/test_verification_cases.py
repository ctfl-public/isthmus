import unittest
import numpy as np
import trimesh
import os
from mapping import *
try:
    from numba import cuda
    cuda_available = cuda.is_available()
except ImportError:
    cuda_available = False

# read in output
def read_output(ndims=3):

    if ndims == 3:
        f = open('./voxel_tri/triangle_voxels_0.dat')
    else:
        f = open('./voxel_tri/line_voxels_0.dat')
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
                raise Exception("ERROR: unable to read vox-tri association file")
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
Test a 8x8x8 voxel cube centered in a 10x10x10 cell grid
10x10x10 cell grid makes a 11x11x11 corner grid
Cell side length: 1
Voxel side length: 0.667

Grid should have following volume fractions for line at z=0,y=0:
    Corner at center: 1.0
    1st out: 140/144
    2nd out: 13/18
    3rd out: 7/18
    4th out: 1/12
    Corners at edge of grid: 0.0
"""
def voxel_division_test():

    # voxel cube initialization
    v_size = 2/3
    cube_lo = np.asarray([-4*v_size]*3)
    nvoxs = np.asarray([8,8,8])

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
    lims = np.asarray([[-5,-5,-5], [5,5,5]])
    ncells = np.asarray([10,10,10])

    # divide volumes
    mc_system = marchingWindows(lims, ncells, v_size, voxels, 'vox2surf.surf', 0)
    corners = [crn for crn in mc_system.corner_grid.corners if abs(crn.position[2]) < 0.5 and abs(crn.position[1]) < 0.5]

    # test resulting grid corners at center and going outward

    epsilon = 1e-6
    for c in corners:
        if (any(abs(c.position) > 4.5)):
            if (abs(c.volume - 0) > epsilon):
                return False
        elif (any(abs(c.position) > 3.5)):
            if (abs(c.volume - 1/12) > epsilon):
                return False
        elif (any(abs(c.position) > 2.5)):
            if (abs(c.volume - 7/18) > epsilon):
                return False
        elif (any(abs(c.position) > 1.5)):
            if (abs(c.volume - 13/18) > epsilon):
                return False
        elif (any(abs(c.position) > 0.5)):
            if (abs(c.volume - 140/144) > epsilon):
                return False
        else:
            if (abs(c.volume - 1) > epsilon):
                return False
        

    return True


"""
TEST MARCHING CUBES SURFACE CREATION
marching cubes 4x4x4 grid of side length 4e-6
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
    gl = 4e-6
    # this cube length gives 0.75 volume fraction at corner at origin, 0 everywhere else
    vl = (0.75**(1/3))*(gl/4)

    # voxel cube initialization (2x2x2)
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
    ncells = np.asarray([4,4,4])

    # create surface
    mc_system = marchingWindows(lims, ncells, v_size, voxels, 'vox2surf.surf', 0, weight=False)
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

    epsilon = 1e-6*(2e-6) # 1/2 a micro-grid size
    for i in range(8):
        if (correct_triangles[i].compare_tris(surf_tris[i], epsilon) == False):
            return False

    return True



"""
This tests correct association and fractionizing of triangles to voxels

Ok, counteroffer: make a cube with voxels slightly misaligned to triangles,
so all interior face voxels should have a nice value
"""
def voxel_association_test(gpu=False):
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


    # voxel cube initialization (26x26x26)
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
    mc_system = marchingWindows(lims, ncells, v_size, voxels, 'vox2surf.surf', 0, gpu=gpu)
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

    # check all 'internal face' surface voxels, i.e. not near edges or corners
    pl_vox_lim = (nline/2)*v_size/4
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
        epsilon = 2e-6*kvt
        for fv in range(len(face_voxels[i])):
            cv = face_voxels[i][fv]
            if (abs(vox_vals[cv] - kvt*(vox_face_area/tri_area)) > epsilon):
                print('Face {}: Computed vox scalar {}, True vox scalar {}, Diff {}, Epsilon {}'.format(
                        i + 1, 
                        vox_vals[cv], 
                        kvt*(vox_face_area/tri_area), 
                        abs(vox_vals[cv] - kvt*(vox_face_area/tri_area)),
                        epsilon
                        ))
                return False

    return True

"""
TEST VOXEL VOLUME DIVISION BEFORE MARCHING CUBES
Test a 8x8 voxel square centered in a 10x10 cell grid
10x10 cell grid makes a 11x11 corner grid
Cell side length: 1
Voxel side length: 0.667

Grid should have following volume fractions for line at z=0,y=0:
    Corner at center: 1.0
    1st out: 140/144
    2nd out: 13/18
    3rd out: 7/18
    4th out: 1/12
    Corners at edge of grid: 0.0
"""
def voxel2D_division_test():

    # voxel cube initialization
    v_size = 2/3
    cube_lo = np.asarray([-4*v_size]*2)
    nvoxs = np.asarray([8,8])

    voxels = []
    for j in range(nvoxs[1]):
        y = cube_lo[1] + (0.5 + j)*v_size
        for i in range(nvoxs[0]):
            x = cube_lo[0] + (0.5 + i)*v_size
            voxels.append([x,y])
    voxels = np.asarray(voxels)

    # grid initialization
    lims = np.asarray([[-5,-5], [5,5]])
    ncells = np.asarray([10,10])

    # divide volumes
    mc_system = marchingWindows(lims, ncells, v_size, voxels, 'vox2surf.surf', 0, ndims=2)
    corners = [crn for crn in mc_system.corner_grid.corners if abs(crn.position[1]) < 0.5]

    # test resulting grid corners at center and going outward
    epsilon = 1e-6
    for c in corners:
        if (any(abs(c.position) > 4.5)):
            if (abs(c.volume - 0) > epsilon):
                return False
        elif (any(abs(c.position) > 3.5)):
            if (abs(c.volume - 1/12) > epsilon):
                return False
        elif (any(abs(c.position) > 2.5)):
            if (abs(c.volume - 7/18) > epsilon):
                return False
        elif (any(abs(c.position) > 1.5)):
            if (abs(c.volume - 13/18) > epsilon):
                return False
        elif (any(abs(c.position) > 0.5)):
            if (abs(c.volume - 140/144) > epsilon):
                return False
        else:
            if (abs(c.volume - 1) > epsilon):
                return False
        

    return True

class SurfEdgeSC:
    def __init__(self, vs=[]):
        self.verts = np.asarray(vs)
        if (vs == []):
            self.init = False
        else:
            self.init = True

    # compare edge vertices (order does not matter)
    # assumes edges not degenerate
    def compare_edges(self, stri, epsilon):
        # do self's 2 verts each have a matching vert in stri?
        matching = [False, False]

        for i in range(2): 
            for j in range(2):
                if (all(abs(self.verts[i] - stri.verts[j]) < epsilon)):
                    if (matching[i] == True):
                        return False
                    matching[i] = True

        return all(matching)

class SurfEdge:
    def __init__(self, vs=[]):
        self.verts = np.asarray(vs)
        self.plane = -1 # -1 is invalid plane

        # basis vectors created from triangle
        self.l = self.verts[1] - self.verts[0]
        self.theta = np.arctan2(self.l[1], self.l[0])
        ntheta = self.theta - np.pi/2 # outward normal of border edge
        self.normal = np.array([np.cos(ntheta), np.sin(ntheta)])

        plane_normals = np.asarray([[-1.,  0.],
                                    [ 1.,  0.],
                                    [ 0., -1.],
                                    [ 0.,  1.]])
        for i in range(4):
            pn = plane_normals[i]
            if (all(abs(self.normal - pn) < 1e-6)):
                self.plane = i
                break

def surface2D_creation_test():
    # grid length (4x4) and voxel square length (2x2)
    gl = 4e-6
    # this cube length gives 0.75 volume fraction at corner at origin, 0 everywhere else
    vl = (0.75**(1/2))*(gl/4)

    # voxel cube initialization
    v_size = vl/2
    cube_lo = np.asarray([-v_size]*2)
    nvoxs = np.asarray([2,2])

    voxels = []
    for j in range(nvoxs[1]):
        y = cube_lo[1] + (0.5 + j)*v_size
        for i in range(nvoxs[0]):
            x = cube_lo[0] + (0.5 + i)*v_size
            voxels.append([x,y])
    voxels = np.asarray(voxels)

    # grid initialization
    lims = np.asarray([[-gl/2]*2, [gl/2]*2])
    ncells = np.asarray([4,4])

    # create surface
    mc_system = marchingWindows(lims, ncells, v_size, voxels, 'vox2surf.surf', 0, weight=False, ndims=2)
    new_tris = mc_system.verts[mc_system.faces]

    # should be 4 edges
    if (len(new_tris) != 4):
        return False
    
    # checking which quadrant edge centroid is in,
    # put triangle vertices in surf_tris
    centroid_quad = np.asarray([(sum(v)/2 > 0) for v in new_tris])

    surf_edges = [SurfEdgeSC() for i in range(4)]

    quad_indices = [[0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1]]
    for i in range(len(centroid_quad)):
        cq = centroid_quad[i]
        for j in range(len(quad_indices)):
            if (all(cq == quad_indices[j])):
                if (surf_edges[j].init == True):
                    return False
                surf_edges[j].verts = new_tris[i]
                surf_edges[j].init = True
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
    correct_edges = [
        SurfEdgeSC(vs=[[-lc,   0],
                       [  0, -lc]]),
        SurfEdgeSC(vs=[[-lc,   0],
                       [  0,  lc]]),
        SurfEdgeSC(vs=[[ lc,   0],
                       [  0, -lc]]),
        SurfEdgeSC(vs=[[ lc,   0],
                       [  0,  lc]])
    ]

    epsilon = 1e-6*(2e-6) # 1/2 a micro-grid size
    for i in range(4):
        if (correct_edges[i].compare_edges(surf_edges[i], epsilon) == False):
            return False

    return True

"""
This tests correct association and fractionizing of triangles to voxels

Ok, counteroffer: make a cube with voxels slightly misaligned to triangles,
so all interior face voxels should have a nice value
"""
def voxel2D_association_test(gpu=False):
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

    # voxel cube initialization (26x26)
    cube_lo = np.asarray([-vl/2]*2)
    nvoxs = np.asarray([nline,nline])

    voxels = []
    for j in range(nvoxs[1]):
        y = cube_lo[1] + (0.5 + j)*v_size
        for i in range(nvoxs[0]):
            x = cube_lo[0] + (0.5 + i)*v_size
            voxels.append([x,y])
    voxels = np.asarray(voxels)

    # grid initialization
    lims = np.asarray([[-gl/2]*2, [gl/2]*2])
    ncells = np.asarray([ref, ref])

    # create surface
    mc_system = marchingWindows(lims, ncells, v_size, voxels, 'vox2surf.surf', 0, gpu=gpu, ndims=2)
    new_tris = mc_system.verts[mc_system.faces]

    surf_tris = [SurfEdge(t) for t in new_tris]

    # read triangle-voxel assignment
    # two matching dictionaries:
    # tri_voxs   is dict of tri_id -> list of voxel ids
    # tri_sfracs is dict of tri_id -> list of voxel fracs
    tri_voxs, tri_sfracs = read_output(ndims=2)

    # assign values to triangles
    # faces are xlo (0), xhi (1), ylo (2), yhi (3)
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
    vox_file.write('x,y,v\n')
    for i in range(len(voxels)):
        vox_file.write('{},{},{}\n'.format(voxels[i][0],voxels[i][1],vox_vals[i]))
    vox_file.close()

    # check all 'internal face' surface voxels, i.e. not near edges or corners
    pl_vox_lim = (nline/2)*v_size/4
    norm_vox_lim = (nline/2 - 1)*v_size

    # collect voxels of each 'internal face', using face code above
    face_voxels = [[],[],[],[]]
    for i in range(len(voxels)):
        # xlo or xhi
        if (abs(voxels[i][1]) < pl_vox_lim):
            if (voxels[i][0] < -norm_vox_lim):
                face_voxels[0].append(i)
            elif (voxels[i][0] > norm_vox_lim):
                face_voxels[1].append(i)
        # ylo or yhi
        elif (abs(voxels[i][0]) < pl_vox_lim):
            if (voxels[i][1] < -norm_vox_lim):
                face_voxels[2].append(i)
            elif (voxels[i][1] > norm_vox_lim):
                face_voxels[3].append(i)

    vox_face_area = v_size
    tri_area = clength
    for i in range(4):
        kvt = (i + 1)*kv
        epsilon = 2e-6*kvt
        for fv in range(len(face_voxels[i])):
            cv = face_voxels[i][fv]
            if (abs(vox_vals[cv] - kvt*(vox_face_area/tri_area)) > epsilon):
                print('Face {}: Computed vox scalar {}, True vox scalar {}, Diff {}, Epsilon {}'.format(
                        i + 1, 
                        vox_vals[cv], 
                        kvt*(vox_face_area/tri_area), 
                        abs(vox_vals[cv] - kvt*(vox_face_area/tri_area)),
                        epsilon
                        ))
                return False

    return True

def safe_mkdir(path, name):
    full_path = path + name + '/'
    try:
        if not os.path.exists(full_path):
            os.mkdir(full_path)
    except OSError as err:
            print(err)


class TestVoxelDivision(unittest.TestCase):
    def test_voxel_division(self):
        self.assertTrue(voxel_division_test())

    def test_voxel2D_division(self):
        self.assertTrue(voxel2D_division_test())

class TestSurfaceCreation(unittest.TestCase):
    def test_surface_creation(self):
        self.assertTrue(surface_creation_test())

    def test_surface2D_creation(self):
        self.assertTrue(surface2D_creation_test())

class TestVoxelAssociation(unittest.TestCase):
    def test_voxel_association(self):
        self.assertTrue(voxel_association_test())

    def test_voxel2D_association(self):
        self.assertTrue(voxel2D_association_test())
        
    def test_voxel_association_gpu(self):
        if not cuda_available:
            self.skipTest("Skipping GPU tests (Numba or CUDA not available)")
        self.assertTrue(voxel_association_test(gpu=True))



if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    pathg= os.getcwd() + '/'
    safe_mkdir(pathg, 'test_results')
    os.chdir('./test_results')

    unittest.main(verbosity=2)

