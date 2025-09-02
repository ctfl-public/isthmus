import numpy as np
import os
import sys
import time
import copy
from Marching_Cubes import marching_cubes, mesh_surface_area
from scipy.spatial import cKDTree

# Check if Numba is available, use isthmus_gpu
try:
    import numba
    from isthmus_gpu import get_intersection_area_gpu
    numba_available = True
except ImportError:
    numba_available = False

# need csv, dev, grids, voxel_data, and voxel_tri folders

#%% Individual geometric elements used in grids

class Line:
    def __init__(self, endpts, locs=[]):
        self.endpts = np.array(endpts) # [[x1, y1], [x2, y2]]
        self.a = self.endpts[0] # [x,y]
        self.b = self.endpts[1]
        self.length = np.linalg.norm(self.b - self.a)
        transend = np.transpose(self.endpts)
        self.xlo = np.array([min(transend[0]), min(transend[1])])
        self.xhi = np.array([max(transend[0]), max(transend[1])])
        self.theta = np.arctan2((self.b[1] - self.a[1]), (self.b[0] - self.a[0]))
        self.vert_indices = []
        if len(locs) == 0:
            self.locs = [-1, -1]
        elif len(locs) == 2:
            self.locs = [locs[0], locs[1]]
        else:
            print('ERROR: invalid line endpoint positions')
            sys.exit(1)

        self.voxel_ids = []    # index of each owned voxel in global array
        self.voxel_scalar_fracs = [] # fraction of triangle values to assign to each voxel

class Triangle:
    def __init__(self, verts, identity, ncell):
        self.vertices = verts  # [[x1, y1, z1], [x2,y2,z2], [x3,y3,z3]]
        self.id = identity     # own triangle index
        self.cell = ncell      # owning cell object
        self.voxel_ids = []    # index of each owned voxel in global array
        self.voxel_scalar_fracs = [] # fraction of triangle values to assign to each voxel
        self.centroid = np.average(verts, axis=0)
        # axis-aligned bounding box limits
        trans_verts = np.transpose(self.vertices) # [[xs], [ys], [zs]]
        self.lo = np.array([min(c) for c in trans_verts])
        self.hi = np.array([max(c) for c in trans_verts])
        
        # basis vectors created from triangle
        u = self.vertices[1] - self.vertices[0]
        v = self.vertices[2] - self.vertices[0]
        n = np.cross(u, v)
        self.normal = n/np.linalg.norm(n) # outward normal

        # epsilon based on triangle size for floating point comparisons
        self.epsilon = 1e-4*max([max(x) - min(x) for x in np.transpose(self.vertices)])

        # precompute triangle edges normals to be used in clip_sh
        plane_normal = [np.cross(self.vertices[i]-self.vertices[i-1], self.normal) for i in range(3)]
        self.plane_normal = [n / np.linalg.norm(n) for n in plane_normal]

    # distance to nearest point on triangle, by combining normal and planar components
    def check_overlap(self, face):
        # first check alignment of face with triangle (is the face visible to tri based on outward normals)
        if (np.dot(face.n, self.normal) < 0):
            return False, []
        
        # project face onto triangle plane, should be a parallelogram (?)

        proj_face = np.array([face.xs[i] - self.normal*np.dot(self.normal, face.xs[i] - self.vertices[0]) for i in range(4)])
        trans_face = np.transpose(proj_face)
        pf_lo = np.array([min(c) for c in trans_face])
        pf_hi = np.array([max(c) for c in trans_face])

        epsilon = 1e-3*Voxel.size
        # then a fast AABB test between triangle and projected face
        for i in range(3):
            if (pf_lo[i] - self.hi[i] > epsilon or epsilon < self.lo[i] - pf_hi[i]):
                return False, []

        # then use the separating axis theorem; should be 5 possible separation axes (need to be normalized to unit?)
        test_axes = [np.cross(self.vertices[2] - self.vertices[1], self.normal), # axes normal to each tri edge
                     np.cross(self.vertices[1] - self.vertices[0], self.normal),
                     np.cross(self.vertices[0] - self.vertices[2], self.normal),
                     np.cross(proj_face[2]     - proj_face[1],     self.normal), # axes normal to parallelogram edges
                     np.cross(proj_face[1]     - proj_face[0],     self.normal)]
        
        # project triangle and parallelogram onto each axis, see if separation is possible
        for ax in test_axes:
            tri_ax  = [np.dot(self.vertices[i], ax) for i in range(len(self.vertices))]
            face_ax = [np.dot(proj_face[i], ax)     for i in range(len(proj_face))]

            if (min(tri_ax) > max(face_ax) or max(tri_ax) < min(face_ax)):
                return False, []

        # if no axes are separable, there is overlap
        return True, proj_face

# get length of the intersection of two line segments
def get_intersection_length(projections, bases):
    inter_lines = []
    for b_index, base in enumerate(bases):
        proj = projections[b_index]
        diff = base.b - base.a
        base_len = np.linalg.norm(diff)
        if base_len < 1e-20:
            inter_lines.append(0)
        else:
            ind = 0 if abs(diff[0]) > abs(diff[1]) else 1

            # t of base a is 0, base b is 1
            t_a = (proj.a[ind] - base.a[ind])/diff[ind]
            t_b = (proj.b[ind] - base.a[ind])/diff[ind]

            # check bounding line
            if (t_a < 0 and t_b < 0) or (t_a > 1 and t_b > 1):
                inter_lines.append(0)
            else:
            # clip projected line to base line, t = [0, 1]
                t_a = max(t_a, 0)
                t_b = max(t_b, 0)

                t_a = min(t_a, 1)
                t_b = min(t_b, 1)
                t_diff = abs(t_b - t_a)
                inter_lines.append(t_diff*base_len)
                
    return inter_lines

# get_intersection_area function for batch processing
def get_intersection_area(proj_faces, tri_normal, tri_plane_normal, tri_vertices, tri_epsilon):
    # find overlapping area
    all_clipped_points = clip_sh(proj_faces, tri_plane_normal, tri_vertices, tri_epsilon) 
    polygon_areas = []
    for intr_indx, clipped_points in enumerate(all_clipped_points):
        if len(clipped_points) < 3:
            polygon_areas.append(0)
            continue
        # rotate overlap polygon into xy plane
        rotated_points = orient_polygon_xy(clipped_points, tri_normal[intr_indx]) 
        # get area with shoelace formula
        polygon_areas.append(polygon_area(rotated_points)) 
        progress_bar(intr_indx, len(all_clipped_points), '    finding intersection areas')
    return polygon_areas

# Sutherland-Hodgman polygon clipping
# inputs are vertices of subject (to be clipped) and vertices
# of window (the clipper)
def clip_sh(subjects, tri_plane_normal, tri_vertices, tri_epsilon):
    final_pts = []
    for intr_indx, subject in enumerate(subjects):
        # clipping operation
        in_pts = subject
        for i in range(3):
            out_pts = []

            for j in range(len(in_pts)):
                p1 = in_pts[j - 1]
                p2 = in_pts[j]

                # compute intersection with infinite edge
                p1_in, p2_in, intersect = segment_plane_intersection(p1, p2, tri_plane_normal[intr_indx][i], tri_vertices[intr_indx][i], tri_epsilon[intr_indx])

                if (p2_in):
                    if (not p1_in):
                        out_pts.append(intersect)
                    out_pts.append(p2)
                elif (p1_in):   # and not p2_in
                    out_pts.append(intersect)
                # if p1 and p2 both outside, do nothing, delete line segment

            in_pts = out_pts

        # remove duplicate vertices
        final_pts.append([])
        for i in range(len(out_pts)):
            dupe = False
            for j in range(i + 1, len(out_pts)):
                if (all(abs(out_pts[j] - out_pts[i]) < tri_epsilon[intr_indx])):
                    dupe = True
                    break
            if not dupe:
                final_pts[-1].append(out_pts[i])

        progress_bar(intr_indx, len(subjects), '    clipping polygons')

    return final_pts

# for line segment of points p1 and p2, does it intersect plane defined by
# outward unit normal n, passing through point q
# return inside/outside determinations for p1,p2 and intersection point
# 'in' means inside or on plane; out means strictly outside
def segment_plane_intersection(p1, p2, n, q, epsilon):
    intersect = np.zeros(3)
    p1_dist = np.dot(p1 - q, n)
    p2_dist = np.dot(p2 - q, n)

    p1_in = False
    p2_in = False
    if (p1_dist < epsilon):
        p1_in = True
    if (p2_dist < epsilon):
        p2_in = True
    # if one in and other out, there is an intersection
    if (p1_in + p2_in == 1):
        if (p1_in):
            if (abs(p1_dist) < epsilon):
                intersect = p1
            else:
                frac = abs(p1_dist)/(abs(p2_dist) + abs(p1_dist))
                intersect = p1 + frac*(p2 - p1)
        # p2 is inside
        else:
            if (abs(p2_dist) < epsilon):
                intersect = p2
            else:
                frac = abs(p1_dist)/(abs(p2_dist) + abs(p1_dist))
                intersect = p1 + frac*(p2 - p1)

    return p1_in, p2_in, intersect

def orient_polygon_xy(verts, normal):
    theta = np.arccos(normal[2])
    epsilon = 1e-4
    if (theta < epsilon or np.pi - theta < epsilon):
        return np.array([[verts[i][0], verts[i][1]] for i in range(len(verts))])
    else:
        # need finite rotation to align polygon
        # get unit vector of axis around which to rotate
        axis = np.cross(normal, [0,0,1])
        ax_len = np.sqrt(axis[0]**2 + axis[1]**2 + axis[2]**2)
        axis /= ax_len

        # theta is already known, and if i did my math right, it should always
        # be a positive rotation; use 3D rotation matrix
        R = np.outer(axis, axis)*(1 - np.cos(theta))
        R += np.identity(3)*np.cos(theta)
        R += np.array([ [       0, -axis[2],  axis[1]],
                        [ axis[2],        0, -axis[0]],
                        [-axis[1],  axis[0],        0]])*np.sin(theta)
        return np.array([np.matmul(R, v)[:-1] for v in verts])


# shoelace formula (trapezoid rule); for 2D XY POLYGONS ONLY
def polygon_area(verts):
    area = 0
    for i in range(len(verts)):
        p1 = verts[i - 1]
        p2 = verts[i]
        area += (p1[1] + p2[1])*(p1[0] - p2[0])

    return abs(area*0.5)

def get_tri_area(verts):

    # herons formula = sqrt(s(s - a)(s - b)(s - c)) for triangle lengths a,b,c, s= half-perimeter
    a = np.linalg.norm(verts[2] - verts[1])
    b = np.linalg.norm(verts[1] - verts[0])
    c = np.linalg.norm(verts[0] - verts[2])

    s = (a + b + c)/2
    area = np.sqrt(s*(s - a)*(s - b)*(s - c))

    return area
   
def get_longest_side(verts):
    L0 = np.linalg.norm(verts[1] - verts[0])
    L1 = np.linalg.norm(verts[2] - verts[1])
    L2 = np.linalg.norm(verts[0] - verts[2])
    sides = np.array([L0, L1, L2])
    max_len = np.argmax(sides)
    if max_len == 0:
        AC = [1, 0]
    elif max_len == 1:
        AC = [2, 1]
    else:
        AC = [0, 2]
    return AC

class Voxel:
    size = 0
    n_surfvoxels = 0
    def __init__(self, xs, i, j, k, index):
        self.position = xs # centroid position
        self.indices = np.array([i,j,k]) # position in 3D array
        self.id = index   # own voxel index in flattened array
        self.lo = np.array(xs - 0.5*Voxel.size) # bounding box lower limit
        self.hi = self.lo + Voxel.size # bounding box higher limit
        self.oid = -1 # index in original voxel list argument
        self.type = -1 # integer depth into voxel structure, 0 is edge voxel
        self.weight = 0 # weighting applied to volume in volume division
        self.finalized = False # whether type has been determined yet
        self.surface = False # flag for being edge (type 0) voxel

        # placeholders for surface voxel data
        self.surf_id = None
        self.triangle_ids = [] # index of each owned triangle
        self.closest_triangle_id = -1 # initialize to invalid id
        self.closest_triangle_dist = 1e12 # initialize to massive number
        self.closest_triangle_cell = -1 # initialize to invalid id
        self.faces = [] # 6 owned face objects of voxel cube

    def generate(self, oid):
        self.oid = oid
        self.type = 0
        self.weight = 1

    def convert2surfvoxel(self):
        self.surface = True
        self.surf_id = Voxel.n_surfvoxels # voxel index in surface voxel list
        Voxel.n_surfvoxels += 1

        # corners: first four are zlo x-y plane, then last four are zhi x-y plane
        cs  =  [self.lo + np.array([0,0,0])*Voxel.size,
                self.lo + np.array([1,0,0])*Voxel.size,
                self.lo + np.array([0,1,0])*Voxel.size,
                self.lo + np.array([1,1,0])*Voxel.size,
                self.lo + np.array([0,0,1])*Voxel.size,
                self.lo + np.array([1,0,1])*Voxel.size,
                self.lo + np.array([0,1,1])*Voxel.size,
                self.lo + np.array([1,1,1])*Voxel.size
                ]

        self.faces.append(Voxel_Face(0, cs[2], cs[0], cs[4], cs[6])) # xlo
        self.faces.append(Voxel_Face(1, cs[1], cs[3], cs[7], cs[5])) # xhi
        self.faces.append(Voxel_Face(2, cs[0], cs[1], cs[5], cs[4])) # ylo
        self.faces.append(Voxel_Face(3, cs[3], cs[2], cs[6], cs[7])) # yhi
        self.faces.append(Voxel_Face(4, cs[2], cs[3], cs[1], cs[0])) # zlo
        self.faces.append(Voxel_Face(5, cs[4], cs[5], cs[7], cs[6])) # zhi

class Voxel_Face:
    def __init__(self, tipo, x1, x2, x3, x4):
        # type (tipo) is 0-5: xlo, xhi, ylo, yhi, zlo, zhi face
        self.type = tipo
        # corners in CCW order: bottom-left, bottom-right, top-right, top-left looking inwards
        self.xs = [x1, x2, x3, x4]
        # unit outward normal
        self.n = np.cross(self.xs[1] - self.xs[0], self.xs[3] - self.xs[0])
        self.n = self.n/np.sqrt(self.n[0]**2 + self.n[1]**2 + self.n[2]**2)
        self.exposed = False

class Voxel2D:
    size = 0
    n_surfvoxels = 0
    def __init__(self, xs, i, j, index):
        self.position = xs # centroid position
        self.indices = np.array([i,j]) # position in 2D array
        self.id = index   # own voxel index in flattened array
        self.lo = np.array(xs - 0.5*Voxel2D.size) # bounding box lower limit
        self.hi = self.lo + Voxel2D.size # bounding box higher limit
        self.oid = -1 # index in original voxel list argument
        self.type = -1 # integer depth into voxel structure, 0 is edge voxel
        self.weight = 0 # weighting applied to volume in volume division
        self.finalized = False # whether type has been determined yet
        self.surface = False # flag for being edge (type 0) voxel

        # placeholders for surface voxel data
        self.surf_id = None
        self.triangle_ids = [] # index of each owned triangle
        self.closest_triangle_id = -1 # initialize to invalid id
        self.closest_triangle_dist = 1e12 # initialize to massive number
        self.closest_triangle_cell = -1 # initialize to invalid id
        self.faces = [] # 6 owned face objects of voxel cube

    def generate(self, oid):
        self.oid = oid
        self.type = 0
        self.weight = 1

    def convert2surfvoxel(self):
        self.surface = True
        self.surf_id = Voxel2D.n_surfvoxels # voxel index in surface voxel list
        Voxel2D.n_surfvoxels += 1

        # corners
        cs  =  [self.lo + np.array([0,0])*Voxel2D.size,
                self.lo + np.array([1,0])*Voxel2D.size,
                self.lo + np.array([0,1])*Voxel2D.size,
                self.lo + np.array([1,1])*Voxel2D.size]

        self.faces.append(Voxel_Face2D(0, cs[2], cs[0])) # xlo
        self.faces.append(Voxel_Face2D(1, cs[1], cs[3])) # xhi
        self.faces.append(Voxel_Face2D(2, cs[0], cs[1])) # ylo
        self.faces.append(Voxel_Face2D(3, cs[3], cs[2])) # yhi

class Voxel_Face2D:
    def __init__(self, tipo, x1, x2):
        # type (tipo) is 0-3: xlo, xhi, ylo, yhi face
        self.type = tipo
        # corners in CCW order around perimeter of voxel
        self.xs = [x1, x2]
        # unit outward normal
        if self.type == 0:
            self.n = [-1,0]
        elif self.type == 1:
            self.n = [1,0]
        elif self.type == 2:
            self.n = [0,-1]
        else:
            self.n = [0,1]
        self.n = np.array(self.n)
        self.exposed = False

# this class is for corners in the grid, i.e. each MC_Cell corresponds to
# 8 MC_Corners
class MC_Corner:
    """
    Parameters
    ----------
    p: corner position
    
    i, j, k: indices in grid
    """
    ## @param p corner position
    def __init__(self, p, i, j, k):
        self.position = p # [x,y,z]
        self.indices = np.array([i,j,k]) # indices in the grid
        self.volume = 0 # volume fraction of cell filled by voxel material
        self.voxels = [] # voxel ids owned by corner

class MC_Corner2D:
    ## @param p corner position
    def __init__(self, p, i, j):
        self.position = p # [x,y]
        self.indices = np.array([i,j]) # indices in the grid
        self.volume = 0 # volume fraction of cell filled by voxel material
        self.inside = -1 # 1 if inside, 0 if outside, -1 if unassigned
        self.voxels = [] # voxel ids owned by corner

# this is the unit cell of the marching cubes grid, with position, owned voxels,
# and owned triangles
class MC_Cell:
    def __init__(self, i):
        self.surface_voxels = [] # surface voxel objects owned by this cell
        self.triangles = [] # triangles owned by this cell
        self.id = i       # index in cell_grid

class MC_Cell2D:
    def __init__(self, corn_list, i, j, index):
        self.corners = np.array(corn_list)
        self.indices = np.array([i,j])
        self.id = index
        self.xlo = np.array(self.corners[0].position) # [i,j] indices for [x,y] of minimum corner
        self.xhi = np.array(self.corners[2].position) # same for max corner
        self.cell_len = self.xhi - self.xlo # length of cell sides
        self.center = (self.xhi + self.xlo)/2

        # marching squares topology index
        self.type = -1

        # empty except for leaf cells
        self.borders = [] # edges, used for marching squares interpolation
        self.surface_voxels = []  # list of voxel objects owned by cell
        self.neighbors = []

    def set_topology(self):
        self.type = self.corners[3].inside*8 + \
                    self.corners[2].inside*4 + \
                    self.corners[1].inside*2 + \
                    self.corners[0].inside
    
        # available locs of cell edges
        diff = np.array([[ 0.0, -0.5],  # bottom
                         [ 0.5,  0.0],  # right
                         [ 0.0,  0.5],  # top
                         [-0.5,  0.0]]) # left

        # fully outside or inside, no surface; further labels are for inside regions
        if self.type == 0 or self.type == 15:
            self.borders = []

        # two surface elements

        # these two (5 & 10) are saddle points, only one of the possible solutions is used here
        elif self.type == 5 or self.type == 10:
            # bottom left and top right diagonal
            if self.type == 5:
                loc1 = [0, 1]
                loc2 = [2, 3]
            # top left and bottom right diagonal
            elif self.type == 10:
                loc1 = [1, 2]
                loc2 = [3, 0]
            a1 = self.center + self.cell_len*diff[loc1[0]]
            b1 = self.center + self.cell_len*diff[loc1[1]]
            a2 = self.center + self.cell_len*diff[loc2[0]]
            b2 = self.center + self.cell_len*diff[loc2[1]]
            self.borders = [Line([a1, b1], loc1), Line([a2, b2], loc2)]

        # one surface elements
        else:
            # bottom left
            if self.type == 1:
                loc = [0, 3]
            # bottom right
            elif self.type == 2:
                loc = [1, 0]
            # bottom half
            elif self.type == 3:
                loc = [1, 3]
            # top right
            elif self.type == 4:
                loc = [2, 1]
            # right half
            elif self.type == 6:
                loc = [2, 0]
            # all but top left
            elif self.type == 7:
                loc = [2, 3]
            # top left
            elif self.type == 8:
                loc = [3, 2]
            # left half
            elif self.type == 9:
                loc = [0, 2]
            # all but top right
            elif self.type == 11:
                loc = [1, 2]
            # top half
            elif self.type == 12:
                loc = [3, 1]
            # all but bottom right
            elif self.type == 13:
                loc = [0, 1]
            # all but bottom left
            elif self.type == 14:
                loc = [3, 0]
            else:
                print('ERROR: invalid type {} for marching squares cell')
                sys.exit(1)
            a = self.center + self.cell_len*diff[loc[0]]
            b = self.center + self.cell_len*diff[loc[1]]
            self.borders = [Line([a,b], loc)]
        
    def interpolate(self):
        new_borders = []
        thresh = 0.5
        for brd in self.borders:
            loc = brd.locs
            new_endpts = []
            for i in range(2):
                pt_new = [brd.endpts[i][0], brd.endpts[i][1]]
                if loc[i] == 0:
                    corn1 = self.corners[0]
                    corn2 = self.corners[1]
                    x_new = ( (thresh - corn1.volume)/(corn2.volume - corn1.volume) )*(corn2.position[0] - corn1.position[0]) + corn1.position[0]
                    pt_new[0] = x_new
                elif loc[i] == 1:
                    corn1 = self.corners[1]
                    corn2 = self.corners[2]
                    y_new = ( (thresh - corn1.volume)/(corn2.volume - corn1.volume) )*(corn2.position[1] - corn1.position[1]) + corn1.position[1]
                    pt_new[1] = y_new
                elif loc[i] == 2:
                    corn1 = self.corners[2]
                    corn2 = self.corners[3]
                    x_new = ( (thresh - corn1.volume)/(corn2.volume - corn1.volume) )*(corn2.position[0] - corn1.position[0]) + corn1.position[0]
                    pt_new[0] = x_new
                elif loc[i] == 3:
                    corn1 = self.corners[3]
                    corn2 = self.corners[0]
                    y_new = ( (thresh - corn1.volume)/(corn2.volume - corn1.volume) )*(corn2.position[1] - corn1.position[1]) + corn1.position[1]
                    pt_new[1] = y_new
                else:
                    print('ERROR: invalid edge type {}'.format(loc[0]))
                    sys.exit(1)
                new_endpts.append(np.array(pt_new))
            new_borders.append(Line([new_endpts[0], new_endpts[1]], loc))
        self.borders = new_borders

#%% Main system class where all the magic happens
class MC_System:  
    """! @brief Holder of the keys of the kingdom
    Welcome to the isthmus experience! This program assumes minimal overlap of pixels,
    
    Parameter
    ---------
    lims: [lo, hi] array
        The bounding box enclosing the geometry. A 2x3 numpy array representing the domain limits of the grid in 3D space. 
    ncells: [nx, ny, nz] integers
        No. of cells in x, y, and z directions.
    voxel_size: float
        Voxel edge length.
    voxels: [[x, y, z], ...] ndarray
        Array of voxels positions.
    name: 
        Name of output surface file.
    call_no:
        Call number to append with the output file that associate triangles to voxels. 
    gpu: bool
        If True, use GPU for calculations.

    """
    def __init__(self, lims, ncells, voxel_size, voxels, name, call_no, gpu=False, weight=True, ndims=3):
        print('Executing marching cubes...')

        # initialize system variables
        self.gpu = gpu and numba_available  # Use GPU only if Numba is available
        self.ndims = ndims # 2D or 3D
        self.weight_flag = weight # whether to weight voxels by layer
        self.grid_lims = lims
        self.verts = []
        self.faces = []
        
        # remove surface file if exists so if routine fails, error will occur in calling program
        if os.path.exists(name):
            os.remove(name)
        
        # check validity of grid being created and voxel data
        self.check_grid(lims, ncells)
        self.check_voxels(lims, ncells, voxel_size, np.transpose(voxels))

        # initialize system variables
        Voxel.size = Voxel2D.size = voxel_size

        # organize voxels and divide volumes among grid corners
        self.vox_grid = self.sort_voxels(voxels)
        self.surface_voxels = self.weight_voxels(lims, ncells)
        self.corner_grid = Corner_Grid(lims, ncells + 1, self.vox_grid)

        if self.ndims == 3:
            # prepare marching cubes volume grid, and create mesh
            self.create_surface()
            
            # write SPARTA-compliant surface
            self.write_surface(name)
            
            # find voxels on the surface and organize these surface voxels and triangles into cells
            self.cell_grid = Cell_Grid(lims, ncells, self.surface_voxels, self.faces, self.verts, self.gpu)
            
            # associate voxels to triangles
            self.write_triangle_voxels(call_no)
        else:
            self.cell_grid = Cell_Grid2D(lims, ncells, self.corner_grid)
            # create marching squares surface
            self.surf_lines = self.create_surface2D()

            # convert Line objects to vertices and faces, then write surface file
            self.convert_lines()
            self.write_surface2D(name)

            self.cell_grid.voxels_to_edges(self.surface_voxels, self.surf_lines)

            self.write_line_voxels(call_no)
        
    ## check validity of grid limits and number of cells
    def check_grid(self, lims, ncells):
        if self.ndims != 2 and self.ndims != 3:
            raise Exception("System must be 2D or 3D")

        if (lims.shape != (2,self.ndims)):
            raise Exception("Invalid grid limits given")
            
        if (ncells.shape != (self.ndims,)):
            raise Exception("Invalid numbers of grid cells given")
            
        for i in range(self.ndims):
            if (lims[1][i] <= lims[0][i]):
                raise Exception("Invalid grid limits given (limits inverted)")
            if (not np.issubdtype(ncells[i], np.integer)):
                raise Exception("Numbers of grid cells must be integers")
            
    # check validity of voxel positions and size
    def check_voxels(self, lims, ncells, voxel_size, positions):
        cell_length = (lims[1] - lims[0])/ncells # length of cell in [x,y,z] directions
        if (any(cell_length < voxel_size)):
            if self.ndims == 3:
                exc = "Voxel size {:.2e} is larger than marching cubes grid cell dimension(s) {:.2e} {:.2e} {:.2e} ".format( \
                       voxel_size, cell_length[0], cell_length[1], cell_length[2])
            else:
                exc = "Voxel size {:.2e} is larger than marching cubes grid cell dimension(s) {:.2e} {:.2e}".format( \
                                voxel_size, cell_length[0], cell_length[1])
            raise Exception(exc)
        if (not voxel_size > 0):
            raise Exception("Voxel size is invalid")
        
        if (len(positions) != self.ndims):
            raise Exception("Invalid voxel coordinates given")
        
        # bounding box for voxel centroids
        voxc_lims = np.array([[min(pxs) for pxs in positions],  # mins
                              [max(pxs) for pxs in positions]]) # maxs
        # bounding box for acceptable voxel positions with appropriate buffer
        if self.weight_flag:
            Lmax = 1.5*max(cell_length) + voxel_size
        else:
            Lmax = 0.5*(max(cell_length) + voxel_size)
        buffer_lims = np.array([lims[0] + Lmax, lims[1] - Lmax])

        # test buffer box for positive area
        if any([buffer_lims[0][i] >= buffer_lims[1][i] for i in range(self.ndims)]):
            raise Exception("Insufficient buffer added to marching windows grid")
        # test voxel positions for not trespassing into buffer zone
        if any([buffer_lims[0][i] > voxc_lims[0][i] for i in range(self.ndims)]) or any([buffer_lims[1][i] < voxc_lims[1][i] for i in range(self.ndims)]):
            raise Exception("Insufficient buffer added to marching windows grid for voxel set")
    
    # vox_cs are [[x1,y1,z1], [x2,y2,z2],...] of centroids
    def sort_voxels(self, vox_cs):
        # initialize voxels and limits of voxel grid to be used
        first_vox = vox_cs[0]
        nvoxs = np.ceil((first_vox - self.grid_lims[0])/Voxel.size)
        vcx_lo = first_vox - nvoxs*Voxel.size
        nvoxs += np.ceil((self.grid_lims[1] - first_vox)/Voxel.size)
        vcx_hi = vcx_lo + nvoxs*Voxel.size
        nvoxs = (nvoxs).astype(int) + 1
        
        # create voxel space grid, -1 if nothing, vox id if something
        vox_grid = Voxel_Grid([vcx_lo, vcx_hi], nvoxs)
        
        # populate voxel space
        vox_elno = (np.ones(len(vox_cs))*-1).astype(int)
        for i in range(len(vox_cs)):
            ind = np.rint((vox_cs[i] - vcx_lo)/Voxel.size).astype(int)
            n = vox_grid.get_element(ind)
            vox_elno[i] = n
            if (vox_grid.voxels[n].type != -1):
                print('WARNING: overwriting voxel with another in same position')
            vox_grid.voxels[n].generate(i)
        return vox_grid
        
    def weight_voxels(self, lims, ncells):
        # set voxel weights to something other than 0 or -1
        surface_voxels = []
        if self.weight_flag:
            cell_length = max((lims[1] - lims[0])/ncells) # length of cell in [x,y,z] directions
            cv_ratio = cell_length/Voxel.size
            w_max = np.ceil((3*cv_ratio/2) - 0.5)
            w_min = np.floor(-(3*cv_ratio/2) - 0.5)
            level = 0
            while level <= w_max or (-level - 1) >= w_min:
                for n in range(len(self.vox_grid.voxels)):
                    vox = self.vox_grid.voxels[n]
                    if vox.finalized == False:
                        if vox.type == level:
                            self.vox_grid.check_surrounded_solid(n)
                            if level == 0 and vox.type == 0:
                                vox.convert2surfvoxel()
                                surface_voxels.append(vox)
                        elif vox.type == -(level + 1):
                            self.vox_grid.check_surrounded_void(n)
                level += 1
                assert(level < 1000)
        else:
            for n in range(len(self.vox_grid.voxels)):
                vox = self.vox_grid.voxels[n]
                if vox.type == 0:
                    self.vox_grid.check_surrounded_solid(n)
                    if vox.type == 0:
                        vox.convert2surfvoxel()
                        surface_voxels.append(vox)

        # set weights and find exposed faces
        if self.weight_flag:
            for n in range(len(self.vox_grid.voxels)):
                vox = self.vox_grid.voxels[n]
                dvox = (0.5 + vox.type)
                vox.weight = 0.5*(1 + dvox*(2/(3*cv_ratio)))
                vox.weight = min(vox.weight, 1.0)
                vox.weight = max(0.0, vox.weight)
                if vox.surface == True:
                    self.vox_grid.check_exposed_faces(n)
        else:
            for n in range(len(self.vox_grid.voxels)):
                vox = self.vox_grid.voxels[n]
                if vox.type < 0:
                    vox.weight = 0.0
                else:
                    vox.weight = 1.0
                if vox.surface == True:
                    self.vox_grid.check_exposed_faces(n)

        return surface_voxels
    
    # produce surface with marching cubes from corner grid
    def create_surface(self):
        print('Creating surface mesh...')
        cg = self.corner_grid
        corner_volumes = np.asarray([[[0.0]*cg.dims[0]]*cg.dims[1]]*cg.dims[2])
        for n in range(len(cg.corners)):
            ind = cg.get_indices(n)
            corner_volumes[ind[2]][ind[1]][ind[0]] = cg.corners[n].volume # marching cubes requires [z,y,x] order

        verts, faces, normals, values = marching_cubes(volume= corner_volumes, level=0.5)
        self.corner_volumes = corner_volumes
        self.verts = np.fliplr(verts) # marching_cubes() outputs in z,y,x order
        self.faces = faces
        # purging degenerates
        # 1. Points cannot be duplicates of each other
        # Create a KDTree for efficient nearest-neighbor lookup
        tree = cKDTree(self.verts)
        p_eps = 1e-7*Voxel.size # this is a small epsilon to determine if points are the 'same'
        duplicates = tree.query_pairs(p_eps)
        
        # Initialize duplicates array with -1 values
        # -1 not duplicate, otherwise index of what it duplicates
        dupes = np.full(len(self.verts), -1, dtype=int)

        # Union-Find data structure
        parent = np.arange(len(self.verts))
        def find(x):
            while x != parent[x]:
                parent[x] = parent[parent[x]]  # Path compression
                x = parent[x]
            return x
        def union(x, y):
            root_x = find(x)
            root_y = find(y)
            if root_x != root_y:
                parent[root_y] = root_x
        for i, j in duplicates:
            union(i, j)

        # Assign each point to its duplicate root, to make sure no duplicates are left
        for i in range(len(dupes)):
            root = find(i)
            if root != i:
                dupes[i] = root

        # replace all duplicate points with 'original' point
        revealed_faces = np.array([p if dupes[p] == -1 else dupes[p] for p in self.faces.flatten()])
        revealed_faces.resize((len(self.faces), 3))

        # 2. Triangles must have a set of 3 unique points
        revealed_faces = np.array([f for f in revealed_faces if len(set(f)) == 3])
        # reassign vertices after transformation
        # 3. Triangles cannot be degenerate (collinear)
        #       3a. separate degenerates from full triangles
        area_eps = 1e-8*Voxel.size # if area less than this, it's 'zero'              #### need to be changed according voxel resolution
        degen_tris = []
        degen_edges = []
        full_tris = []
        for f in revealed_faces:
            vs = self.verts[f]
            a = get_tri_area(vs)
            if a < area_eps:
                degen_tris.append(f)
                degen_edges.append(set(f[get_longest_side(vs)]))
            else:
                full_tris.append(f)
        degen_tris = np.array(degen_tris)
        degen_edges = np.array(degen_edges)
        full_tris = np.array(full_tris)
        #       3b. delete pairs of degenerates that share an edge
        dupes = np.zeros(len(degen_edges)).astype(int)
        for i in range(len(degen_edges)):
            if dupes[i] == 0:
                for j in range(i + 1, len(degen_edges)):
                    if dupes[j] == 0:
                        if degen_edges[i] == degen_edges[j]:
                            dupes[j] = 1
                            dupes[i] = 1
        degen_edges = degen_edges[dupes == 0] # if edge shared by two degens, no connectivity
        degen_tris = degen_tris[dupes == 0]   # issue, just delete it

        #       3c. repair connectivity for full triangles sharing an edge with a degenerate
        for i in range(len(degen_edges)):
            # roughly, to fix connectivity where quad ABCM has degen triangle ABC
            # and full triangle ACM, switch it instead to triangles ABM and BCM,
            # deleting ACM from the full_tris list
            de = degen_edges[i] # de is {A, C}
            for j in range(len(full_tris)): # f is triangle A,C,M
                f = full_tris[j]
                if de.issubset(f):
                    dt = degen_tris[i] # triangle A,B,C
                    dl = list(de)
                    A = dl[0]
                    C = dl[1]

                    M = f[~np.isin(f, dl)][0] # M is full tri vertex not shared by degen triangle
                    B = dt[~np.isin(dt, dl)][0] # B is degen tri vertex not shared by full triangle
                    new_tri1 = np.array([v if v != A else B for v in f])
                    new_tri2 = np.array([v if v != C else M for v in dt])

                    full_tris[j] = new_tri1 # replace ACM with new triangle
                    full_tris = np.append(full_tris, np.array([new_tri2]), axis=0) # append other new triangle
                    break

        self.faces = full_tris


        self.transform_surface()


    # marching_cubes() gives origin of (0,0,0) and cell size of 1; this rescales the surface properly
    def transform_surface(self):
        cg = self.corner_grid
        
        # scale vertices from a cell length of 1 to the proper cell length
        cell_len_array = np.array([cg.cell_length]*len(self.verts)).astype(float)
        self.verts *= cell_len_array
        
        # translate to proper coordinates
        translations = np.array([cg.lims[0]]*len(self.verts))
        self.verts += translations
    
    # produce surface with marching cubes from corner grid
    def create_surface2D(self):
        print('Creating surface mesh...')
        surf_lines = []
        for n in range(len(self.cell_grid.cells)):
            c_cell = self.cell_grid.cells[n]
            inside = np.array([c_cell.corners[i].inside == 1 for i in range(4)])
            if not(all(inside)) and not(all(~inside)):
                c_cell.set_topology()
                c_cell.interpolate()
                for bd in c_cell.borders:
                    surf_lines.append(bd)

        return surf_lines
    
    def convert_lines(self):
        # I know this is inefficient, I'm just trying to get something done quickly for 2D

        # collect all vertices into list and record vertex indices in each Line object
        vert_list = []
        for i in range(len(self.surf_lines)):
            ln = self.surf_lines[i]
            vert_list.append(ln.a)
            ln.vert_indices.append(2*i)
            vert_list.append(ln.b)
            ln.vert_indices.append(2*i + 1)

        # get rid of duplicate points and re-number vertex indices in Lines
        self.verts = []
        vert_ids = { i:-1 for i in range(len(vert_list))} # key is index in vert_list, value is (non-duped) index in self.verts
        for i in range(len(vert_list)):
            vert_candidate = vert_list[i]
            dupe_flag = False
            og_pt = None

            eps = 1e-6*min(self.cell_grid.cell_length)
            for j in range(len(self.verts)):
                if all(abs(self.verts[j] - vert_candidate) < eps):
                    og_pt = j
                    dupe_flag = True
                    break
            if dupe_flag:
                vert_ids[i] = og_pt
            else:
                self.verts.append(vert_candidate)
                vert_ids[i] = len(self.verts) - 1
        self.verts = np.array(self.verts)

        # re-index vertices in each Line and fill self.faces
        new_surf_lines = []
        for n in range(len(self.cell_grid.cells)):
            c_cell = self.cell_grid.cells[n]
            inside = np.array([c_cell.corners[i].inside == 1 for i in range(4)])
            if not(all(inside)) and not(all(~inside)):
                new_bds = []
                for bd in c_cell.borders:
                    bd.vert_indices[0] = vert_ids[bd.vert_indices[0]]
                    bd.vert_indices[1] = vert_ids[bd.vert_indices[1]]
                    if bd.vert_indices[0] != bd.vert_indices[1]:
                        new_bd = Line([self.verts[bd.vert_indices[0]], self.verts[bd.vert_indices[1]]])
                        new_bd.vert_indices = [bd.vert_indices[0], bd.vert_indices[1]]
                        self.faces.append(new_bd.vert_indices)
                        new_surf_lines.append(new_bd)
                        new_bds.append(new_bd)
                c_cell.borders = new_bds
        self.surf_lines = new_surf_lines
        self.faces = np.array(self.faces)

    # write surface of triangles to disk, the argument is the name of the file
    def write_surface(self, name):
        """! @warning Test warning for doxygen
        """
        print('Writing SPARTA file of surface...')
        surf_file = open(name, "w")
        surf_file.write('surf file from isthmus\n\n')
        surf_file.write('{p} points\n{t} triangles\n\nPoints\n\n'.format(p = len(self.verts), t = len(self.faces)))
        for i in range(len(self.verts)):
            surf_file.write('{b} {x} {y} {z}\n'.format(b = i + 1, x = self.verts[i][0], \
                                                       y = self.verts[i][1], z = self.verts[i][2]))
    
        # order of points is flipped so sparta marks inside and outside correctly (DON'T INVERT IN INPUT SCRIPT)
        surf_file.write('\nTriangles\n\n')
        for i in range(len(self.faces)):
            surf_file.write('{b} {p1} {p2} {p3}\n'.format(b = i + 1, p1 = self.faces[i][0] + 1, \
                                                        p2 = self.faces[i][1] + 1, p3 = self.faces[i][2] + 1)) 
        surf_file.close() 
        
    # write surface of triangles to disk, the argument is the name of the file
    def write_surface2D(self, name):
        print('Writing SPARTA file of 2D surface...')
        surf_file = open(name, "w")
        surf_file.write('surf file from isthmus\n\n')
        surf_file.write('{p} points\n{t} lines\n\nPoints\n\n'.format(p = len(self.verts), t = len(self.faces)))
        for i in range(len(self.verts)):
            surf_file.write('{b} {x} {y}\n'.format(b = i + 1, x = self.verts[i][0], \
                                                   y = self.verts[i][1]))
    
        # order of points is flipped so sparta marks inside and outside correctly (DON'T INVERT IN INPUT SCRIPT)
        surf_file.write('\nLines\n\n')
        for i in range(len(self.faces)):
            surf_file.write('{b} {p1} {p2}\n'.format(b = i + 1, p1 = self.faces[i][1] + 1, \
                                                     p2 = self.faces[i][0] + 1)) 
        surf_file.close()

    def write_triangle_voxels(self,call_no):
        directory = 'voxel_tri'
        if not os.path.exists(directory):
            os.makedirs(directory)
        f = open(os.path.join(directory, 'triangle_voxels_'+str(call_no)+'.dat'), 'w')
        f.write('{nt} total triangles\n\n'.format(nt = len(self.cell_grid.triangles)))
        for t in self.cell_grid.triangles:
            f.write('start id {ti}\n'.format(ti=t.id + 1))
            for v in range(len(t.voxel_ids)):
                f.write('    {vi} {svf}\n'.format(vi=t.voxel_ids[v], svf=t.voxel_scalar_fracs[v]))
            f.write('end id {ti}\n'.format(ti=t.id + 1))
        f.close()

    def write_line_voxels(self,call_no):
        directory = 'voxel_tri'
        if not os.path.exists(directory):
            os.makedirs(directory)
        f = open(os.path.join(directory, 'line_voxels_'+str(call_no)+'.dat'), 'w')
        f.write('{nt} total lines\n\n'.format(nt = len(self.surf_lines)))
        for i, t in enumerate(self.surf_lines):
            f.write('start id {ti}\n'.format(ti=i + 1))
            for v in range(len(t.voxel_ids)):
                f.write('    {vi} {svf}\n'.format(vi=t.voxel_ids[v], svf=t.voxel_scalar_fracs[v]))
            f.write('end id {ti}\n'.format(ti=i + 1))
        f.close()
    
    def get_surface_area(self):
        return mesh_surface_area(self.verts, self.faces)

#%% Grid class and derived classes

class Grid:
    def __init__(self, lims, dims):
        self.lims = lims    # grid domain limits, [[xlo,ylo,zlo], [xhi,yhi,zhi]]
        self.dims = dims # [nx, ny, nz], no. of elements in each direction
        self.ndims = len(dims)
        if self.ndims == 3:
            self.coords = [[],[],[]] # list of possible x,y, and z coordinates
        else:
            self.coords = [[],[]] # possible x and y coordinates
        
    # elements are stored in 1d arrays, so 3d indices (x,y,z) are fed here to get that 1d index
    def get_element(self, ind):
        # i,j,k are x,y, and z indices
        if self.ndims == 3:
            return ind[2]*self.dims[1]*self.dims[0] + ind[1]*self.dims[0] + ind[0]
        else:
            return ind[1]*self.dims[0] + ind[0]
    
    # get 3d coordinates (x,y,z) from 1d position in array
    def get_indices(self,n):
        ind = []
        # a,b,c are x,y, and z indices
        if self.ndims == 3:
            c = int(n/(self.dims[1]*self.dims[0]))
            ind.insert(0, c)
            n = n % (self.dims[1]*self.dims[0])
        b = int(n/self.dims[0])
        ind.insert(0, b)
        n = n % self.dims[0]
        a = n
        ind.insert(0, a)
        return ind
    
    # see if indices are valid for ind = [i,j,k] for x,y,z
    def valid_element(self, ind):
        for i in range(self.ndims):
            if (ind[i] < 0):
                return False
            if not (ind[i] < self.dims[i]):
                return False
        return True
  
class Voxel_Grid(Grid):
    def __init__(self, lims, dims):
        super().__init__(lims, dims)
        if self.ndims == 3:
            self.voxels = np.array([[[Voxel(self.lims[0] + Voxel.size*np.array([i,j,k]), i, j, k, self.get_element([i,j,k])) \
                                    for i in range(self.dims[0])] for j in range(self.dims[1])]  for k in range(self.dims[2])])
        else:
            self.voxels = np.array([[Voxel2D(self.lims[0] + Voxel.size*np.array([i,j]), i, j, self.get_element([i,j])) \
                                    for i in range(self.dims[0])] for j in range(self.dims[1])])            
        self.voxels = self.voxels.flatten()
        for i in range(self.ndims): # possible x,y,z corner coordinates
            self.coords[i] = list(np.linspace(self.lims[0][i], self.lims[1][i], self.dims[i]))
        
    def check_surrounded_solid(self, n):
        # check 6 (4) cardinal neighbors, initially assume surrounded by voxels
        vox = self.voxels[n]
        surrounded = True
        for d in range(self.ndims):
            for s in [-1, 1]:
                ind = self.get_indices(n)
                ind[d] += s
                if self.valid_element(ind):
                    neighbor = self.voxels[self.get_element(ind)]
                    if neighbor.type < vox.type:
                        surrounded = False
                        break
            if not(surrounded):
                break

        if surrounded:
            vox.type += 1
        else:
            vox.finalized = True
                
    def check_surrounded_void(self, n):
        # check 6 (4) cardinal neighbors, initially assume surrounded by voids
        vox = self.voxels[n]
        surrounded = True
        for d in range(self.ndims):
            for s in [-1, 1]:
                ind = self.get_indices(n)
                ind[d] += s
                if self.valid_element(ind):
                    neighbor = self.voxels[self.get_element(ind)]
                    if neighbor.type > vox.type:
                        surrounded = False
                        break
            if not(surrounded):
                break

        if surrounded:
            vox.type -= 1
        else:
            vox.finalized = True

    def check_exposed_faces(self, n):
        vox = self.voxels[n]
        f_count = 0
        for d in range(self.ndims):
            for s in [-1, 1]:
                ind = self.get_indices(n)
                ind[d] += s
                if self.valid_element(ind):
                    neighbor = self.voxels[self.get_element(ind)]
                    if neighbor.type < 0:
                        vox.faces[f_count].exposed = True
                else:
                    vox.faces[f_count].exposed = True
                f_count += 1

# grid of corners, used to feed volume fractions to marching cubes function
class Corner_Grid(Grid):
    """
    Represents a 3D grid of corner points where each corner stores information about voxel assignments and volume fractions. 
    
    Attributes:
    ----------
    lims: 
    A 2x3 numpy array representing the domain limits of the grid in 3D space. 
    It contains the lower (lims[0]) and upper (lims[1]) bounds for the x, y, and z directions.

    dims: 
    A numpy array representing the number of cells along each dimension (x, y, z). 
    This helps define how many corner points are in the grid.

    cell_length: 
    This represents the distance between corners along each dimension (x, y, z). 
    It is computed from the domain limits and dimensions.

    corners: 
    A list containing instances of the MC_Corner class. 
    Each MC_Corner represents a corner point in the 3D grid and contains information like position and volume.

    coords: 
    A list containing all possible coordinates for the x, y, and z dimensions, defining the positions of all corners within the grid.
    """
    def __init__(self, lims, dims, vox_grid):
        print('Dividing voxel volumes for surface creation...')
        super().__init__(lims, dims)
        self.cell_length = (self.lims[1] - self.lims[0])/(self.dims - 1) # length of cell in [x,y,z] directions
        if self.ndims == 3:
            self.corners = np.array([[[MC_Corner(self.lims[0] + self.cell_length*[i,j,k], i, j, k) \
                                    for i in range(dims[0])] for j in range(dims[1])]  for k in range(dims[2])])
        else:
            self.corners = np.array([[MC_Corner2D(self.lims[0] + self.cell_length*[i,j], i, j) \
                                    for i in range(dims[0])] for j in range(dims[1])])            
        self.corners = self.corners.flatten()
        for i in range(self.ndims): # possible x,y,z corner coordinates
            self.coords[i] = list(np.linspace(self.lims[0][i], self.lims[1][i], self.dims[i]))
        self.associate_voxels(vox_grid) # assign all voxels to owning corner and divide volumes        

    ## @defgroup Associators
    ## @{
    def associate_voxels(self, vox_grid):
        voxels = vox_grid.voxels
        i = 0
        for v in voxels:
             ind = (np.rint(((v.position - self.lims[0])/self.cell_length))).astype(int) # x,y,z corner indices
             self.corners[self.get_element(ind)].voxels.append(v)
             progress_bar(i+1, len(voxels), 'assigning voxels to corners')
             i += 1
        self.divide_volumes() # divide volumes between corners
    ## @}

    def divide_volumes(self):
        max_unique_dist = 0.5*(self.cell_length - Voxel.size)
        self.active_flags = [] # flags to divide penetrating voxel volumes
        if self.ndims == 3:
            voxel_volume = Voxel.size**3
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        self.active_flags.append([i,j,k])
        else:
            voxel_volume = Voxel.size**2
            for i in range(2):
                for j in range(2):
                    self.active_flags.append([i,j])
        self.active_flags = np.array(self.active_flags)

        for i in range(len(self.corners)):
            c = self.corners[i]
            weighted_voxs = [vox for vox in c.voxels if vox.weight > 1e-6]
            for v in weighted_voxs:
                displ = v.position - c.position
                dist = np.array([abs(x) for x in displ])
                if all(dist < max_unique_dist):
                    c.volume += voxel_volume*v.weight
                else: # if voxel is not fully inside one corner's space, divide between corners
                    self.divide_voxel(c, displ, max_unique_dist, v.weight)
            progress_bar(i+1, len(self.corners), 'dividing voxel volumes')

        prd = np.prod(self.cell_length) # volume (area) of corner region
        for c in self.corners:
            c.volume /= prd
            if c.volume >= 0.5:
                c.inside = 1
            else:
                c.inside = 0

            
    # divide voxel volume between multiple corners
    def divide_voxel(self, c, diff, min_pen_distance, v_weight):
        penetration = np.array([abs(diff[i]) - min_pen_distance[i] for i in range(self.ndims)])
        pen_flag = np.zeros(self.ndims).astype(int) # flag for penetration in [x,y,z]; 0 if none, 1 if in positive direction, -1 if negative
        for i in range(self.ndims):
            if (penetration[i] > 0):
                pen_flag[i] = 1 if diff[i] > 0 else -1
            else:
                penetration[i] = 0
        
        # 0 means inside current corner region, 1 means penetration region
        # i.e. i,j,k = [0,0,0] means current corner region, i,j,k = [1,1,1] means all penetration
        for active_flag in self.active_flags:
            c_lengths = [(penetration[m] if active_flag[m] else Voxel.size - penetration[m]) for m in range(self.ndims)]
            vol_index = c.indices + active_flag*pen_flag
            self.corners[self.get_element(vol_index)].volume += np.prod(c_lengths)*v_weight

# this is where voxels and triangles are located and connected to each other
class Cell_Grid(Grid):
    neighbor_increments = np.array([])
    for z in range(-1, 2):
        for y in range(-1, 2):
            for x in range(-1,2):
                neighbor_increments = np.append(neighbor_increments, [x,y,z])
    neighbor_increments = np.reshape(neighbor_increments, (27, 3)).astype(int)
    
    def __init__(self, lims, dims, surface_voxels, faces, verts, gpu=False):
        print('Associating voxels to surface triangles...')
        super().__init__(lims, dims)
        self.gpu = gpu
        self.cell_length = (self.lims[1] - self.lims[0])/self.dims # length of cell in [x,y,z] directions
        self.cells = np.array([MC_Cell(i) for i in range(np.prod(dims))])
        for i in range(3): # center of cell
            self.coords[i] = list(np.linspace(self.lims[0][i] + 0.5*self.cell_length[i], \
                                              self.lims[1][i] - 0.5*self.cell_length[i], self.dims[i]))
             
        self.associate_voxels(surface_voxels) # associate voxels on surface to cells
        self.associate_triangles(faces, verts)   # associate triangles to cells
        self.voxels_to_triangles()

        
    ## @defgroup Associators
    ## These functions determine which cells to place given points in and places them
    ## in appropriate groups
    ## @{
    # associate edge voxels to cells
    def associate_voxels(self, surface_voxels):
        for v in surface_voxels:
            ind = ((v.position - self.lims[0])/self.cell_length).astype(int) # x,y,z corner indices
            self.cells[self.get_element(ind)].surface_voxels.append(v)

    # associate triangles to cells
    ## @return tri_cell_ids an array of cell ids indexed by face indices
    def associate_triangles(self, faces, verts):
        self.triangles = []
        for i in range(len(faces)):
            centroid = np.average(verts[faces[i][:]], axis=0)
            ind = ((centroid - self.lims[0])/self.cell_length).astype(int) # cell indices [x,y,z]
            n = self.get_element(ind)
            self.triangles.append(Triangle(verts[faces[i][:]], i, n))
            self.cells[n].triangles.append(self.triangles[-1])
    ## @}    
    
    # associate each triangle to voxels based on inward normal view of voxel faces
    def voxels_to_triangles(self):        
        proj_fs = []    # projected faces
        tri_normal = []
        tri_plane_normal = []
        tri_vertices = []
        tri_epsilon = []
        c_voxels = []

        print('    associating voxels...')
        # First assign voxels to each triangle in each cell
        for c in self.cells:
            if len(c.triangles):
                # collect all voxels in current and neighboring cells
                ind = self.get_indices(c.id)
                c_voxels.append([])
                for ni in Cell_Grid.neighbor_increments:
                    n_ind = ind + ni
                    if (self.valid_element(n_ind)):
                        c_voxels[-1] += self.cells[self.get_element(n_ind)].surface_voxels
                
                # project eligible exposed voxel faces onto triangle plane
                for t in c.triangles:
                    for vox in c_voxels[-1]:
                        for f in vox.faces:
                            if (f.exposed) and (np.dot(f.n, t.normal) > 0):
                                    proj_fs.append(np.array([f.xs[i] - t.normal*np.dot(t.normal, f.xs[i] - t.vertices[0]) for i in range(4)]))
                                    tri_normal.append(t.normal)
                                    tri_plane_normal.append(t.plane_normal)
                                    tri_vertices.append(t.vertices)
                                    tri_epsilon.append(t.epsilon)

        # Find area of overlap between projected faces and triangles
        # Use GPU or CPU version based on the flag
        if self.gpu:
            print("        Running get_intersection_area on GPU...")
            v_areas = get_intersection_area_gpu(proj_fs, tri_normal, tri_plane_normal, tri_vertices, tri_epsilon)
        else:
            v_areas = get_intersection_area(proj_fs, tri_normal, tri_plane_normal, tri_vertices, tri_epsilon)

        # Collect voxel face areas together
        area_idx = 0
        c_idx = 0
        for c in self.cells:
            if len(c.triangles):
                for t in c.triangles:
                    t_area = get_tri_area(t.vertices)
                    for vox in c_voxels[c_idx]:
                        for f in vox.faces:
                            if f.exposed and np.dot(f.n, t.normal) > 0:
                                v_id = vox.oid
                                # Assign the computed area to the voxel-triangle pair
                                if v_id in t.voxel_ids:
                                    ind = t.voxel_ids.index(v_id)
                                    t.voxel_scalar_fracs[ind] += v_areas[area_idx]
                                else:
                                    if v_areas[area_idx] > t_area * 1e-6:  # Ensure it's significant
                                        t.voxel_ids.append(v_id)
                                        t.voxel_scalar_fracs.append(v_areas[area_idx])
                                area_idx += 1  # Move to next computed area
                c_idx += 1

        # Normalize scalar fractions by total voxel face area intercepted by the triangle
        low_area = 0
        for t in self.triangles:
            t.voxel_scalar_fracs = np.array(t.voxel_scalar_fracs)
            total_area = t.voxel_scalar_fracs.sum()
            if (total_area < 1e-6*get_tri_area(t.vertices)):                       #### need to be changed according voxel resolution   
                low_area += 1
                print('Uh oh, no voxel face area available for this triangle')
                print(t.vertices)
                print(t.voxel_scalar_fracs)
                print()
            t.voxel_scalar_fracs = t.voxel_scalar_fracs / total_area
        if (low_area):
            print('WARNING: {:.1f} % of triangles have (near-)zero area intersected by voxel faces'.format(100*low_area/len(self.triangles)))

class Cell_Grid2D(Grid):
    neighbor_increments = np.array([])
    for y in range(-1, 2):
        for x in range(-1,2):
            neighbor_increments = np.append(neighbor_increments, [x,y])
    neighbor_increments = np.reshape(neighbor_increments, (9, 2)).astype(int)
    
    def __init__(self, lims, dims, corner_grid):
        super().__init__(lims, dims)
        self.cell_length = (self.lims[1] - self.lims[0])/self.dims # length of cell in [x,y] directions
        cg = corner_grid.corners
        self.cells = []
        for j in range(dims[1]):
            c_line = []
            for i in range(dims[0]):
                c_cg = [cg[corner_grid.get_element([i,j])], cg[corner_grid.get_element([i+1,j])], cg[corner_grid.get_element([i+1,j+1])], cg[corner_grid.get_element([i,j+1])]]
                c_line.append(MC_Cell2D(c_cg, i, j, self.get_element([i,j])))
            self.cells.append(c_line)
        
        self.cells = np.array(self.cells).flatten()
        for i in range(self.ndims): # center of cell
            self.coords[i] = list(np.linspace(self.lims[0][i] + 0.5*self.cell_length[i], \
                                              self.lims[1][i] - 0.5*self.cell_length[i], self.dims[i]))
    
    def associate_voxels(self, surface_voxels):
        for v in surface_voxels:
            ind = ((v.position - self.lims[0])/self.cell_length).astype(int) # x,y,z corner indices
            self.cells[self.get_element(ind)].surface_voxels.append(v)

    # associate each triangle to voxels based on inward normal view of voxel faces
    def voxels_to_edges(self, all_surface_voxels, sys_surf_lines):
        # associate surface voxels to each cell
        self.associate_voxels(all_surface_voxels)
        
        proj_fs = []    # projected faces
        tris = []
        c_voxels = []

        # First assign voxels to each triangle in each cell
        for c in self.cells:
            if len(c.borders):
                # collect all voxels in current and neighboring cells
                ind = self.get_indices(c.id)
                c_voxels.append([])
                for ni in Cell_Grid2D.neighbor_increments:
                    n_ind = ind + ni
                    if (self.valid_element(n_ind)):
                        c_voxels[-1] += self.cells[self.get_element(n_ind)].surface_voxels
                
                # project eligible exposed voxel faces onto triangle plane
                for t in c.borders:
                    ntheta = t.theta - np.pi/2 # outward normal of border edge
                    t_norm = np.array([np.cos(ntheta), np.sin(ntheta)])
                    for vox in c_voxels[-1]:
                        for f in vox.faces:
                            if (f.exposed) and (np.dot(f.n, t_norm) > 0):
                                    proj_fs.append(Line([f.xs[i] - t_norm*np.dot(t_norm, f.xs[i] - t.a) for i in range(2)]))
                                    tris.append(t)

        v_areas = get_intersection_length(proj_fs, tris)

        # Collect voxel face areas together
        area_idx = 0
        c_idx = 0
        for c in self.cells:
            if len(c.borders):
                for t in c.borders:
                    ntheta = t.theta - np.pi/2 # outward normal of border edge
                    t_norm = np.array([np.cos(ntheta), np.sin(ntheta)])
                    for vox in c_voxels[c_idx]:
                        for f in vox.faces:
                            if f.exposed and np.dot(f.n, t_norm) > 0:
                                v_id = vox.oid
                                # Assign the computed area to the voxel-triangle pair
                                if v_id in t.voxel_ids:
                                    ind = t.voxel_ids.index(v_id)
                                    t.voxel_scalar_fracs[ind] += v_areas[area_idx]
                                else:
                                    if v_areas[area_idx] > t.length * 1e-6:  # Ensure it's significant
                                        t.voxel_ids.append(v_id)
                                        t.voxel_scalar_fracs.append(v_areas[area_idx])
                                area_idx += 1  # Move to next computed area
                c_idx += 1

        # Normalize scalar fractions by total voxel face area intercepted by the triangle
        low_area = 0
        for t in sys_surf_lines:
            t.voxel_scalar_fracs = np.array(t.voxel_scalar_fracs)
            total_area = t.voxel_scalar_fracs.sum()
            if (total_area < 1e-6*np.linalg.norm(t.b - t.a)):  
                low_area += 1
                print('Uh oh, no voxel face area available for this triangle')
                print(t.a)
                print(t.b)
                print(t.voxel_scalar_fracs)
                print()
            t.voxel_scalar_fracs = t.voxel_scalar_fracs / total_area
        if (low_area):
            print('WARNING: {:.1f} % of triangles have (near-)zero area intersected by voxel faces'.format(100*low_area/len(sys_surf_lines)))

#%% Miscellaneous Functions

def progress_bar(c, total, message):
    finished = 0
    if np.ceil(100*(c + 1)/total) == progress_bar.c_decade:
        sys.stdout.write('\r')
        finished = np.rint(10*(c + 1)/total).astype(int)
        sys.stdout.write('    ' + message + '... [' + '='*finished + ' '*(10 - finished) + ']')
        sys.stdout.flush()
        progress_bar.c_decade += 10
    if finished == 10:
        sys.stdout.write('\n')
        progress_bar.c_decade = 10
progress_bar.c_decade = 10

# generate a factor of +/- mag% to a number for testing 
def noise_gen(mag):
    return (2*np.random.rand() - 1)*(mag/100) + 1