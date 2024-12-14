import numpy as np
import os
import sys
import time
import copy
from Marching_Cubes import marching_cubes, mesh_surface_area
from scipy.spatial import cKDTree

# need csv, dev, grids, voxel_data, and voxel_tri folders

#%% Individual geometric elements used in grids
class Triangle:
    def __init__(self, verts, identity, ncell):
        self.vertices = verts  # [[x1, y1, z1], [x2,y2,z2], [x3,y3,z3]]
        self.id = identity     # own triangle index
        self.cell = ncell      # owning cell object
        self.voxel_ids = []    # index of each owned voxel in global array
        self.s_voxel_ids = []  # index of owned voxel in surface array
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

        epsilon = 1e-3*Surface_Voxel.size
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
        
    def get_intersection_area(self, proj_face):
        clipped_points = clip_sh(proj_face, self) # find overlapping area
        if len(clipped_points) < 3:
            return 0
        rotated_points = orient_polygon_xy(clipped_points, self.normal) # rotate overlap polygon into xy plane
        return polygon_area(rotated_points) # get area with shoelace formula

# Sutherland-Hodgman polygon clipping
# inputs are vertices of subject (to be clipped) and vertices
# of window (the clipper)
def clip_sh(subject, clip_tri):
    # clipping operation
    in_pts = subject
    out_pts = []
    for i in range(len(clip_tri.vertices)):
        clip_edge = clip_tri.vertices[i] - clip_tri.vertices[i - 1]
        plane_normal = np.cross(clip_edge, clip_tri.normal)
        plane_normal /= np.linalg.norm(plane_normal)
        
        out_pts = []

        for j in range(len(in_pts)):
            p1 = in_pts[j - 1]
            p2 = in_pts[j]

            # compute intersection with infinite edge
            p1_in, p2_in, intersect = segment_plane_intersection(p1, p2, plane_normal, clip_tri.vertices[i], clip_tri.epsilon)

            if (p2_in):
                if (not p1_in):
                    out_pts.append(intersect)
                out_pts.append(p2)
            elif (p1_in):
                out_pts.append(intersect)
            # if p1 and p2 both outside, do nothing, delete line segment

        in_pts = out_pts

    final_pts = []
    for i in range(len(out_pts)):
        dupe = False
        for j in range(i + 1, len(out_pts)):
            if (all(abs(out_pts[j] - out_pts[i]) < clip_tri.epsilon)):
                dupe = True
                break
        if not dupe:
            final_pts.append(out_pts[i])

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
    # if one in and other out,
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

class Surface_Voxel:
    size = 0
    def __init__(self, xs, index):
        self.position = xs
        low_corner = np.array(xs - 0.5*Surface_Voxel.size)
        # corners: first four are zlo x-y plane, then last four are zhi x-y plane
        cs  =  [low_corner + np.array([0,0,0])*Surface_Voxel.size,
                low_corner + np.array([1,0,0])*Surface_Voxel.size,
                low_corner + np.array([0,1,0])*Surface_Voxel.size,
                low_corner + np.array([1,1,0])*Surface_Voxel.size,
                low_corner + np.array([0,0,1])*Surface_Voxel.size,
                low_corner + np.array([1,0,1])*Surface_Voxel.size,
                low_corner + np.array([0,1,1])*Surface_Voxel.size,
                low_corner + np.array([1,1,1])*Surface_Voxel.size
                ]
        # axis-aligned bounding box
        self.lo = cs[0]
        self.hi = cs[-1]
        self.id = index   # own voxel index
        self.surf_id = Surface_Voxel.n_svoxels # voxel index in surface voxel list
        Surface_Voxel.n_svoxels += 1
        self.triangle_ids = [] # index of each owned triangle
        self.closest_triangle_id = -1 # initialize to invalid id
        self.closest_triangle_dist = 1e12 # initialize to massive number
        self.closest_triangle_cell = -1 # initialize to invalid id
        self.faces = []
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

# this is the unit cell of the marching cubes grid, with position, owned voxels,
# and owned triangles
class MC_Cell:
    def __init__(self, i):
        self.surface_voxels = [] # surface voxel objects owned by this cell
        self.triangles = [] # triangles owned by this cell
        self.id = i       # index in cell_grid

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

    """
    def __init__(self, lims, ncells, voxel_size, voxels, name, call_no):
        print('Executing marching cubes...')

        Surface_Voxel.n_svoxels = 0
        self.verts = []
        self.faces = []
        
        # remove surface file if exists so if routine fails, error will occur in calling program
        if os.path.exists(name):
            os.remove(name)
        
        # check validity of grid being created and voxel data
        self.check_grid(lims, ncells)
        self.check_voxels(lims, ncells, voxel_size, np.transpose(voxels))
        
        # initialize system variables
        self.voxels = voxels
        Surface_Voxel.size = voxel_size
        
        # prepare marching cubes volume grid, and create mesh
        self.corner_grid = Corner_Grid(lims, ncells + 1, self.voxels, Surface_Voxel.size)       
        self.create_surface(voxel_size)
        
        # write SPARTA-compliant surface
        self.write_surface(name)
        
        # find voxels on the surface and organize these surface voxels and triangles into cells
        self.surface_voxels = self.sort_voxels()
        self.cell_grid = Cell_Grid(lims, ncells, self.surface_voxels, self.faces, self.verts)
        
        # associate voxels to triangles
        self.write_triangle_voxels(call_no)
        
    ## check validity of grid limits and number of cells
    def check_grid(self, lims, ncells):
        if (lims.shape != (2,3)):
            raise Exception("Invalid grid limits given")
            
        if (ncells.shape != (3,)):
            raise Exception("Invalid numbers of grid cells given")
            
        for i in range(3):
            if (lims[1][i] <= lims[0][i]):
                raise Exception("Invalid grid limits given (limits inverted)")
            if (not np.issubdtype(ncells[i], np.integer)):
                raise Exception("Numbers of grid cells must be integers")
            
    # check validity of voxel positions and size
    def check_voxels(self, lims, ncells, voxel_size, positions):
        cell_length = (lims[1] - lims[0])/ncells # length of cell in [x,y,z] directions
        if (any(cell_length < voxel_size)):
            raise Exception("Voxel size {:.2e} is larger than marching cubes grid cell dimension(s) {:.2e} {:.2e} {:.2e} ".format( \
                            voxel_size, cell_length[0], cell_length[1], cell_length[2]))
            
        if (not voxel_size > 0):
            raise Exception("Voxel size is invalid")
        
        if (len(positions) != 3):
            raise Exception("Invalid voxel coordinates given")
        
        for i in range(3):
            # i is 0,1,2 for x,y, or z; positions is list of coordinates
            border = lims[0][i] + 0.05*(voxel_size + cell_length[i])
            if (any(x < border for x in positions[i])):
                raise Exception("Voxel(s) outside of acceptable grid space")
            border = lims[1][i] - 0.05*(voxel_size + cell_length[i])
            if (any(x > border for x in positions[i])):
                raise Exception("Voxel(s) outside of acceptable grid space")
    
    # voxels are [[x1,y1,z1], [x2,y2,z2],...]
    def sort_voxels(self):
        print('Sorting which voxels are on the surface...')
        # initialize voxels and limits of voxel grid to be used
        voxs = self.voxels
        v_lims = np.concatenate((np.min(voxs, axis=0), np.max(voxs, axis=0))).reshape((2,3))
        diff = np.rint((v_lims[1] - v_lims[0])/Surface_Voxel.size).astype(int) # closest regular grid distance from origin
        v_lims[1] = v_lims[0] + Surface_Voxel.size*diff
        
        # create voxel space grid, -1 if nothing, vox id if something
        vox_grid = Voxel_Grid(v_lims, diff + 1)
        
        # populate voxel space
        vox_elno = (np.ones(len(voxs))*-1).astype(int)
        for i in range(len(voxs)):
            ind = np.rint((voxs[i] - v_lims[0])/Surface_Voxel.size).astype(int)
            n = vox_grid.get_element(ind[0],ind[1],ind[2])
            vox_elno[i] = n
            if (vox_grid.vox_space[n] != -1):
                print('WARNING: overwriting voxel with another in same position')
            vox_grid.vox_space[n] = i
        
        # now, based on neighbors, get voxel types for 1st layer of surface
        surface_voxels = []
        for v in range(len(voxs)):
            found_flag = 0
            for d in range(3):
                for i in range(-1, 2, 2):
                    ind = list(vox_grid.get_indices(vox_elno[v]))
                    ind[d] += i
                    if (vox_grid.valid_element(ind)):
                        other_ind = int(vox_grid.get_element(ind[0],ind[1],ind[2]))
                        if (vox_grid.vox_space[other_ind] == -1):
                            surface_voxels.append(Surface_Voxel(voxs[v], v))
                            found_flag = 1
                    else:
                        surface_voxels.append(Surface_Voxel(voxs[v], v))
                        found_flag = 1
                    if found_flag:
                        break
                if found_flag:
                    break
            progress_bar(v+1, len(voxs), 'finding surface voxels')

        # for found surface voxels, determine which faces are exposed
        for sv in surface_voxels:
            v = sv.id
            counter = 0
            for d in range(3): # x, y, z directions
                for i in range(-1, 2, 2): # negative and positive neighbor
                    ind = list(vox_grid.get_indices(vox_elno[v]))
                    ind[d] += i
                    if (vox_grid.valid_element(ind)):
                        other_ind = int(vox_grid.get_element(ind[0],ind[1],ind[2]))
                        if (vox_grid.vox_space[other_ind] == -1):
                            sv.faces[counter].exposed = True
                    else:
                        sv.faces[counter].exposed = True
                    counter += 1
            progress_bar(v+1, len(voxs), 'finding exposed voxel faces')

        return surface_voxels
    
    # produce surface with marching cubes from corner grid
    def create_surface(self,voxel_size):
        print('Creating surface mesh...')
        cg = self.corner_grid
        corner_volumes = np.asarray([[[0.0]*cg.dims[0]]*cg.dims[1]]*cg.dims[2])
        for n in range(len(cg.corners)):
            a,b,c = cg.get_indices(n)
            corner_volumes[c][b][a] = cg.corners[n].volume # marching cubes requires [z,y,x] order

        verts, faces, normals, values = marching_cubes(volume= corner_volumes, level=0.5)
        self.corner_volumes = corner_volumes
        self.verts = np.fliplr(verts) # marching_cubes() outputs in z,y,x order
        self.faces = faces
        # purging degenerates
        # 1. Points cannot be duplicates of each other
        # Create a KDTree for efficient nearest-neighbor lookup
        tree = cKDTree(self.verts)
        p_eps = 1e-7*voxel_size # this is a small epsilon to determine if points are the 'same'
        duplicates = tree.query_pairs(p_eps)
        
        # Initialize duplicates array with -1 values
        # -1 not duplicate, otherwise index of what it duplicates
        dupes = np.full(len(self.verts), -1, dtype=int)
        for i, j in duplicates:
            dupes[j] = i
        # replace all duplicate points with 'original' point
        revealed_faces = np.array([p if dupes[p] == -1 else dupes[p] for p in self.faces.flatten()])
        revealed_faces.resize((len(self.faces), 3))

        # 2. Triangles must have a set of 3 unique points
        revealed_faces = np.array([f for f in revealed_faces if len(set(f)) == 3])
        # reassign vertices after transformation
        # 3. Triangles cannot be degenerate (collinear)
        #       3a. separate degenerates from full triangles
        area_eps = 1e-8*voxel_size # if area less than this, it's 'zero'              #### need to be changed according voxel resolution
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
    
    # write surface of triangles to disk, the argument is the name of the file
    def write_surface(self, name):
        """! @warning Don't use naughty words for the filename or Vikram will be mad
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
        
    def write_triangle_voxels(self,call_no):
        f = open('voxel_tri/triangle_voxels_'+str(call_no)+'.dat', 'w')
        f.write('{nt} total triangles\n\n'.format(nt = len(self.cell_grid.triangles)))
        for t in self.cell_grid.triangles:
            f.write('start id {ti}\n'.format(ti=t.id + 1))
            for v in range(len(t.voxel_ids)):
                f.write('    {vi} {svf}\n'.format(vi=t.voxel_ids[v], svf=t.voxel_scalar_fracs[v]))
            f.write('end id {ti}\n'.format(ti=t.id + 1))
        f.close()
    
    def get_surface_area(self):
        return mesh_surface_area(self.verts, self.faces)

#%% Grid class and derived classes

class Grid:
    def __init__(self, lims, dims):
        self.lims = lims    # grid domain limits, [[xlo,ylo,zlo], [xhi,yhi,zhi]]
        self.dims = dims # [nx, ny, nz], no. of elements in each direction
        self.coords = [[],[],[]] # list of possible x,y, and z coordinates
        
    # elements are stored in 1d arrays, so 3d indices (x,y,z) are fed here to get that 1d index
    def get_element(self,i,j,k):
        # i,j,k are x,y, and z indices
        return k*self.dims[1]*self.dims[0] + j*self.dims[0] + i
    
    # get 3d coordinates (xky,z) from 1d position in array
    def get_indices(self,n):
        # a,b,c are x,y, and z indices
        c = int(n/(self.dims[1]*self.dims[0]))
        n = n % (self.dims[1]*self.dims[0])
        b = int(n/self.dims[0])
        n = n % self.dims[0]
        a = n
        return a,b,c
    
    # see if indices are valid for ind = [i,j,k] for x,y,z
    def valid_element(self, ind):
        for i in range(3):
            if (ind[i] < 0):
                return False
            if not (ind[i] < self.dims[i]):
                return False
        return True
  
class Voxel_Grid(Grid):
    def __init__(self, lims, dims):
        super().__init__(lims, dims)
        self.vox_space = np.ones(np.prod(dims))*-1   
        
    
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
    def __init__(self, lims, dims, voxels, voxel_size):
        print('Dividing voxel volumes for surface creation...')
        super().__init__(lims, dims)
        self.cell_length = (self.lims[1] - self.lims[0])/(self.dims - 1) # length of cell in [x,y,z] directions
        self.corners = np.array([[[MC_Corner(self.lims[0] + self.cell_length*[i,j,k], i, j, k) \
                                   for i in range(dims[0])] for j in range(dims[1])]  for k in range(dims[2])])
        self.corners = self.corners.flatten()
        for i in range(3): # possible x,y,z corner coordinates
            self.coords[i] = list(np.linspace(self.lims[0][i], self.lims[1][i], self.dims[i]))
        self.associate_voxels(voxels, voxel_size) # assign all voxels to owning corner and divide volumes        

    ## @defgroup Associators mwah mwah
    ## @note jenny come back the kids miss you
    ## @{
    def associate_voxels(self, voxels, voxel_size): 
        i = 0
        for v in voxels:
             ind = (np.rint(((v - self.lims[0])/self.cell_length))).astype(int) # x,y,z corner indices
             self.corners[self.get_element(ind[0],ind[1],ind[2])].voxels.append(v)
             progress_bar(i+1, len(voxels), 'finding surface voxels')
             i += 1
        self.divide_volumes(voxel_size) # divide volumes between corners
    ## @}

    def divide_volumes(self, voxel_size):
        max_unique_dist = 0.5*(self.cell_length - voxel_size)
        voxel_volume = voxel_size**3
        i = 0
        for c in self.corners:
            if len(c.voxels):
                for v in c.voxels:
                    displ = v - c.position
                    dist = np.array([abs(x) for x in displ])
                    if all(dist <= max_unique_dist):
                        c.volume += voxel_volume
                    else: # if voxel is not fully inside one corner's space, divide between corners
                        self.divide_voxel(c, v, voxel_size, max_unique_dist)
            progress_bar(i+1, len(self.corners), 'dividing voxel volumes')
            i += 1

        prd = np.prod(self.cell_length)
        for c in self.corners:
            c.volume /= prd
            
    # divide voxel volume between multiple corners
    def divide_voxel(self, c, v, voxel_size, mud):
        diff = v - c.position # distance between voxel center and corner in each dimension
        min_pen_distance = mud # minimum distance in each dimension to penetrate into another corner
        
        penetration = np.array([abs(diff[i]) - min_pen_distance[i] for i in range(3)])
        pen_flag = np.array([0,0,0]) # flag for penetration in [x,y,z]; 0 if none, 1 if in positive direction, -1 if negative
        for i in range(3):
            if (penetration[i] > 0):
                pen_flag[i] = 1 if diff[i] > 0 else -1
            else:
                penetration[i] = 0
        
        # 0 means inside current corner region, 1 means penetration region
        # i.e. i,j,k = [0,0,0] means current corner region, i,j,k = [1,1,1] means all penetration
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    active_flag = np.array([i,j,k])
                    c_lengths = np.array([0,0,0])
                    c_lengths = [(penetration[m] if active_flag[m] else voxel_size - penetration[m]) for m in range(3)]
                    vol_index = c.indices + active_flag*pen_flag
                    self.corners[self.get_element(vol_index[0], vol_index[1], vol_index[2])].volume += np.prod(c_lengths)
        
# this is where voxels and triangles are located and connected to each other
class Cell_Grid(Grid):
    neighbor_increments = np.array([])
    for z in range(-1, 2):
        for y in range(-1, 2):
            for x in range(-1,2):
                neighbor_increments = np.append(neighbor_increments, [x,y,z])
    neighbor_increments = np.reshape(neighbor_increments, (27, 3)).astype(int)
    
    def __init__(self, lims, dims, surface_voxels, faces, verts):
        print('Associating voxels to surface triangles...')
        super().__init__(lims, dims)
        self.cell_length = (self.lims[1] - self.lims[0])/self.dims # length of cell in [x,y,z] directions
        self.cells = np.array([MC_Cell(i) for i in range(np.prod(dims))])
        for i in range(3): # center of cell
            self.coords[i] = list(np.linspace(self.lims[0][i] + 0.5*self.cell_length[i], \
                                              self.lims[1][i] - 0.5*self.cell_length[i], self.dims[i]))
             
        self.associate_voxels(surface_voxels) # associate voxels on surface to cells
        self.associate_triangles(faces, verts)   # associate triangles to cells
        self.voxels_to_triangles()

        
    ## @defgroup Associators these boys put data together like a matchmaker kissy kissy
    ## These functions determine which cells to place given points in and places them
    ## in appropriate groups
    ## @{
    # associate edge voxels to cells
    def associate_voxels(self, surface_voxels):
        self.surface_voxels = surface_voxels            
        for v in surface_voxels:
            ind = ((v.position - self.lims[0])/self.cell_length).astype(int) # x,y,z corner indices
            self.cells[self.get_element(ind[0],ind[1],ind[2])].surface_voxels.append(v)

    # associate triangles to cells
    ## @return tri_cell_ids an array of cell ids indexed by face indices
    def associate_triangles(self, faces, verts):
        self.triangles = []
        for i in range(len(faces)):
            centroid = np.average(verts[faces[i][:]], axis=0)
            ind = ((centroid - self.lims[0])/self.cell_length).astype(int) # cell indices [x,y,z]
            n = self.get_element(ind[0],ind[1],ind[2])
            self.triangles.append(Triangle(verts[faces[i][:]], i, n))
            self.cells[n].triangles.append(self.triangles[-1])
    ## @}    
    
    # associate each triangle to voxels based on inward normal view of voxel faces
    def voxels_to_triangles(self):        
        # first assign voxels to each triangle in each cell
        for c in self.cells:
            if len(c.triangles):
                # collect all voxels in current and neighboring cells
                ind = list(self.get_indices(c.id))
                c_voxels = []
                for ni in Cell_Grid.neighbor_increments:
                    n_ind = ind + ni
                    if (self.valid_element(n_ind)):
                        c_voxels += self.cells[self.get_element(n_ind[0], n_ind[1], n_ind[2])].surface_voxels
                
                # project eligible exposed voxel faces onto triangle plane and test for intersection
                for t in c.triangles:
                    v_ids = []
                    sv_ids = []
                    v_areas = []
                    t_area = get_tri_area(t.vertices)
                    for vox in c_voxels:
                        for f in vox.faces:
                            if (f.exposed):
                                if (np.dot(f.n, t.normal) > 0):
                                    proj_f = np.array([f.xs[i] - t.normal*np.dot(t.normal, f.xs[i] - t.vertices[0]) for i in range(4)])
                                    area = t.get_intersection_area(proj_f) # find area of overlap between projected face and triangle
                                    vox.triangle_ids.append(t.id)
                                    vox.triangle_ids = list(set(vox.triangle_ids)) # prevent duplicates in list
                                    v_ids.append(vox.id)
                                    sv_ids.append(vox.surf_id)
                                    v_areas.append(area)

                    # collect voxel face areas together
                    for i in range(len(v_ids)):
                        if (v_ids[i] in t.voxel_ids):
                            ind = t.voxel_ids.index(v_ids[i])
                            t.voxel_scalar_fracs[ind] += v_areas[i]
                        else:
                            if (v_areas[i] > t_area*1e-6):                      #### need to be changed according voxel resolution
                                t.voxel_ids.append(v_ids[i])
                                t.s_voxel_ids.append(sv_ids[i])
                                t.voxel_scalar_fracs.append(v_areas[i])

            progress_bar(c.id + 1, len(self.cells), 'associating voxels    ')

        # now normalize scalar fractions by total voxel face area intercepted by the triangle
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
