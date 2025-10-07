import numpy as np
import sys
from utilities import progress_bar
from geometry import get_intersection_area, get_intersection_length, get_tri_area

# Check if Numba is available, use isthmus_gpu
try:
    import numba
    from geometry_gpu import get_intersection_area_gpu
    numba_available = True
except ImportError:
    numba_available = False

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


class MC_Corner:
    """
    This class is for corners in the grid, i.e. each MC_Cell corresponds to
    8 MC_Corners

    Parameters
    ----------
    p: np.ndarray
        corner position.
    
    i, j, k: ints
        indices in grid.
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


class MC_Cell:
    """
    This class is for cells in the grid, i.e. each MC_Grid corresponds to
    1 MC_Cell

    Parameters
    ----------
    i: int
        index in grid.
    """
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



class Grid:
    """
    Base class for all grid types.

    Parameters
    ----------
    lims : array-like of shape (2, D)
        Lower/upper limits per dimension.
    dims : array-like of int, shape (D,)
        Number of voxels per dimension.
    """
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
    """
    A grid (box) of voxels. All voxels are initialized with self.oid=, self.type=-1, and self.weight=0.

    Parameters
    ----------
    lims : array-like of shape (2, D)
        Lower/upper limits per dimension.
    dims : array-like of int, shape (D,)
        Number of voxels per dimension.

    Attributes
    ----------
    voxels : np.ndarray
        Flattened array of voxel objects.
    coords : list[list[float]]
        Possible corner coordinates per dimension.
    """
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
    Represents a 3D/2D grid of corner points where each corner stores information about voxel assignments and volume fractions.

    Parameters
    ----------
    lims : array-like of shape (2, D)
        Lower/upper limits per dimension.
    dims : array-like of int, shape (D,)
        Number of voxels per dimension.
    vox_grid : Voxel_Grid
        The voxel grid associated with this corner grid.

    Attributes
    ----------
    cell_length: array-like of shape (D,)
        This represents the distance between corners along each dimension (x, y, z). 
        It is computed from the domain limits and dimensions.

    corners: list[MC_Corner]
        A list containing instances of the MC_Corner class. 
        Each MC_Corner represents a corner point in the 3D grid and contains information like position and volume.

    coords: list[list[float]]
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
        i = 0
        weighted_voxs = [vox for vox in vox_grid.voxels if vox.weight > 1e-6]
        for v in weighted_voxs:
             ind = (np.rint(((v.position - self.lims[0])/self.cell_length))).astype(int) # x,y,z corner indices
             self.corners[self.get_element(ind)].voxels.append(v)
             progress_bar(i+1, len(weighted_voxs), 'assigning voxels to corners')
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
            for v in c.voxels:
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


