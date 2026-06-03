import numpy as np
import os
from Marching_Cubes import marching_cubes, mesh_surface_area
from scipy.spatial import cKDTree
from grid import Voxel, Voxel2D, Corner_Grid, Cell_Grid, Cell_Grid2D, \
    Voxel_Grid, Line, numba_available
from geometry import get_tri_area, get_longest_side

#%% Main system class where all the magic happens
class marchingWindows:  
    """
    Welcome to the isthmus experience! This program assumes minimal overlap of pixels.

    Parameters
    ----------
    lims: array-like of shape (2, D)
        The bounding box enclosing the geometry. 
        A [lo, hi] array representing the domain limits of the grid in 3D space. 
    ncells: [nx, ny, nz] integers
        No. of cells in x, y, and z directions.
    voxel_size: float
        Voxel edge length.
    voxels: [[x, y, z], ...] ndarray
        Array of voxels positions.
    name: str
        Name of output surface file.
    call_no: int
        Call number to append with the output file that associate triangles to voxels. 
    gpu: bool
        If True, use GPU for calculations.
    weight: bool
        If True, use the weighting method. It weights the voxels by their position from nearest edge voxel.
    ndims: int
        Number of dimensions (2 or 3).

    Attributes
    ----------
    corner_volumes: ndarray
        D array of corner volumes used for marching cubes.
    verts: ndarray
        Array of vertices in the surface mesh.
    faces: ndarray
        Array of faces in the surface mesh.
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
        self.vox_grid = self.sort_voxels(voxels, ncells)
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
    def sort_voxels(self, vox_cs, ncells):
        # initialize voxels and limits of voxel grid to be used
        cell_length = max((self.grid_lims[1] - self.grid_lims[0])/ncells) # length of cell in [x,y,z] directions
        cv_ratio = cell_length/Voxel.size
        buffer = np.ceil((3*cv_ratio/2) + 0.5)*Voxel.size
        vox_xs = np.transpose(vox_cs)
        first_vox = vox_cs[0]
        xlo = [min(x) - 2*buffer for x in vox_xs]
        xhi = [max(x) + 2*buffer for x in vox_xs]

        nvoxs = np.ceil((first_vox - xlo)/Voxel.size)
        vcx_lo = first_vox - nvoxs*Voxel.size
        nvoxs += np.ceil((xhi - first_vox)/Voxel.size)
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
