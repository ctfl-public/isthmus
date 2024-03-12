import numpy as np
from Marching_Cubes import marching_cubes, mesh_surface_area
import imageio

# program assumes minimal overlap of pixels

# this is where all the magic happens, basically everything else is owned by
# this class
class MC_System:  
    def __init__(self, lims, ncells, voxel_size, voxels, name, cell_tri=False):
        # check validity of grid being created and voxel data
        self.check_grid(lims, ncells)
        self.check_voxels(lims, ncells, voxel_size, np.transpose(voxels))
        
        # initialize system variables
        self.voxel_size = voxel_size
        self.voxels = voxels
        
        # prepare grid to feed to marching cubes
        self.corner_grid = Corner_Grid(lims, ncells + 1)       
        self.corner_grid.associate_voxels(self.voxels, self.voxel_size) # assign voxels to owning corner and divide volumes        
        
        # execute marching cubes and scale/translate surface
        self.verts, self.faces = self.create_surface()
        self.transform_surface()
        
        # write SPARTA-compliant surface
        self.cell_grid = Cell_Grid(lims, ncells)
        self.write_surface(name)
        
        # associate voxels to triangles by way of the containing cell
        self.cell_grid.associate_voxels(self.voxels) # associate voxels to cells
        self.tri_cell_ids = self.cell_grid.associate_triangles(self.verts, self.faces) # associate triangles to cells
        if cell_tri:
            self.write_triangle_cells()
        self.voxel_triangle_ids = self.cell_grid.voxels_to_triangles(self.verts, self.voxels) # associate voxels to triangle in same cell
        self.write_voxel_triangles()

        
    # check validity of grid limits and number of cells
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
            raise Exception("Voxel size is larger than marching cubes grid cell dimension(s)")
            
        if (not voxel_size > 0):
            raise Exception("Voxel size is invalid")
        
        if (len(positions) != 3):
            raise Exception("Invalid voxel coordinates given")
        
        for i in range(3):
            # i is 0,1,2 for x,y, or z; positions is list of coordinates
            border = lims[0][i] + 0.5*(voxel_size + cell_length[i])
            if (any(x < border for x in positions[i])):
                raise Exception("Voxel(s) outside of acceptable grid space")
            border = lims[1][i] - 0.5*(voxel_size + cell_length[i])
            if (any(x > border for x in positions[i])):
                raise Exception("Voxel(s) outside of acceptable grid space")
    
    # produce surface with marching cubes from corner grid
    def create_surface(self): 
        cg = self.corner_grid
        corner_volumes = np.asarray([[[0.0]*cg.dims[0]]*cg.dims[1]]*cg.dims[2])
        for n in range(len(cg.corners)):
            a,b,c = cg.get_indices(n)
            corner_volumes[c][b][a] = cg.corners[n].volume # marching cubes requires [z,y,x] order

        verts, faces, normals, values = marching_cubes( \
               volume= corner_volumes, level=0.5, \
               gradient_direction='descent', \
               allow_degenerate=False)
            
        verts = np.fliplr(verts) # marching_cubes() outputs in z,y,x order
        
        return verts, faces

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
        surf_file = open(name, "w")
        surf_file.write('surf file from isthmus\n\n')
        surf_file.write('{p} points\n{t} triangles\n\nPoints\n\n'.format(p = len(self.verts), t = len(self.faces)))
        for i in range(len(self.verts)):
            surf_file.write('{b} {x} {y} {z}\n'.format(b = i + 1, x = self.verts[i][0], \
                                                       y = self.verts[i][1], z = self.verts[i][2]))
    
        # order of points is flipped so sparta marks inside and outside correctly (DON'T INVERT IN INPUT SCRIPT)
        surf_file.write('\n\nTriangles\n\n')
        for i in range(len(self.faces)):
            surf_file.write('{b} {p1} {p2} {p3}\n'.format(b = i + 1, p1 = self.faces[i][0] + 1, \
                                                        p2 = self.faces[i][1] + 1, p3 = self.faces[i][2] + 1)) 
        surf_file.close() 
        
    def write_voxel_triangles(self):
        f = open('voxel_triangles.dat', 'w')
        f.write('vox_idx,tri_id\n') # voxel index, triangle id
        for i in range(len(self.voxel_triangle_ids)):
            if (self.voxel_triangle_ids[i] != -1): # if voxel is assigned to a triangle
                f.write('{v_idx},{tri_id}\n'.format(v_idx= i, tri_id= int(self.voxel_triangle_ids[i])))
        f.close()

    def write_triangle_cells(self):
        gr = self.cell_grid
        f = open('triangle_cells.dat', 'w')
        f.write('tri,n,xc,yc,zc\n') # triangle id, cell integer id, cell x index, y index, and z index
        for i in range(len(self.tri_cell_ids)):
            cell = self.tri_cell_ids[i]
            x,y,z = gr.get_indices(cell)
            f.write('{tri},{n},{xc},{yc},{zc}\n'.format(tri= i + 1, n= int(cell), xc= int(x), yc= int(y), zc= int(z)))
        f.close()
    
    def get_surface_area(self):
        return mesh_surface_area(self.verts, self.faces)

# superclass for marching cubes cell grid and corner grid
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
    
    # determine in which element a point in space resides
    def point_association(self, p):
        # associate with binary search in each of 3 dimensions
        indices = []
        epsilon = min(self.cell_length)*0.0001 # for floating point issues, 0.01% of cell length
        for i in range(3):
            c_high = self.dims[i]
            c_low = -1
            c_index = int((c_high + c_low)/2) # cell index
            found_bounds = [False, False] # lower and upper bounds of cell are/n't correct
            last_c_index = -1
            while (not all(found_bounds)):
                if (found_bounds[0]):
                    c_low = c_index
                    c_index = int((c_high + c_low)/2)
                if (found_bounds[1]):
                    c_high = c_index
                    c_index = int((c_high + c_low)/2)
                found_bounds = [False, False]
                
                # low and high are bounds of corner space for the 'c_index' element
                low = self.coords[i][c_index] - self.cell_length[i]*0.5
                high = low + self.cell_length[i]
                if (p[i] >= low):
                    found_bounds[0] = True
                if (p[i] < high):
                    found_bounds[1] = True
                
                # if c_index is being repeated, likely a floating point error for point being near an element boundary
                if (last_c_index == c_index):
                    if (abs(p[i] - low) < epsilon or abs(p[i] - high) < epsilon):
                        found_bounds[0] = True
                        found_bounds[1] = True
                    else:
                        print('For voxel position {p}:'.format(p=p[i]))
                        raise Exception("Point unable to be found")
                last_c_index = c_index
            indices.append(c_index)
            
        return indices # x,y,z indices
        
# this class is for corners in the grid, i.e. each MC_Cell corresponds to
# 8 MC_Corners
class MC_Corner:
    def __init__(self, p, i, j, k):
        self.position = p # [x,y,z]
        self.indices = np.array([i,j,k]) # indices in the grid
        self.volume = 0 # volume fraction of cell filled by voxel material
        self.voxels = [] # voxel ids owned by corner
        
    
# grid of corners, used to feed volume fractions to marching cubes function
class Corner_Grid(Grid):
    def __init__(self, lims, dims):
        super().__init__(lims, dims)
        self.cell_length = (self.lims[1] - self.lims[0])/(self.dims - 1) # length of cell in [x,y,z] directions
        self.corners = np.array([[[MC_Corner(self.lims[0] + self.cell_length*[i,j,k], i, j, k) \
                                   for i in range(dims[0])] for j in range(dims[1])]  for k in range(dims[2])])
        self.corners = self.corners.flatten()
        for i in range(3): # possible x,y,z corner coordinates
            self.coords[i] = list(np.linspace(self.lims[0][i], self.lims[1][i], self.dims[i]))
        
    def associate_voxels(self, voxels, voxel_size):    
        self.voxels_to_corners(voxels) # associate voxels to corners
        self.divide_volumes(voxel_size) # divide volumes between corners
    
    def voxels_to_corners(self, voxels):
        for v in voxels:
             ind = self.point_association(v) # x,y,z corner indices 
             self.corners[self.get_element(ind[0],ind[1],ind[2])].voxels.append(v)

    def divide_volumes(self, voxel_size):
        max_unique_dist = 0.5*(self.cell_length - voxel_size)
        voxel_volume = voxel_size**3
        for c in self.corners:
            if len(c.voxels):
                for v in c.voxels:
                    displ = v - c.position
                    dist = np.array([abs(x) for x in displ])
                    if all(dist <= max_unique_dist):
                        c.volume += voxel_volume
                    else: # if voxel is not fully inside one corner's space, divide between corners
                        self.divide_voxel(c, v, voxel_size, max_unique_dist)
            
        for c in self.corners:
            c.volume /= np.prod(self.cell_length)
            
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
        

# this is the unit cell of the marching cubes grid, with position, owned voxels,
# and owned triangles
class MC_Cell:
    def __init__(self):
        self.voxels = [] # voxel positions owned by this cell
        self.v_inds = [] # voxel global indices owned by this cell
        
        self.triangles = [] # triangle vertex ids owned by this cell
        self.t_inds = [] # triangle global indices
        
        self.vertices = [] # vertices of triangles owned by this cell
        self.vert_inds = [] # vertex global indices
                    
# this is where voxels and triangles are located and connected to each other
class Cell_Grid(Grid):
    def __init__(self, lims, dims):
        super().__init__(lims, dims)
        self.cell_length = (self.lims[1] - self.lims[0])/self.dims # length of cell in [x,y,z] directions
        self.cells = np.array([[[MC_Cell() for i in range(dims[0])] for j in range(dims[1])] for k in range(dims[2])])
        self.cells = self.cells.flatten()  
        for i in range(3):
            self.coords[i] = list(np.linspace(self.lims[0][i] + 0.5*self.cell_length[i], \
                                              self.lims[1][i] - 0.5*self.cell_length[i], self.dims[i]))
    # associate voxels to cells
    def associate_voxels(self, voxels):
        for i in range(len(voxels)):
             ind = self.point_association(voxels[i]) # x,y,z corner indices 
             self.cells[self.get_element(ind[0],ind[1],ind[2])].voxels.append(voxels[i])
             self.cells[self.get_element(ind[0],ind[1],ind[2])].v_inds.append(i)
    
    # associate points to cells
    def associate_points(self, ps):
        i = 0
        for p in ps:
            ind = ((p - self.lims[0])/self.cell_length).astype(int) # cell indices [x,y,z]
            n = self.get_element(ind[0],ind[1],ind[2])
            self.cells[n].vertices.append(p)
            self.cells[n].vert_inds.append(i)
            i += 1
         
    # associate triangles to cells
    def associate_triangles(self, verts, faces):
        tri_cell_ids = np.ones(len(faces))*-1
        for i in range(len(faces)):
            centroid = np.average(verts[faces[i][:]], axis=0)
            ind = ((centroid - self.lims[0])/self.cell_length).astype(int) # cell indices [x,y,z]
            n = self.get_element(ind[0],ind[1],ind[2])
            self.cells[n].triangles.append(faces[i])
            self.cells[n].t_inds.append(i)
            tri_cell_ids[i] = n
        return tri_cell_ids
        
    # associate each voxel to 0 or 1 triangles, return this list
    def voxels_to_triangles(self, verts, voxels):
        triangles = np.ones(len(voxels))*-1
        for c in self.cells:
            if len(c.triangles) < 1: # if no triangles in cell, don't assign voxels in cell to a triangle
                continue
            elif (len(c.triangles) == 1): # if only one triangle, assign all voxels in cell to it
                for i in range(len(c.v_inds)):
                    triangles[c.v_inds[i]] = c.t_inds[0] + 1 # increment by 1 because SPARTA triangle ids start from 1
            else: 
                # if multiple triangles, choose for each voxel the triangle with the closest centroid
                for i in range(len(c.voxels)): 
                    vt_distances = []
                    for f in c.triangles:
                        centroid = np.average(verts[f[:]], axis=0)
                        dist = np.linalg.norm(centroid - c.voxels[i])
                        vt_distances.append(dist)
                    chosen_tri = vt_distances.index(min(vt_distances))
                    triangles[c.v_inds[i]] = c.t_inds[chosen_tri] + 1
        
        return triangles
    
"""
To-do
1. clean and push to main
2. testing suite for quality, unit tests; surface quality (vox inside and surface area) vox2triangle quality and MC robustness
3. tif input allowed (test and push)

- removal of duplicate geometry (?)
- implement parallelization (Luis)
- implement different voxel-triangle mapping procedures
- interface with sparta
- find voxels for 0-voxel triangles
- review warnings in compilation
"""
