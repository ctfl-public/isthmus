import numpy as np
from Marching_Cubes import marching_cubes, mesh_surface_area

# POSITIONS and LENGTHS in x,y,z order
# INDICES in z,y,x order
# program assumes minimal overlap of pixels

# this is where all the magic happens, basically everything else is owned by
# this class
class MC_System:  
    def __init__(self, lims, ncells, voxel_size, voxels):
        # check validity of grid being created and voxel data
        self.check_grid(lims, ncells)
        self.check_voxels(lims, ncells, voxel_size, np.transpose(voxels))
        
        # initialize system variables
        self.voxel_size = voxel_size
        self.voxels = voxels
        
        # execute marching cubes
        self.corner_grid = Corner_Grid(lims, ncells + 1)
        
        self.corner_grid.associate_voxels(self.voxels, self.voxel_size) # assign voxels to owning corner and divide volumes
        
        self.verts, self.faces = self.create_surface()
        
        self.transform_surface()
        
        self.write_surface()
        
        self.cell_grid = Cell_Grid(lims, ncells)
        
        self.cell_grid.associate_voxels(self.voxels)
        
        self.cell_grid.associate_triangles(self.verts, self.faces)
        
        self.voxel_triangle_ids = self.cell_grid.voxels_to_triangles(self.verts, self.voxels)
        
    
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

        return verts, faces

    def transform_surface(self):
        cg = self.corner_grid
        
        # scale vertices from a cell length of 1 to the proper cell length
        cell_len_array = np.array([cg.cell_length]*len(self.verts)).astype(float)
        self.verts *= cell_len_array
        
        # translate to proper coordinates
        translations = np.array([cg.lims[0]]*len(self.verts))
        self.verts += translations
        
    def write_surface(self):
        surf_file = open("vox2mesh.surf", "w")
        surf_file.write('surf file from isthmus\n\n')
        surf_file.write('{p} points\n{t} triangles\n\nPoints\n\n'.format(p = len(self.verts), t = len(self.faces)))
        for i in range(len(self.verts)):
            surf_file.write('{b} {x} {y} {z}\n'.format(b = i + 1, x = self.verts[i][0], \
                                                       y = self.verts[i][1], z = self.verts[i][2]))
    
        # order of points is flipped so sparta marks inside and outside correctly (DON'T INVERT IN INPUT SCRIPT)
        surf_file.write('\n\nTriangles\n\n')
        for i in range(len(self.faces)):
            surf_file.write('{b} {p1} {p2} {p3}\n'.format(b = i + 1, p1 = self.faces[i][2] + 1, \
                                                        p2 = self.faces[i][1] + 1, p3 = self.faces[i][0] + 1)) 
        surf_file.close() 

class Grid:
    def __init__(self, lims, dims):
        self.lims = lims    # grid domain limits, [[xlo,ylo,zlo], [xhi,yhi,zhi]]
        self.dims = dims # [nx, ny, nz], no. of elements in each direction
        self.coords = [[],[],[]] # list of possible x,y, and z coordinates
        

    def get_element(self,i,j,k):
        # i,j,k are x,y, and z indices
        return k*self.dims[1]*self.dims[0] + j*self.dims[0] + i
    
    def get_indices(self,n):
        # a,b,c are x,y, and z indices
        c = int(n/(self.dims[1]*self.dims[0]))
        n = n % (self.dims[1]*self.dims[0])
        b = int(n/self.dims[0])
        n = n % self.dims[0]
        a = n
        return a,b,c
    
    def point_association(self, p):
        # associate with binary search in each dimension
        indices = []
        for i in range(3):
            c_high = self.dims[i]
            c_low = -1
            c_index = int((c_high + c_low)/2) # cell index
            found_bounds = [False, False] # lower and upper bounds of cell are/n't correct
            while (not all(found_bounds)):
                if (found_bounds[0]):
                    c_low = c_index
                    c_index = int((c_high + c_low)/2)
                if (found_bounds[1]):
                    c_high = c_index
                    c_index = int((c_high + c_low)/2)
                found_bounds = [False, False]
                
                # low and high are bounds of corner space for the 'c_index' point
                low = self.coords[i][c_index] - self.cell_length[i]*0.5
                high = low + self.cell_length[i]
                if (p[i] >= low):
                    found_bounds[0] = True
                if (p[i] < high):
                    found_bounds[1] = True
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
        
    

class Corner_Grid(Grid):
    def __init__(self, lims, dims):
        super().__init__(lims, dims)
        self.cell_length = (self.lims[1] - self.lims[0])/(self.dims - 1) # length of cell in [x,y,z] directions
        self.corners = np.array([[[MC_Corner(self.lims[0] + self.cell_length*[i,j,k], i, j, k) for i in range(dims[0])] \
                                                                                      for j in range(dims[1])] \
                                                                                      for k in range(dims[2])])
        self.corners = self.corners.flatten()
        for i in range(3):
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
                    else:
                        self.divide_voxel(c, v, voxel_size, max_unique_dist)
            
        for c in self.corners:
            c.volume /= np.prod(self.cell_length)
            
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
    def __init__(self, c):
        self.center = c # center of cell position
        self.voxels = [] # voxel positions owned by this cell
        self.v_inds = [] # voxel global indices owned by this cell
        self.triangles = [] # triangle vertex ids owned by this cell
        self.t_inds = [] # triangle global indices
                    
    
class Cell_Grid(Grid):
    def __init__(self, lims, dims):
        super().__init__(lims, dims)
        self.cell_length = (self.lims[1] - self.lims[0])/self.dims # length of cell in [x,y,z] directions
        self.cells = np.array([[[MC_Cell(self.lims[0] + 0.5*self.cell_length + self.cell_length*[i,j,k]) for i in range(dims[0])] \
                                                                                      for j in range(dims[1])] \
                                                                                      for k in range(dims[2])])
        self.cells = self.cells.flatten()  
        for i in range(3):
            self.coords[i] = list(np.linspace(self.lims[0][i] + 0.5*self.cell_length[i], \
                                              self.lims[1][i] - 0.5*self.cell_length[i], self.dims[i]))
    
    def associate_voxels(self, voxels):
        for i in range(len(voxels)):
             ind = self.point_association(voxels[i]) # x,y,z corner indices 
             self.cells[self.get_element(ind[0],ind[1],ind[2])].voxels.append(voxels[i])
             self.cells[self.get_element(ind[0],ind[1],ind[2])].v_inds.append(i)

    def associate_triangles(self, verts, faces):
        for i in range(len(faces)):
            centroid = np.average(verts[faces[i][:]], axis=0)
            ind = self.point_association(centroid)
            self.cells[self.get_element(ind[0],ind[1],ind[2])].triangles.append(faces[i])
            self.cells[self.get_element(ind[0],ind[1],ind[2])].t_inds.append(i)
            
    def voxels_to_triangles(self, verts, voxels):
        triangles = np.ones(len(voxels))*-1
        for c in self.cells:
            if len(c.triangles) < 1:
                continue
            elif (len(c.triangles) == 1):
                for i in range(len(c.v_inds)):
                    triangles[c.v_inds[i]] = c.t_inds[0] + 1 # increment by 1 because SPARTA triangle ids start from 1
            else:
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
- parallelize
- ADD COMMENTS
- return triangle list
- review warnings in compilation
- set up testing suite for surface quality (vox inside and surface area) vox2triangle quality and MC robustness
"""



"""
# grid data: cells, corners, triangles, and vertices
class MC_Grid:
    def __init__(self, lims, ncells):
        self.lims = lims    # grid domain limits, [[xlo,ylo,zlo], [xhi,yhi,zhi]]
        self.ncells = ncells # [nx, ny, nz], no. of cells in each direction
        self.cell_length = (self.lims[1] - self.lims[0])/self.ncells # length of cell in [x,y,z] directions
        self.ncorners = self.ncells + 1 # no. of corners in each direction
        
        # generate corners, start from lower limits, increment by cell lengths in each direction
        p = self.lims[0]
        nc = self.ncorners
        cl = self.cell_length
        self.corners = np.array([[[MC_Corner(p + cl*[i,j,k]) for i in range(nc[0])] for j in range(nc[1])] for k in range(nc[2])])
        self.corner_coords = np.array([[(p[i] + cl[i]*j) for j in range(nc[i])] for i in range(3)])
        self.corner_volumes = np.asarray([[[0.0]*self.ncorners[0]]*self.ncorners[1]]*self.ncorners[2])
        
        cell_func = lambda x : MC_Cell(x.position, x.position + cl)
        np_cell_func = np.vectorize(cell_func)
        self.cells = np_cell_func(self.corners)
        for i in range(3):
            self.cells = np.delete(self.cells, -1, i)
        

class MC_Surface:
    def __init__(self):
        self.verts = [] # list of vertices, each defined by [x,y,z] position
        self.faces = [] # list of triangles, each defined by [v1, v2, v3] vertices
        self.tri_cells = [] # z,y,x indices for owning cell of face
    

              
# this is where all the magic happens, basically everything else is owned by
# this class
class MC_System:
    
    
    def __init__(self, lims, ncells, voxel_size, voxels):

        # generate voxel objects in grid
        self.voxels = np.array([Voxel(pos[0], pos[1], pos[2]) for pos in voxels])
    
        # create surface, associate between elements, write SPARTA mesh file
        self.associate_voxels() # associate voxels to cells and corners, divide volumes
        self.create_surface() # execute marching cubes on processed grid
        self.transform_mesh() # re-center and scale surface to actual coordinates
        self.associate_triangles() # associate voxels to triangles
        self.mesh.write_surface() # write SPARTA compliant surface file to disk

            
    def associate_voxels(self):
        
        for i in range(len(self.voxels)):
            # associate voxel to owning cell with binary search in each dimension
            p = self.voxels[i].position
            cell_indices = self.find_cell(p)
            self.voxels[i].cell = cell_indices # z,y,x indices
            self.cells[cell_indices[0]][cell_indices[1]][cell_indices[2]].voxels.append(i)
            
            # associate voxel to owning corner
            cell_center = self.cells[cell_indices[0]][cell_indices[1]][cell_indices[2]].get_center()
            diff = p - cell_center
            booleize = lambda x : 1 if x >= 0 else 0
            np_booleize = np.vectorize(booleize)
            diff_bool = np.flip(np_booleize(diff))
            corner_indices = cell_indices + diff_bool # z,y,x indices
            self.voxels[i].corner = corner_indices
            self.corners[corner_indices[0]][corner_indices[1]][corner_indices[2]].voxels.append(i)
            
            # divide volumes between corners
            pc = self.corners[corner_indices[0]][corner_indices[1]][corner_indices[2]].position
            diff = p - pc # distance in each dimension from owned corner
            min_pen_distance = 0.5*(self.cell_length - self.voxel_size)
            
            penetration = np.array([0,0,0]) # penetration distance in x,y,z directions
            pen_flag = np.array([0,0,0])   # flag for penetration in directions
            for i in range(3):
                potential_pen = abs(diff[i]) - min_pen_distance[i]
                if (potential_pen > 0):
                    penetration[i] = potential_pen
                    pen_flag[i] = 1 if diff[i] > 0 else -1
            pen_flag = np.flip(pen_flag) # flags in z,y,x direction
            
            x_vol = penetration[0]*(self.voxel_size - penetration[1])*(self.voxel_size - penetration[2])
            y_vol = penetration[1]*(self.voxel_size - penetration[0])*(self.voxel_size - penetration[2])
            z_vol = penetration[2]*(self.voxel_size - penetration[1])*(self.voxel_size - penetration[1])
            xy_vol = penetration[0]*penetration[1]*(self.voxel_size - penetration[2])
            xz_vol = penetration[0]*penetration[2]*(self.voxel_size - penetration[1])
            yz_vol = penetration[1]*penetration[2]*(self.voxel_size - penetration[0])
            xyz_vol = np.prod(penetration)
            
            self.corners[corner_indices[0]][corner_indices[1]][corner_indices[2]].volume += self.voxel_size**3 - x_vol - y_vol - z_vol - \
                                                                                             xy_vol - xz_vol - yz_vol - xyz_vol
            self.corners[corner_indices[0]][corner_indices[1]][corner_indices[2] + pen_flag[2]].volume += x_vol
            self.corners[corner_indices[0]][corner_indices[1] + pen_flag[1]][corner_indices[2]].volume += y_vol
            self.corners[corner_indices[0] + pen_flag[0]][corner_indices[1]][corner_indices[2]].volume += z_vol
            self.corners[corner_indices[0]][corner_indices[1] + pen_flag[1]][corner_indices[2] + pen_flag[2]].volume += xy_vol
            self.corners[corner_indices[0] + pen_flag[0]][corner_indices[1]][corner_indices[2] + pen_flag[2]].volume += xz_vol
            self.corners[corner_indices[0] + pen_flag[0]][corner_indices[1] + pen_flag[1]][corner_indices[2]].volume += yz_vol
            self.corners[corner_indices[0] + pen_flag[0]][corner_indices[1] + pen_flag[1]][corner_indices[2] + pen_flag[2]].volume += xyz_vol
         
        # scale voxel volumes assigned to each corner as a fraction of cell volume
        for k in range(self.ncorners[2]):
            for j in range(self.ncorners[1]):
                for i in range(self.ncorners[0]):
                    self.corners[k][j][i].volume /= np.prod(self.cell_length)
                    self.corner_volumes[k][j][i] = self.corners[k][j][i].volume
                    

    
  
    def associate_triangles(self):
        m = self.mesh 
        m.tri_cells = np.ones(m.faces.shape)*-1 # z,y,x indices for owning cell of face
        
        # associate faces to cells
        for i in range(len(m.faces)):
            avg_coords = np.average(m.verts[m.faces[i,:]], axis=0) # average all vertices of current triangle
            cell_indices = self.find_cell(avg_coords)
            m.tri_cells[i] = cell_indices
            self.cells[cell_indices[0]][cell_indices[1]][cell_indices[2]].triangles.append(i)
            
        # associate voxels to faces
        for k in range(self.ncells[2]):
            for j in range(self.ncells[1]):
                for i in range(self.ncells[0]):
                    c = self.cells[k][j][i]
                    if (len(c.triangles) < 1):
                        continue
                    elif (len(c.triangles) == 1):
                        for v in range(len(c.voxels)):
                            self.voxels[c.voxels[v]].triangle = c.triangles[0]
                    else:
                        for v in range(len(c.voxels)):
                            vt_distances = [] # distances from the voxel to that face
                            avail_triangles = [] # triangle indices
                            for f in range(len(c.triangles)):
                                avail_triangles.append(c.triangles[f])
                                ct = m.faces[c.triangles[f]] # current triangle's point indices
                                tri_points = np.array([m.verts[x] for x in ct]) # ct's actual point coordinates
                                centroid = np.average(tri_points, axis=0)
                                
                                dist = np.linalg.norm(centroid - self.voxels[c.voxels[v]].position)
                                vt_distances.append(dist)
                            chosen_tri = vt_distances.index(min(vt_distances))
                            self.voxels[c.voxels[v]].triangle = chosen_tri
                                
                        
                            PQ = tri_points[2] - tri_points[0]
                            PR = tri_points[1] - tri_points[0]
                            n = np.cross(PR, PQ) # get INWARD unit normal of triangle
                            n = n/np.linalg.norm(n)
                            
 """