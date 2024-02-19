import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from time import time
from Marching_Cubes import marching_cubes, mesh_surface_area

# POSITIONS and LENGTHS in x,y,z order
# INDICES in z,y,x order
# program assumes minimal overlap of pixels

class MC_Cell:
    def __init__(self, lo, hi):
        self.lo = lo
        self.hi = hi
        self.voxels = []
        
    def get_center(self):
        center = (self.hi + self.lo)/2
        return center

class MC_Corner:
    def __init__(self, p):
        self.position = p # [x,y,z]
        self.voxels = []
        self.volume = 0

class Voxel:
    def __init__(self, x, y, z):
        self.position = np.array([x,y,z])
        self.cell = np.zeros(3) # z,y,x indices for the owning cell
        self.corner = np.zeros(3) # z,y,x indices for owning corner
        
class MC_Mesh:
    def __init__(self, verts, faces):
        self.verts = verts
        self.faces = faces

class MC_Grid:
    def __init__(self, lims, ncells, voxel_size, voxels):
        self.mesh = MC_Mesh([],[])
        
        self.dims = ['x', 'y', 'z']

        self.lims = np.array(lims)     # domain limits, [[xlo,ylo,zlo], [xhi,yhi,zhi]]
        self.ncells = np.array(ncells) # [nx, ny, nz], no. of cells in each direction
        self.cell_length = (self.lims[1] - self.lims[0])/self.ncells
        self.cell_volume = np.prod(self.cell_length)
        self.ncorners = self.ncells + 1 
        
        # generate corners, start from lower limits, increment by cell lengths in each direction
        p = lims[0]
        nc = self.ncorners
        cl = self.cell_length
        self.corners = np.array([[[MC_Corner(p + cl*[i,j,k]) for i in range(nc[0])] for j in range(nc[1])] for k in range(nc[2])])
        self.corner_coords = np.array([[(p[i] + cl[i]*j) for j in range(nc[i])] for i in range(len(self.dims))])
        self.corner_volumes = np.asarray([[[0.0]*self.ncorners[0]]*self.ncorners[1]]*self.ncorners[2])
        
        cell_func = lambda x : MC_Cell(x.position, x.position + cl)
        np_cell_func = np.vectorize(cell_func)
        self.cells = np_cell_func(self.corners)
        for i in range(3):
            self.cells = np.delete(self.cells, -1, i)

        # generate voxel objects in grid
        self.voxel_size = voxel_size
        self.voxel_volume = voxel_size**3
        self.voxels = np.array([Voxel(pos[0], pos[1], pos[2]) for pos in voxels])
        
        if (any(self.cell_length < voxel_size)):
            raise Exception("Voxel size is larger than marching cubes grid cell dimension(s)")
        self.associate_voxels()
        self.create_mesh()
        self.transform_mesh()
        self.associate_triangles()
        
    def associate_voxels(self):
        
        for i in range(len(self.voxels)):
            # associate voxel to owning cell with binary search in each dimension
            p = self.voxels[i].position
            cell_indices = []
            for j in range(len(p)-1, -1, -1):
                corner_ps = self.corner_coords[j]
                c_high = self.ncells[j]
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
                    if (p[j] >= corner_ps[c_index]):
                        found_bounds[0] = True
                    if (p[j] < corner_ps[c_index + 1]):
                        found_bounds[1] = True
                cell_indices.append(c_index)
            cell_indices = np.array(cell_indices)
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
            
            self.corners[corner_indices[0]][corner_indices[1]][corner_indices[2]].volume += self.voxel_volume - x_vol - y_vol - z_vol - \
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
                    self.corners[k][j][i].volume /= self.cell_volume
                    self.corner_volumes[k][j][i] = self.corners[k][j][i].volume
                    
    def create_mesh(self):
        
        ms = np.zeros([self.ncorners[0], self.ncorners[1], self.ncorners[2]])
        ms[:, :, :] = 1
        ms = ms.astype(bool)
            
        verts, faces, normals, values = marching_cubes( \
               volume= self.corner_volumes, level=0.5, \
               gradient_direction='descent', \
               allow_degenerate=False, mask=ms)
        
        self.mesh.verts = verts
        self.mesh.faces = faces
        
    def transform_mesh(self):
        m = self.mesh
        
        # scale vertices from a cell length of 1 to the proper cell length
        cell_len_array = np.array([self.cell_length]*len(m.verts)).astype(float)
        m.verts *= cell_len_array
        
        # translate to proper coordinates
        translations = np.array([self.corners[0][0][0].position]*len(m.verts))
        m.verts += translations
  
    def associate_triangles(self):
        bleh = 1

def generate_test_voxels(v_size, ns, lims):
    rs = 1
    xc = ((np.random.rand(ns))*(lims[1][0] - lims[0][0] - 2*rs)) + lims[0][0] + rs
    yc = ((np.random.rand(ns))*(lims[1][1] - lims[0][1] - 2*rs)) + lims[0][1] + rs
    zc = ((np.random.rand(ns))*(lims[1][2] - lims[0][2] - 2*rs)) + lims[0][2] + rs
    r = np.array([rs]*ns)

    xs = []
    ys = []
    zs = []
    for i in range(ns):
        sphere_x, sphere_y, sphere_z = make_sphere(xc[i], yc[i], zc[i], r[i], v_size)
        xs += sphere_x
        ys += sphere_y
        zs += sphere_z
        
    return xs, ys, zs

def make_sphere(xc, yc, zc, r, v_size):
    nvox_1d = int(2*r/v_size)
    if (nvox_1d % 2):
        nvox_1d += 1
    nvox_1d = int(nvox_1d/2 + 0.1)
    
    xs = []
    ys = []
    zs = []
    for i in range(nvox_1d*2):
        x = -nvox_1d*v_size + 0.5*v_size + i*v_size
        for j in range(nvox_1d*2):
            y = -nvox_1d*v_size + 0.5*v_size + j*v_size
            for k in range(nvox_1d*2):
                z = -nvox_1d*v_size + 0.5*v_size + k*v_size
                if (np.sqrt(x**2 + y**2 + z**2) < r):
                    xs.append(x + xc)
                    ys.append(y + yc)
                    zs.append(z + zc)
    return xs, ys, zs    

def plot_results(verts, faces, lo, hi, proj=False):
    tris = Poly3DCollection(verts[faces])
    tris.set_edgecolor('k')
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')
    ax.add_collection3d(tris)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2]) 
    if (proj):
        ax.view_init(elev=90, azim=90, roll=0)
    plt.tight_layout()
    #plt.savefig('isthmus_test', dpi=400)
    plt.show()
    

    
    """
    To-do
    - associate voxels to triangles
    - check to see if voxels are far enough inside
    - refactor init and associate_voxels to be smaller  
    - quality checks based on voxels being inside geometry/surface area/volume
    
    """


np.random.seed(199)
ns = 3


lo = [-2.5, -2.5, -2.5]
hi = [2.5, 2.5, 2.5]
lims = [lo, hi]
ncells = [50, 50, 50]

vss = [0.1, 0.05, 0.04, 0.035, 0.03, 0.028]

mc_time = []
nvox = []
for i in range(len(vss)):
    v_size = vss[i]
    
    xs, ys, zs = generate_test_voxels(v_size, ns, lims)
    nvox.append(len(xs))
    
    voxels = np.transpose(np.array([xs, ys, zs]))
    
    t1 = time()
    mc_grid = MC_Grid(lims, ncells, v_size, voxels)
    mc_time.append(time() - t1)

    plot_results(mc_grid.mesh.verts, mc_grid.mesh.faces, lo, hi)

nvox = np.array(nvox)
mc_time = np.array(mc_time)
plt.figure()
a, b = np.polyfit(nvox, mc_time, 1)
plt.plot(nvox, a*nvox + b, color='blue', label='Best Fit Line')
plt.scatter(nvox, mc_time, color='red', label='Isthmus Output')
plt.xlabel('Number of voxels')
plt.ylabel('Time for Marching Cubes System')
plt.title('50x50x50 MC Grid')
plt.legend()
plt.grid()

mc_time = []
mc_cells = []
ncells = [[50, 50, 50], [60,60,60], [70,70,70], [80,80,80]]
v_size = 0.04
xs, ys, zs = generate_test_voxels(v_size, ns, lims)
total_voxels = len(xs)
voxels = np.transpose(np.array([xs, ys, zs]))
for i in range(len(ncells)):
    
    mc_cells.append(np.prod(np.array(ncells[i])))
    
    t1 = time()
    mc_grid = MC_Grid(lims, ncells[i], v_size, voxels)
    mc_time.append(time() - t1)

    plot_results(mc_grid.mesh.verts, mc_grid.mesh.faces, lo, hi)
    
mc_cells = np.array(mc_cells)
mc_time = np.array(mc_time)
plt.figure()
a, b = np.polyfit(mc_cells, mc_time, 1)
plt.plot(mc_cells, a*mc_cells + b, color='blue', label='Best Fit Line')
plt.scatter(mc_cells, mc_time, color='red', label='Isthmus Output')
plt.xlabel('Number of MC Cells')
plt.ylabel('Time for Marching Cubes System')
plt.title(str(total_voxels) + ' voxels')
plt.legend()
plt.grid()
