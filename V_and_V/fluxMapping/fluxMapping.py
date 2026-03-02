# %%
# If not added to the PYTHONPATH, uncomment below two lines to add the ISTHMUS source directory.
import os,sys
sys.path.append('../../src')

from isthmus import marchingWindows
import numpy as np
import pandas as pd
print('ISTHMUS module loaded\n')

# user parameter selection
vox_size = 1  # voxel side length
cell_len = 1  # will be adjusted if domain lengths are not multiples of this
angle = 0    # angle in degrees between cube alignment and the voxel/grid alignment (rotated around z-axis)

class Voxel:
    def __init__(self, x):
        self.x = np.array(x) # [x,y,z] of center
        self.type = -1       # 1 for internal, 0 for edge, -1 for void
        self.scalar = 0

    def fill(self):
        self.type = 1

def get_tri_area(verts):
    # herons formula = sqrt(s(s - a)(s - b)(s - c)) for triangle lengths a,b,c, s= half-perimeter
    a = np.linalg.norm(verts[2] - verts[1])
    b = np.linalg.norm(verts[1] - verts[0])
    c = np.linalg.norm(verts[0] - verts[2])

    s = (a + b + c)/2
    area = np.sqrt(s*(s - a)*(s - b)*(s - c))

    return area

class Triangle:
    def __init__(self, tri_obj):
        self.id = tri_obj.id
        self.vertices = tri_obj.vertices
        self.scalar = 0
        self.area = get_tri_area(self.vertices)
        self.owned_vox_ids = tri_obj.voxel_ids
        self.owned_sfracs = tri_obj.voxel_scalar_fracs

def get_voxs(vox_grid): 
    c_voxs = []
    for vox in vox_grid.flatten():
        if vox.type >= 0:
            c_voxs.append(vox)
    return c_voxs

def apply_scalars(v_rate, tris, c_voxs, vox_grid):
    total_tri_scalar = 0
    total_vox_scalar = 0
    # apply volume loss (scalar) to each triangle and convert to voxel-based
    for surf in tris:
        surf.scalar = v_rate*surf.area
        total_tri_scalar += surf.scalar
        for i in range(len(surf.owned_vox_ids)):
            vox = c_voxs[surf.owned_vox_ids[i]]
            vox.scalar += surf.scalar*surf.owned_sfracs[i]
    print('\nTriangle Scalar Total: {:.5e}'.format(total_tri_scalar))

    # check conservation
    for vox in vox_grid.flatten():
        total_vox_scalar += vox.scalar
    print('Voxel Scalar Total:    {:.5e}'.format(total_vox_scalar))

    mapping_error = 100*(total_vox_scalar - total_tri_scalar)/(total_tri_scalar)
    print('Vox vs. Surface Total Scalar Error: {:.2e} %'.format(mapping_error))
    return mapping_error

def safe_mkdir(path, name):
    full_path = path + name + '/'
    try:
        if not os.path.exists(full_path):
            os.mkdir(full_path)
    except OSError as err:
            print(err)

os.chdir(os.path.dirname(os.path.abspath(__file__)))
pathg= os.getcwd() + '/'
safe_mkdir(pathg, 'results')
os.chdir('./results')

# Case initialization
hlen = 5
lims = np.array([[-10, -10, -10], [10, 10, 10]])
ncells = np.rint((lims[1] - lims[0])/cell_len).astype(int)
cell_len = (lims[1] - lims[0])/ncells
print('Voxel size: {}'.format(vox_size))
print('Cell size [x,y,z]: [{},{},{}]'.format(cell_len[0],cell_len[1],cell_len[2]))
a_rad = angle*(np.pi/180)
coord_transf = [[ np.cos(a_rad), np.sin(a_rad), 0],
                [-np.sin(a_rad), np.cos(a_rad), 0],
                [             0,             0, 1]]

nx_vox = np.ceil((lims[1] - lims[0])/vox_size).astype(int)
vg = [[[Voxel(lims[0] + (np.array([i,j,k]) + 0.5)*vox_size)
        for i in range(nx_vox[0])] for j in range(nx_vox[1])] for k in range(nx_vox[2])]
vox_grid = np.array(vg)
nvoxs = 0
for cvox in vox_grid.flatten():
    v_prime = np.matmul(coord_transf, cvox.x)
    if all([abs(v) < hlen for v in v_prime]):
        cvox.fill()
        nvoxs += 1
print('Voxels Created: {}\n'.format(nvoxs))

weight = False
gpu = False
c_voxs = get_voxs(vox_grid)

# generate surface from voxels, assign surface elements to voxels
resultsMC = marchingWindows(lims, ncells, vox_size, [v.x for v in c_voxs], 'vox2surf.surf', 0,
                            weight=weight, gpu=gpu)

# classify surface voxels identified by marchingWindows()
for sv in resultsMC.surface_voxels:
    c_vox = c_voxs[sv.oid]
    c_vox.type = 0

# organize data for triangles in created surface
triangles = [Triangle(tri) for tri in resultsMC.cell_grid.triangles]

# apply a uniform triangle-based flux and convert to voxel scalars via flux mapping
flux = 0.125
map_error = apply_scalars(flux, triangles, c_voxs, vox_grid)

# %%
