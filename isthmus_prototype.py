import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from Marching_Cubes import marching_cubes, mesh_surface_area

def plot_nice(ax, lims, file, xy=False):
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_zlabel("z-axis")
    ax.set_xlim(lims[0][0], lims[0][1])
    ax.set_ylim(lims[1][0], lims[1][1])
    ax.set_zlim(lims[2][0], lims[2][1])  
    if (xy):
        ax.view_init(elev=90, azim=90, roll=0)
    plt.tight_layout()
    plt.savefig(file, dpi=400)
    plt.show()
  
def main():
    """
    args: voxel positions, voxel size, mc grid lims, mc grid refinement
    1. assign voxels to cells
    2. assign voxels to corners with volumes
    3. feed volume grid to marching cubes function with threshold
    4. scale and translate mesh to appropriate coordinates
    5. associate voxels to surface elements ('faces')
    
    Notes: the only change that should be necessary to the marching cubes function
          is to have it return a data structure that relates each face in the mesh
          to a cell and a corner in that cell where applicable (dictionary?)
    
          mask or gradient direction options needed in marching cubes?

    generate_test_voxels()
    voxel_assignment()
    verts, faces, normals, values, association = measure.marching_cubes( \
        volume=, level=, allow_degenerate=False)
    transform_mesh()
    associate_voxels()
    """
     
    size = 51
    # 3D array of corner values to give to marching cubes
    ellip_corner_list = np.asarray([[[[0,0,0]]*size]*size]*size).astype(float)
    ellip_base = np.zeros((size,size,size))
    for i in range(size):
        for j in range(size):
            for k in range(size):
                x = (i-int(size/2))*.1
                y = (j-int(size/2))*.1
                z = (k-int(size/2))*.1
                ellip_corner_list[i][j][k] = [x,y,z]
                ellip_base[i][j][k] = np.sqrt(x**2 + y**2 + z**2)
    
               
    thresh = ((size-1)/2 + 0.5)*0.1
    ellip_bool = ellip_base < thresh

    # calculate corner positions around the origin and scaled by sc
    xs = np.zeros((size,size,size))
    ys = np.zeros((size,size,size))
    zs = np.zeros((size,size,size))
    for i in range(size):
        for j in range(size):
            for k in range(size):
                xs[i][j][k] = ellip_corner_list[i][j][k][0]
                ys[i][j][k] = ellip_corner_list[i][j][k][1]
                zs[i][j][k] = ellip_corner_list[i][j][k][2]
    
    # create voxel cells from corners
    cubes = np.zeros((size-1,size-1,size-1))
    for i in range(len(cubes)):
        for j in range(len(cubes[i])):
            for k in range(len(cubes[i][j])):
                total = 0
                for m in range(2):
                    for n in range(2):
                        for o in range(2):
                            if (ellip_bool[i+m][j+n][k+o]):
                                total += 1
                if total > 3:
                    cubes[i][j][k] = 1
    
    cubes = cubes.astype(bool)
    colors = np.empty(cubes.shape, dtype=object)
    colors[cubes] = 'red'
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    
    # these are NOT the voxels that we are inputting, just graphing voxels to see the grid
    ax.voxels(xs, ys, zs, cubes, facecolors=colors, edgecolor='k')
    xlims = [-2.5, 2.5]
    ylims = [-2.5, 2.5]
    zlims = [-2.5, 2.5]
    plot_nice(ax, [xlims,ylims,zlims], 'cubes')
    
    ms = np.zeros(ellip_base.shape)
    ms[:, :, :] = 1
    ms = ms.astype(bool)
    # Use marching cubes to obtain the surface mesh of these ellipsoids
    verts, faces, normals, values = marching_cubes( \
        volume=ellip_base, level=thresh, \
        gradient_direction='descent', \
        allow_degenerate=False, mask=ms)
    
    scale_factors = np.array([[(x - 1) for x in ellip_base.shape]]*len(verts)).astype(float)
    grid_size = ellip_corner_list[-1][-1][-1] - ellip_corner_list[0][0][0]
    scale_factors /= np.array(grid_size)
    verts = verts/scale_factors
    
    translations = np.array([ellip_corner_list[0][0][0]]*len(verts))
    verts += translations
    
    # Display resulting triangular mesh using Matplotlib
    fig1 = plt.figure(figsize=(10, 10))
    ax1 = fig1.add_subplot(projection='3d')
    
    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor('k')
    ax1.add_collection3d(mesh)
    plot_nice(ax1, [xlims,ylims,zlims], 'mcubes')
    
    # quality checks - surface area
    radius = ((size-1)/2)*0.1
    analytical_sa = 4*np.pi*radius**2
    mc_sa = mesh_surface_area(verts, faces)
    print('Surface area error: ' + '{:.1f}'.format(100*(mc_sa - analytical_sa)/analytical_sa) + ' %')

if __name__=='__main__':
    main()