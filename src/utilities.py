import numpy as np
import sys
import imageio

def tiff_slicer(fileName, x, y, z, l, voxel_size, lb= 5):
    """
    Slices a 3D TIFF image into a smaller voxelized cube.

    Args:
        fileName (str): Path to the input TIFF file.
        x (float): x in the in-plane direction
        y (float): y in the in-plane direction
        z (float): z is the through thickness direction
        l (float): cube length
        voxel_size (float): size of the voxel
        lb (float): buffer length

    Returns:
        tuple: (voxs, lims) A tuple containing the voxel coordinates and the limits of the sliced region.
    """
    lo = [-lb, -lb, -lb]  

    hi = [l/2 + lb, (l + lb), (l + lb)]

    lims = np.array([lo, hi])

    voxs = []
    voxelMTX = loadData(fileName)
    sample_mtx = voxelMTX[int(z):int(z+l/2),int(y-l/2):int(y + l/2),int(x-l/2):int(x + l/2)]

    for i in range(int(l)):
        for j in range(int(l)):
            for k in range(int(l/2)):
                if sample_mtx[k,j,i] == 1:
                    voxs.append([k,j,i])

    lims = lims*voxel_size
    voxs = np.array(voxs)*voxel_size

    return voxs, lims

def loadData(surf):
    if surf.endswith('.tif'):
        # Load the TIFF file
        image_volume = imageio.volread(surf)
    
    elif surf.endswith('.txt') or surf.endswith('.dat'):
        tempdata = np.loadtxt(surf, skiprows=2)
        xmax = int(max(tempdata[:, 0]))
        ymax = int(max(tempdata[:, 1]))
        zmax = int(max(tempdata[:, 2]))
        image_volume = np.zeros((xmax, ymax, zmax), dtype='int')
        for val in tempdata:
            image_volume[int(val[0]) - 1, int(val[1]) - 1, int(val[2]) - 1] = int(val[3])
            
    return image_volume


def progress_bar(c, total, message):
    """
    Displays a simple textual progress bar in the console.

    The bar updates every 10% of progress. Once the total is reached,
    it prints a newline and resets.

    Args:
        c (int): Current iteration index (0-based).
        total (int): Total number of iterations.
        message (str): Message to display alongside the progress bar.

    Returns:
        None: Prints progress bar directly to stdout.

    Notes:
        - Uses `progress_bar.c_decade` as a static attribute to track
          the next update threshold (default = 10).
        - The bar length is fixed to 10 characters (each "=" equals 10%).

    Example:
        >>> import time
        >>> for i in range(100):
        ...     progress_bar(i, 100, "Processing")
        ...     time.sleep(0.05)
        # Output:
        #     Processing... [=         ]
        #     Processing... [==        ]
        #     ...
        #     Processing... [==========]
    """
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


def noise_gen(mag):
    """
    Generates a random noise factor within ±(mag/100) of 1.
    """
    return (2*np.random.rand() - 1)*(mag/100) + 1


def readVoxelTri(fileName):
    """
    Reads a voxel-triangle association file.
    Returns mappings of triangles to their associated voxels and scalar fractions.
    """
    f = open(fileName)
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
                raise Exception("ERROR: unable to read triangle_voxels.dat")
    tri_voxs = {triangle_ids[i] : owned_voxels[i] for i in range(len(triangle_ids))}
    tri_sfracs = {triangle_ids[i] : owned_sfracs[i] for i in range(len(triangle_ids)) }

    for tv in tri_voxs.values():
        for v in range(len(tv)):
            for v2 in range(v+1, len(tv)):
                if tv[v] == tv[v2]:
                    raise Exception("ERROR: surface voxel double-assigned to a triangle")
    return tri_voxs,tri_sfracs