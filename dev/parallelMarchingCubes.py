#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 20:59:38 2024

@author: lch285
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
import trimesh
from Marching_Cubes import marching_cubes, mesh_surface_area
import multiprocessing as mp
import time
import imageio

def plotVoxels(label_image):
    # Create a figure
    fig = plt.figure()
    
    # Plot the original visualization
    ax1 = fig.add_subplot(111, projection='3d')
    # Get indices of non-zero values in label_image
    nonzero_indices = np.nonzero(label_image)
    
    # Scatter plot using non-zero indices and color each point based on its label
    ax1.scatter(nonzero_indices[0], nonzero_indices[1], nonzero_indices[2], c=label_image[nonzero_indices], cmap='viridis', edgecolor='black')

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Fiber Detection and centerline')
    plt.show()

def generate_stl(voxelMTX, filename='output.stl', translation = np.array([0,0,0])):
    # Use marching cubes to obtain the mesh
    verts, faces, normals, values, Dict = marching_cubes(voxelMTX)
                                                        
    # Apply translation
    verts += translation
    
    # Create a mesh object
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    
    # Ensure the normals are pointing outward
    mesh.faces = np.flip(mesh.faces, axis=1) 
    
    return mesh

def combine_meshes(meshes):
    # Initialize lists to store vertices and faces
    vertices_combined = []
    faces_combined = []
    vertex_offset = 0
    
    # Concatenate vertices and faces of each mesh
    for mesh in meshes:
        vertices_combined.append(mesh.vertices)
        faces_combined.append(mesh.faces + vertex_offset)
        vertex_offset += len(mesh.vertices)
    
    # Concatenate all vertices and faces arrays
    vertices_combined = np.concatenate(vertices_combined)
    faces_combined = np.concatenate(faces_combined)
    
    # Adjust vertices given voxel size
    vertices_combined *= voxelSize
    
    # Create a new mesh
    combined_mesh = trimesh.Trimesh(vertices=vertices_combined, faces=faces_combined)
    
    return combined_mesh

def generate_stl_parallel(args):
    data, translation, filename = args
    mesh = generate_stl(data, filename, translation)
    return mesh

def createPadding(image_volume):
    # Set the desired padding size (in pixels) for each dimension
    padding_size = 1  # Adjust this value based on your needs
    
    # Create a padded volume with the same data type as the original image
    padded_volume = np.zeros(
        (image_volume.shape[0] + 2 * padding_size,
         image_volume.shape[1] + 2 * padding_size,
         image_volume.shape[2] + 2 * padding_size),
        dtype=image_volume.dtype,
    )
    
    # Calculate padding ranges for each dimension
    x_range = slice(padding_size, padding_size + image_volume.shape[0])
    y_range = slice(padding_size, padding_size + image_volume.shape[1])
    z_range = slice(padding_size, padding_size + image_volume.shape[2])
    
    # Copy the original image into the center of the padded volume
    padded_volume[x_range, y_range, z_range] = image_volume
    
    
    #Load padded volume
    binary_volume = np.squeeze(np.array(padded_volume))
    
    return binary_volume

def createCircle(r=50,domainSize=150):

    voxelMTX = np.zeros((domainSize,domainSize,domainSize))
    
    # Create grid coordinates
    x, y, z = np.ogrid[:domainSize, :domainSize, :domainSize]
    
    # Calculate distances to the center for all points
    distances = np.sqrt((x )**2 + (y )**2 + (z )**2)
    
    # Set voxel values based on distance from the center
    voxelMTX[distances <= r] = 1

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
    
if __name__ == '__main__':
    fileName = '/Users/vijaybmohan/Desktop/Cam/isthmus/50.tif'
    voxelSize = 1
    
    # Load in Tif
    voxelMTX = loadData(fileName)
    
    # Add padding to the volume
    voxelMTX = createPadding(voxelMTX)
    
    # Get dimesnion with padding
    xdim, ydim, zdim = voxelMTX.shape
    
    print('Starting to create surface in Parallel!')
    # Number of voxels per split
    divisionVoxels = 30

    
    splitListX = np.arange(0,xdim+1,divisionVoxels)
    splitListY = np.arange(0,ydim+1,divisionVoxels)
    splitListZ = np.arange(0,zdim+1,divisionVoxels)
    
    if splitListX[-1] != xdim:
        splitListX = np.append(splitListX, xdim)
    if splitListY[-1] != ydim:
        splitListY = np.append(splitListY, ydim)
    if splitListZ[-1] != zdim:
        splitListZ = np.append(splitListZ, zdim)
    
    splitMTX = []
    
    for i in range(len(splitListX)-1):
        for j in range(len(splitListY)-1):
            for k in range(len(splitListY)-1):
                tempMTX = voxelMTX[splitListX[i]:splitListX[i+1]+1,splitListY[j]:splitListY[j+1]+1,splitListZ[k]:splitListZ[k+1]+1]
                tempXdim, tempYdim, tempZim = tempMTX.shape
                if np.sum(tempMTX) != 0 and np.sum(tempMTX) != tempXdim*tempYdim*tempZim:
                    # plotVoxels(tempMTX)
                    splitMTX.append([tempMTX,[splitListX[i],splitListY[j],splitListZ[k]]])
    
    startTime = time.time()
    with mp.Pool(processes=mp.cpu_count()) as pool: # Adjust the number of processes as needed
        args = [(data, translation, str(num) + '.stl') for num, (data, translation) in enumerate(splitMTX)]
        meshesList = pool.map(generate_stl_parallel, args)
    
    combinedMesh = combine_meshes(meshesList)
    endTime = time.time()
    print('It took %i sec to genereate the surface in parallel with %i divisions!'%((endTime-startTime),divisionVoxels))
    
    stlFileName = "originalJoined.stl"
    
    print('Saving stl as',stlFileName)
    
    # Save the joined mesh
    combinedMesh.export(stlFileName, file_type='stl_ascii')
    
    print(combinedMesh.is_watertight)
    print(combinedMesh.is_volume)
    