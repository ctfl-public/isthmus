# This is Luis' parallel marching cubes setup
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
Created on Sun Feb 18 20:59:38 2024

@author: lch285

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

"""






# This is vox2stl.py

"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
Created on Mon Aug 21 16:12:17 2023

@author: ctfl


import numpy as np
import trimesh
from skimage import measure
import imageio
import random
import tifffile as tiff
import itertools
from scipy.spatial import ConvexHull
import tripy
import pymesh
from collections import defaultdict
import multiprocessing
import pymeshlab as ml

def stlSoothing(FileName, vertices,faces):
    # # Load mesh from stl
    # mesh_data = mesh.Mesh.from_file(basefilename)
    
    # Load mesh from vertices and faces
    trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    #Saving STL before filtering
    # The base file is being saved here. Operations need to be added here.
    basestl = FileName+"no-smoothing.stl"
    trimesh_mesh.export(basestl, file_type="stl_ascii")
    
    if (laplacian+humphrey+taubin) > 1:
        raise AssertionError("ERROR: FILTERS WILL BE OVERWRITTEN. SET ANY ONE FILTER VALUE TO 1") 
    # Apply filter iteration i
    if laplacian == 1:
        smoothed_mesh = trimesh.smoothing.filter_laplacian(trimesh_mesh, lamb=0.5, iterations=iter, volume_constraint=True)
    if humphrey == 1:
        smoothed_mesh = trimesh.smoothing.filter_humphrey(trimesh_mesh, alpha=0.1, beta=0.5, iterations=iter, laplacian_operator=None)
    if taubin == 1:
        smoothed_mesh = trimesh.smoothing.filter_taubin(trimesh_mesh, lamb=0.5, nu=0.5, iterations=iter, laplacian_operator=None)
    
    smoothed_vertices = smoothed_mesh.vertices
    smoothed_faces = smoothed_mesh.faces

    ##FIX NON manifold edges: start                                      
    non_manifold_edges = find_non_manifold_edges(vertices, faces)
    non_manifold_edges_smoothed = find_non_manifold_edges(smoothed_vertices, smoothed_faces)
    print("Non-manifold edges base file: ", non_manifold_edges)
    print("Non-manifold edges smmoth file: ", non_manifold_edges_smoothed)                           
    fixed_mesh = fix_non_manifold_edges(smoothed_mesh)
    smoothed_mesh.vertices = fixed_mesh.vertices
    smoothed_mesh.faces = fixed_mesh.faces
    non_manifold_edges_smoothed = find_non_manifold_edges(smoothed_mesh.vertices, smoothed_mesh.faces)
    print("Non-manifold edges smmoth file: THIS SHOULD BE NULL/EMPTY like [] ", non_manifold_edges_smoothed)
    ##FIX NON manifold edges: stop

    ##PYMESHLAB start: ADDING FUNCTION FOR THE REMOVAL OF HANGING FIBERS
    #PYMESHLAB and TRIMESH, mesh reading formats are different.
    pymesh_smoothed_mesh = ml.Mesh(smoothed_mesh.vertices, smoothed_mesh.faces)
    meshlab_server = ml.MeshSet(verbose=True)
    meshlab_server.add_mesh(pymesh_smoothed_mesh, mesh_name="smoothed_mesh", set_as_current=True)
    if pde_mode == 1:
        meshlab_server.apply_filter("remove_isolated_pieces_wrt_face_num", mincomponentsize=min_face_count, removeunref=True)
        modified_mesh = meshlab_server.current_mesh()
    #meshlab_server.apply_filter('surface_reconstruction_screened_poisson', visiblelayer=False,depth=8,fulldepth=5,cgdepth=0,scale=1.1,samplespernode=1.5,pointweight=4,iters=8,confidence=False,preclean=False,threads=16) ###THIS FILTER SHOULD SERVE THE PURPOSE
    meshlab_server.apply_filter('remove_zero_area_faces')
    #meshlab_server.apply_filter('compute_normal_per_vertex', weightmode='By Angle')
    #meshlab_server.apply_filter('compute_normal_per_face')
    modified_mesh = meshlab_server.current_mesh()
    # FIXING non manifold
    #meshlab_server.apply_filter('repair_non_manifold_edges_by_removing_faces')   ## removing faces 
    #meshlab_server.apply_filter('repair_non_manifold_edges_by_splitting_vertices') ## SPLIT VERTICES
    #meshlab_server.apply_filter('repair_non_manifold_vertices_by_splitting', vertdispratio=0) # vertdispration 0 or 0.1
    #modified_mesh = meshlab_server.current_mesh()
    smoothed_mesh = trimesh.Trimesh(vertices=modified_mesh.vertex_matrix(), faces=modified_mesh.face_matrix())
    ##PYMESHLAB stop    

    ##CHECK NORMAL DIRECTION and watertight: start
    #check_and_fix_normals(smoothed_mesh)
    non_manifold_edges_smoothed = find_non_manifold_edges(smoothed_mesh.vertices, smoothed_mesh.faces)
    smoothed_mesh.vertices, smoothed_mesh.faces = make_mesh_watertight(smoothed_mesh.vertices, smoothed_mesh.faces)
    print("Non-manifold edges smmoth file: ", non_manifold_edges_smoothed)
    print("TRIMESH CHECK VOLUME ------> "+str(is_volume_tri(smoothed_mesh.vertices, smoothed_mesh.faces)))
    print("TRIMESH CHECK WATERTIGHT --> "+str(is_watertight_tri(smoothed_mesh.vertices, smoothed_mesh.faces)))
    ##is_volume_tri() --> makes sure mesh is correctly defined(watertight, manifold checks, normal direction)
    ##CHECK NORMAL DIRECTION: stop

    # Save the smoothed mesh as a new ASCII.stl file
    basefilename = FileName+"laplacian-"+str(iter)+".stl"
    smoothed_mesh.export(basefilename, file_type="stl_ascii")
    print(smoothed_mesh)
    mesh_surface_area = compute_surface_area(smoothed_vertices,smoothed_faces)
    print(f"Mesh Surface Area after smoothing: {mesh_surface_area} square units")
    
    file.write(str(mesh_surface_area)+' ')

    min_extents = np.min(vertices, axis=0)
    max_extents = np.max(vertices, axis=0)

    print(f"Minimum Extents before smoothing (X, Y, Z): {min_extents}")
    print(f"Maximum Extents before smoothing (X, Y, Z): {max_extents}")

    file.write(str(min_extents[0])+' '+str(min_extents[1])+' '+str(min_extents[2])+' ')
    file.write(str(max_extents[0])+' '+str(max_extents[1])+' '+str(max_extents[2])+' ')
    

    min_extents = np.min(smoothed_vertices, axis=0)
    max_extents = np.max(smoothed_vertices, axis=0)

    print(f"Minimum Extents after smoothing (X, Y, Z): {min_extents}")
    print(f"Maximum Extents after smoothing (X, Y, Z): {max_extents}")
  
    file.write(str(min_extents[0])+' '+str(min_extents[1])+' '+str(min_extents[2])+' ')
    file.write(str(max_extents[0])+' '+str(max_extents[1])+' '+str(max_extents[2])+' ')

    mesh_volume = compute_volume(smoothed_vertices, smoothed_faces)
    print(f"Mesh Volume 2: {mesh_volume} cubic units")
    file.write(str(mesh_volume)+' ')
 
    return basefilename

def check_and_fix_normals(mesh):
    # Access the normals of the triangles
    triangle_normals = mesh.face_normals

    # Calculate the weighted sum of triangle normals
    weighted_sum = np.sum(mesh.area_faces * triangle_normals, axis=0)

    # Normalize the overall face normal
    overall_face_normal = weighted_sum / np.linalg.norm(weighted_sum)

    # Check if all normals are pointing outward
    are_normals_outward = np.all(np.dot(triangle_normals, overall_face_normal) > 0)

    if not are_normals_outward:
        # Reverse the normals for the triangles with inward normals
        inverted_normals = triangle_normals[np.dot(triangle_normals, overall_face_normal) <= 0]
        mesh.faces_normal[np.dot(triangle_normals, overall_face_normal) <= 0] = -inverted_normals

        # Update the mesh with the corrected normals
        mesh.update_normals()

        print("Normals have been corrected.")
    else:
        print("All normals are already pointing outward.")

def fix_non_manifold_edges(mesh):
    mesh, _ = pymesh.remove_duplicated_faces(mesh)
    
    return mesh


def make_mesh_watertight(vertices, faces):
    # Identify holes in the mesh
    holes = find_mesh_holes(vertices, faces)

    # Fill holes by triangulating each hole
    for hole in holes:
        triangles = tripy.earclip(hole)
        for triangle in triangles:
            faces.append([vertices.index(vertex) for vertex in triangle])

    return vertices, faces

def find_mesh_holes(vertices, faces):
    # Identify holes using a simple approach
    edge_count = defaultdict(int)

    # Count occurrences of each edge
    for face in faces:
        for i in range(3):
            edge = tuple(sorted([face[i], face[(i + 1) % 3]]))
            edge_count[edge] += 1

    # Find open edges (incident on only one face)
    open_edges = [edge for edge, count in edge_count.items() if count == 1]

    # Extract connected vertices to form holes
    holes = []
    while open_edges:
        hole = []
        current_edge = open_edges.pop()
        hole.append(current_edge[0])

        while True:
            hole.append(current_edge[1])
            next_edge = [edge for edge in edge_count.keys() if current_edge[1] in edge and edge_count[edge] == 1]
            
            if not next_edge:
                break
            
            next_edge = next_edge[0]
            edge_count[next_edge] -= 1
            if edge_count[next_edge] == 0:
                open_edges.remove(next_edge)
            current_edge = next_edge

        holes.append(hole)

    return holes

def delete_non_manifold_edges(vertices, faces):
    edge_count = defaultdict(int)

    # Count occurrences of each edge
    for face in faces:
        for i in range(3):
            edge = tuple(sorted([face[i], face[(i + 1) % 3]]))
            edge_count[edge] += 1

    non_manifold_edges = set()

    # Check for non-manifold edges (incident on more than two faces)
    for edge, count in edge_count.items():
        if count != 2:
            non_manifold_edges.add(edge)

    # Delete non-manifold edges
    for edge in non_manifold_edges:
        # Find the faces incident on the non-manifold edge
        incident_faces = [face for face in faces if set(edge).issubset(face)]

        # Remove adjacent faces
        for face in incident_faces:
            faces.remove(face)

    # Clean up unused vertices
    used_vertices = {vertex for face in faces for vertex in face}
    vertices = [vertex for i, vertex in enumerate(vertices) if i in used_vertices]

    return vertices, faces


def find_non_manifold_edges(vertices, faces):
    edge_count = defaultdict(int)

    # Count occurrences of each edge
    for face in faces:
        for i in range(3):
            edge = tuple(sorted([face[i], face[(i + 1) % 3]]))
            edge_count[edge] += 1

    non_manifold_edges = set()

    # Check for non-manifold edges (incident on more than two faces)
    for edge, count in edge_count.items():
        if count != 2:
            non_manifold_edges.add(edge)

    return list(non_manifold_edges)

def is_watertight(vertices, faces):
    vertices_np = np.array(vertices)
    hull = ConvexHull(vertices_np)

    return len(hull.vertices) == len(vertices)

#Check if a mesh has all the properties required to represent a valid volume,
#rather than just a surface.These properties include being watertight, having
#consistent winding and outward facing normals.

def is_volume_tri(vertices, faces):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return mesh.is_volume

def is_watertight_tri(vertices, faces):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return mesh.is_watertight

def compute_volume(vertices, faces):
        volume = 0.0
        for face in faces:
            v0, v1, v2 = vertices[face]
            volume += np.dot(v0, np.cross(v1, v2)) / 6.0
        return abs(volume)

def compute_surface_area(vertices, faces):
      area = 0.0
      for face in faces:
        v0, v1, v2 = vertices[face]
        e1 = v1 - v0
        e2 = v2 - v0
        cross_product = np.cross(e1, e2)
        area += 0.5 * np.linalg.norm(cross_product)
      return area

def getMesh(binary_volume,length,voxel_size):
    
    # Create a mesh using the marching cubes algorithm
    vertices, faces, _, _ = measure.marching_cubes(binary_volume)
    
    # Swap x and z to make it match the tif
    # Indices of the columns you want to swap
    column_index1 = 0  # Replace with the index of the first column
    column_index2 = 2  # Replace with the index of the second column
    
    # Swap the columns using array indexing
    vertices[:, [column_index1, column_index2]] = vertices[:, [column_index2, column_index1]]
 
    # Convert vertices to physical coordinates using voxel size
    vertices = vertices * voxel_size
    #print(binary_volume[:5, :, :])
    return vertices, faces

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


def getstl(surfacename, tifvoxelsize, temp_number,volumeLength, corner):
    
    print('start reading tif to creat stl temp %s\n'%(temp_number))
    
    # Create image_volume
    if surfacename.endswith('.tif'):
        image_volume = imageio.volread(surfacename)
        print(np.shape(image_volume))
        
    elif surfacename.endswith('.txt') or surfacename.endswith('.dat'):
        tempdata =np.loadtxt(surfacename, skiprows=2)
        xmax =int(max(tempdata[:,0]))
        ymax =int(max(tempdata[:,1]))
        zmax =int(max(tempdata[:,2]))
        image_volume = np.zeros((xmax,ymax,zmax),dtype='int')
        for val in tempdata:
            image_volume[int(val[0])-1,int(val[1])-1,int(val[2])-1] = int(val[3])
            
    
    print('Finish reading tif to creat stl temp %s\n'%(temp_number))
    
        
    # Surface file name
    tempName = surfacename[:-4]+'_V%i_%i-%i-%i-%s-'%(temp_number,corner[0],corner[1],corner[2],volumeLength)
    
    if volumeLength != 'Full':
        # adjust nummber volumelength 
        volumeLength = int(volumeLength/tifvoxelsize)
        
        print('Volume is %s in voxels and voxelsize is %s\n'%(volumeLength,tifvoxelsize))
    
        # Crop the volume of interest
        temp_volume = image_volume[corner[0]:corner[0]+volumeLength,corner[1]:corner[1]+volumeLength,corner[2]:corner[2]+volumeLength]
        
        print('temp_volume is %s\n'%(str(np.shape(temp_volume))))
        
    else: # Create stl from full tif
        temp_volume = image_volume
    
    if np.sum(temp_volume) == 0:
        print('Empty Volume, there is no material!')
        return 'Empty'
    # Make a binary MTX
    temp_volume = temp_volume/ np.max(temp_volume)
    
    # Calculate porosity
    fulltempVolume = temp_volume.shape[0]*temp_volume.shape[1]*temp_volume.shape[2]
    materialVolume = np.sum(temp_volume/np.max(temp_volume))    
    porosity = round((fulltempVolume - materialVolume)/fulltempVolume , 4)
    
    # Creat padding around volume 
    print('Creating padding temp %s'%(surfacename))
    binary_volume = createPadding(temp_volume)
    
    # Remove Ilands
    if pde_mode == 1:
        chen = tempName+str(".dat")
        #ovito= tempName+str("ovito.dat")
        file1 = open(chen,'w')
        #f = open(ovito,'w')
        print(np.shape(binary_volume))
        #print("H")
        print(np.shape(temp_volume))
        #chen files
        x_i,y_j,z_k = np.shape(binary_volume)
        file1.write(str(x_i-1)+' '+str(y_j-1)+' '+str(z_k-1)+' '+str(tifvoxelsize*10**-6)+'\n')
        file1.write("i j k voxel")
        count = 1
        for i in range(x_i):
            for j in range(y_j):
                for k in range(z_k):
                    if binary_volume[i, j, k] == 1:
                       value = int(binary_volume[i, j, k])
                       file1.write(f"\n{i} {j} {k} {value}")
                       count = count + 1
                       print(f"Point ({i}, {j}, {k}) has a value of 1")
                    else:
                        print(f"Point ({i}, {j}, {k}) does not have a value of 1")#print(f"Point ({i}, {j}, {k}): {temp_volume[i, j, k]}")
        pcount = 1
        #f.write('ITEM: TIMESTEP\n')
        #f.write('0\n')
        #f.write('ITEM: NUMBER OF ATOMS\n')
        #f.write(str(count-1)+"\n")
        #f.write('ITEM: BOX BOUNDS pp pp pp\n')
        #f.write(str(0)+"  "+str(x_i)+"\n")
        #f.write(str(0)+"  "+str(y_j)+"\n")
        #f.write(str(0)+"  "+str(z_k)+"\n")
        #f.write('ITEM: ATOMS id x y z voxel')

        #for i in range(x_i):
        #    for j in range(y_j):
        #        for k in range(z_k):
        #            if binary_volume[i, j, k] == 1:
        #               value = int(binary_volume[i, j, k])
        #               f.write(f"\n{pcount} {i} {j} {k} {value}")
        #               pcount = pcount + 1

        file1.close()

        #f.close()
    # Get vertices and faces 
    print('Starting mesh creation for temp %s'%(tempName))
    vertices, faces = getMesh(binary_volume,volumeLength,tifvoxelsize)

    ### GET SURFACE AREA
    mesh_surface_area = compute_surface_area(vertices, faces)
    print(f"Mesh Surface Area: {mesh_surface_area} square units")
    file.write(str(mesh_surface_area)+' ')
    
    ## SA / material_vol
    lengthbyarea = materialVolume/mesh_surface_area
    print(f"LengthByArea-Vol/SA: {lengthbyarea, materialVolume}")
    file.write(str(materialVolume)+' '+str(lengthbyarea)+' ')

    # Perform smoothing 
    tempName = stlSoothing(tempName, vertices,faces)   
    print('Starting tif saving for  temp %s'%(str(tempName)))
    # Save the volume as a 3D TIFF file (uncomment to double check stl)[:-4]
    tiff.imsave(tempName[:-4]+'.tif', binary_volume[1:-1,1:-1,1:-1]) 

def voxel2stl(croppingFlags, cropSettings, surfaceSettings):
    
    laplacian, humphrey,taubin, iter, min_face_count, pde_mode = surfaceSettings
    
    normalFlag, cornerFlag = croppingFlags
    
    if normalFlag:
        filenames, filevoxels, numVolumes, volumeLength = cropSettings
    
    if cornerFlag:
        filenames, filevoxels, cornersMTX, volumeLength = cropSettings
        
    
    # matrix to store Volumes data 
    sim_mtx = [] #
    if normalFlag:
        if volumeLength != 0 and numVolumes != 0:
            tempMTX= np.zeros(len(filenames),dtype= 'int')
            for temp in range(numVolumes):
                tempName = random.choice(filenames)
                tempNameIndex = filenames.index(tempName)
                tempMTX[tempNameIndex] += 1
     
    temp_number = 0
    for surf in filenames:
        tempNameIndex = filenames.index(surf)
        if surf.endswith('.tif'):
            print('loading tif')
            image_volume = imageio.volread(surf)
            print('Finish loading tif')
        elif surf.endswith('.txt') or surf.endswith('.dat'):
            tempdata =np.loadtxt(surf,skiprows=2)
            xmax =int(max(tempdata[:,0]))
            ymax =int(max(tempdata[:,1]))
            zmax =int(max(tempdata[:,2]))
            image_volume = np.zeros((xmax,ymax,zmax),dtype='int')
            for val in tempdata:
                image_volume[int(val[0])-1,int(val[1])-1,int(val[2])-1] = int(val[3])
                
        # max voxel length in x, y, and z
        voxelsLegth = (image_volume.shape[0],image_volume.shape[1],image_volume.shape[2])
        
        if normalFlag:
            if volumeLength == 0: # create stl from all the volume
                corner = np.zeros(3, dtype='int')
                fullvolume = 'Full'
                getstl(surf, filevoxels[tempNameIndex], temp_number,fullvolume, corner)
                temp_number += 1
            
            elif numVolumes == 0:
                # Number of volumes in each direction
                dimX = int(voxelsLegth[0]*filevoxels[tempNameIndex]/volumeLength)
                dimY = int(voxelsLegth[0]*filevoxels[tempNameIndex]/volumeLength)
                dimZ = int(voxelsLegth[0]*filevoxels[tempNameIndex]/volumeLength)
                
                totalvolumes = dimX*dimY*dimZ
                # if totalvolumes > 10:
                #     print('Do this in parallel, %i volumes'%(totalvolumes))
                #     return
                xCorners = [i*int(volumeLength/filevoxels[tempNameIndex]) for i in range(dimX)]
                yCorners = [i*int(volumeLength/filevoxels[tempNameIndex]) for i in range(dimY)]
                zCorners = [i*int(volumeLength/filevoxels[tempNameIndex]) for i in range(dimZ)]
                corners = list(itertools.product(xCorners,yCorners,zCorners))
                for corner in corners:
                    getstl(surf, filevoxels[tempNameIndex], temp_number,volumeLength, corner)
                    temp_number += 1
                    
            else:
                for times in range(tempMTX[tempNameIndex]):
                    
                    print('corner',times,'of',surf)
                    if numVolumes != 0:
                         
                        # name of iteration volume
                        # tempName = filenames[:-4]+'_V%i'%(temp)
                        
                        corner = np.zeros(3, dtype='int')
                        # get random temp corner
                        corner[0] = random.randint(0,voxelsLegth[0]-volumeLength) 
                        corner[1] = random.randint(0,voxelsLegth[1]-volumeLength)
                        corner[2] = random.randint(0,voxelsLegth[2]-volumeLength)
                        # center crop
                        # corner[0] = (voxelsLegth[0]-volumeLength)/2 #random.randint(0,voxelsLegth[0]-volumeLength) 
                        # corner[1] = (voxelsLegth[1]-volumeLength)/2 #random.randint(0,voxelsLegth[1]-volumeLength)
                        # corner[2] = (voxelsLegth[2]-volumeLength)/2 #random.randint(0,voxelsLegth[2]-volumeLength)

                        
                        getstl(surf, filevoxels[tempNameIndex], temp_number,volumeLength, corner)
                        temp_number += 1

        elif cornerFlag:
            for corner in cornersMTX:
                getstl(surf, filevoxels[tempNameIndex], temp_number,volumeLength, corner)
                temp_number += 1

    
        
if __name__ == '__main__':
    
    random.seed(500)
    filenames = ['5656um_ari05_Filled_transformed.tif'] # One or more
    filevoxels = [2.9009] # One or more
    laplacian, humphrey,taubin = 1,0,0 # Any one filter should be set to 1, rest 0  
    iter = 2 # iterations for the Filter
    min_face_count = 25000
    pde_mode = 0 # 0 or 1 -> in 1 hanging fibers will be removed according to min_face_count.
    
    surfaceSettings = laplacian, humphrey,taubin, iter, min_face_count, pde_mode
    
    normalFlag = 0
    cornerFlag = 1
    croppingFlags = normalFlag, cornerFlag
    
    if normalFlag:
        numVolumes = 1 # zero for Lego
        volumeLength = 400 # zero for full volume
        cropSettings = filenames, filevoxels, numVolumes, volumeLength
    
    elif cornerFlag:
        cornersMTX = np.array([[0,1020,0],[0,1020,68]]) # One or more in Array
        volumeLength = 200 # One or more in Array
        
        cropSettings = filenames, filevoxels, cornersMTX, volumeLength
        
    
    file_path = "properties-"+str(laplacian)+str(humphrey)+str(taubin)+"-"+str(iter)+".dat"
    file = open(file_path, 'w')

    voxel2stl(croppingFlags,cropSettings, surfaceSettings)
    file.close()
    """