#
def parseResultsMC(resultsMC,iteration):
    """
    Reads ISTHMUS marching cubes results :resultsMC: and returns volumes, faces, vertices.
    Writes an STL file 'grids/grid_:iteration:.stl'.
    """
    #
    import trimesh
    cornerVolumes = resultsMC.corner_volumes
    faces = resultsMC.faces
    vertices = resultsMC.verts
    combinedMesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    combinedMesh.export('grids/grid_'+str(iteration)+'.stl', file_type='stl_ascii')
    cVolFrac = np.sum(cornerVolumes)/(nCells[0]*nCells[1]*nCells[2])
    #
    return cornerVolumes, faces, vertices
#
def readVoxelTri(name):
    f = open(name)
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
#
def readReactionSPARTA(fileName,timescale,timestepDSMC):
    """
    Reads file :fileName: in SPARTA surface format and returns the mass of CO formed at each surface triangle.
    Scales proportionally by a recession timescale :timescale: and inversely by the timestep from SPARTA :timestepDSMC:.
    """
    #
    # Initialize variables
    timeFlag = 0
    ind = 0
    COFormed = []
    #
    # Read reation file from SPARTA
    f = open(fileName,'r')
    for num, line in enumerate(f, 1):
        if 'ITEM: TIMESTEP' in line:
            timeFlag += 1
        if timeFlag == 2:
            ind += 1
        if timeFlag == 2 and ind > 9:
            s=tuple(line.split())
            COFormed.append([float(s[0]),float(s[1])*(12*10**-3)*fnum*timescale/(avog*timestepDSMC)])
    f.close()
    #
    return np.array(COFormed)
