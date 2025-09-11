import numpy as np
import trimesh
import os
import imageio
import warnings
from isthmus import readVoxelTri
import json

class ablationCase:
    def __init__(self):
        # Create required directories if they don't already exist
        dirs = ['grids','voxel_data','voxel_tri']
        for d in dirs:
            os.makedirs(d,exist_ok=True)
        print('Directories created')

        # in number of voxels
        width = 200
        height = 100
        buffer = 5  
        voxelSize = 3.3757e-6  # in meters

        # timescale and some quantities for DSMC
        self.timescale = 2
        self.timestepDSMC = 7.5e-9
        self.fnum = 14866.591116363918
        self.avog = 6.022*10**23

        self.voxelSize = voxelSize
        lo = [-buffer, -buffer, -buffer]
        hi = [height + buffer, (width + buffer), (width + buffer)]
        self.lims = voxelSize*np.array([lo, hi])
        self.nCells = np.array([int(height),int(width),int(width)])

        # load voxels from tiff file
        fileName = 'sample.tif'
        voxelMatrix = imageio.volread(fileName)
        voxs = []
        for i in range(int(width)):
            for j in range(int(width)):
                for k in range(int(height)):
                    if voxelMatrix[k,j,i] == 1:
                        voxs.append([k,j,i])
        self.voxs = np.array(voxs)*self.voxelSize
        print(f'{len(voxs):d} voxels loaded from sample')


    def runDSMC(self, step):
        """
        Runs DSMC simulation using SPARTA.
        """

        # Instead of running DSMC, we will read pre-calculated reaction files
        COFormed = self._readReactionSPARTA('reactionFiles/surf_react_sparta_'+str(step)+'.out')
        self.COFormed = COFormed[COFormed[:, 0].argsort()]

    def ablate(self, step):
        """
        Ablates the material based on the mass of CO formed at each surface triangle.
        Updates the voxel list by removing voxels that have completely ablated.
        """
        # read volume fraction of the material
        with open('volFrac.dat') as f:
            cVolFrac = f.readline().strip('\n')
        # 
        # Read voxel data 
        with open('voxel_data/voxelData_'+str(step-1)+'.dat') as f: 
            lines = (line for line in f if not line.startswith('#')) 
            voxs_alt = np.loadtxt(lines, delimiter=',', skiprows=0) 
        # 
        # Associate voxels to tirangles (The flux mapping file)
        tri_voxs,tri_sfracs = readVoxelTri('voxel_tri/triangle_voxels_'+str(step-1)+'.dat')

        #Triangles check between sparta and isthmus
        if len(self.COFormed) != len(tri_voxs):
            warnings.warn("No of triangles in sparta is not equal to isthmus, debug!!!")
         
        # Calculate mass of carbon associated with each voxel
        volFracC = float(cVolFrac)
        volC = volFracC*(self.lims[1,0]-self.lims[0,0])*(self.lims[1,1]-self.lims[0,1])*(self.lims[1,2]-self.lims[0,2])
        massC = volC*1800
        massCVox = massC/len(voxs_alt)
        # 
        # Calculate the mass of carbon removed from each voxel
        cRemovedVox = np.zeros((len(voxs_alt)))
        for i in range(len(self.COFormed)):
            vox_no = np.array((tri_voxs[(i+1)]),dtype = int)
            sfracs = np.array((tri_sfracs[(i+1)]),dtype = float)
            for k in range(len(vox_no)):
                cRemovedVox[vox_no[k]] = cRemovedVox[vox_no[k]] + sfracs[k] * self.COFormed[i,1]
        cRemovedVox[:] = cRemovedVox[:] + voxs_alt[:,3]
        # 
        # Remove voxels
        voxs_alt = np.column_stack((voxs_alt[:,0:3],cRemovedVox))
        for i in range(len(cRemovedVox)):
            if cRemovedVox[i] > massCVox:
                voxs_alt[i,:] = 0
        self.voxs_alt = voxs_alt[~np.all(voxs_alt == 0, axis=1)] 

        self.voxs = voxs_alt[:,0:3]

    def postProcess(self, cornerVolumes, vertices, faces, iteration):
        """
        Writes an STL file 'grids/grid_:iteration:.stl'.
        Writes a voxel data file 'voxel_data/voxel_data_:iteration:.dat'.
        Writes a volume fraction file 'volFrac.dat'.
        """
        # Write the stl file
        combinedMesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        combinedMesh.export('grids/grid_'+str(iteration)+'.stl', file_type='stl_ascii')

        # Write coordinate voxel data
        if iteration == 0:
            cRemovedVox = np.zeros((len(self.voxs),1))
            self.voxs_alt = np.column_stack((self.voxs,cRemovedVox))
        f = open('voxel_data/voxel_data_'+str(iteration)+'.dat','w+')
        for i in range(len(self.voxs_alt)):
            f.write(str(self.voxs_alt[i,0])+','+str(self.voxs_alt[i,1])+','+str(self.voxs_alt[i,2])+','+str(self.voxs_alt[i,3])+'\n')
        f.close()
        
        #
        # Write the file containing volume fraction of the material
        cVolFrac = np.sum(cornerVolumes)/(self.nCells[0]*self.nCells[1]*self.nCells[2])
        f = open('volFrac.dat','w+')
        f.write(str(cVolFrac)+'\n')
        f.close()
    #

    def clean(self):
        # Remove temporary files
        os.remove('voxelData')
        os.remove('voxelTri')
        os.remove('volFrac.dat')
        print('Temporary directories removed')



    def _readReactionSPARTA(self,fileName):
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
                COFormed.append([float(s[0]),float(s[1])*(12*10**-3)*self.fnum*self.timescale/(self.avog*self.timestepDSMC)])
        f.close()
        #
        return np.array(COFormed)



class multiPhaseCase:
    def __init__(self):
        # Create required directories if they don't already exist
        dirs = ['grids','voxel_data','voxel_tri']
        for d in dirs:
            os.makedirs(d,exist_ok=True)
        print('Directories created')

        # in number of voxels
        width = 200
        height = 100
        buffer = 5  
        voxelSize = 3.3757e-6  # in meters

        # timescale and some quantities for DSMC
        self.timescale = 1
        self.timestepDSMC = 7.5e-9
        self.fnum = 14866.591116363918
        self.avog = 6.022*10**23

        self.voxelSize = voxelSize
        lo = [-buffer, -buffer, -buffer]
        hi = [height + buffer, (width + buffer), (width + buffer)]
        self.lims = voxelSize*np.array([lo, hi])
        self.nCells = np.array([int(height),int(width),int(width)])

        self.rate_of_ablation_fiber = 1
        self.rate_of_ablation_matrix = 20
        self.voxs_types = {}

        # load voxels from tiff file
        fileName = 'sample_multiphase.tif'
        voxelMatrix = imageio.volread(fileName)
        voxs_layers = []
        voxs = []
        for i in range(int(width)):
            for j in range(int(width)):
                for k in range(int(height)):
                    if voxelMatrix[k,j,i] == 0:
                        voxs.append([k,j,i])
                        voxs_layers.append([k*self.voxelSize,j*self.voxelSize,i*self.voxelSize,len(voxs),'matrix'])
                    else:
                        voxs.append([k,j,i])
                        voxs_layers.append([k*self.voxelSize,j*self.voxelSize,i*self.voxelSize,len(voxs),'fiber'])
        self.voxs = np.array(voxs)*self.voxelSize
        self.voxs_layers = voxs_layers
        print(f'{len(voxs):d} voxels loaded from sample')

        


    def runDSMC(self, step):
        """
        Runs DSMC simulation using SPARTA.
        """

        # Instead of running DSMC, we will read pre-calculated reaction files
        COFormed = self._readReactionSPARTA('reactionFiles/surf_react_sparta_'+str(step)+'.out')
        self.COFormed = COFormed[COFormed[:, 0].argsort()]

    def ablate(self, step):
        """
        Ablates the material based on the mass of CO formed at each surface triangle.
        Updates the voxel list by removing voxels that have completely ablated.
        """
        # read volume fraction of the material
        with open('volFrac.dat') as f:
            cVolFrac = f.readline().strip('\n')
        # 
        # Read voxel data 
        with open('voxel_data/voxelData_'+str(step-1)+'.dat') as f: 
            lines = (line for line in f if not line.startswith('#')) 
            voxs_alt = np.loadtxt(lines, delimiter=',', skiprows=0) 
        # 
        # Associate voxels to tirangles (The flux mapping file)
        tri_voxs,tri_sfracs = readVoxelTri('voxel_tri/triangle_voxels_'+str(step-1)+'.dat')
        
        #Triangles check between sparta and isthmus
        if len(self.COFormed) != len(tri_voxs):
            warnings.warn("No of triangles in sparta is not equal to isthmus, debug!!!")
         
        # Calculate mass of carbon associated with each voxel
        volFracC = float(cVolFrac)
        volC = volFracC*(self.lims[1,0]-self.lims[0,0])*(self.lims[1,1]-self.lims[0,1])*(self.lims[1,2]-self.lims[0,2])
        massC = volC*1800
        massCVox = massC/len(voxs_alt)
        # 
        # Calculate the mass of carbon removed from each voxel
        cRemovedVox = np.zeros((len(voxs_alt)))
        for i in range(len(self.COFormed)):
            vox_no = np.array((tri_voxs[(i+1)]),dtype = int)
            sfracs = np.array((tri_sfracs[(i+1)]),dtype = float)
            for k in range(len(vox_no)):
                cRemovedVox[vox_no[k]] = cRemovedVox[vox_no[k]] + sfracs[k] * self.COFormed[i,1]
        cRemovedVox[:] = cRemovedVox[:] + voxs_alt[:,3]
        # 
        # Remove voxels
        voxs_alt = np.column_stack((voxs_alt[:,0:3],cRemovedVox))
        for i in range(len(cRemovedVox)):
            if cRemovedVox[i] > massCVox:
                voxs_alt[i,:] = 0
        self.voxs_alt = voxs_alt[~np.all(voxs_alt == 0, axis=1)] 

        self.voxs = voxs_alt[:,0:3]

    def postProcess(self, cornerVolumes, vertices, faces, iteration):
        """
        Writes an STL file 'grids/grid_:iteration:.stl'.
        Writes a voxel data file 'voxel_data/voxel_data_:iteration:.dat'.
        Writes a volume fraction file 'volFrac.dat'.
        """
        # Write the stl file
        combinedMesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        combinedMesh.export('grids/grid_'+str(iteration)+'.stl', file_type='stl_ascii')

        # Write coordinate voxel data
        if iteration == 0:
            cRemovedVox = np.zeros((len(self.voxs),1))
            self.voxs_alt = np.column_stack((self.voxs,cRemovedVox))
        f = open('voxel_data/voxel_data_'+str(iteration)+'.dat','w+')
        for i in range(len(self.voxs_alt)):
            f.write(str(self.voxs_alt[i,0])+','+str(self.voxs_alt[i,1])+','+str(self.voxs_alt[i,2])+','+str(self.voxs_alt[i,3])+'\n')
        f.close()
        
        #
        # Write the file containing volume fraction of the material
        cVolFrac = np.sum(cornerVolumes)/(self.nCells[0]*self.nCells[1]*self.nCells[2])
        f = open('volFrac.dat','w+')
        f.write(str(cVolFrac)+'\n')
        f.close()

        self.voxs_types.update({'structure_voxs': self.voxs_layers, 
                                'rate_of_ablation_fiber': self.rate_of_ablation_fiber, 
                                'rate_of_ablation_matrix': self.rate_of_ablation_matrix})
        with open('voxel_data/types'+str(iteration)+'.dat', 'w+') as file:
            json.dump(self.voxs_types, file, indent=4)
    #

    def clean(self):
        # Remove temporary files
        os.remove('voxelData')
        os.remove('voxelTri')
        os.remove('volFrac.dat')
        print('Temporary directories removed')



    def _readReactionSPARTA(self,fileName):
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
                COFormed.append([float(s[0]),float(s[1])*(12*10**-3)*self.fnum*self.timescale/(self.avog*self.timestepDSMC)])
        f.close()
        #
        return np.array(COFormed)

