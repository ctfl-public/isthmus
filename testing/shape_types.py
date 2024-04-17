# -*- coding: utf-8 -*-
"""
shapes
^w^
"""
import numpy as np

# roughly, size is a 'half-length' for shapes

class Ellipsoid():
    def __init__(self, size):
        self.a = size # x dimension should be very thin .25
        self.b = size
        self.c = size # z dimension should be very long 1.5
    
    def is_voxel_valid(self,x,y,z):
        return (x/self.a)**2 + (y/self.b)**2 + (z/self.c)**2 < 1
    
    def get_shape_area(self):
        big_sum = (self.a*self.b)**1.6 + (self.a*self.c)**1.6 + (self.b*self.c)**1.6
        return 4*np.pi*(big_sum/3)**(1/1.6) # surface area of ellipsoid
    
    def get_shape_volume(self):
        return (4/3)*np.pi*self.a*self.b*self.c

class Cylinder():
    def __init__(self, size):
        self.length = size*2
        self.radius = self.length/6
        
    def is_voxel_valid(self,x,y,z):
        return (x**2 + y**2 < self.radius**2 and abs(z) < self.length/2)
    
    def get_shape_area(self):
        circle_sum = 2*np.pi*self.radius**2
        return circle_sum + np.pi*2*self.radius*self.length # surface area of cylinder
    
    def get_shape_volume(self):
        return self.length*np.pi*self.radius**2
    
class Cube():
    def __init__(self, size):
        self.hlength = size
        
    def is_voxel_valid(self,x,y,z):
        return (abs(x) < self.hlength and abs(y) < self.hlength and abs(z) < self.hlength)
        
    def get_shape_area(self):
        return 6*(self.hlength*2)**2
    
    def get_shape_volume(self):
        return (self.size*2)**3
    
def make_shape(v_size, lims, size, shape):
    print('Generating '+ shape + ' volume of voxels...', end='')
    
    voxs = []
    diff = (lims[1] - lims[0])
    nvox_1d = (diff/v_size).astype(int)
    for i in range(3):
        if (nvox_1d[i] % 2):
            nvox_1d[i] += 1
    nvox_1d = (nvox_1d/2 + 0.1).astype(int)
    
    if shape == 'ellipsoid':
        s_obj = Ellipsoid(size)
    elif shape == 'cylinder':
        s_obj = Cylinder(size)
    elif shape == 'cube':
        s_obj = Cube(size)
    else:
        raise Exception('Invalid shape type given')
    
    for i in range(nvox_1d[0]*2):
        x = -nvox_1d[0]*v_size + 0.5*v_size + i*v_size
        for j in range(nvox_1d[1]*2):
            y = -nvox_1d[1]*v_size + 0.5*v_size + j*v_size
            for k in range(nvox_1d[2]*2):
                z = -nvox_1d[2]*v_size + 0.5*v_size + k*v_size
                if (s_obj.is_voxel_valid(x,y,z)):
                    voxs.append([x,y,z])
        
    print(' {:d} voxels created'.format(len(voxs)))
    
    return np.array(voxs), s_obj.get_shape_area(), s_obj.get_shape_volume()
