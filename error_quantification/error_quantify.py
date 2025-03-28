"""
Error Prediction Calculations for Isthmus
Ethan Huff (ya boi)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import ConvexHull
import sys
import time

# %% estimate error of approximating polygons using curves

eps_cs = [0.0001, 0.001, 0.01, 0.05] # acceptable area defect ratio relative to original polygon area, A_defect/A_polygon
n = np.linspace(3, 20, 18).astype(int) # number of sides of polygon

# radius of curvature ratio wrt polygon circumradius
plt.plot(n, np.cos(np.pi/n), color='pink', linestyle='dashed', label='Inscribed Radius')
for eps_c in eps_cs:
    curve_ratio = np.sqrt(eps_c * np.sin(2*np.pi/n) / (2*(np.tan(np.pi/n) - np.pi/n)))
    plt.plot(n, curve_ratio, label='e = {:.2f}%'.format(eps_c*100))
plt.grid()
plt.xticks(ticks=n, labels=n)
plt.xlabel('Number of Sides')
plt.ylabel('Curvature Radius/Circumradius')
plt.legend()
plt.xlim(3, 20)
plt.ylim(0, 1.2)


def get_area(points):
    max_y_mag = max(abs(np.transpose(points)[1]))
    area = 0
    apts = np.array([[pt[0], pt[1] + 1.2*max_y_mag] for pt in points])
    for i in range(len(apts)):
        pt1 = apts[i - 1]
        pt2 = apts[i]
        area += (pt2[1] + pt1[1])*(pt2[0] - pt1[0])
    area *= 0.5
    return abs(area)

# %% approximate polygons using curves at corners
class Arc:
    def __init__(self, center, radius, ext_angles=[]):
        self.center = np.array(center)
        self.radius = radius

        self.full = False
        self.ext_angles = np.array([0, 2*np.pi])
        if len(ext_angles) == 0:
            self.full = True
        else:
            self.ext_angles[0] = ext_angles[0]
            self.ext_angles[1] = ext_angles[1]

        # [x,y] pairs to have a numerical approximation of the arc
        self.num_pts = []
        nsides = 300
        if (self.ext_angles[1] - self.ext_angles[0]) > 0:
            dt = (self.ext_angles[1] - self.ext_angles[0])/nsides
        else:
            dt = (self.ext_angles[1] - self.ext_angles[0] + 2*np.pi)/nsides
        for i in range(nsides + 1):
            x = self.radius*np.cos(self.ext_angles[0] + i*dt) + self.center[0]
            y = self.radius*np.sin(self.ext_angles[0] + i*dt) + self.center[1]
            self.num_pts.append([x,y])
        self.num_pts = np.array(self.num_pts)
        transend = np.transpose(self.num_pts)
        self.xlo = np.array([min(transend[0]), min(transend[1])])
        self.xhi = np.array([max(transend[0]), max(transend[1])])

    def get_ends(self):
        a = np.array([self.radius*np.cos(self.ext_angles[0]) + self.center[0],
                      self.radius*np.sin(self.ext_angles[0]) + self.center[1]])
        b = np.array([self.radius*np.cos(self.ext_angles[1]) + self.center[0],
                      self.radius*np.sin(self.ext_angles[1]) + self.center[1]])
        return np.array([a,b])

    
    def plot(self, clr):
        coords = np.transpose(self.num_pts)
        plt.plot(coords[0], coords[1], color=clr)

    def get_x_from_y(self, y):
        diff = np.sqrt(self.radius**2 - (y - self.center[1])**2)
        if abs(diff) < 1e-5:
            xs = [self.center[0]]
        else:
            xs = [self.center[0] - diff, self.center[0] + diff]
        fxs = []
        for x in xs:
            theta = np.arctan2(y - self.center[1], x - self.center[0])
            if theta < 0:
                theta += 2*np.pi
            if self.ext_angles[1] > self.ext_angles[0]:
                if theta >= self.ext_angles[0] and theta <= self.ext_angles[1]:
                    fxs.append(x)
            else:
                if (self.ext_angles[0] < theta and theta <= 2*np.pi + 1e-12) or \
                    -1e-12 < theta and theta < self.ext_angles[1]:
                    fxs.append(x)
        return fxs
        
    def get_y_from_x(self, x):
        diff = np.sqrt(self.radius**2 - (x - self.center[0])**2)
        if abs(diff) < 1e-5:
            ys =  [self.center[1]]
        else:
            ys =  [self.center[1] - diff, self.center[1] + diff]
        fys = []
        for y in ys:
            theta = np.arctan2(y - self.center[1], x - self.center[0])
            if theta < 0:
                theta += 2*np.pi
            if self.ext_angles[1] > self.ext_angles[0]:
                if theta >= self.ext_angles[0] and theta <= self.ext_angles[1]:
                    fys.append(y)
            else:
                if (self.ext_angles[0] < theta and theta <= 2*np.pi + 1e-12) or \
                    -1e-12 < theta and theta < self.ext_angles[1]:
                    fys.append(y)
        return fys
            
class Line:
    def __init__(self, endpts, locs=[]):
        self.vertical = 0
        self.endpts = np.array(endpts) # [[x1, y1], [x2, y2]]
        self.a = self.endpts[0] # [x,y]
        self.b = self.endpts[1]
        transend = np.transpose(self.endpts)
        self.xlo = np.array([min(transend[0]), min(transend[1])])
        self.xhi = np.array([max(transend[0]), max(transend[1])])
        self.radius = 1e6
        self.theta = np.arctan2((self.b[1] - self.a[1]), (self.b[0] - self.a[0]))
        if len(locs) == 0:
            self.locs = [-1, -1]
        elif len(locs) == 2:
            self.locs = [locs[0], locs[1]]
        else:
            print('ERROR: invalid line endpoint positions')
            sys.exit(1)
        
        self.m = self.bi = 0

        if (abs(self.theta) < 1e-4):
            self.vertical = -1 # horizontal
        elif (abs(abs(self.theta) - np.pi/2) < 1e-4):
            self.vertical = 1 # vertical
        else:
            self.m = (self.b[1] - self.a[1])/(self.b[0] - self.a[0])
            self.bi = self.a[1] - self.m*self.a[0]

    def get_ends(self):
        return self.endpts

    def plot(self, clr):
        ppts = np.transpose([self.a, self.b])
        plt.plot(ppts[0], ppts[1], color=clr)

    def get_y_from_x(self, x):
        if self.vertical == 1:
            return None
        elif self.vertical == -1:
            return self.a[1]
        else:
            return self.m*x + self.bi
        
    def get_x_from_y(self, y):
        if self.vertical == 1:
            return self.a[0]
        elif self.vertical == -1:
            return None
        else:
            return (y - self.bi)/self.m

class Polygon:
    def __init__(self, sides):
        self.sides = np.array([Line([sides[i - 1], sides[i]]) for i in range(len(sides))])
        self.nsides = len(sides)
        transsides = np.transpose(sides)
        self.xlo = np.array([min(transsides[0]), min(transsides[1])])
        self.xhi = np.array([max(transsides[0]), max(transsides[1])])

    def check_point_inside(self, pt):
        for s in self.sides:
            poly_line = np.append(s.b - s.a, [0])
            point_line = np.append(pt - s.a, [0])
            zcomp = np.cross(poly_line, point_line)[2]
            if (zcomp < -1e-12):
                return False
        return True


# ordered list of oriented line segments
class RegularPolygon:
    def __init__(self, radius, nsides):
        self.sides = []
        self.nsides = nsides
        self.radius = radius # circumcircle
        self.bbox = np.array([[-radius, -radius], [radius, radius]])

        dtheta = 2*np.pi/self.nsides
        r = self.radius
        for i in range(self.nsides):
            # angles of corners on this line segment
            th1 = np.pi/2 + dtheta*i
            th2 = th1 + dtheta
            # [[x1, y1], [x2, y2]]
            self.sides.append(Line([[r*np.cos(th1), r*np.sin(th1)], [r*np.cos(th2), r*np.sin(th2)]]))

    def plot(self, color):
        for s in self.sides:
            s.plot(color)

    def check_point_inside(self, pt):
        for s in self.sides:
            poly_line = np.append(s.b - s.a, [0])
            point_line = np.append(pt - s.a, [0])
            zcomp = np.cross(poly_line, point_line)[2]
            if (zcomp < -1e-12):
                return False
        return True

    def make_polycurve(self, curve_ratio):
        curve_sides = []
        curve_rad = self.radius*curve_ratio
        for i in range(len(self.sides)):
            s1 = self.sides[i - 1]
            s2 = self.sides[i]
            # angle ABC is phi
            A = s1.a
            B = s1.b
            C = s2.b
            phi = np.arccos(np.dot(A - B, C - B)/(np.linalg.norm(A - B)*np.linalg.norm(C - B)))

            # corner of current side
            # get curvature center from corner
            theta = np.arctan2(B[1], B[0])
            r = np.linalg.norm(B)
            lr = curve_rad/np.tan(phi/2)
            curve_center = (r - (np.sqrt(lr**2 + curve_rad**2)))*np.array([np.cos(theta), np.sin(theta)])

            # the curved sector has an angle of (pi - phi) centered at theta
            angle1 = theta - (np.pi - phi)/2
            if (angle1 < 0):
                angle1 += 2*np.pi
            angle2 = theta + (np.pi - phi)/2
            if (angle2 < 0):
                angle2 += 2*np.pi
            curve_sides.append(Arc(curve_center, curve_rad, [angle1, angle2]))

            # remove length lr from s2 on each side to make room for arc
            midpt = (B + C)/2 # [xc, yc]
            dx = C - midpt # [dx, dy]
            lm = np.linalg.norm(dx)
            ndx = ((lm - lr)/lm) * dx
            curve_sides.append(Line([midpt - ndx, midpt + ndx]))


        polycurve = Polycurve(curve_sides, self)
        return polycurve

class Polycurve:
    def __init__(self, sides, source):
        # ordered list of arcs and lines
        self.sides = sides
        self.polygon = source # original polygon
        self.min_curve_rad = 1e6
        for s in self.sides:
            if (s.radius < self.min_curve_rad):
                self.min_curve_rad = s.radius

    def plot(self, color):
        for s in self.sides:
            s.plot(color)

class Voxel:
    def __init__(self, x, v_size):
        self.x = np.array(x) # [x,y] of center
        self.size = v_size # side length
        self.type = -1
        self.weight = 0

    def binary_fill(self):
        self.weight = 1 # smoothed volume weight
        self.type = 0

    def plot(self, cl):
        xc = self.x[0]
        yc = self.x[1]
        hv = self.size/2
        xs = [xc - hv, xc + hv, xc + hv, xc - hv, xc - hv]
        ys = [yc - hv, yc - hv, yc + hv, yc + hv, yc - hv]
        plt.plot(xs, ys, color=cl)

class Cell:
    def __init__(self, xlo, xhi, corners):
        # coordinates
        self.xlo = np.array(xlo)
        self.xhi = np.array(xhi)
        self.center = (self.xlo + self.xhi)/2
        self.borders = []

        # length of square cell
        self.cell_len = (xhi - xlo)[0]

        # list of corner objects associated
        # [bottom left, bottom right, top right, top left]
        self.corners = np.array(corners)
        assert len(corners) == 4, 'ERROR: invalid number of corners ({}) provided to cell'.format(len(corners))

        self.type = 0 # 1 in, 0 out, -1 mixed
        self.voxels = [] # list of voxel objects

    def add_voxel(self, vox):
        self.voxels.append(vox)

    def plot(self, clr):
        xs = [self.xlo[0], self.xhi[0], self.xhi[0], self.xlo[0], self.xlo[0]]
        ys = [self.xlo[1], self.xlo[1], self.xhi[1], self.xhi[1], self.xlo[1]]
        plt.plot(xs, ys, color=clr)

    def set_topology(self):
        self.type = self.corners[3].inside*8 + \
                    self.corners[2].inside*4 + \
                    self.corners[1].inside*2 + \
                    self.corners[0].inside
    
        # available locs of cell edges
        diff = np.array([[ 0.0, -0.5],  # bottom
                         [ 0.5,  0.0],  # right
                         [ 0.0,  0.5],  # top
                         [-0.5,  0.0]]) # left

        # fully outside or inside, no surface; further labels are for inside regions
        if self.type == 0 or self.type == 15:
            self.borders = []

        # two surface elements
        elif self.type == 5 or self.type == 10:
            # bottom left and top right diagonal
            if self.type == 5:
                loc1 = [0, 1]
                loc2 = [2, 3]
            # top left and bottom right diagonal
            elif self.type == 10:
                loc1 = [1, 2]
                loc2 = [3, 0]
            a1 = self.center + self.cell_len*diff[loc1[0]]
            b1 = self.center + self.cell_len*diff[loc1[1]]
            a2 = self.center + self.cell_len*diff[loc2[0]]
            b2 = self.center + self.cell_len*diff[loc2[1]]
            self.borders = [Line([a1, b1], loc1), Line([a2, b2], loc2)]

        # one surface elements
        else:
            # bottom left
            if self.type == 1:
                loc = [0, 3]
            # bottom right
            elif self.type == 2:
                loc = [1, 0]
            # bottom half
            elif self.type == 3:
                loc = [1, 3]
            # top right
            elif self.type == 4:
                loc = [2, 1]
            # right half
            elif self.type == 6:
                loc = [2, 0]
            # all but top left
            elif self.type == 7:
                loc = [2, 3]
            # top left
            elif self.type == 8:
                loc = [3, 2]
            # left half
            elif self.type == 9:
                loc = [0, 2]
            # all but top right
            elif self.type == 11:
                loc = [1, 2]
            # top half
            elif self.type == 12:
                loc = [3, 1]
            # all but bottom right
            elif self.type == 13:
                loc = [0, 1]
            # all but bottom left
            elif self.type == 14:
                loc = [3, 0]
            else:
                print('ERROR: invalid type {} for marching squares cell')
                sys.exit(1)
            a = self.center + self.cell_len*diff[loc[0]]
            b = self.center + self.cell_len*diff[loc[1]]
            self.borders = [Line([a,b], loc)]
        
    def interpolate(self):
        new_borders = []
        thresh = 0.499
        for brd in self.borders:
            loc = brd.locs
            new_endpts = []
            for i in range(2):
                pt_new = [brd.endpts[i][0], brd.endpts[i][1]]
                if loc[i] == 0:
                    corn1 = self.corners[0]
                    corn2 = self.corners[1]
                    x_new = ( (thresh - corn1.frac)/(corn2.frac - corn1.frac) )*(corn2.x[0] - corn1.x[0]) + corn1.x[0]
                    pt_new[0] = x_new
                elif loc[i] == 1:
                    corn1 = self.corners[1]
                    corn2 = self.corners[2]
                    y_new = ( (thresh - corn1.frac)/(corn2.frac - corn1.frac) )*(corn2.x[1] - corn1.x[1]) + corn1.x[1]
                    pt_new[1] = y_new
                elif loc[i] == 2:
                    corn1 = self.corners[2]
                    corn2 = self.corners[3]
                    x_new = ( (thresh - corn1.frac)/(corn2.frac - corn1.frac) )*(corn2.x[0] - corn1.x[0]) + corn1.x[0]
                    pt_new[0] = x_new
                elif loc[i] == 3:
                    corn1 = self.corners[3]
                    corn2 = self.corners[0]
                    y_new = ( (thresh - corn1.frac)/(corn2.frac - corn1.frac) )*(corn2.x[1] - corn1.x[1]) + corn1.x[1]
                    pt_new[1] = y_new
                else:
                    print('ERROR: invalid edge type {}'.format(loc[0]))
                    sys.exit(1)
                new_endpts.append(np.array(pt_new))
            new_borders.append(Line([new_endpts[0], new_endpts[1]], loc))
        self.borders = new_borders

            

class Corner:
    def __init__(self, x):
        self.x = x
        self.inside = 0 # 1 if inside, 0 if outside
        self.frac = 0

class Grid:
    def __init__(self, origin, ncells, cell_len, scale_flag=False):
        self.scale_flag = scale_flag
        self.xlo = origin
        self.xhi = origin + ncells*cell_len
        self.ncells = ncells
        self.cell_len = cell_len
        self.mc_surf = [] # list of lines
        self.cells = []

        # possible x and y coordinates of corners in grid
        self.ylines = []
        self.xlines=  []
        for j in range(ncells[1] + 1):
            self.ylines.append(self.xlo[1] + self.cell_len*j)
        for i in range(ncells[0] + 1):
            self.xlines.append(self.xlo[0] + self.cell_len*i)

        # create grid corners
        self.corners = []
        for j in range(ncells[1] + 1):
            corner_line = []
            for i in range(ncells[0] + 1):
                cx = self.xlo + self.cell_len*np.array([i,j])
                corner_line.append(Corner(cx))
            self.corners.append(corner_line)

        # create cells of grid
        for j in range(ncells[1]):
            cell_line = []
            for i in range(ncells[0]):
                cxlo = self.xlo + self.cell_len*np.array([i,j])
                cxhi = cxlo + np.ones(2)*self.cell_len
                cell_line.append(Cell(cxlo, cxhi, [self.corners[j][i], 
                                                   self.corners[j][i + 1], 
                                                   self.corners[j + 1][i + 1], 
                                                   self.corners[j + 1][i]]))
            self.cells.append(cell_line)


    def sort(self, polycurve):
        final_coords = []
        for s in polycurve.sides:
            if type(s) == Line:
                x_intersect_pts = []
                for xl in self.xlines:
                    if xl >= s.xlo[0] and xl <= s.xhi[0]:
                        c_y = s.get_y_from_x(xl)
                        if c_y != None:
                            x_intersect_pts.append([xl, c_y])
                y_intersect_pts = []
                for yl in self.ylines:
                    if yl >= s.xlo[1] and yl <= s.xhi[1]:
                        c_x = s.get_x_from_y(yl)
                        if c_x != None:
                            y_intersect_pts.append([c_x, yl])
                final_coords += x_intersect_pts
                final_coords += y_intersect_pts
            elif type(s) == Arc:
                x_intersect_pts = []
                for xl in self.xlines:
                    if xl >= s.xlo[0] and xl <= s.xhi[0]:
                        c_ys = s.get_y_from_x(xl)
                        if c_ys != None:
                            for i in range(len(c_ys)):
                                x_intersect_pts.append([xl, c_ys[i]])
                y_intersect_pts = []
                for yl in self.ylines:
                    if yl >= s.xlo[1] and yl <= s.xhi[1]:
                        c_xs = s.get_x_from_y(yl)
                        if c_xs != None:
                            for i in range(len(c_xs)):
                                y_intersect_pts.append([c_xs[i], yl])
                final_coords += x_intersect_pts
                final_coords += y_intersect_pts
        final_coords = np.array(final_coords)
        return final_coords

    def create_voxels(self, polygon, voxel_ratio):
        voxel_ratio = int(voxel_ratio)
        voxel_size = self.cells[0][0].cell_len/voxel_ratio
        ny = len(self.cells)
        nx = len(self.cells[0])
        vg = [[Voxel([self.xlo[0] + (i + 0.5)*voxel_size, self.xlo[1] + (j + 0.5)*voxel_size], voxel_size)
               for i in range(nx*voxel_ratio)] for j in range(ny*voxel_ratio)]
        self.vox_grid = np.array(vg)
        for j in range(len(self.cells)):
            for i in range(len(self.cells[j])):
                c_cell = self.cells[j][i]
                corner_inside = []
                inside_flag = 0 # 0 edge, 1 inside, -1 outside
                # check if cell is inside (assumes convex polygon)
                for cn in c_cell.corners:
                    cn_flag = polygon.check_point_inside(cn.x)
                    corner_inside.append(cn_flag)
                corner_inside = np.array(corner_inside)
                if all(corner_inside == True):
                    inside_flag = 1
                elif all(corner_inside == False):
                    inside_flag = -1
                for n in range(voxel_ratio):
                    for m in range(voxel_ratio):
                        if inside_flag == 1:
                            vox_flag = True
                        elif inside_flag == -1:
                            vox_flag = False
                        else:
                            diff = np.array([m, n])
                            vx = c_cell.xlo + voxel_size*(0.5 + diff)
                            vox_flag = polygon.check_point_inside(vx)
                        
                        vy = voxel_ratio*j + n
                        vx = voxel_ratio*i + m
                        c_vox = self.vox_grid[vy][vx]
                        if vox_flag:
                            c_vox.binary_fill()
                        if j == 0 or i == 0 or j == len(self.vox_grid) - 1 or i == len(self.vox_grid[0]) - 1:
                            c_vox.weight = 0
                            c_vox.type = int(-1e5)
                        c_cell.add_voxel(c_vox)

        # set voxel weights to something other than 0 or 1
        if self.scale_flag:
            level = 0
            w_max = np.ceil((3*voxel_ratio/2) - 0.5)
            w_min = np.floor(-(3*voxel_ratio/2) - 0.5)
            while level <= w_max or (-level - 1) >= w_min:
                t1 = level
                t2 = -level - 1
                for j in range(1, len(self.vox_grid) - 1):
                    for i in range(1, len(self.vox_grid[j]) - 1):
                        vox = self.vox_grid[j][i]
                        if vox.type == t1:
                            # check 4 cardinal neighbors, initially assume surrounded by voxels
                            surrounded = True

                            if self.vox_grid[j][i - 1].type < vox.type:
                                surrounded = False
                            if self.vox_grid[j - 1][i].type < vox.type:
                                surrounded = False
                            if self.vox_grid[j][i + 1].type < vox.type:
                                surrounded = False
                            if self.vox_grid[j + 1][i].type < vox.type:
                                surrounded = False

                            if surrounded:
                                vox.type = t1 + 1
                        
                        elif vox.type == t2:
                            # check 4 cardinal neighbors, initially assume surrounded by voxels
                            surrounded = True
                            if self.vox_grid[j][i - 1].type > vox.type:
                                surrounded = False
                            if self.vox_grid[j - 1][i].type > vox.type:
                                surrounded = False
                            if self.vox_grid[j][i + 1].type > vox.type:
                                surrounded = False
                            if self.vox_grid[j + 1][i].type > vox.type:
                                surrounded = False

                            if surrounded:
                                vox.type = t2 - 1
                level += 1
                assert(level < 1000)
            
            for j in range(len(self.vox_grid)):
                for i in range(len(self.vox_grid[j])):
                    vox = self.vox_grid[j][i]
                    vox.weight = 0.5*(1 + (0.5 + vox.type)*(2/(3*voxel_ratio)))
                    vox.weight = min(vox.weight, 1.0)
                    vox.weight = max(0.0, vox.weight)



        # fill corners of cells with voxel mass
        for j in range(len(self.cells)):
            for i in range(len(self.cells[j])):
                c_cell = self.cells[j][i]
                for vox in c_cell.voxels:
                    if vox.weight > 1e-12:
                        # split area between corners
                        # left-right (x axis), then bottom-top (y axis)
                        ylow = [0, 0]
                        for i in range(2):
                            ind = (c_cell.center[i] - (vox.x[i] - 0.5*vox.size))/vox.size
                            if ind > 1.0:
                                ylow[i] = 1.0
                            elif ind < 0:
                                ylow[i] = 0.0
                            else:
                                ylow[i] = ind

                        c_cell.corners[0].frac += vox.weight*(ylow[0]*ylow[1])
                        c_cell.corners[1].frac += vox.weight*((1 - ylow[0])*ylow[1])
                        c_cell.corners[2].frac += vox.weight*((1 - ylow[0])*(1 - ylow[1]))
                        c_cell.corners[3].frac += vox.weight*(ylow[0]*(1 - ylow[1]))
        
        for j in range(len(self.corners)):
            for i in range(len(self.corners[j])):
                c_corn = self.corners[j][i]
                c_corn.frac /= voxel_ratio**2 # scale area relative to cell
                if c_corn.frac > 0.499:
                    c_corn.inside = 1
                else:
                    c_corn.inside = 0

    def simple_ms_surface(self):
        for j in range(len(self.cells)):
            for i in range(len(self.cells[j])):
                c_cell = self.cells[j][i]
                inside = np.array([c_cell.corners[a].frac + 1e-8 < 0.5 for a in range(4)])
                if not(all(inside)) and not(all(~inside)):
                    c_cell.set_topology()
                    # for ccb in c_cell.borders:
                    #     ccb.plot('green')
                    c_cell.interpolate()
                    for ccb in c_cell.borders:
                        self.mc_surf.append(ccb)

    def plot_ms_surface(self):
        for j in range(len(self.cells)):
            for i in range(len(self.cells[j])):
                c_cell = self.cells[j][i]
                inside = np.array([c_cell.corners[a].frac + 1e-8 < 0.5 for a in range(4)])
                if not(all(inside)) and not(all(~inside)):
                    for ccb in c_cell.borders:
                        ccb.plot('black')


    def fill(self):
        c0 = []
        c25 = []
        c50 = []
        c75 = []
        c100 = []
        for j in range(len(self.corners)):
            for i in range(len(self.corners[j])):
                c_corner = self.corners[j][i]
                mfrac = round(c_corner.frac*4)
                if (mfrac == 0):
                    c0.append(c_corner.x)
                elif (mfrac == 1):
                    c25.append(c_corner.x)
                elif (mfrac == 2):
                    c50.append(c_corner.x)
                elif (mfrac == 3):
                    c75.append(c_corner.x)
                elif (mfrac == 4):
                    c100.append(c_corner.x)
                else:
                    print('ERROR: Invalid corner fraction {}'.format(c_corner.frac))
                    sys.exit(1)
        c0 = np.transpose(c0)
        if (len(c0) > 0):
            plt.scatter(c0[0], c0[1], color='purple')
        c25 = np.transpose(c25)
        if (len(c25) > 0):
            plt.scatter(c25[0], c25[1], color='blue')
        c50 = np.transpose(c50)
        if (len(c50) > 0):
            plt.scatter(c50[0], c50[1], color='green')
        c75 = np.transpose(c75)
        if (len(c75) > 0):
            plt.scatter(c75[0], c75[1], color='orange')
        c100 = np.transpose(c100)
        if (len(c100) > 0):
            plt.scatter(c100[0], c100[1], color='red')
    
        for j in range(len(self.cells)):
            for i in range(len(self.cells[j])):
                for vx in self.cells[j][i].voxels:
                    if vx.type == -1:
                        vx.plot('grey')
                    elif vx.type == 0:
                        vx.plot('orange')

    def plot(self, clr):
        for cl in self.cells:
            for c in cl:
                c.plot(clr)

# minimum distance of point x to any part of the polygon
def hausdorff_distance(polygon, x):
    dist = 1e12
    norm = [0,0,0]
    for line in polygon.sides:
        ax = np.append(x - line.a, [0])
        ab = np.append(line.b - line.a, [0])
        line_length = np.linalg.norm(ab)
        cross = np.cross(ax, ab)
        frac = np.dot(ax, ab)/line_length**2
        if frac < 0:
            c_dist = np.linalg.norm(ax)
            c_norm = -ax/c_dist
        elif frac > 1:
            vec = np.append(x - line.b, [0])
            c_dist = np.linalg.norm(vec)
            c_norm = -vec/c_dist
        else:
            norm_cross = np.linalg.norm(cross)
            if norm_cross < 1e-10:
                c_dist = 0
                c_norm = np.zeros(3)
            else:
                c_dist = norm_cross/line_length
                bign = np.cross(ab, cross)
                c_norm = -bign/np.linalg.norm(bign)
        
        if c_dist < abs(dist):
            dist = c_dist
            if cross[2] > 0:
                dist *= -1
            norm = [c_norm[i] for i in range(3)]
    return dist, norm



vrs = [30, 20, 10, 5]

nsides = 5
circumradius = 2
vox_size = 0.01
signed_hausdorff = []
for x in [False, True]:
    c_signed_hausdorff = []
    for vox_ratio in vrs:
        t0 = time.time()
        polygon = RegularPolygon(circumradius, nsides)

        # plt.figure()
        # plt.gca().set_aspect('equal')
        # polygon.plot('blue')

        l_c = vox_size*vox_ratio
        ncr = (int(1.2*circumradius/l_c + 4)*np.ones(2)).astype(int)
        offset = np.array([0, 0.1])*l_c
        origin = -ncr*l_c + offset
        ncells = 2*ncr
        grid = Grid(origin, ncells, l_c, scale_flag=x)

        # create voxelized surface from approximating polygon
        grid.create_voxels(polygon, vox_ratio)

        # get a (simplified) marching squares surface from voxel grid
        grid.simple_ms_surface()
        # grid.plot_ms_surface()

        csd = []
        for mcs in grid.mc_surf:
            dist1, norm1 = hausdorff_distance(polygon, mcs.a)
            csd.append(dist1)

            dist2, norm2 = hausdorff_distance(polygon, mcs.b)
            csd.append(dist2)

            # if dist2 > 0:
            #     plt.plot([mcs.b[0], mcs.b[0] + dist2*norm2[0]],[mcs.b[1], mcs.b[1] + dist2*norm2[1]], color='green')
            # else:
            #     plt.plot([mcs.b[0], mcs.b[0] - dist2*norm2[0]],[mcs.b[1], mcs.b[1] - dist2*norm2[1]], color='red')
            # if dist1 > 0:
            #     plt.plot([mcs.a[0], mcs.a[0] + dist1*norm1[0]],[mcs.a[1], mcs.a[1] + dist1*norm1[1]], color='green')
            # else:
            #     plt.plot([mcs.a[0], mcs.a[0] - dist1*norm1[0]],[mcs.a[1], mcs.a[1] - dist1*norm1[1]], color='red')
        #signed_hausdorff.append(np.array(csd)/l_c)
        c_signed_hausdorff.append(np.array(csd)/l_c)

        # plt.xlim(-2.5, 2.5)
        # plt.ylim(-2.5, 2.5)
        # grid.plot('black')
        # polycurve.plot('red')
        # grid.fill()
        print('Time for {}: {:.2f}'.format(vox_ratio, time.time() - t0))
        print('Min: {:.2f} Max: {:.2f}'.format(min(c_signed_hausdorff[-1]), max(c_signed_hausdorff[-1])))
    
    signed_hausdorff.append(c_signed_hausdorff)
    print()

plt.figure()
sfac = 2
poss = np.linspace(1, len(signed_hausdorff[0]), len(signed_hausdorff[0]))*sfac
sh0 = plt.boxplot(signed_hausdorff[0], patch_artist=True, positions=poss-0.3)
for box in sh0['boxes']:
    box.set_facecolor('red')
sh1 = plt.boxplot(signed_hausdorff[1], patch_artist=True, positions=poss+0.3)
for box in sh1['boxes']:
    box.set_facecolor('blue')
plt.plot([0, max(poss)+sfac], [0, 0], color='black')
plt.grid()
plt.xticks(poss, vrs)
plt.title('double-counted hausdorff')
plt.xlabel('Cell / Voxel Size Ratio')
plt.ylabel('Hausdorff Distance / Cell Length')
plt.xlim(0, max(poss)+sfac)
plt.ylim(-0.5, 0.5)


# %%

# voxel fill ratios for corners
vox_ratios = np.linspace(2, 10, 5).astype(int)
allowed_fills = []

plt.figure()
for vr in vox_ratios:
    nmax = vr**2
    c_allowed = []
    for n in range(nmax + 1):
        c_allowed.append(n/nmax)
    plt.scatter(c_allowed, [vr]*len(c_allowed))
xts = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
plt.xticks(ticks=xts, labels=xts)
plt.grid()

# %%
