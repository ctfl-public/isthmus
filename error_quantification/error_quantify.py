# %%
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

# %%
            
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
    def __init__(self, vertices, lines):
        self.sides = lines
        self.nsides = len(self.sides)

        self.empty_flag = 1
        if len(vertices):
            self.empty_flag = 0
            transverts = np.transpose(vertices)
            self.xlo = np.array([min(transverts[0]), min(transverts[1])])
            self.xhi = np.array([max(transverts[0]), max(transverts[1])])

    @classmethod
    def point_polygon(cls, vertices):
        sides = np.array([Line([vertices[i - 1], vertices[i]]) for i in range(len(vertices))])
        return cls(vertices, sides)

    @classmethod
    def line_polygon(cls, lines):
        vertices = []
        for sd in lines:
            vertices.append(sd.a)
            vertices.append(sd.b)
        return cls(vertices, lines)

    @classmethod
    def regular_polygon(cls, radius, nsides):
        vertices = []
        dtheta = 2*np.pi/nsides
        for i in range(nsides):
            # angles of corners on this line segment
            theta = np.pi/2 + dtheta*i
            vertices.append([radius*np.cos(theta), radius*np.sin(theta)])
        return cls.point_polygon(vertices)

    # # shoelace formula for 2D area of polygon
    # def get_polygon_area(self):
    #     max_y_mag = max(abs(np.transpose(self.vertices)[1]))
    #     area = 0
    #     apts = np.array([[pt[0], pt[1] + 1.2*max_y_mag] for pt in self.vertices])
    #     for i in range(len(apts)):
    #         pt1 = apts[i - 1]
    #         pt2 = apts[i]
    #         area += (pt2[1] + pt1[1])*(pt2[0] - pt1[0])
    #     area *= 0.5
    #     return abs(area)

    def check_point_inside(self, pt):
        for s in self.sides:
            poly_line = np.append(s.b - s.a, [0])
            point_line = np.append(pt - s.a, [0])
            zcomp = np.cross(poly_line, point_line)[2]
            if (zcomp < -1e-12):
                return 0
        return 1
    
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
        hv = self.size/2*0.95
        xs = [xc - hv, xc + hv, xc + hv, xc - hv, xc - hv]
        ys = [yc - hv, yc - hv, yc + hv, yc + hv, yc - hv]
        plt.plot(xs, ys, color=cl)

class Cell:
    corner_grid = []
    leaf_cell_lens = []
    full_polygon = None
    grid = None
    def __init__(self, ixlo, ixhi, parent):
        self.ixlo = np.array(ixlo) # [i,j] indices for [x,y] of minimum corner
        self.ixhi = np.array(ixhi) # same for max corner
        self.ncells = self.ixhi - self.ixlo  # [nx, ny] for cells in each direction
        self.cell_len = self.ncells*Cell.leaf_cell_lens # length of cell sides
        self.child_cells = []
        self.parent = parent

        # bounding box of corner objects, [bottom left, bottom right, top right, top left]
        i_min = self.ixlo[0]
        j_min = self.ixlo[1]
        i_max = self.ixhi[0]
        j_max = self.ixhi[1]
        self.corners = np.array([Cell.corner_grid[j_min][i_min], Cell.corner_grid[j_min][i_max],
                                 Cell.corner_grid[j_max][i_max], Cell.corner_grid[j_max][i_min]])
        self.xlo = self.corners[0].x
        self.xhi = self.corners[2].x
        self.center = (self.xhi + self.xlo)/2
        
        # 1 in , 0 out, -1 mixed
        self.in_flag = -1
        # marching squares topology index
        self.type = -1

        # empty except for leaf cells
        self.borders = [] # edges, used for marching squares interpolation
        self.voxels = []  # list of voxel objects owned by cell

        if all(self.ncells == 1):
            Cell.grid.cell_grid[ixlo[1]][ixlo[0]] = self
        elif any(self.ncells < 1):
            print('ERROR: fatal error in cell creation')
            exit(1)
        else:
            # split the longer dimension in roughly half
            a_dim = 0 if self.ncells[0] > self.ncells[1] else 1 # x (or y)
            b_dim = 1 if a_dim == 0 else 0            # y (or x)
            delta1 = int(self.ncells[a_dim]/2)    # lower half
            delta2 = self.ncells[a_dim] - delta1  # upper half

            # create lower half cell
            diff = np.zeros(2).astype(int)
            diff[a_dim] = delta1
            diff[b_dim] = self.ncells[b_dim]
            self.child_cells.append(Cell(ixlo, ixlo + diff, self))

            # create upper half cell
            diff[a_dim] = delta2
            self.child_cells.append(Cell(ixhi - diff, ixhi, self))

    @classmethod
    def root_cell(cls, ixlo, grid):
        # reset static variables and reinitialize
        cls.grid = grid
        cls.corner_grid = grid.corners # save the corner objects of entire grid
        cls.leaf_cell_lens = (grid.corners[1][1].x - grid.corners[0][0].x)
        cls.full_polygon = None
        return cls(ixlo, grid.ncells, None)

    @classmethod
    def root_sort_inout(cls, root, polygon):
        cls.full_polygon = polygon
        root.sort_inout(polygon)

    def clip_polyline(self, polyline):
        new_lines = []
        ext_xlo = self.xlo - 1e-15
        ext_xhi = self.xhi + 1e-15
        # check existence of polyline and overall bounding box first
        if not(polyline.empty_flag) and all(polyline.xlo < ext_xhi) and all(polyline.xhi > ext_xlo):
            # line by line
            for sd in polyline.sides:
                # check line bounding box
                if all(sd.xlo < ext_xhi) and all(sd.xhi > ext_xlo):
                    og_line = [sd.a, sd.b]
                    extrema = [min, max]
                    clipped_line = []
                    for i in range(2):
                        if all(og_line[i] > ext_xlo) and all(og_line[i] < ext_xhi):
                            clipped_line.append(og_line[i])
                        else:
                            # check for first intersection on 'i' side
                            intersects = []
                            t_vals = []
                            dx = [sd.b[0] - sd.a[0], sd.b[1] - sd.a[1]]
                            interim = np.ones(2)
                            for j in range(2):
                                k = (j + 1) % 2
                                if (abs(dx[j]) > 1e-30):
                                    for xlim in [self.xlo, self.xhi]:
                                        t = (xlim[j] - sd.a[j])/dx[j]
                                        y_int = t*dx[k] + sd.a[k]
                                        if t > 0 and t < 1 and y_int > ext_xlo[k] and y_int < ext_xhi[k]:
                                            t_vals.append(t)
                                            interim[j] = xlim[j]
                                            interim[k] = y_int
                                            intersects.append(interim)
                                
                            # select best intersect if any valid
                            if len(intersects):
                                ind = t_vals.index(extrema[i](t_vals))
                                clipped_line.append(intersects[ind])

                    if len(clipped_line) == 2:
                        new_lines.append(Line(clipped_line))
                    # elif len(clipped_line) == 1:
                    #     print('ERROR: failure to clip line to cell')
                    #     exit(1)

        return Polygon.line_polygon(new_lines)

    def sort_inout(self, polyline, inherit=-1):
        new_polyline = polyline
        if inherit == -1:
            # create new polyline with just relevant sections
            new_polyline = self.clip_polyline(polyline)

            # if no polyline -> use full poly to check CENTER point
            if len(new_polyline.sides) < 1:
                self.in_flag = Cell.full_polygon.check_point_inside(self.center)
        else:
            self.in_flag = inherit

        for c in self.child_cells:
            c.sort_inout(new_polyline, inherit=self.in_flag)

    def add_voxel(self, vox):
        self.voxels.append(vox)

    def plot(self):
        xlo = self.corners[0].x + 0.05*Cell.leaf_cell_lens
        xhi = self.corners[2].x - 0.05*Cell.leaf_cell_lens
        xs = [xlo[0], xhi[0], xhi[0], xlo[0], xlo[0]]
        ys = [xlo[1], xlo[1], xhi[1], xhi[1], xlo[1]]
        ltj = ['red', 'blue', 'green']
        plt.plot(xs, ys, color=ltj[self.in_flag])

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
        self.inside = -1 # 1 if inside, 0 if outside, -1 if unassigned
        self.frac = 0

class Grid:
    def __init__(self, polygon, origin, ncells, cell_len, scale_flag=False):
        self.scale_flag = scale_flag
        self.xlo = origin
        self.xhi = origin + ncells*cell_len
        self.ncells = ncells
        self.cell_len = cell_len

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

        # recursively create grid cells and reference in grid format
        self.cell_grid = np.zeros((ncells[1], ncells[0])).astype(int).tolist()
        self.root_cell = Cell.root_cell([0, 0], self)

        # sort cells by polygon
        Cell.root_sort_inout(self.root_cell, polygon)

    def sort(self, polygon):
        final_coords = []
        for s in polygon.sides:
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
        final_coords = np.array(final_coords)
        return final_coords

    def create_voxels(self, polygon, voxel_ratio):
        voxel_ratio = int(voxel_ratio)
        voxel_size = Cell.leaf_cell_lens[0]/voxel_ratio
        ny = self.root_cell.ncells[1]
        nx = self.root_cell.ncells[0]
        vg = [[Voxel([self.xlo[0] + (i + 0.5)*voxel_size, self.xlo[1] + (j + 0.5)*voxel_size], voxel_size)
               for i in range(nx*voxel_ratio)] for j in range(ny*voxel_ratio)]
        self.vox_grid = np.array(vg)
        for j in range(len(self.cell_grid)):
            for i in range(len(self.cell_grid[j])):
                c_cell = self.cell_grid[j][i]
                for n in range(voxel_ratio):
                    for m in range(voxel_ratio):
                        if c_cell.in_flag == 1:
                            vox_flag = True
                        elif c_cell.in_flag == 0:
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
                        if j == 0 or i == 0 or j == len(self.cell_grid) - 1 or i == len(self.cell_grid[0]) - 1:
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
        for j in range(len(self.cell_grid)):
            for i in range(len(self.cell_grid[j])):
                c_cell = self.cell_grid[j][i]
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
        for j in range(len(self.cell_grid)):
            for i in range(len(self.cell_grid[j])):
                c_cell = self.cell_grid[j][i]
                inside = np.array([c_cell.corners[a].frac + 1e-8 < 0.5 for a in range(4)])
                if not(all(inside)) and not(all(~inside)):
                    c_cell.set_topology()
                    c_cell.interpolate()

    def plot_ms_surface(self):
        for j in range(len(self.cell_grid)):
            for i in range(len(self.cell_grid[j])):
                for ccb in self.cell_grid[j][i].borders:
                    ccb.plot('pink')

    def plot_cells(self):
        for cl in range(len(self.cell_grid)):
            for c in range(len(self.cell_grid[cl])):
                self.cell_grid[cl][c].plot()

    def plot_voxels(self):
        for vl in self.vox_grid:
            for v in vl:
                if v.type == 0:
                    v.plot('black')

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

vrs = [30, 20, 10, 5, 1]

nsides = 5
circumradius = 2
vox_size = 0.01
signed_hausdorff = []
for x in [False]: #, True]:
    c_signed_hausdorff = []
    for vox_ratio in vrs:
        t0 = time.time()
        polygon = Polygon.regular_polygon(circumradius, nsides)

        l_c = vox_size*vox_ratio
        ncr = (int(1.2*circumradius/l_c + 1)*np.ones(2)).astype(int)
        offset = np.array([0, 0.1])*l_c
        origin = -ncr*l_c + offset
        ncells = 2*ncr
        grid = Grid(polygon, origin, ncells, l_c, scale_flag=x)
        
        # create voxelized surface from polygon
        grid.create_voxels(polygon, vox_ratio)

        # get a (simplified) marching squares surface from voxel grid
        grid.simple_ms_surface()

        # out red, in blue, mix black
        # plt.figure()
        # plt.gca().set_aspect('equal')
        # polygon.plot('purple')
        # grid.plot_cells()
        # grid.plot_voxels()
        # grid.plot_ms_surface()


#         csd = []
#         for mcs in grid.mc_surf:
#             dist1, norm1 = hausdorff_distance(polygon, mcs.a)
#             csd.append(dist1)

#             dist2, norm2 = hausdorff_distance(polygon, mcs.b)
#             csd.append(dist2)

#             # if dist2 > 0:
#             #     plt.plot([mcs.b[0], mcs.b[0] + dist2*norm2[0]],[mcs.b[1], mcs.b[1] + dist2*norm2[1]], color='green')
#             # else:
#             #     plt.plot([mcs.b[0], mcs.b[0] - dist2*norm2[0]],[mcs.b[1], mcs.b[1] - dist2*norm2[1]], color='red')
#             # if dist1 > 0:
#             #     plt.plot([mcs.a[0], mcs.a[0] + dist1*norm1[0]],[mcs.a[1], mcs.a[1] + dist1*norm1[1]], color='green')
#             # else:
#             #     plt.plot([mcs.a[0], mcs.a[0] - dist1*norm1[0]],[mcs.a[1], mcs.a[1] - dist1*norm1[1]], color='red')
#         #signed_hausdorff.append(np.array(csd)/l_c)
#         c_signed_hausdorff.append(np.array(csd)/l_c)

        # plt.xlim(-2.5, 2.5)
        # plt.ylim(-2.5, 2.5)
        # grid.fill()
        tf = time.time() - t0
        tsc = 1000*tf/(len(grid.cell_grid)*len(grid.cell_grid[0]))
        print('Time for {}: {:.2f} s ({:.2f} ms per cell)'.format(vox_ratio, tf, tsc))
        # print('Min: {:.2f} Max: {:.2f}'.format(min(c_signed_hausdorff[-1]), max(c_signed_hausdorff[-1])))
    
#     signed_hausdorff.append(c_signed_hausdorff)
#     print()

# plt.figure()
# sfac = 2
# poss = np.linspace(1, len(signed_hausdorff[0]), len(signed_hausdorff[0]))*sfac
# sh0 = plt.boxplot(signed_hausdorff[0], patch_artist=True, positions=poss-0.3)
# for box in sh0['boxes']:
#     box.set_facecolor('red')
# sh1 = plt.boxplot(signed_hausdorff[1], patch_artist=True, positions=poss+0.3)
# for box in sh1['boxes']:
#     box.set_facecolor('blue')
# plt.plot([0, max(poss)+sfac], [0, 0], color='black')
# plt.grid()
# plt.xticks(poss, vrs)
# plt.title('double-counted hausdorff')
# plt.xlabel('Cell / Voxel Size Ratio')
# plt.ylabel('Hausdorff Distance / Cell Length')
# plt.xlim(0, max(poss)+sfac)
# plt.ylim(-0.5, 0.5)

# %%