# %%
"""
Error Prediction Calculations for Isthmus
Ethan Huff (ya boi)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib import colors
import sys
import time
import copy

# %%
fig_save = False
glob_dpi = 100
class Line:
    def __init__(self, endpts, locs=[]):
        self.endpts = np.array(endpts) # [[x1, y1], [x2, y2]]
        self.a = self.endpts[0] # [x,y]
        self.b = self.endpts[1]
        self.length = np.linalg.norm(self.b - self.a)
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

        self.vox_refs = []
        self.vox_fracs = []
        self.scalar = 0

    def get_ends(self):
        return self.endpts

    def plot(self, clr, lst):
        ppts = np.transpose([self.a, self.b])
        plt.plot(ppts[0], ppts[1], color=clr, linestyle=lst, linewidth=2, zorder=20)

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
    def regular_polygon(cls, radius, nsides, rot):
        vertices = []
        dtheta = 2*np.pi/nsides
        rad_rot = (np.pi/180)*rot
        for i in range(nsides):
            # angles of corners on this line segment
            theta = np.pi/2 + dtheta*i + rad_rot
            vertices.append([radius*np.cos(theta), radius*np.sin(theta)])
        return cls.point_polygon(vertices)

    def get_perim(self):
        perim = 0
        for line in self.sides:
            perim += np.linalg.norm(line.b - line.a)
        return perim

    # shoelace formula for 2D area of polygon
    def get_polygon_area(self):
        max_y_mag = max(abs(self.xlo[1]), abs(self.xhi[1]))
        area = 0
        for line in self.sides:
            pt1 = line.a + [0, 1.2*max_y_mag]
            pt2 = line.b + [0, 1.2*max_y_mag]
            area += (pt2[1] + pt1[1])*(pt2[0] - pt1[0])
        area *= 0.5
        return abs(area)

    def check_point_inside(self, pt):
        for s in self.sides:
            poly_line = np.append(s.b - s.a, [0])
            point_line = np.append(pt - s.a, [0])
            zcomp = np.cross(poly_line, point_line)[2]
            if (zcomp < -1e-12):
                return 0
        return 1
    
    def plot(self, color, lst):
        for s in self.sides:
            s.plot(color, lst)


class Voxel:
    def __init__(self, x, v_size, loc):
        self.x = np.array(x) # [x,y] of center
        self.size = v_size # side length
        self.type = -1
        self.weight = 0
        self.finalized = False
        self.integrity = 0.0
        self.loc = loc

        crns = [self.x + np.array([-1,-1])*v_size/2,
                self.x + np.array([1,-1])*v_size/2,
                self.x + np.array([1,1])*v_size/2,
                self.x + np.array([-1,1])*v_size/2]
        self.faces = [Voxel_Face(0, crns[3], crns[0]),
                      Voxel_Face(1, crns[1], crns[2]),
                      Voxel_Face(2, crns[0], crns[1]),
                      Voxel_Face(3, crns[2], crns[3])]
        
        self.proj_surfaces = []
        self.proj_area = 0
        self.scalar = 0
        self.cell_ref = None

    def binary_fill(self):
        self.weight = 1 # smoothed volume weight
        self.type = 0
        self.integrity = 1.0

    def plot(self, cl):
        xc = self.x[0]
        yc = self.x[1]
        hv = self.size/2*0.95
        xs = [xc - hv, xc + hv, xc + hv, xc - hv, xc - hv]
        ys = [yc - hv, yc - hv, yc + hv, yc + hv, yc - hv]
        plt.plot(xs, ys, color=cl, zorder=0)

class BareVox:
    def __init__(self, x, integ):
        self.x = x
        self.integrity = integ

class Voxel_Face:
    def __init__(self, tipo, x1, x2):
        # type (tipo) is 0-3: xlo, xhi, ylo, yhi
        self.type = tipo
        # corners in CCW order
        self.xs = [x1, x2]
        # unit outward normal
        self.n = np.zeros(3)
        if self.type == 0:
            self.n[0] = -1
        elif self.type == 1:
            self.n[0] = 1
        elif self.type == 2:
            self.n[1] = -1
        elif self.type == 3:
            self.n[1] = 1
        else:
            print('ERROR: invalid voxel face type {:d}'.format(tipo))
        self.exposed = False

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
        self.subgrid = []
        self.subgrid_in = []
        self.subroot = None
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
        self.neighbors = []

        if all(self.ncells == 1):
            Cell.grid.cell_grid[ixlo[1]][ixlo[0]] = self
        elif any(self.ncells < 1):
            print('ERROR: fatal error in cell creation')
            sys.exit(1)
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
                            for j in range(2):
                                k = (j + 1) % 2
                                if (abs(dx[j]) > 1e-30):
                                    for xlim in [self.xlo, self.xhi]:
                                        t = (xlim[j] - sd.a[j])/dx[j]
                                        y_int = t*dx[k] + sd.a[k]
                                        if t > 0 and t < 1 and y_int > ext_xlo[k] and y_int < ext_xhi[k]:
                                            t_vals.append(t)
                                            interim = np.ones(2)
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
                    #     sys.exit(1)

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

        if self.in_flag == -1 and all(self.ncells == 1):
            self.subroot = Subcell.root_cell(self, round(Grid.vox_ratio/Grid.grid_ratio), new_polyline, Cell.full_polygon)

    def add_voxel(self, vox):
        self.voxels.append(vox)
        vox.cell_ref = self

    def plot(self):
        xlo = self.corners[0].x
        xhi = self.corners[2].x
        xs = [xlo[0], xhi[0], xhi[0], xlo[0], xlo[0]]
        ys = [xlo[1], xlo[1], xhi[1], xhi[1], xlo[1]]
        ltj = [(0, 112/255, 192/255)]*3
        plt.plot(xs, ys, color=ltj[self.in_flag], zorder=5)

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


# cells within the leaf cells down to the scale of voxels
class Subcell:
    full_polyline = None
    voxel_ratio = 0
    vox_len = 0
    def __init__(self, ixlo, nvoxs, parent, leaf=None):
        self.ixlo = ixlo
        self.ixhi = ixlo + nvoxs
        self.nvoxs = nvoxs  # [nx, ny] for voxels in each direction
        self.child_cells = []
        self.parent = parent
        self.leaf = leaf
        if leaf == None:
            self.leaf = parent.leaf

        # bounding box of corner objects, [bottom left, bottom right, top right, top left]
        self.xlo = self.leaf.xlo + Subcell.vox_len*self.ixlo
        self.xhi = self.xlo + Subcell.vox_len*self.nvoxs
        self.center = (self.xhi + self.xlo)/2
        
        # 1 in , 0 out, -1 mixed
        self.in_flag = -1

        if all(self.nvoxs == 1):
            self.leaf.subgrid[ixlo[1]][ixlo[0]] = self
        elif any(self.nvoxs < 1):
            print('ERROR: fatal error in subcell creation')
            sys.exit(1)
        else:
            # split the longer dimension in roughly half
            a_dim = 0 if nvoxs[0] > nvoxs[1] else 1 # x (or y)
            b_dim = 1 if a_dim == 0 else 0          # y (or x)
            delta1 = int(self.nvoxs[a_dim]/2)    # lower half
            delta2 = self.nvoxs[a_dim] - delta1  # upper half

            # create lower half cell
            diff = np.zeros(2).astype(int)
            diff[a_dim] = delta1
            diff[b_dim] = self.nvoxs[b_dim]
            self.child_cells.append(Subcell(self.ixlo, diff, self))

            # create upper half cell
            diff[a_dim] = delta2
            self.child_cells.append(Subcell(self.ixhi - diff, diff, self))

    @classmethod
    def root_cell(cls, c_leaf, voxel_ratio, polyline, full_poly):
        cls.full_polyline = full_poly
        cls.voxel_ratio = voxel_ratio
        cls.vox_len = Cell.leaf_cell_lens[0]/voxel_ratio
        c_leaf.subgrid = np.zeros((voxel_ratio, voxel_ratio)).astype(int).tolist()
        new_root = cls(np.array([0,0]), np.array([voxel_ratio, voxel_ratio]), None, leaf=c_leaf)
        new_root.sort_inout(polyline)
        c_leaf.subgrid_in = np.array([[True if c_leaf.subgrid[n][m].in_flag == 1 else False for m in range(voxel_ratio)] for n in range(voxel_ratio)])
        return new_root


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
                            for j in range(2):
                                k = (j + 1) % 2
                                if (abs(dx[j]) > 1e-30):
                                    for xlim in [self.xlo, self.xhi]:
                                        t = (xlim[j] - sd.a[j])/dx[j]
                                        y_int = t*dx[k] + sd.a[k]
                                        if t > 0 and t < 1 and y_int > ext_xlo[k] and y_int < ext_xhi[k]:
                                            t_vals.append(t)
                                            interim = np.ones(2)
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
            if len(new_polyline.sides) < 1 or all(self.nvoxs == 1):
                self.in_flag = Subcell.full_polyline.check_point_inside(self.center)
        else:
            self.in_flag = inherit

        for c in self.child_cells:
            c.sort_inout(new_polyline, inherit=self.in_flag)

class Corner:
    def __init__(self, x):
        self.x = x
        self.inside = -1 # 1 if inside, 0 if outside, -1 if unassigned
        self.frac = 0

class Grid:
    vox_ratio = 0
    grid_ratio = 0
    vox_len = 0
    def __init__(self, polygon, origin, ncells, cell_len, vox_ratio, grid_ratio, scale_flag=True):
        Grid.vox_ratio = vox_ratio
        Grid.grid_ratio = grid_ratio
        self.scale_flag = scale_flag
        self.xlo = origin
        self.xhi = origin + ncells*cell_len
        self.ncells = ncells
        self.cell_len = cell_len
        self.nvox = 0
        self.mc_surf = []

        self.polygon_perim = polygon.get_perim()

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

        # set neighbor cells
        nnlim = 1
        for j in range(len(self.cell_grid)):
            for i in range(len(self.cell_grid[j])):
                c_cell = self.cell_grid[j][i]
                for nj in range(j - nnlim, j + nnlim + 1):
                    if nj > -1 and nj < len(self.cell_grid):
                        for ni in range(i - nnlim, i + nnlim + 1):
                            if ni > -1 and ni < len(self.cell_grid[j]):
                                c_cell.neighbors.append(self.cell_grid[nj][ni])

    def create_voxels(self, voxel_ratio):
        voxel_ratio = int(voxel_ratio)
        voxel_size = Cell.leaf_cell_lens[0]/voxel_ratio
        Grid.vox_len = voxel_size
        ny = self.root_cell.ncells[1]
        nx = self.root_cell.ncells[0]
        vg = [[Voxel([self.xlo[0] + (i + 0.5)*voxel_size, self.xlo[1] + (j + 0.5)*voxel_size], voxel_size, [i,j])
               for i in range(nx*voxel_ratio)] for j in range(ny*voxel_ratio)]
        self.vox_grid = np.array(vg)
        for j in range(len(self.cell_grid)):
            for i in range(len(self.cell_grid[j])):
                c_cell = self.cell_grid[j][i]
                # whole cell is inside or outside
                if c_cell.in_flag == 1 or c_cell.in_flag == 0:
                    for n in range(voxel_ratio):
                        for m in range(voxel_ratio):
                            vy = voxel_ratio*j + n
                            vx = voxel_ratio*i + m
                            c_vox = self.vox_grid[vy][vx]
                            if c_cell.in_flag:
                                c_vox.binary_fill()
                                self.nvox += 1
                            c_cell.add_voxel(c_vox)
                # cell is mixed
                else:
                    # split cell into regions like when cell grid was created
                    # return subgrid of voxel_ratio**2 of valid positions
                    subgrid_in = c_cell.subgrid_in
                    for n in range(len(subgrid_in)):
                        for m in range(len(subgrid_in[n])):
                            vy = voxel_ratio*j + n
                            vx = voxel_ratio*i + m
                            c_vox = self.vox_grid[vy][vx]
                            if subgrid_in[n][m]:
                                c_vox.binary_fill()
                                self.nvox += 1
                            c_cell.add_voxel(c_vox)


    def weight_voxels(self, vr):
        # set voxel weights to something other than 0 or -1
        level = 0
        w_max = np.ceil((3*vr/2) - 0.5)
        w_min = np.floor(-(3*vr/2) - 0.5)
        while level <= w_max or (-level - 1) >= w_min:
            t1 = level
            t2 = -level - 1
            for j in range(0, len(self.vox_grid)):
                for i in range(0, len(self.vox_grid[j])):
                    vox = self.vox_grid[j][i]
                    if vox.finalized == False:
                        if vox.type == t1:
                            # check 4 cardinal neighbors, initially assume surrounded by voxels
                            surrounded = True

                            if i > 0 and self.vox_grid[j][i - 1].type < vox.type:
                                surrounded = False
                            if j > 0 and self.vox_grid[j - 1][i].type < vox.type:
                                surrounded = False
                            if i < len(self.vox_grid[j]) - 1 and self.vox_grid[j][i + 1].type < vox.type:
                                surrounded = False
                            if j < len(self.vox_grid) - 1 and self.vox_grid[j + 1][i].type < vox.type:
                                surrounded = False

                            if surrounded:
                                vox.type = t1 + 1
                            else:
                                vox.finalized = True
                        
                        elif vox.type == t2:
                            # check 4 cardinal neighbors, initially assume surrounded by voxels
                            surrounded = True
                            if i > 0 and self.vox_grid[j][i - 1].type > vox.type:
                                surrounded = False
                            if j > 0 and self.vox_grid[j - 1][i].type > vox.type:
                                surrounded = False
                            if i < len(self.vox_grid[j]) - 1 and self.vox_grid[j][i + 1].type > vox.type:
                                surrounded = False
                            if j < len(self.vox_grid) - 1 and self.vox_grid[j + 1][i].type > vox.type:
                                surrounded = False

                            if surrounded:
                                vox.type = t2 - 1
                            else:
                                vox.finalized = True
            level += 1
            assert(level < 1000)


        # set weights and find exposed faces
        for j in range(len(self.vox_grid)):
            for i in range(len(self.vox_grid[j])):
                vox = self.vox_grid[j][i]
                dvox = (0.5 + vox.type)
                vox.weight = 0.5*(1 + dvox*(2/(3*vr)))
                vox.weight = min(vox.weight, 1.0)
                vox.weight = max(0.0, vox.weight)
                if vox.type == 0:
                    if i == 0 or self.vox_grid[j][i - 1].type < 0:
                        vox.faces[0].exposed = True
                    if i == len(self.vox_grid[j]) - 1 or self.vox_grid[j][i + 1].type < 0:
                        vox.faces[1].exposed = True
                    if j == 0 or self.vox_grid[j - 1][i].type < 0:
                        vox.faces[2].exposed = True
                    if j == len(self.vox_grid) - 1 or self.vox_grid[j + 1][i].type < 0:
                        vox.faces[3].exposed = True

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
                c_corn.frac /= vr**2 # scale area relative to cell
                if c_corn.frac > 0.499:
                    c_corn.inside = 1
                else:
                    c_corn.inside = 0

    # associate each triangle to voxels based on inward normal view of voxel faces
    def voxels_to_surfaces(self):
        total_surf_len = 0
        total_vox_len = 0
        # first assign voxels to each triangle in each cell
        for j in range(len(self.cell_grid)):
            for i in range(len(self.cell_grid[j])):
                c_cell = self.cell_grid[j][i]
                if len(c_cell.borders):
                    # collect all voxels in current and neighboring cells
                    c_voxels = []
                    for neighbor in c_cell.neighbors:
                        for vox in neighbor.voxels:
                            if vox.type == 0:
                                c_voxels.append(vox)
                    
                    # project eligible exposed voxel faces onto triangle plane and test for intersection
                    for t in c_cell.borders:
                        v_refs = []      # voxel ids 
                        v_areas = []    # intersected area
                        t_area = np.linalg.norm(t.b - t.a) # triangle area
                        total_surf_len += t_area
                        ntheta = t.theta - np.pi/2 # outward normal angle
                        t_norm = np.array([np.cos(ntheta), np.sin(ntheta)])
                        for vox in c_voxels:
                            for f in vox.faces:
                                if (f.exposed) and (np.dot(f.n[:2], t_norm) > 0):
                                    proj_f = Line([f.xs[i] - t_norm*np.dot(t_norm, f.xs[i] - t.a) for i in range(2)])
                                    
                                    # start here
                                    #area = np.random.random()*t_area # find area of overlap between projected face and triangle
                                    area = get_intersection_area(proj_f, t)
                                    total_vox_len += area
                                    vox.proj_surfaces.append(t)
                                    vox.proj_surfaces = list(set(vox.proj_surfaces)) # prevent duplicates in list
                                    vox.proj_area += area
                                    v_refs.append(vox)
                                    v_areas.append(area)

                        # collect voxel face areas together
                        for i in range(len(v_refs)):
                            if (v_refs[i] in t.vox_refs):
                                ind = t.vox_refs.index(v_refs[i])
                                t.vox_fracs[ind] += v_areas[i]
                            else:
                                if (v_areas[i] > t_area*1e-6):  #### need to be changed according voxel resolution
                                    t.vox_refs.append(v_refs[i])
                                    t.vox_fracs.append(v_areas[i])

        print('Projected vs. Surf Perimeter Error: {:.2f} %'.format(100*(total_vox_len - total_surf_len)/(total_surf_len)))

        # now normalize scalar fractions by total voxel face area intercepted by the triangle
        low_area = 0
        for t in self.mc_surf:
            t.vox_fracs = np.array(t.vox_fracs)
            total_area = t.vox_fracs.sum()
            if (total_area < 1e-6*np.linalg.norm(t.b - t.a)):  #### need to be changed according voxel resolution   
                low_area += 1
                print('Uh oh, no voxel face area available for this triangle')
                print(t.a)
                print(t.b)
                print(t.vox_fracs)
                print()
            t.vox_fracs = t.vox_fracs / total_area
        if (low_area):
            print('WARNING: {:.1f} % of triangles have (near-)zero area intersected by voxel faces'.format(100*low_area/len(self.mc_surf)))

    # theta_fn is function of theta for the scalar per unit length value
    def apply_scalars(self, theta_fn):
        all_thetas = np.linspace(-np.pi, np.pi, 100)
        analytical = theta_fn(all_thetas) + 2

        avg_error = 0

        surf_pts = []
        total_tri_scalar = 0
        total_vox_scalar = 0
        for surf in self.mc_surf:
            centroid = (surf.b + surf.a)/2
            theta = np.arctan2(centroid[1], centroid[0])
            surf.scalar = (theta_fn(theta) + 2)*surf.length
            total_tri_scalar += surf.scalar
            surf_pts.append([theta, surf.scalar/surf.length])
            for i in range(len(surf.vox_refs)):
                vox = surf.vox_refs[i]
                vox.scalar += surf.scalar*surf.vox_fracs[i]
        transsurf = np.transpose(surf_pts)

        vox_guess = []
        nguess = 0
        for j in range(len(self.vox_grid)):
            for i in range(len(self.vox_grid[j])):
                vox = self.vox_grid[j][i]
                total_vox_scalar += vox.scalar
                if vox.type == 0:
                    theta = np.arctan2(vox.x[1], vox.x[0])
                    analyt = theta_fn(theta) + 2
                    nguess += 1
                    if vox.proj_area == 0:
                        print('WARNING: vox has no projected area')
                        vox_guess.append([theta, 0])
                        avg_error += abs(100*(0 - analyt)/analyt)
                    else:
                        vguess = vox.scalar/vox.proj_area
                        if vguess < 0.5 or vguess > 3.5:
                            print('WARNING: voxel flux outside of graphing range')
                        vox_guess.append([theta, vguess])
                        avg_error += abs(100*(vguess - analyt)/analyt)

        avg_error /= nguess
        print('Vox vs. Surface Total Scalar Error: {:.2f} %'.format(100*(total_vox_scalar - total_tri_scalar)/(total_tri_scalar)))

        return avg_error

    def simple_ms_surface(self):
        ms_perim = 0
        for j in range(len(self.cell_grid)):
            for i in range(len(self.cell_grid[j])):
                c_cell = self.cell_grid[j][i]
                inside = np.array([c_cell.corners[a].frac + 1e-8 < 0.5 for a in range(4)])
                if not(all(inside)) and not(all(~inside)):
                    c_cell.set_topology()
                    c_cell.interpolate()
                    for bd in c_cell.borders:
                        self.mc_surf.append(bd)
                        ms_perim += np.linalg.norm(bd.b - bd.a)
        print('Surf vs. Analytical Perimeter Error: {:.2f} %'.format(100*(ms_perim - self.polygon_perim)/(self.polygon_perim)))


    def save_voxs(self):
        c_voxs = []
        for j in range(len(self.vox_grid)):
            for i in range(len(self.vox_grid[j])):
                vox = self.vox_grid[j][i]
                if vox.type >= 0:
                    c_voxs.append(BareVox(vox.x, vox.integrity))
        return c_voxs
    
    def save_surf(self):
        return copy.deepcopy(self.mc_surf)

    def reset(self):
        for j in range(len(self.vox_grid)):
            for i in range(len(self.vox_grid[j])):
                vox = self.vox_grid[j][i]
                vox.finalized = False
                vox.scalar = 0
                if vox.type < 0:
                    vox.type = -1
                else:
                    vox.type = 0
                for k in range(4):
                    vox.faces[k].exposed = False
        
        for j in range(len(self.corners)):
            for i in range(len(self.corners[j])):
                self.corners[j][i].frac = 0
        for j in range(len(self.cell_grid)):
            for i in range(len(self.cell_grid[j])):
                self.cell_grid[j][i].borders = []
        self.mc_surf.clear()

    def plot_ms_surface(self):
        pts = []
        for j in range(len(self.cell_grid)):
            for i in range(len(self.cell_grid[j])):
                for ccb in self.cell_grid[j][i].borders:
                    ccb.plot('green', 'solid')
                    pts.append(ccb.a)
        pts = np.transpose(pts)
        plt.scatter(pts[0], pts[1], color='green', edgecolors='black', s=25, zorder=25)

    def plot_cells(self, white=True, rad=0.05):
        for cl in range(len(self.cell_grid)):
            for c in range(len(self.cell_grid[cl])):
                self.cell_grid[cl][c].plot()

        clr = (0, 112/255, 192/255)
        for cl in self.corners:
            for c in cl:
                if white:
                    fc = 'white'
                else:
                    if abs(c.frac - 1.0) < 1e-6:
                        fc = tuple([186/255]*3)
                    elif abs(c.frac) < 1e-6:
                        fc = [1.0]*3
                    elif c.frac >= 0.5:
                        hue = (-140*c.frac + 350)/360
                        vc = [hue, 1.0, 1.0]
                        fc = colors.hsv_to_rgb(vc)
                    else:
                        hue = (-100*c.frac + 40)
                        if hue < 0:
                            hue += 360
                        hue /= 360
                        vc = [hue, 1.0, 1.0]
                        fc = colors.hsv_to_rgb(vc)
                circ = Circle((c.x[0], c.x[1]), radius=rad, edgecolor=clr, facecolor=fc, linewidth=1.2, zorder=10)
                plt.gca().add_patch(circ)

    def plot_vols(self):
        cl = (0, 112/255, 192/255)
        for yl in self.ylines:
            plt.plot([self.xlo[0], self.xhi[0]], [yl + 0.5*self.cell_len]*2, color=cl, linestyle='dotted', zorder=5)
        for xl in self.xlines:
            plt.plot([xl + 0.5*self.cell_len]*2, [self.xlo[1], self.xhi[1]], color=cl, linestyle='dotted', zorder=5)

    def plot_voxels(self, opacity=1.0):
        vox_gray = tuple([186/255]*3)
        for vl in self.vox_grid:
            for v in vl:
                if v.type >= 0:
                    vxlo = v.x - 0.5*Grid.vox_len
                    square = Rectangle((vxlo[0], vxlo[1]), Grid.vox_len, Grid.vox_len, edgecolor='black', facecolor=vox_gray, linewidth=1.2, alpha=opacity, zorder=0)
                    plt.gca().add_patch(square)

    def plot_wvoxels(self):
        for vl in self.vox_grid:
            for v in vl:
                if abs(v.weight - 1.0) < 1e-5:
                    vox_gray = tuple([186/255]*3)
                elif abs(v.weight) < 1e-5:
                    vox_gray = [1.0]*3
                elif v.weight >= 0.5:
                    hue = (-140*v.weight + 350)/360
                    vc = [hue, 1.0, 1.0]
                    vox_gray = colors.hsv_to_rgb(vc)
                else:
                    hue = (-100*v.weight + 40)
                    if hue < 0:
                        hue += 360
                    hue /= 360
                    vc = [hue, 1.0, 1.0]
                    vox_gray = colors.hsv_to_rgb(vc)
                vxlo = v.x - 0.5*Grid.vox_len
                square = Rectangle((vxlo[0], vxlo[1]), Grid.vox_len, Grid.vox_len, edgecolor='black', facecolor=vox_gray, linewidth=1.2)
                plt.gca().add_patch(square)

# get length of the intersection of two line segments
def get_intersection_area(base, proj):
    diff = base.b - base.a
    base_len = np.linalg.norm(diff)
    if base_len < 1e-20:
        return 0
    ind = 0 if abs(diff[0]) > abs(diff[1]) else 1

    # t of base a is 0, base b is 1
    t_a = (proj.a[ind] - base.a[ind])/diff[ind]
    t_b = (proj.b[ind] - base.a[ind])/diff[ind]

    # check bounding line
    if (t_a < 0 and t_b < 0) or (t_a > 1 and t_b > 1):
        return 0
    else:
    # clip projected line to base line, t = [0, 1]
        t_a = max(t_a, 0)
        t_b = max(t_b, 0)

        t_a = min(t_a, 1)
        t_b = min(t_b, 1)
        t_diff = abs(t_b - t_a)
        return t_diff*base_len

shape_dict = {'tri0'   : (3, 0),
              'quad0'  : (4, 0),
              'quad45' : (4, 45),
              'pent0'  : (5, 0),
              'triac0': (30, 0)}
# technically 30 sided is triacontagon, tridecagon is 13
shape_data = list(shape_dict.keys())
pabl = 0

fig_save = False
glob_dpi = 500

grid_ratio = [4, 8]
voxel_ratio = [16]
s_name = 'triac0'


for vr in voxel_ratio:
    for gr in grid_ratio:
        if vr >= gr:
            circumradius = 2
            
            print('GR: {:d}, VR: {:d}:'.format(gr, vr))
            t0 = time.time()

            shape = shape_dict[s_name]
            polygon = Polygon.regular_polygon(circumradius, shape[0], shape[1])

            v_len = circumradius/vr
            l_c = circumradius/gr
            vox_bb_len = circumradius + 0.5*v_len
            l_max = 1.5*l_c + 0.5*v_len
            nc_scalar = np.ceil((vox_bb_len + l_max) / l_c) + 1e-5
            ncr = (nc_scalar*np.ones(2)).astype(int)

            origin = -ncr*l_c
            ncells = 2*ncr

            grid = Grid(polygon, origin, ncells, l_c, vr, gr)

            # create voxelized surface from polygon
            grid.create_voxels(vr/gr)

            fig_size = 5
            # sz is side length in 1/72"
            sz = 0.09
            
            sz *= 72

            big_rad = 0.07
            dpiv = 300

            if gr == 8:
                # plain voxels
                plt.figure(figsize=(fig_size,fig_size))
                plt.gca().set_aspect('equal')
                grid.plot_voxels()
                plt.ylim(-2.5,2.5)
                plt.xlim(-2.5,2.5)
                plt.axis('off')
                plt.xticks([])
                plt.yticks([])
                plt.savefig('mm_1c', dpi=dpiv, bbox_inches='tight')

                # voxels with cells
                plt.figure(figsize=(fig_size,fig_size))
                plt.gca().set_aspect('equal')
                grid.plot_voxels()
                grid.plot_cells(rad=big_rad)
                plt.axis('off')
                plt.ylim(-2.5,2.5)
                plt.xlim(-2.5,2.5)
                plt.xticks([])
                plt.yticks([])  
                plt.savefig('mm_1d', dpi=dpiv, bbox_inches='tight')
            else:
                # plain voxels
                plt.figure(figsize=(fig_size,fig_size))
                plt.gca().set_aspect('equal')
                grid.plot_voxels()
                plt.ylim(-2.5,2.5)
                plt.xlim(-2.5,2.5)
                plt.axis('off')
                plt.xticks([])
                plt.yticks([])
                plt.savefig('mm_1a', dpi=dpiv, bbox_inches='tight')

                # voxels with cells
                plt.figure(figsize=(fig_size,fig_size))
                plt.gca().set_aspect('equal')
                grid.plot_voxels()
                grid.plot_cells(rad=big_rad)
                plt.axis('off')
                plt.ylim(-2.5,2.5)
                plt.xlim(-2.5,2.5)
                plt.xticks([])
                plt.yticks([])
                plt.savefig('mm_1b', dpi=dpiv, bbox_inches='tight')


                # voxels with volume-divided cells
                plt.figure(figsize=(fig_size,fig_size))
                plt.gca().set_aspect('equal')
                grid.plot_voxels()
                grid.plot_cells()
                grid.plot_vols()
                plt.ylim(0,2.5)
                plt.xlim(0,2.5)
                plt.axis('off')
                plt.xticks([])
                plt.yticks([])
                plt.savefig('mm_2a', dpi=dpiv, bbox_inches='tight')


                grid.weight_voxels(vr/gr)

                # weighted voxels
                plt.figure(figsize=(fig_size,fig_size))
                plt.gca().set_aspect('equal')
                grid.plot_wvoxels()
                grid.plot_cells()
                grid.plot_vols()
                plt.ylim(0,2.5)
                plt.xlim(0,2.5)
                plt.axis('off')
                plt.xticks([])
                plt.yticks([])
                plt.savefig('mm_2b', dpi=dpiv, bbox_inches='tight')


                # nodal volumes
                plt.figure(figsize=(fig_size,fig_size))
                plt.gca().set_aspect('equal')
                grid.plot_cells(white=False)
                grid.plot_vols()
                plt.ylim(0,2.5)
                plt.xlim(0,2.5)
                plt.axis('off')
                plt.xticks([])
                plt.yticks([])
                plt.savefig('mm_3b', dpi=dpiv, bbox_inches='tight')


                # zoom out nodal volumes
                plt.figure(figsize=(fig_size,fig_size))
                plt.gca().set_aspect('equal')
                grid.plot_cells(white=False, rad=big_rad)
                plt.ylim(-2.5,2.5)
                plt.xlim(-2.5,2.5)
                plt.axis('off')
                plt.xticks([])
                plt.yticks([])
                plt.savefig('mm_4a', dpi=dpiv, bbox_inches='tight')



                # get a (simplified) marching squares surface from voxel grid
                grid.simple_ms_surface()

                # zoom out nodal volumes with surface
                plt.figure(figsize=(fig_size,fig_size))
                plt.gca().set_aspect('equal')
                grid.plot_cells(white=False, rad=big_rad)
                grid.plot_ms_surface()
                plt.ylim(-2.5,2.5)
                plt.xlim(-2.5,2.5)
                plt.axis('off')
                plt.xticks([])
                plt.yticks([])
                plt.savefig('mm_4b', dpi=dpiv, bbox_inches='tight')


                # same but with voxels
                plt.figure(figsize=(fig_size,fig_size))
                plt.gca().set_aspect('equal')
                grid.plot_cells(white=False, rad=big_rad)
                grid.plot_voxels(opacity=0.1)
                grid.plot_ms_surface()
                plt.ylim(-2.5,2.5)
                plt.xlim(-2.5,2.5)
                plt.axis('off')
                plt.xticks([])
                plt.yticks([])
                plt.savefig('mm_4c', dpi=dpiv, bbox_inches='tight')


                # just surface
                plt.figure(figsize=(fig_size,fig_size))
                plt.gca().set_aspect('equal')
                grid.plot_ms_surface()
                plt.ylim(-2.5,2.5)
                plt.xlim(-2.5,2.5)
                plt.axis('off')
                plt.xticks([])
                plt.yticks([])
                plt.savefig('mm_4d', dpi=dpiv, bbox_inches='tight')


            # divide voxels among surface elements
            # grid.voxels_to_surfaces()

            # flux_error = grid.apply_scalars(np.cos)
            # if fig_save:
            #     plt.savefig(s_name + '_geom', dpi=glob_dpi, bbox_inches='tight')



# %%