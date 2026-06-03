import numpy as np
from utilities import progress_bar

# get length of the intersection of two line segments
def get_intersection_length(projections, bases):
    inter_lines = []
    for b_index, base in enumerate(bases):
        proj = projections[b_index]
        diff = base.b - base.a
        base_len = np.linalg.norm(diff)
        if base_len < 1e-20:
            inter_lines.append(0)
        else:
            ind = 0 if abs(diff[0]) > abs(diff[1]) else 1

            # t of base a is 0, base b is 1
            t_a = (proj.a[ind] - base.a[ind])/diff[ind]
            t_b = (proj.b[ind] - base.a[ind])/diff[ind]

            # check bounding line
            if (t_a < 0 and t_b < 0) or (t_a > 1 and t_b > 1):
                inter_lines.append(0)
            else:
            # clip projected line to base line, t = [0, 1]
                t_a = max(t_a, 0)
                t_b = max(t_b, 0)

                t_a = min(t_a, 1)
                t_b = min(t_b, 1)
                t_diff = abs(t_b - t_a)
                inter_lines.append(t_diff*base_len)
                
    return inter_lines



# get_intersection_area function for batch processing
def get_intersection_area(proj_faces, tri_normal, tri_plane_normal, tri_vertices, tri_epsilon):
    # find overlapping area
    all_clipped_points = clip_sh(proj_faces, tri_plane_normal, tri_vertices, tri_epsilon) 
    polygon_areas = []
    for intr_indx, clipped_points in enumerate(all_clipped_points):
        if len(clipped_points) < 3:
            polygon_areas.append(0)
            continue
        # rotate overlap polygon into xy plane
        rotated_points = orient_polygon_xy(clipped_points, tri_normal[intr_indx]) 
        # get area with shoelace formula
        polygon_areas.append(polygon_area(rotated_points)) 
        progress_bar(intr_indx, len(all_clipped_points), '    finding intersection areas')
    return polygon_areas

# Sutherland-Hodgman polygon clipping
# inputs are vertices of subject (to be clipped) and vertices
# of window (the clipper)
def clip_sh(subjects, tri_plane_normal, tri_vertices, tri_epsilon):
    final_pts = []
    for intr_indx, subject in enumerate(subjects):
        # clipping operation
        in_pts = subject
        for i in range(3):
            out_pts = []

            for j in range(len(in_pts)):
                p1 = in_pts[j - 1]
                p2 = in_pts[j]

                # compute intersection with infinite edge
                p1_in, p2_in, intersect = segment_plane_intersection(p1, p2, tri_plane_normal[intr_indx][i], tri_vertices[intr_indx][i], tri_epsilon[intr_indx])

                if (p2_in):
                    if (not p1_in):
                        out_pts.append(intersect)
                    out_pts.append(p2)
                elif (p1_in):   # and not p2_in
                    out_pts.append(intersect)
                # if p1 and p2 both outside, do nothing, delete line segment

            in_pts = out_pts

        # remove duplicate vertices
        final_pts.append([])
        for i in range(len(out_pts)):
            dupe = False
            for j in range(i + 1, len(out_pts)):
                if (all(abs(out_pts[j] - out_pts[i]) < tri_epsilon[intr_indx])):
                    dupe = True
                    break
            if not dupe:
                final_pts[-1].append(out_pts[i])

        progress_bar(intr_indx, len(subjects), '    clipping polygons')

    return final_pts

# for line segment of points p1 and p2, does it intersect plane defined by
# outward unit normal n, passing through point q
# return inside/outside determinations for p1,p2 and intersection point
# 'in' means inside or on plane; out means strictly outside
def segment_plane_intersection(p1, p2, n, q, epsilon):
    intersect = np.zeros(3)
    p1_dist = np.dot(p1 - q, n)
    p2_dist = np.dot(p2 - q, n)

    p1_in = False
    p2_in = False
    if (p1_dist < epsilon):
        p1_in = True
    if (p2_dist < epsilon):
        p2_in = True
    # if one in and other out, there is an intersection
    if (p1_in + p2_in == 1):
        if (p1_in):
            if (abs(p1_dist) < epsilon):
                intersect = p1
            else:
                frac = abs(p1_dist)/(abs(p2_dist) + abs(p1_dist))
                intersect = p1 + frac*(p2 - p1)
        # p2 is inside
        else:
            if (abs(p2_dist) < epsilon):
                intersect = p2
            else:
                frac = abs(p1_dist)/(abs(p2_dist) + abs(p1_dist))
                intersect = p1 + frac*(p2 - p1)

    return p1_in, p2_in, intersect

def orient_polygon_xy(verts, normal):
    theta = np.arccos(normal[2])
    epsilon = 1e-4
    if (theta < epsilon or np.pi - theta < epsilon):
        return np.array([[verts[i][0], verts[i][1]] for i in range(len(verts))])
    else:
        # need finite rotation to align polygon
        # get unit vector of axis around which to rotate
        axis = np.cross(normal, [0,0,1])
        ax_len = np.sqrt(axis[0]**2 + axis[1]**2 + axis[2]**2)
        axis /= ax_len

        # theta is already known, and if i did my math right, it should always
        # be a positive rotation; use 3D rotation matrix
        R = np.outer(axis, axis)*(1 - np.cos(theta))
        R += np.identity(3)*np.cos(theta)
        R += np.array([ [       0, -axis[2],  axis[1]],
                        [ axis[2],        0, -axis[0]],
                        [-axis[1],  axis[0],        0]])*np.sin(theta)
        return np.array([np.matmul(R, v)[:-1] for v in verts])


# shoelace formula (trapezoid rule); for 2D XY POLYGONS ONLY
def polygon_area(verts):
    area = 0
    for i in range(len(verts)):
        p1 = verts[i - 1]
        p2 = verts[i]
        area += (p1[1] + p2[1])*(p1[0] - p2[0])

    return abs(area*0.5)

def get_tri_area(verts):

    # herons formula = sqrt(s(s - a)(s - b)(s - c)) for triangle lengths a,b,c, s= half-perimeter
    a = np.linalg.norm(verts[2] - verts[1])
    b = np.linalg.norm(verts[1] - verts[0])
    c = np.linalg.norm(verts[0] - verts[2])

    s = (a + b + c)/2
    area = np.sqrt(s*(s - a)*(s - b)*(s - c))

    return area
   
def get_longest_side(verts):
    L0 = np.linalg.norm(verts[1] - verts[0])
    L1 = np.linalg.norm(verts[2] - verts[1])
    L2 = np.linalg.norm(verts[0] - verts[2])
    sides = np.array([L0, L1, L2])
    max_len = np.argmax(sides)
    if max_len == 0:
        AC = [1, 0]
    elif max_len == 1:
        AC = [2, 1]
    else:
        AC = [0, 2]
    return AC