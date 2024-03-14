# -*- coding: utf-8 -*-
"""
sparta out vs sparta in triangles
"""

import pandas as pd
import numpy as np
import copy

# read in output
def read_output(name):
    # read in created data
    f = open(name)
    surf_lines = f.readlines()
    npoints = int(surf_lines[2].split()[0]) # number of vertices
    f.close()
    
    points = surf_lines[7:7+npoints]
    points = np.array([x.split() for x in points])
    points = pd.DataFrame(data=points, columns=['id', 'x', 'y', 'z'], dtype=np.double).set_index(['id'], verify_integrity=True)
    points = points.set_index(points.index.astype(int))
    
    tris = surf_lines[10+npoints:]
    tris = np.array([x.split() for x in tris])
    tris = pd.DataFrame(data=tris, columns=['id', 'p1', 'p2', 'p3'], dtype=np.double).set_index(['id'], verify_integrity=True).astype(int)
    tris = tris.set_index(tris.index.astype(int))
    
    tris['v1x'] = 0.0
    tris['v1y'] = 0.0
    tris['v1z'] = 0.0
    tris['v2x'] = 0.0
    tris['v2y'] = 0.0
    tris['v2z'] = 0.0
    tris['v3x'] = 0.0
    tris['v3y'] = 0.0
    tris['v3z'] = 0.0
    for i in range(len(tris)):
        tris['v1x'].iloc[i] = points['x'].loc[tris['p1'].iloc[i]]
        tris['v1y'].iloc[i] = points['y'].loc[tris['p1'].iloc[i]]
        tris['v1z'].iloc[i] = points['z'].loc[tris['p1'].iloc[i]]
        tris['v2x'].iloc[i] = points['x'].loc[tris['p2'].iloc[i]]
        tris['v2y'].iloc[i] = points['y'].loc[tris['p2'].iloc[i]]
        tris['v2z'].iloc[i] = points['z'].loc[tris['p2'].iloc[i]]
        tris['v3x'].iloc[i] = points['x'].loc[tris['p3'].iloc[i]]
        tris['v3y'].iloc[i] = points['y'].loc[tris['p3'].iloc[i]]
        tris['v3z'].iloc[i] = points['z'].loc[tris['p3'].iloc[i]]
           
    return points, tris

def read_data(name):
    # read in created data
    f = open(name)
    surf_lines = f.readlines()
    f.close()
    
    ntris = int(surf_lines[3])
    tris = surf_lines[9:9+ntris]
    tris = np.array([x.split() for x in tris])
    cols = ['id', 'v1x', 'v1y', 'v1z', 'v2x', 'v2y', 'v2z', 'v3x', 'v3y', 'v3z', 'f1', 'f2']
    tris = pd.DataFrame(data=tris, columns=cols, dtype=np.double).set_index(['id'], verify_integrity=True)
    tris = tris.set_index(tris.index.astype(int))    
    
    return tris

# 1 is output from sparta, 2 is output from isthmus 

# raw input and output geometry read in, points and triangles
print('Reading data... ', end='')
p_in, t_in = read_output('vox2surf.surf')
p_out, t_out = read_output('final.surf')
t_data = read_data('forces.dat')
print('done')

epsilon_sq = (1e-6)**2
pc = 0
print('Are triangles of data equivalent to isthmus? ', end='')
sq_norm1 = (t_data['v1x'] - t_in['v1x'])**2 + (t_data['v1y'] - t_in['v1y'])**2 + (t_data['v1z'] - t_in['v1z'])**2
sq_norm2 = (t_data['v2x'] - t_in['v2x'])**2 + (t_data['v2y'] - t_in['v2y'])**2 + (t_data['v2z'] - t_in['v2z'])**2
sq_norm3 = (t_data['v3x'] - t_in['v3x'])**2 + (t_data['v3y'] - t_in['v3y'])**2 + (t_data['v3z'] - t_in['v3z'])**2

for i in range(len(sq_norm1)):
    if (sq_norm1.iloc[i] and sq_norm2.iloc[i] and sq_norm3.iloc[i]):
        pc += 1
pc *= 100/len(t_data)

print('{:.2f}% yes'.format(pc))

pc = 0
print('Are triangles of data equivalent to sparta-outputted mesh? ', end='')
sq_norm1 = (t_data['v1x'] - t_out['v1x'])**2 + (t_data['v1y'] - t_out['v1y'])**2 + (t_data['v1z'] - t_out['v1z'])**2
sq_norm2 = (t_data['v2x'] - t_out['v2x'])**2 + (t_data['v2y'] - t_out['v2y'])**2 + (t_data['v2z'] - t_out['v2z'])**2
sq_norm3 = (t_data['v3x'] - t_out['v3x'])**2 + (t_data['v3y'] - t_out['v3y'])**2 + (t_data['v3z'] - t_out['v3z'])**2

for i in range(len(sq_norm1)):
    if (sq_norm1.iloc[i] and sq_norm2.iloc[i] and sq_norm3.iloc[i]):
        pc += 1
pc *= 100/len(t_data)

print('{:.2f}% yes'.format(pc))

"""
p_out_dup = p_out[p_out.duplicated()]
p_out_red = p_out[~p_out.duplicated()]

p_out_red = p_out_red.sort_values(by=['x','y','z'])
p_in = p_in.sort_values(by=['x','y','z'])

p_diff_array = np.array(p_out_red) - np.array(p_in)
for i in range(len(p_diff_array)):
    if (np.linalg.norm(p_diff_array[i]) > epsilon):
        raise Exception('Point index {d} is not equal'.format(d=i))
print('Points are equivalent...')

"""
"""
dupes = [[] for x in range(len(p_out_red))]
for i in range(len(p_out_dup)):
    cp = p_out_dup.iloc[i]
    print(len(p_out_dup) - i)
    for i in range(len(p_out_red)):
        if (all(abs(p_out_red.iloc[i] - cp) < epsilon)):
            dupes[i].append(cp.name)
            break
"""
"""
dupes = []
uniques = []
for i in range(len(p_out_red)):
    reduced_point = np.array(p_out_red.iloc[i])
    p_out_dup['norm'] = (p_out_dup['x'] - reduced_point[0])**2 + (p_out_dup['y'] - reduced_point[1])**2 + (p_out_dup['z'] - reduced_point[2])**2
    c_dupes = p_out_dup[p_out_dup['norm'] < epsilon]
    p_out_dup = p_out_dup[p_out_dup['norm'] >= epsilon]
    uniques.append(p_out_red.iloc[i].name)
    dupes.append([p_out_red.iloc[i].name] + list(c_dupes.index))

test = pd.DataFrame(np.array([dupes[i][j] for i in range(len(dupes)) for j in range(len(dupes[i]))]))
if test.duplicated().sum():
    raise Exception('Oopsies, duplicate point filtering didn\'t work')
for i in range(len(uniques)):
    unique = p_out.loc[uniques[i]]
    for j in range(0, len(dupes[i])):
        dupe = p_out.loc[dupes[i][j]]
        if np.linalg.norm(unique - dupe) > epsilon:
            raise Exception('Oopsies, duplicate point filtering didn\'t work')

sparta_to_isthmus = {dupes[i][j] : uniques[i] for i in range(len(uniques)) for j in range(len(dupes[i]))}

t_out_translated = copy.deepcopy(t_out)
for i in range(len(t_out)):
    for j in range(3):
        t_out_translated.iloc[i,j] = sparta_to_isthmus[t_out.iloc[i,j]]

"""
"""
p_out.insert(len(p_in.columns), 'dupes', [[] for x in p_out.index])
for i in range(len(p_out)):
    for j in range(i + 1, len(p_out)):
        if (all(p_out.iloc[i] == p_out.iloc[j])):
            p_out.iloc[i].loc['dupes'].append(j)
"""
"""

output_points = []
for i in range(len(t_out)):
    output_points.append(t_out.iloc[i][0])
    output_points.append(t_out.iloc[i][1])
    output_points.append(t_out.iloc[i][2])
    
output_points = set(output_points)

t_in_inds = t_in.index
t_out_inds = t_out.index
"""