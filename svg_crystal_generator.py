import math
import numpy as np


# ZB: width="1000" height="700" viewBox="-9 -16 25 25">
svg = """<?xml version="1.0" encoding="UTF-8"?>

<svg xmlns="http://www.w3.org/2000/svg" version="2.0"
      width="1920" height="900" viewBox="-2 -7 12 10">
    <defs>
        <radialGradient id="Gahighlight" cx="50%" cy="50%" r="50%" fx="50%" fy="50%">
            <stop offset="0%"   stop-color="#55ff55" />
            <stop offset="100%" stop-color="#449944" />
        </radialGradient>
        <radialGradient id="Ashighlight" cx="50%" cy="50%" r="50%" fx="50%" fy="50%">
            <stop offset="0%"   stop-color="#ff5555" />
            <stop offset="100%" stop-color="#994444" />
        </radialGradient>
        <linearGradient id="linehighlight" gradientUnits="userSpaceOnUse" gradientTransform="rotate(90)">
            <stop offset="0"   stop-color="#424242" />
            <stop offset="0.5" stop-color="#999999" />
            <stop offset="1"   stop-color="#424242" />
        </linearGradient>
    </defs>
"""


elements = []
camera_position = np.array((50.0, 20.0, 10.0))
fov = 100
line_outline_width = 0.03
line_outline_color = '#000000'
line_bind_width = 0.1
line_bind_color = '#555555'
r_Ga = 2
r_As = 1.5

def add_sphere(x, y, z, r, atom_type):
    pos = np.array([x, y, z, 1]) @ model_to_world @ world_to_camera @ camera_to_projected
    scale = pos[3] if pos[3] != 0 else 1
    camera_distance = abs(np.linalg.norm(np.array([x, y, z] - camera_position)))
    radius = 15.0 / np.tan(np.deg2rad(0.5*fov)) * r / np.sqrt(camera_distance**2 - r**2)
    elements.append((
        'circle', pos[2], {
            'cx': pos[0],
            'cy': -pos[1],
            'r': radius,  #r/np.sqrt(abs(scale)),
            # 'opacity': '0.6',
            'fill': 'url(#{}highlight)'.format(atom_type)}))


def add_line(pos_from, pos_to, color, width):
    pos_from = np.array([*pos_from, 1]) @ model_to_world @ world_to_camera @ camera_to_projected
    pos_to = np.array([*pos_to, 1]) @ model_to_world @ world_to_camera @ camera_to_projected
    elements.append((
        'line', max(pos_from[2], pos_to[2]) + 1, {
            'x1': pos_from[0],
            'y1': -pos_from[1],
            'x2': pos_to[0],
            'y2': -pos_to[1],
            'stroke': color,
            # 'stroke': 'url(#linehighlight)',
            'stroke-width': width}))


def get_projection(near, far, fov):
    n = near
    f = far
    fov = np.deg2rad(fov)
    a = - f/(f-n)
    b = - 2 * f*n/(f-n)
    s = 1/(np.tan(0.5*fov))
    return np.array([
        [  s,  0,  0,  0],
        [  0,  s,  0,  0],
        [  0,  0,  a,  b],
        [  0,  0, -1,  0]])


def lookat(pos_from, pos_to):
    forward = np.array(pos_from) - np.array(pos_to)
    forward /= np.linalg.norm(forward)
    right = np.cross([0, 0, 1], forward) 
    if np.allclose(right, 0):
        right = np.cross([0, 1, 0], forward) 
    up = np.cross(forward, right)

    return np.array([
        [ *right,    0 ],
        [ *up,       0 ],
        [ *forward,  0 ],
        [ *pos_from, 1 ]]).T


world_to_camera = lookat(camera_position, (0, 0, 0))
# print(world_to_camera)

model_to_world = np.array([
    [ 1, 0, 0,   0 ],
    [ 0, 1, 0,   0 ],
    [ 0, 0, 1,   0 ],
    [ 0, 0, 0,   1 ]])

camera_to_projected = get_projection(near=1, far=10000, fov=fov)

def build_zb_structure():
    a = 4.066
    atom_gas = [
        (    0,     0,     0),
        (    a,     0,     0),
        (    0,     a,     0),
        (    0,     0,     a),
        (    a,     a,     0),
        (    a,     0,     a),
        (    0,     a,     a),
        (    a,     a,     a),
        (0.5*a, 0.5*a,     0),
        (0.5*a,     0, 0.5*a),
        (0,     0.5*a, 0.5*a),
        (0.5*a, 0.5*a,     a),
        (0.5*a,     a, 0.5*a),
        (a,     0.5*a, 0.5*a),
    ]

    atom_ass = [
        (0.25*a, 0.25*a, 0.25*a),
        (0.75*a, 0.75*a, 0.25*a),
        (0.75*a, 0.25*a, 0.75*a),
        (0.25*a, 0.75*a, 0.75*a),
    ]

    outline_ends = [
        ((0, 0, 0), (a, 0, 0)),
        ((0, 0, 0), (0, a, 0)),
        ((0, 0, 0), (0, 0, a)),
        ((a, 0, a), (a, 0, 0)),
        ((a, 0, a), (0, 0, a)),
        ((a, 0, a), (a, a, a)),
        ((a, a, 0), (a, 0, 0)),
        ((a, a, 0), (0, a, 0)),
        ((a, a, 0), (a, a, a)),
        ((0, a, a), (a, a, a)),
        ((0, a, a), (0, a, 0)),
        ((0, a, a), (0, 0, a)),
    ]

    bind_ends = [
        (atom_ass[0], atom_gas[0]),
        (atom_ass[0], atom_gas[8]),
        (atom_ass[0], atom_gas[9]),
        (atom_ass[0], atom_gas[10]),
        (atom_ass[1], atom_gas[4]),
        (atom_ass[1], atom_gas[8]),
        (atom_ass[1], atom_gas[12]),
        (atom_ass[1], atom_gas[13]),
        (atom_ass[2], atom_gas[5]),
        (atom_ass[2], atom_gas[9]),
        (atom_ass[2], atom_gas[11]),
        (atom_ass[2], atom_gas[13]),
        (atom_ass[3], atom_gas[6]),
        (atom_ass[3], atom_gas[10]),
        (atom_ass[3], atom_gas[11]),
        (atom_ass[3], atom_gas[12]),
    ]
    return atom_gas, atom_ass, outline_ends, bind_ends


def build_wz_structure():
    a = 4.053
    c = 6.680
    third_rot = 2*np.pi/3
    c_rot_third = np.array([
        [math.cos(third_rot), -math.sin(third_rot), 0],
        [math.sin(third_rot), math.cos(third_rot), 0],
        [0,0,1]]) 
    corner_a = np.array((0, 0, 0))
    corner_b = np.array((a, 0, 0))
    corner_c = np.dot(c_rot_third, corner_b)
    corner_e = np.array((0, 0, c))

    atom_gas = [
        corner_a,
        corner_b,
        corner_c,
        corner_b + corner_c,
        corner_e + corner_a,
        corner_e + corner_b,
        corner_e + corner_c,
        corner_e + corner_b + corner_c,
        0.5*corner_e + 2*corner_b/3 + 1*corner_c/3,
    ]

    atom_ass = [pos + 3*corner_e/8 for pos in atom_gas if pos[2] < c]

    outline_ends = [
        (corner_a, corner_b),
        (corner_a, corner_c),
        (corner_a, corner_e),
        (corner_b, corner_b + corner_c),
        (corner_c, corner_b + corner_c),
        (corner_e + corner_a, corner_e + corner_b),
        (corner_e + corner_a, corner_e + corner_c),
        (corner_e + corner_b, corner_e + corner_b + corner_c),
        (corner_e + corner_c, corner_e + corner_b + corner_c),
        (corner_b, corner_b + corner_e),
        (corner_c, corner_c + corner_e),
        (corner_b + corner_c, corner_b + corner_c + corner_e),
    ]

    bind_ends = [(pos, pos - 3*corner_e/8) for pos in atom_ass]
    bind_ends += [
        (atom_ass[0], atom_gas[8]),
        (atom_ass[1], atom_gas[8]),
        (atom_ass[3], atom_gas[8]),
        (atom_ass[4], atom_gas[4]),
        (atom_ass[4], atom_gas[5]),
        (atom_ass[4], atom_gas[7]),
    ]

    return atom_gas, atom_ass, outline_ends, bind_ends


def add_structure(sphere_data, line_data):
    for atom_list, radius, name in sphere_data:
        for x, y, z in atom_list:
            add_sphere(x, y, z, radius, name)

    for end_list, color, width in line_data:
        for start, end in end_list:
            add_line(start, end, color, width)


def add_GaAs_structure(atom_gas, atom_ass, outline_ends, bind_ends):
    sphere_data = [
        (atom_gas, r_Ga, 'Ga'),
        (atom_ass, r_As, 'As')
    ]

    line_data = [
        (outline_ends, line_outline_color, line_outline_width),
        (bind_ends, line_bind_color, line_bind_width)
    ]
    add_structure(sphere_data, line_data)


add_GaAs_structure(*build_zb_structure())
# add_GaAs_structure(*build_wz_structure())



elements.sort(key=lambda element: -element[1])
for name, z, attributes in elements:
    svg += '    <{} {} />\n'.format(name, ' '.join(('{}="{}"'.format(key, value) for key, value in attributes.items())))

svg += "</svg>\n"

with open('../../data/Tmp/test.svg', 'w') as f:
    f.write(svg)
