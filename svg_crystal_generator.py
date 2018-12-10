import math
import sys
import numpy as np


def get_projection(near, far, fov):
    n = near
    f = far
    fov = np.deg2rad(fov)
    a = - (n+f)/(f-n)
    b = - 2 * f*n/(f-n)
    s = 1/(np.tan(0.5*fov))
    return np.array([
        [  s,  0,  0,  0],
        [  0,  s,  0,  0],
        [  0,  0,  a,  b],
        [  0,  0, -1,  0]])


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac), 0],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab), 0],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc, 0],
                     [0, 0, 0, 1]])


def rotation_matrix_x(theta):
    return np.array([
        [math.cos(theta), -math.sin(theta), 0],
        [math.sin(theta),  math.cos(theta), 0],
        [0,0,1]])


def lookat(pos_from, pos_to, up=np.array([0.0, 0.0, 1.0]), roll=0):
    forward = np.array(pos_from) - np.array(pos_to)
    forward /= np.linalg.norm(forward)
    up /= np.linalg.norm(up)
    right = np.cross(up, forward)
    if np.allclose(right, 0):
        print('ERROR: lookat, up || forward')
        right = np.cross([0, 1, 0], forward) 

    up = np.cross(forward, right)

    a = -np.dot(right, pos_from)
    b = -np.dot(up, pos_from)
    c = -np.dot(forward, pos_from)
    look_matrix = np.array([
        [ *right,    a ],
        [ *up,       b ],
        [ *forward,  c ],
        [ *pos_from, 1 ]]).T
    if roll == 0:
        return look_matrix
    roll_matrix = rotation_matrix(forward, roll);

    return roll_matrix @ look_matrix


class VectorFigure3d:
    def __init__(self, fov, camera_position, roll):
        self.fov = fov
        self.camera_position = camera_position
        self.elements = []
        self.model_to_world = np.array([
            [ 1, 0, 0,   0 ],
            [ 0, 1, 0,   0 ],
            [ 0, 0, 1,   0 ],
            [ 0, 0, 0,   1 ]])
        self.world_to_camera = lookat(camera_position, (0, 0, 0), roll=roll)
        self.camera_to_projected = get_projection(near=1, far=10000, fov=fov)
        self.model_to_projected = self.model_to_world @ self.world_to_camera @ self.camera_to_projected


    def write(self, filename):
        # ZB: width="1000" height="700" viewBox="-9 -16 25 25">
        svg = """\
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" version="2.0"
      width="1920" height="900" viewBox="0 -1 1.5 1.5">
    <defs>
        <radialGradient id="Ahighlight" cx="50%" cy="50%" r="50%" fx="50%" fy="50%">
            <stop offset="0%"   stop-color="#55ff55" />
            <stop offset="100%" stop-color="#449944" />
        </radialGradient>
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
    <marker id="VectorArrow" markerUnits="strokeWidth" markerWidth="10" orient="auto" markerHeight="10" refY="2" refX="3.7">
      <path d="M0,0 L0,4 L5,2 z"/>
    </marker>
    </defs>
"""
        self.elements.sort(key=lambda element: -element[1])
        for name, z, attributes in self.elements:
            svg += '    <{} {} />\n'.format(name, ' '.join(('{}="{}"'.format(key, value) for key, value in attributes.items())))

        svg += "</svg>\n"

        with open(filename, 'w') as f:
            f.write(svg)


    def pos_to_camera(self, pos):
        pos = np.array(pos) @ self.model_to_projected
        resolution = 10  # Conversion factor between unit clipspace and view space
        scale = resolution / (pos[3] if pos[3] != 0 else 1)
        pos *= scale
        return pos


    def add_sphere(self, x, y, z, r, atom_type):
        pos = self.pos_to_camera([x, y, z, 1])
        camera_distance = abs(np.linalg.norm(np.array([x, y, z] - self.camera_position)))
        radius = 5.0 / np.tan(np.deg2rad(0.5*self.fov)) * r / np.sqrt(camera_distance**2 - r**2)
        self.elements.append((
            'circle', pos[2], {
                'cx': pos[0],
                'cy': -pos[1],
                'r': radius,
                'fill': 'url(#{}highlight)'.format(atom_type)}))


    def add_line(self, pos_from, pos_to, color, width):
        camera_distance = abs(np.linalg.norm(pos_from - self.camera_position))
        pos_from = self.pos_to_camera([*pos_from, 1])
        pos_to = self.pos_to_camera([*pos_to, 1])
        width = 7 / np.tan(np.deg2rad(0.5*self.fov)) * width / np.sqrt(camera_distance**2 - width**2)
        self.elements.append((
            'line', max(pos_from[2], pos_to[2]) + 1, {
                'x1': pos_from[0],
                'y1': -pos_from[1],
                'x2': pos_to[0],
                'y2': -pos_to[1],
                'stroke': color,
                # 'stroke': 'url(#linehighlight)',
                'stroke-width': width}))


    def add_vector(self, pos_from, pos_to, color, width):
        camera_distance = abs(np.linalg.norm(pos_from - self.camera_position))
        pos_from = self.pos_to_camera([*pos_from, 1])
        pos_to = self.pos_to_camera([*pos_to, 1])
        length = np.linalg.norm(pos_to - pos_from)
        dir = (pos_to - pos_from) / length
        pos_to = pos_from + dir * length * 0.98  # Give space for marker
        width = 7 / np.tan(np.deg2rad(0.5*self.fov)) * width / np.sqrt(camera_distance**2 - width**2)
        self.elements.append((
            'line', max(pos_from[2], pos_to[2]) - 100, {
                'x1': pos_from[0],
                'y1': -pos_from[1],
                'x2': pos_to[0],
                'y2': -pos_to[1],
                'marker-end': 'url(#VectorArrow)',
                'stroke': color,
                'stroke-width': width}))


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
    c_rot_third = rotation_matrix_x(2*np.pi/3)
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


def build_sc_structure():
    a = 4
    atoms = [
        (    0,     0,     0),
        (    a,     0,     0),
        (    0,     a,     0),
        (    0,     0,     a),
        (    a,     a,     0),
        (    a,     0,     a),
        (    0,     a,     a),
        (    a,     a,     a),
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
    return atoms, outline_ends


def build_fcc_structure():
    a = 4
    atoms = [
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

    vectors = [
        ((a, a, 0), (0, 0, a))
    ]

    return atoms, outline_ends, vectors


def build_bcc_structure():
    a = 4
    atoms = [
        (    0,     0,     0),
        (    a,     0,     0),
        (    0,     a,     0),
        (    0,     0,     a),
        (    a,     a,     0),
        (    a,     0,     a),
        (    0,     a,     a),
        (    a,     a,     a),
        (0.5*a, 0.5*a, 0.5*a),
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

    return atoms, outline_ends


def build_hcp_structure():
    a = 4
    c = 7
    sixth_rot = 2*np.pi/6
    c_rot_sixth = np.array([
        [math.cos(sixth_rot), -math.sin(sixth_rot), 0],
        [math.sin(sixth_rot), math.cos(sixth_rot), 0],
        [0,0,1]])
    atoms = [np.array((0, 0, 0)), np.array((a, 0, 0))]
    for i in range(1, 6):
        atoms.append(np.dot(c_rot_sixth, atoms[i]))
    atoms += [pos + (0, 0, c) for pos in atoms]

    outline_ends = []
    for i in range(1, 7):
        outline_ends.append((atoms[i], atoms[(i % 6) + 1]))
        outline_ends.append((atoms[i], atoms[6 + (i % 7) + 1]))
        outline_ends.append((atoms[7 + i], atoms[7 + (i % 6) + 1]))

    vectors = [
        ((0, 0, 0), (0, 0, c))
    ]

    return atoms, outline_ends, vectors


def create_structure(sphere_data, line_data, fov=100,
        camera_position=np.array((50.0, 20.0, 10.0)), up=np.array((0.0 ,0.0, 1.0)),
        # camera_position=np.array((0.0, -50.0, 0.0)), up=np.array((0.0 ,0.0, 1.0)),
        roll=0):
    fig = VectorFigure3d(fov=fov, camera_position=camera_position, roll=roll)
    for atom_list, radius, name in sphere_data:
        for x, y, z in atom_list:
            fig.add_sphere(x, y, z, radius, name)

    for end_list, color, width in line_data:
        for start, end in end_list:
            fig.add_line(start, end, color, width)

    return fig


def create_GaAs_structure(atom_gas, atom_ass, outline_ends, bind_ends, **kwargs):
    line_outline_width = 0.03
    line_outline_color = '#444444'
    line_bind_width = 0.07
    line_bind_color = '#555555'
    r_Ga = 0.4
    r_As = 0.3

    sphere_data = [
        (atom_gas, r_Ga, 'Ga'),
        (atom_ass, r_As, 'As')
    ]

    line_data = [
        (outline_ends, line_outline_color, line_outline_width),
        (bind_ends, line_bind_color, line_bind_width)
    ]
    return create_structure(sphere_data, line_data, **kwargs)


def create_single_atom_structure(atoms, outline_ends, vectors=[]):
    line_outline_width = 0.03
    line_outline_color = '#444444'
    line_vector_width = 0.06
    line_vector_color = '#000000'
    r = 0.4

    sphere_data = [
        (atoms, r, 'A')
    ]

    line_data = [
        (outline_ends, line_outline_color, line_outline_width),
    ]
    fig = create_structure(sphere_data, line_data)
    for vector in vectors:
        fig.add_vector(*vector, line_vector_color, line_vector_width)
    return fig


if __name__ == '__main__':
    structure = sys.argv[1]
    filename = sys.argv[2]
    default_dist = np.linalg.norm((50.0, 20.0, 10.0))
    if structure == 'zb':
        fig = create_GaAs_structure(*build_zb_structure())
    elif structure == 'zb_110':
        cam_pos = np.array((50.0, 50.0, 0.0))
        cam_pos *= default_dist / np.linalg.norm(cam_pos)
        roll = np.arctan2(1, 1/np.sqrt(2))
        fig = create_GaAs_structure(*build_zb_structure(), camera_position=cam_pos, roll=roll)
    elif structure == 'zb_112':
        cam_pos = np.array((40.0, 20.0, 20.0))
        cam_pos /= 1
        roll = np.pi - np.deg2rad(50.77)  # TODO: Where does this number come from?
        fig = create_GaAs_structure(*build_zb_structure(), camera_position=cam_pos, roll=roll)
    elif structure == 'wz':
        fig = create_GaAs_structure(*build_wz_structure())
    elif structure == 'wz_1120':
        c_rot_third = rotation_matrix_x(2*np.pi/3)
        corner_b = np.array((1, 0, 0))
        corner_c = np.dot(c_rot_third, corner_b)
        cam_pos = 50*(corner_b + corner_c)
        fig = create_GaAs_structure(*build_wz_structure(), camera_position=cam_pos)
    elif structure == 'wz_1010':
        corner_b = np.array((4.053, 0, 0))
        cam_pos = corner_b + 10*(rotation_matrix_x(-np.pi/6) @ corner_b)
        fig = create_GaAs_structure(*build_wz_structure(), camera_position=cam_pos)
    elif structure == 'sc':
        fig = create_single_atom_structure(*build_sc_structure())
    elif structure == 'fcc':
        fig = create_single_atom_structure(*build_fcc_structure())
    elif structure == 'bcc':
        fig = create_single_atom_structure(*build_bcc_structure())
    elif structure == 'hcp':
        fig = create_single_atom_structure(*build_hcp_structure())

    fig.write(sys.argv[2])

