import sys

import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')

from diffpy.structure import loadStructure
from pyxem.utils.sim_utils import rotation_list_stereographic
from transforms3d.euler import euler2mat

from figure import save_figure
from figure import TikzAxis
from figure import TikzTable3D
from figure import TikzPlot3D


structure_cub_file = 'D:\\Dokumenter/MTNANO/Prosjektoppgave/Data/Gen/NN_test_data/GaAs_mp-2534_conventional_standard.cif'
structure_hex_file = 'D:\\Dokumenter/MTNANO/Prosjektoppgave/Data/Gen/NN_test_data/GaAs_mp-8883_conventional_standard.cif'


def rotation_euler_to_matrices(rotation_list):
    for r in rotation_list:
        yield euler2mat(*np.deg2rad(r), 'rzxz')

if __name__ == '__main__':
    system = sys.argv[1]
    filename = sys.argv[2]
    if system == 'cubic':
        structure = loadStructure(structure_cub_file)
        corners = [(0, 0, 1), (1, 0, 1), (1, 1, 1)]
    elif system == 'hexagonal':
        structure = loadStructure(structure_hex_file)
        corners = [(0, 0, 0, 1), (1, 0, -1, 0), (1, 1, -2, 0)]
    else:
        print('Unknown crystal system {}'.format(system))

    rotation_list = rotation_list_stereographic(structure, *corners, [0], np.deg2rad(2))
    rotated_points = [np.dot(rotation_matrix, (0, 0, 1))
            for rotation_matrix in rotation_euler_to_matrices(rotation_list)]
    
    axis_styles = {
        'grid': 'major',
        'ticks': 'none',
        'axis equal image': 'true',
        'view': '{110}{20}',
        'z buffer': 'sort',
        'xmax': 1,
        'ymax': 1,
        'zmin': 0,
        'zmax': 1,
        'colormap': """{grad}{
      color(0)=(MaterialBlue);
      color(1)=(MaterialBlue!80!black)}""",
    }
    line_styles = {
        'samples': 21,
        'domain': '0:90',
        'color': 'black!30'
    }
    save_figure(
        filename,
        TikzAxis(
            TikzTable3D(rotated_points),
            TikzPlot3D('{cos(u)}, {sin(u)}, {0}', **line_styles),
            TikzPlot3D('{cos(u)}, {0}, {sin(u)}', **line_styles),
            TikzPlot3D('{0}, {cos(u)}, {sin(u)}', **line_styles),
            **axis_styles))



