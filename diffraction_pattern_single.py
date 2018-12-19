import os
import re
import sys

import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
import diffpy.structure
from PIL import Image
from pyxem.generators.diffraction_generator import DiffractionGenerator
from transforms3d.euler import axangle2mat

from parameters import parameters_parse

from figure import save_figure
from figure import TikzArrow
from figure import TikzImage



def generate_pattern(structure, u, v, w, rot,
        beam_energy_keV, max_excitation_error, target_pattern_dimension_pixels,
        reciprocal_angstrom_per_pixel, simulated_gaussian_sigma):
    generator = DiffractionGenerator(beam_energy_keV, max_excitation_error=max_excitation_error)

    uvw = np.array((u, v, w))
    up = np.array((0.0, 0.0, 1.0))  # Following diffpy, the z-axis is aligned in the crystal and lab frame.
    lattice = structure.lattice
    rotation_angle = np.deg2rad(lattice.angle(up, uvw))  # Because lattice.angle returns degrees...
    rotation_axis = np.cross(lattice.cartesian(up), lattice.cartesian(uvw))

    structure_rotation = axangle2mat(rotation_axis, rotation_angle)
    inplane_rotation = axangle2mat(up, np.deg2rad(rot))
    structure_rotation = structure_rotation @ inplane_rotation 
    lattice_rotated = diffpy.structure.lattice.Lattice(
            structure.lattice.a,
            structure.lattice.b,
            structure.lattice.c,
            structure.lattice.alpha,
            structure.lattice.beta,
            structure.lattice.gamma,
            baserot=structure_rotation)
    # Don't change the original
    structure_rotated = diffpy.structure.Structure(structure)
    structure_rotated.placeInLattice(lattice_rotated)

    half_pattern_size = target_pattern_dimension_pixels // 2
    reciprocal_radius = reciprocal_angstrom_per_pixel*(half_pattern_size - 1)
    sim = generator.calculate_ed_data(structure_rotated, reciprocal_radius, with_direct_beam=False)
    s = sim.as_signal(target_pattern_dimension_pixels, simulated_gaussian_sigma, reciprocal_radius)
    return s.data


def create_patterns(parameters):
    output_dir = parameters['output_dir']
    output_dir = os.path.join(output_dir, 'run_{}'.format(parameters['shortname']))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dp_parameter_regex = re.compile(r'(?P<name>\S+)\s+(?P<u>\S+)\s+(?P<v>\S+)\s+(?P<w>\S+)\s+(?P<rot>\S+)(?P<rest>.*)$')
    arrow_regex = re.compile(r'\((?P<to_x>\d+)\s+(?P<to_y>\d+)\s+"(?P<desc>[^"]*)"\)')

    reciprocal_angstrom_per_pixel = 0.032 # From 110 direction, compared to a_crop
    structures = {}
    for name, value in parameters.items():
        if name.startswith('dp_'):
            match = dp_parameter_regex.match(value)
            structure_name = match.group('name')
            if structure_name not in structures:
                structures[structure_name] = diffpy.structure.loadStructure(
                        parameters['phase_{}_structure_file'.format(structure_name)])
            structure = structures[structure_name]
            pattern = generate_pattern(
                    structure,
                    int(match.group('u')),
                    int(match.group('v')),
                    int(match.group('w')),
                    float(match.group('rot')),
                    parameters['beam_energy_keV'],
                    1/parameters['specimen_thickness'],
                    parameters['target_pattern_dimension_pixels'],
                    parameters['reciprocal_angstrom_per_pixel'],
                    parameters['simulated_gaussian_sigma'])
            pattern *= 255.0/pattern.max()
            pattern = 255.0 - pattern

            arrow_defs = match.group('rest')
            arrows = []
            output_filename = os.path.join(output_dir, 'diffraction_pattern_{}.tex'.format(name[3:]))
            for arrow_def in re.finditer(arrow_regex, arrow_defs):
                arrows.append(TikzArrow(
                    pattern.shape[0] // 2, pattern.shape[1] // 2,
                    int(arrow_def.group('to_x')),
                    pattern.shape[1] - int(arrow_def.group('to_y')),
                    r'\accentcolor, line width=0.10em',
                    text=arrow_def.group('desc'),
                    text_properties='anchor=south west, xshift=0.2em, yshift=0.2em'))

            save_figure(
                output_filename,
                TikzImage(pattern.astype('uint8')),
                *arrows)

if __name__ == "__main__":
    if len(sys.argv) == 2:
        parameters = parameters_parse(sys.argv[1])
        create_patterns(parameters)
    else:
        print("""\
Usage:
  python diffraction_pattern_single.py <parameter file>""")

