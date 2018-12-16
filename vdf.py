import os
import re
import sys

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

import numpy as np

import pyxem as pxm
from skimage.transform import rotate as sk_rotate

from parameters import parameters_parse
from parameters import parameters_save
from figure import save_image
from figure import save_figure
from figure import TikzCircle
from figure import TikzImage
from figure import TikzRectangle
from figure import TikzScalebar


def rotate_2d(angle):
    return np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]])


def vdf_interactive(filename):
    s = pxm.ElectronDiffraction(pxm.load(filename))
    signal_width = s.axes_manager.signal_extent[1] - s.axes_manager.signal_extent[0]
    signal_height = s.axes_manager.signal_extent[3] - s.axes_manager.signal_extent[2]
    middle_x = signal_width / 2
    middle_y = signal_height / 2
    roi = pxm.roi.CircleROI(middle_x, middle_y, r_inner=0, r=signal_width / 50)
    s.plot_interactive_virtual_image(roi=roi)
    fig = plt.figure()

    def onclick(event):
        print("""\
Ended with circle:
cx = {}
cy = {}
r = {}""".format(roi.cx, roi.cy, roi.r))

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()


def vdf(parameters):
    output_dir = parameters['output_dir']
    output_dir = os.path.join(output_dir, 'run_{}_{}'.format(parameters['shortname'], parameters['__date_string']))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    vdf_rois = {}
    vdf_parameter_regex = re.compile(r"""
(?P<dp_x>\S*)\s+
(?P<dp_y>\S*)\s+
(?P<cx>\S*)\s+
(?P<cy>\S*)\s+
(?P<r>\S*)\s+
(?P<rot>\S*)\s+
("(?P<label>[^"]*)"\s+)?
(?P<filename>.*)$""", re.X)
    for name, value in parameters.items():
        if name.startswith('vdf_'):
            match = vdf_parameter_regex.match(value)
            filename = match.group('filename')
            if filename not in vdf_rois:
                vdf_rois[filename] = []
            vdf_rois[filename].append({
                'name': name[4:],
                'cx': float(match.group('cx')),
                'cy': float(match.group('cy')),
                'r': float(match.group('r')),
                'dp_x': int(match.group('dp_x')),
                'dp_y': int(match.group('dp_y')),
                'rot': int(match.group('rot')),
                'label': match.group('label'),
            })

    for filename, rois in vdf_rois.items():
        print('Loading', filename)
        s = pxm.load(filename, lazy=True)
        s = pxm.ElectronDiffraction(s)
        nav_scale_x = s.axes_manager.navigation_axes[0].scale
        nav_scale_y = s.axes_manager.navigation_axes[1].scale
        nav_height, nav_width, sig_height, sig_width = s.data.shape
        dp_centre = np.array([0.5*sig_width, 0.5*sig_height])

        for i, desc in enumerate(rois):
            print('  Generating VDF image for', desc['name'])
            roi = pxm.roi.CircleROI(desc['cx'], desc['cy'], desc['r'])
            vdf = s.get_virtual_image(roi).data.astype('float')
            vdf -= vdf.min()
            vdf *= 255.0 / vdf.max()

            dp = s.inav[desc['dp_x'], desc['dp_y']].data
            dp = sk_rotate(dp, parameters['dp_rotation'], resize=False, preserve_range=True)
            circle_pos = dp_centre + rotate_2d(np.deg2rad(parameters['dp_rotation']))@ (np.array([desc['cx'], sig_height - desc['cy']]) - dp_centre)
            nav_rotation = rotate_2d(np.deg2rad(desc['rot']))

            intensity_elements = [
                TikzImage(vdf.astype('uint8'), angle=desc['rot']),
                TikzCircle(desc['dp_x'], nav_height - desc['dp_y'], 2, r'\accentcolor, fill=\accentcolor',
                    transform=nav_rotation),
            ]
            dp_elements = [
                TikzImage(dp.astype('uint8')),
                TikzCircle(circle_pos[0], circle_pos[1], desc['r'], r'\accentcolor, line width=0.10em',
                    text=desc['label'],
                    text_properties='anchor=south west, xshift=0.3em, yshift=0.3em'),
            ]
            rectangle_name = 'rectangle_{}'.format(desc['name'])
            scalebar_name = 'scalebar_{}'.format(desc['name'])
            if rectangle_name in parameters:
                coords = [int(coord) for coord in parameters[rectangle_name].split()]
                corner_a = np.array([coords[0], nav_height - coords[1]])
                corner_b = np.array([coords[2], nav_height - coords[3]])
                intensity_elements.append(TikzRectangle(*corner_a, *corner_b, r'\accentcolor, line width=0.15em',
                    transform=nav_rotation))
            if scalebar_name in parameters:
                intensity_scale_length = parameters[scalebar_name]
                intensity_scale_physical = nav_scale_x*nav_width if desc['rot'] == 0 else nav_scale_y*nav_height
                scalebar_length = r'\SI{{{}}}{{\nm}}'.format(intensity_scale_length)
                intensity_elements.append(TikzScalebar(intensity_scale_length, intensity_scale_physical, scalebar_length))
                dp_elements.append(TikzScalebar(1, 0.032*sig_width, r'\SI{1}{\per\angstrom}'))

            output_prefix = '{}_{}-{}_{}-{}'.format('vdf', 0, nav_width, 0, nav_height)
            save_figure(
                    os.path.join(output_dir, '{}_loadings_{}_{}.tex'.format(output_prefix, desc['name'], i)),
                    *intensity_elements)
            save_figure(
                os.path.join(output_dir, '{}_factors_{}_{}.tex'.format(output_prefix, desc['name'], i)),
                *dp_elements)

    parameters['methods'] = 'vdf'
    parameters['__save_method_vdf'] = 'decomposition'
    parameters['sample_file'] = next(iter(vdf_rois.keys()))  # At least one of them
    parameters['nav_scale_x'] = 1.28  # TODO: Get from file
    parameters_save(parameters, output_dir)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == 'interactive':
        vdf_interactive(sys.argv[2])
    elif len(sys.argv) == 2:
        parameters = parameters_parse(sys.argv[1])
        vdf(parameters)
    else:
        print(
"""Usage, one of:
    python vdf.py interactive <SPED file>
    python vdf.py <parameter file>""")


