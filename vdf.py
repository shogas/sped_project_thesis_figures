import os
import re
import sys

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

import numpy as np

import pyxem as pxm

from parameters import parameters_parse
from figure import save_image
from figure import save_figure
from figure import TikzCircle
from figure import TikzImage
from figure import TikzScalebar


def vdf_interactive(filename):
    s = pxm.ElectronDiffraction(pxm.load(filename))
    signal_width = s.axes_manager.signal_extent[1] - s.axes_manager.signal_extent[0]
    signal_height = s.axes_manager.signal_extent[3] - s.axes_manager.signal_extent[2]
    middle_x = signal_width / 2
    middle_y = signal_height / 2
    roi = pxm.roi.CircleROI(middle_x, middle_y, r_inner=0, r=signal_width / 50)
    s.plot_interactive_virtual_image(roi=roi)
    plt.show()
    print(
"""Ended with circle:
cx = {}
cy = {}
r = {}""".format(roi.cx, roi.cy, roi.r))


def vdf(parameters):
    output_dir = parameters['output_dir']
    output_dir = os.path.join(output_dir, 'run_{}_{}'.format(parameters['shortname'], parameters['__date_string']))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    vdf_rois = {}
    vdf_parameter_regex = re.compile(r'(?P<dp_x>\S*) (?P<dp_y>\S*) (?P<cx>\S*) (?P<cy>\S*) (?P<r>\S*) (?P<filename>.*)$')
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
            })

    for filename, rois in vdf_rois.items():
        s = pxm.load(filename, lazy=True)
        nav_scale_x = s.axes_manager.navigation_axes[0].scale
        s = pxm.ElectronDiffraction(s)
        for desc in rois:
            roi = pxm.roi.CircleROI(desc['cx'], desc['cy'], desc['r'])
            vdf = s.get_virtual_image(roi).data.astype('float')
            vdf -= vdf.min()
            vdf *= 255.0 / vdf.max()
            nav_height, nav_width, sig_height, sig_width = s.data.shape
            save_figure(
                    os.path.join(output_dir, 'vdf_{}.tex'.format(desc['name'])),
                    TikzImage(vdf.astype('uint8')),
                    TikzScalebar(100, nav_scale_x*nav_width, r'\SI{100}{\nm}'))
            save_figure(
                os.path.join(output_dir, 'vdf_{}_dp.tex'.format(desc['name'])),
                TikzImage(s.inav[desc['dp_x'], desc['dp_y']].data),
                TikzCircle(desc['cx'], sig_height - desc['cy'], desc['r'], r'\accentcolor'),
                TikzScalebar(1, 0.032*sig_width, r'\SI{1}{\per\angstrom}'))


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


