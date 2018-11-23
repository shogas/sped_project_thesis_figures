import os
import re
import sys

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image

import pyxem as pxm

from parameters import parameters_parse


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
    vdf_parameter_regex = re.compile(r'(?P<cx>\S*) (?P<cy>\S*) (?P<r>\S*) (?P<filename>.*)$')
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
                'r': float(match.group('r'))
            })

    for filename, rois in vdf_rois.items():
        s = pxm.ElectronDiffraction(pxm.load(filename))
        for roi_description in rois:
            roi = pxm.roi.CircleROI(roi_description['cx'], roi_description['cy'], roi_description['r'])
            vdf = s.get_virtual_image(roi).data.astype('float')
            vdf -= vdf.min()
            vdf *= 255.0 / vdf.max()
            out_filename = os.path.join(output_dir, 'vdf_{}.tiff'.format(roi_description['name']))
            Image\
                .fromarray(vdf.astype('uint8'))\
                .save(out_filename)


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


