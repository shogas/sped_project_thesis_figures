import sys

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

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


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == 'interactive':
        vdf_interactive(sys.argv[2])
    elif len(sys.argv) == 2:
        parameters = parameters_parse(sys.argv[1])
    else:
        print(
"""Usage, one of:
    python vdf.py interactive <SPED file>
    python vdf.py <parameter file>""")


