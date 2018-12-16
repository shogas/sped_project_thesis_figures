import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import hyperspy.api as hs

from figure import save_figure
from figure import TikzAxis
from figure import TikzTablePlot


def run_scree_plot(filename, result_directory, shortname):
    signal = hs.load(filename, lazy=True)
    signal.decomposition(normalize_poissonian_noise=True, algorithm='svd', output_dimension=100)
    scree_data_full = np.array(signal.get_explained_variance_ratio().data)
    signal = signal.inav[:145, :205]
    signal.decomposition(normalize_poissonian_noise=True, algorithm='svd', output_dimension=100)
    scree_data_quarter = np.array(signal.get_explained_variance_ratio().data)

    axis_styles = {
        'axis_x_line': 'bottom',
        'axis_y_line': 'left',
        'width': r'\textwidth',
    }
    plot_styles = {
        'mark': '*',
        'only marks': 'true',
    }
    color_styles = [{
        'color': 'MaterialBlue',
        'mark_options': '{fill=MaterialBlue}'
    }, {
        'color': 'MaterialRed',
        'mark_options': '{fill=MaterialRed}'
    }]
    save_figure(
        os.path.join(result_directory, 'explained_variance_ratio_{}.tex'.format(shortname)),
        TikzAxis(
            TikzTablePlot(range(len(scree_data_full)), scree_data_full, **plot_styles + color_styles[0]),
            TikzTablePlot(range(len(scree_data_quarter)), scree_data_quarter, **plot_styles + color_styles[1]),
            xlabel='Component index',
            ylabel='Proportion of variance',
            axis_type='semilogy',
            **axis_styles))


if __name__ == '__main__':
    filename = sys.argv[1]
    result_directory = sys.argv[2]
    shortname = sys.argv[3]
    run_scree_plot(filename, result_directory, shortname)
