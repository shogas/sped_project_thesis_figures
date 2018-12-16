import os
import sys

import numpy as np
import pickle
from tqdm import tqdm as report_progress

import matplotlib
matplotlib.use('Qt5Agg')
# matplotlib.use('pgf')
import matplotlib.pyplot as plt


import hyperspy.api as hs
from pyxem.signals.indexation_results import IndexationResults

from common import result_object_file_info
from parameters import parameters_parse
from figure import material_color_palette
from figure import save_figure
from figure import TikzColorbar
from figure import TikzImage
from figure import TikzScalebar


def combine_indexation_results(object_infos):
    total_width  = max((info['x_stop'] for info in object_infos))
    total_height = max((info['y_stop'] for info in object_infos))

    combined_indexation_results_data = None

    for object_info in object_infos:
        print('Tile {}:{}  {}:{} (of {} {})'.format(
            object_info['x_start'], object_info['x_stop'],
            object_info['y_start'], object_info['y_stop'],
            total_width, total_height))

        with open(object_info['filename'], 'rb') as f:
            indexation_results_data = pickle.load(f)

        if combined_indexation_results_data is None:
            combined_indexation_results_data = np.zeros((total_height, total_width, *indexation_results_data.shape[2:4]))

        combined_indexation_results_data[
                object_info['y_start']:object_info['y_stop'],
                object_info['x_start']:object_info['x_stop']] = indexation_results_data

    return IndexationResults(combined_indexation_results_data)


def combine_orientations(result_directory):
    parameters = parameters_parse(os.path.join(result_directory, 'metadata.txt'))
    methods = [
            method.strip() for method in parameters['methods'].split(',')
            if parameters['__save_method_{}'.format(method.strip())] == 'object']

    nav_scale_x = parameters['nav_scale_x']

    object_infos = result_object_file_info(result_directory)
    if 'template_match' in object_infos:
        indexation_results = combine_indexation_results(object_infos['template_match'])

        crystal_map = indexation_results.get_crystallographic_map()
        # crystal_map.save_mtex_map(os.path.join(result_directory, 'mtex_test.csv'))
        phase_map = crystal_map.get_phase_map()
        orientation_map = crystal_map.get_orientation_map()
        reliability_phase_map = crystal_map.get_reliability_map_phase()
        reliability_orientation_map = crystal_map.get_reliability_map_orientation()
        nav_width = phase_map.data.shape[1]

        colored_phases = np.zeros((*phase_map.data.shape, 3))
        colored_phases[phase_map.data == 0, :] = material_color_palette[0][1]  # Red ZB
        colored_phases[phase_map.data == 1, :] = material_color_palette[2][1]  # Blue WZ
        colored_phases *= 255.0
        # middle_slice = slice(int(0.25*nav_width), int(0.75*nav_width))
        # reliability_orientation_map.data *= 255.0 / reliability_orientation_map.data[:, middle_slice].max()
        # reliability_phase_map.data *= 255.0 / reliability_phase_map.data[:, middle_slice].max()
        rel_ori_min = reliability_orientation_map.data.min()
        rel_ori_max = reliability_orientation_map.data.max()
        rel_pha_min = reliability_phase_map.data.min()
        rel_pha_max = reliability_phase_map.data.max()

        reliability_orientation_map.data *= 255.0 / 0.6
        reliability_phase_map.data *= 255.0 / reliability_phase_map.data.max()

        save_figure(
                os.path.join(result_directory, 'phase_map.tex'),
                TikzImage(colored_phases.astype('uint8')),
                TikzScalebar(100, nav_scale_x*nav_width, r'\SI{100}{\nm}'))
        save_figure(
                os.path.join(result_directory, 'reliability_orientation_map.tex'),
                TikzImage(reliability_orientation_map.data.astype('uint8')),
                # TikzScalebar(100, nav_scale_x*nav_width, r'\SI{100}{\nm}'),
                TikzColorbar(rel_ori_min, rel_ori_max, (rel_ori_max-rel_ori_min)/4,
                    'blackwhite', r'0.8\textwidth',
                    at=r'{(0.1\textwidth, 0)}',
                    horizontal=True))
        save_figure(
                os.path.join(result_directory, 'reliability_phase_map.tex'),
                TikzImage(reliability_phase_map.data.astype('uint8')),
                # TikzScalebar(100, nav_scale_x*nav_width, r'\SI{100}{\nm}'))
                TikzColorbar(rel_pha_min, rel_pha_max, (rel_pha_max-rel_pha_min)/4,
                    'blackwhite', r'0.8\textwidth',
                    at=r'{(0.1\textwidth, 0)}',
                    horizontal=True))

        # save_figure(
                # os.path.join(result_directory, 'orientation_map_zb.tex'),
                # TikzImage('orientation_map_zb.png'))
        # save_figure(
                # os.path.join(result_directory, 'orientation_map_color_zb.tex'),
                # TikzImage('orientation_map_color_zb.pdf'))

        # save_figure(
                # os.path.join(result_directory, 'orientation_map_wz.tex'),
                # TikzImage('orientation_map_wz.png'))
        # save_figure(
                # os.path.join(result_directory, 'orientation_map_color_wz.tex'),
                # TikzImage('orientation_map_color_wz.pdf'))


        # dp = hs.load('D:/Dokumenter/MTNANO/Prosjektoppgave/SPED_data_GaAs_NW/gen/Julie_180510_SCN45_FIB_a_three_phase_single_area.hdf5'
        # dp.axes_manager.signal_axes[0].scale = 0.032
        # dp.axes_manager.signal_axes[1].scale = 0.032
        # dp.axes_manager.signal_axes[0].offset = -72*0.032
        # dp.axes_manager.signal_axes[1].offset = -72*0.032
        # print(dp.axes_manager)
        # from pyxem.libraries.diffraction_library import load_DiffractionLibrary
        # diffraction_library_cache_filename = os.path.join(
            # parameters['output_dir'],
            # 'tmp/diffraction_library_{}.pickle'.format(parameters['shortname']))
        # library = load_DiffractionLibrary(diffraction_library_cache_filename, safety=True)
        # indexation_results.plot_best_matching_results_on_signal(dp, ['zb', 'wz'], library)
        # crystal_map.get_reliability_map_phase().plot()
        # crystal_map.get_reliability_map_orientation().plot()
        # plt.show()

        # print(crystal_map.axes_manager)
        # print(crystal_map.data)


if __name__ == '__main__':
    hs.preferences.General.nb_progressbar = False
    hs.preferences.General.show_progressbar = False
    result_directory = sys.argv[1]
    combine_orientations(result_directory)
