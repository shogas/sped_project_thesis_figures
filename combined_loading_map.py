import os
import sys

import numpy as np
from PIL import Image

from tqdm import tqdm as report_progress

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

import pyxem as pxm
from pyxem.generators.indexation_generator import IndexationGenerator
from pyxem.generators.structure_library_generator import StructureLibraryGenerator

from common import result_image_file_info
from parameters import parameters_parse

from figure import save_figure
from figure import TikzImage
from figure import TikzScalebar


def image_l2_norm(image_a, image_b):
    a_max = image_a.max() or 1
    b_max = image_b.max() or 1
    image_a *= 1/a_max
    image_b *= 1/b_max
    return np.linalg.norm(image_a - image_b, ord='fro')


def image_l2_norm_fft(image_a, image_b):
    image_a = np.fft.fft2(image_a)
    image_b = np.fft.fft2(image_b)
    return image_l2_norm(image_a, image_b)


def load_compare_factors(parameters, known_factors):
    zb_1 = np.asarray(Image.open('../../Data/compare_factor_zb_1.png')).astype('float')
    known_factors.append(zb_1 / zb_1.max())
    zb_2 = np.asarray(Image.open('../../Data/compare_factor_zb_2.png')).astype('float')
    known_factors.append(zb_2 / zb_2.max())
    wz = np.asarray(Image.open('../../Data/compare_factor_wz.png')).astype('float')
    known_factors.append(wz / wz.max())
    vac = np.asarray(Image.open('../../Data/compare_factor_vac.png')).astype('float')
    known_factors.append(vac / vac.max())


def classify_compare_l2_norm(parameters, factor, known_factors):
    if len(known_factors) == 0:
        load_compare_factors(parameters, known_factors)

    diffs = [image_l2_norm_fft(factor, known_factor) for known_factor in known_factors]
    best_diff_index = np.argmin(diffs)
    best_diff = diffs[best_diff_index]
    factor_index = best_diff_index
    report_progress.write('    Matched phase {} (difference {})'.format(factor_index, best_diff))

    return factor_index


def classify_l2_norm_normal(parameters, factor, known_factors):
    return classify_l2_norm(parameters, factor, known_factors, image_l2_norm)


def classify_l2_norm_fourier(parameters, factor, known_factors):
    return classify_l2_norm(parameters, factor, known_factors, image_l2_norm_fft)


def classify_l2_norm(parameters, factor, known_factors, norm_func):
    threshold = parameters['classify_l2_norm_threshold']
    if len(known_factors) > 0:
        diffs = [norm_func(factor, known_factor) for known_factor in known_factors]
        best_diff_index = np.argmin(diffs)
        best_diff = diffs[best_diff_index]
        if (best_diff < threshold):
            factor_index = best_diff_index
            # plt.figure()
            # plt.suptitle(' '.join('{:.4f}'.format(d) for d in diffs))
            # for i, (f, d) in enumerate(zip(known_factors, diffs)):
                # plt.subplot(2, 4, i + 1)
                # plt.imshow(f)
            # plt.subplot(2, 4, 6)
            # plt.imshow(factor)
            # plt.show()
            report_progress.write('    Matched phase {} (difference {})'.format(factor_index, best_diff))
        else:
            factor_index = len(known_factors)
            known_factors.append(factor)
            report_progress.write('    New phase {} (difference {})'.format(factor_index, best_diff))
    else:
        factor_index = len(known_factors)
        known_factors.append(factor)

    return factor_index


# TODO(simonhog): From compare/methods/template_match
def create_diffraction_library(parameters, half_pattern_size):
    specimen_thickness = parameters['specimen_thickness']
    beam_energy_keV = parameters['beam_energy_keV']
    reciprocal_angstrom_per_pixel = parameters['reciprocal_angstrom_per_pixel']
    phase_names = [phase_name.strip() for phase_name in parameters['phase_names'].split(',')]
    rotation_list_resolution = np.deg2rad(1)

    phase_descriptions = []
    inplane_rotations = []
    for phase_name in phase_names:
        structure = loadStructure(parameters['phase_{}_structure_file'.format(phase_name)])
        crystal_system = parameters['phase_{}_crystal_system'.format(phase_name)]
        rotations = [float(r.strip()) for r in str(parameters['phase_{}_inplane_rotations'.format(phase_name)]).split(',')]
        phase_descriptions.append((phase_name, structure, crystal_system))
        inplane_rotations.append([np.deg2rad(r) for r in rotations])

    structure_library_generator = StructureLibraryGenerator(phase_descriptions)
    structure_library = structure_library_generator.get_orientations_from_stereographic_triangle(
            inplane_rotations, rotation_list_resolution)
    max_excitation_error = 1/specimen_thickness
    gen = DiffractionGenerator(beam_energy_keV, max_excitation_error=max_excitation_error)
    library_generator = DiffractionLibraryGenerator(gen)
    reciprocal_radius = reciprocal_angstrom_per_pixel*(half_pattern_size - 1)

    diffraction_library = library_generator.get_diffraction_library(
        structure_library,
        calibration=reciprocal_angstrom_per_pixel,
        reciprocal_radius=reciprocal_radius,
        half_shape=(half_pattern_size, half_pattern_size),
        with_direct_beam=False)

    return diffraction_library


diffraction_library = None
def classify_template_match(parameters, factor, known_factors):
    dp = pxm.ElectronDiffraction([[factor]])
    if diffraction_library is None:
        diffraction_library = create_diffraction_library(parameters, dp.data.shape[2])
    pattern_indexer = IndexationGenerator(dp, diffraction_library)
    indexation_results = pattern_indexer.correlate(n_largest=4, keys=phase_names, show_progressbar=False)
    crystal_mapping = indexation_results.get_crystallographic_map(show_progressbar=False)
    phases = crystal_mapping.get_phase_map().data.ravel()
    orientations = crystal_mapping.get_orientation_map().data.ravel()
    for phase, orientation in zip(phases, orientations):
        if (phase, orientation) in known_factors:
            factor_index = known_factors.index((phase, orientation))
            report_progress.write('    Matched phase {}, {}'.format(phase, orientation))
        else:
            factor_index = len(known_factors)
            known_factors.append((phase, orientation))
            report_progress.write('    New phase {}, {}'.format(phase, orientation))

    return factor_index


def combine_loading_map(parameters, method, factor_infos, loading_infos, classify):
    total_width  = max((info['x_stop'] for info in loading_infos))
    total_height = max((info['y_stop'] for info in loading_infos))

    combined_loadings = np.zeros((total_height, total_width, 3))

    colors = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
            [0.5, 0, 0],
            [0, 0.5, 0],
            [0, 0, 0.5],
            [0.5, 0.5, 0],
            [0.5, 0, 0.5],
            [0, 0.5, 0.5],
            ])

    known_factors = []
    factor_info = factor_infos[0]
    last_tile = (0, 0, 0, 0)
    factors = []
    for factor_info, loading_info in report_progress(zip(factor_infos, loading_infos), total=len(factor_infos)):
        tile = (factor_info['x_start'], factor_info['x_stop'],
                factor_info['y_start'], factor_info['y_stop'])
        if tile != last_tile:
            report_progress.write('Tile {}:{}  {}:{} (of {} {})'.format(*tile, total_width, total_height))
            last_tile = tile
        factor = np.asarray(Image.open(factor_info['filename'])).astype('float')
        factor_max = factor.max() or 1
        factor *= 1/factor_max

        factor_index = classify(parameters, factor, known_factors)
        report_progress.write('    Factor index: {}'.format(factor_index))

        loading = np.asarray(Image.open(loading_info['filename']))
        color = colors[factor_index % len(colors)]
        combined_loadings[
                factor_info['y_start']:factor_info['y_stop'],
                factor_info['x_start']:factor_info['x_stop']] += np.outer(loading.ravel(), color).reshape(loading.shape[0], loading.shape[1], 3)
        pixel_count = np.count_nonzero(loading[loading > 10])
        factors.append((factor_index, factor, pixel_count))

    combined_loadings *= 255 / combined_loadings.max()
    return combined_loadings.astype('uint8'), factors


def combine_loading_maps(parameters, result_directory, classification_method, scalebar_nm, rotation):
    shortname = parameters['shortname']
    methods = [
            method.strip() for method in parameters['methods'].split(',')
            if parameters['__save_method_{}'.format(method.strip())] == 'decomposition']
    factor_infos = result_image_file_info(result_directory, 'factors')
    loading_infos = result_image_file_info(result_directory, 'loadings')
    classify = {
        'l2_norm': classify_l2_norm_normal,
        'l2_norm_fourier': classify_l2_norm_fourier,
        'l2_norm_compare': classify_compare_l2_norm,
        'template_match': classify_template_match,
    }[classification_method]
    for (method_name, factor_infos_for_method), loading_infos_for_method in zip(factor_infos.items(), loading_infos.values()):
        allfactors = {}
        allfactor_weights = {}
        combined_loadings, factors = combine_loading_map(
                parameters,
                method_name,
                factor_infos_for_method,
                loading_infos_for_method,
                classify)
        for factor_index, factor, count in factors:
            if factor_index not in allfactors:
                allfactors[factor_index] = []
                allfactor_weights[factor_index] = []
            allfactors[factor_index].append(factor)
            allfactor_weights[factor_index].append(count)

        nav_width = combined_loadings.data.shape[1]
        save_figure(
                os.path.join(result_directory, 'loading_map_{}_{}.tex'.format(shortname, method_name)),
                TikzImage(combined_loadings, rotation),
                TikzScalebar(scalebar_nm, parameters['nav_scale_x']*nav_width, r'\SI{{{}}}{{\nm}}'.format(scalebar_nm)))

        for (factor_index, factor_list), factor_weights in zip(allfactors.items(), allfactor_weights.values()):
            if np.sum(factor_weights) == 0:
                factor_average = factor_list[0]
            else:
                factor_average = np.average(factor_list, weights=factor_weights, axis=0)
                factor_average *= 255.0 / factor_average.max()
            save_figure(
                    os.path.join(result_directory, 'factor_average_{}_{}_{}.tex'.format(shortname, method_name, factor_index)),
                    TikzImage(factor_average.astype('uint8')))


if __name__ == '__main__':
    # TODO(simonhog): Make these less global. known_factors -> general dictionary for data?
    result_directory = sys.argv[1]
    classification_method = sys.argv[2] if len(sys.argv) > 2 else 'l2_norm'
    parameters = parameters_parse(os.path.join(result_directory, 'metadata.txt'))
    parameters['classify_l2_norm_threshold'] = float(sys.argv[3]) if len(sys.argv) > 3 else 5
    scalebar_nm = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    rotation = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    combine_loading_maps(parameters, result_directory, classification_method, scalebar_nm, rotation)
