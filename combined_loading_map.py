from collections import defaultdict
import glob
import os
import sys

import numpy as np
from PIL import Image
from skimage.transform import rotate as sk_rotate

from tqdm import tqdm as report_progress

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

import seaborn as sns

from diffpy.structure import loadStructure
import hyperspy.api as hs
import pyxem as pxm
from pyxem.generators.diffraction_generator import DiffractionGenerator
from pyxem.generators.indexation_generator import IndexationGenerator
from pyxem.generators.library_generator import DiffractionLibraryGenerator
from pyxem.generators.structure_library_generator import StructureLibraryGenerator
from pyxem.libraries.diffraction_library import load_DiffractionLibrary
from pyxem.signals.crystallographic_map import CrystallographicMap
from pyxem.utils.expt_utils import affine_transformation
import hdbscan

from common import result_image_file_info
from parameters import parameters_parse
from combined_orientation import save_crystallographic_map

from figure import save_figure
from figure import TikzAxis
from figure import TikzColorbar
from figure import TikzImage
from figure import TikzRectangle
from figure import TikzScalebar
from figure import TikzTablePlot
from figure import material_color_palette



def image_l1_norm(image_a, image_b):
    a_max = image_a.max() or 1
    b_max = image_b.max() or 1
    image_a_scaled = image_a * (1/a_max)
    image_b_scaled = image_b * (1/b_max)
    return np.linalg.norm(image_a_scaled - image_b_scaled, ord=1)


def image_l1_norm_fft(image_a, image_b):
    image_a = np.fft.fft2(image_a)
    image_b = np.fft.fft2(image_b)
    return image_l1_norm(image_a, image_b)


def image_l2_norm_log(image_a, image_b):
    image_a = np.log1p(image_a)
    image_b = np.log1p(image_b)
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

    return factor_index, None



def classify_l2_norm_log(parameters, factor, known_factors):
    return classify_norm(parameters, factor, known_factors, image_l2_norm_log)


def classify_l1_norm_normal(parameters, factor, known_factors):
    return classify_norm(parameters, factor, known_factors, image_l1_norm)


def classify_l1_norm_fourier(parameters, factor, known_factors):
    return classify_norm(parameters, factor, known_factors, image_l1_norm_fft)




def create_diffraction_library(parameters, pattern_size):
    diffraction_library_cache_filename = os.path.join(
            parameters['output_dir'],
            'tmp/diffraction_library_{}.pickle'.format(parameters['shortname']))
    if os.path.exists(diffraction_library_cache_filename):
        return load_DiffractionLibrary(diffraction_library_cache_filename, safety=True)

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
    half_pattern_size = pattern_size // 2
    reciprocal_radius = reciprocal_angstrom_per_pixel*(half_pattern_size - 1)

    diffraction_library = library_generator.get_diffraction_library(
        structure_library,
        calibration=reciprocal_angstrom_per_pixel,
        reciprocal_radius=reciprocal_radius,
        half_shape=(half_pattern_size, half_pattern_size),
        with_direct_beam=False)


    diffraction_library.pickle_library(diffraction_library_cache_filename)

    return diffraction_library


diffraction_library = None
def classify_template_match(parameters, factor, known_factors):
    phase_names = [phase_name.strip() for phase_name in parameters['phase_names'].split(',')]
    dp = pxm.ElectronDiffraction([[factor]])
    global diffraction_library
    if diffraction_library is None:
        diffraction_library = create_diffraction_library(parameters, dp.data.shape[2])
    pattern_indexer = IndexationGenerator(dp, diffraction_library)
    indexation_results = pattern_indexer.correlate(n_largest=4, keys=phase_names, show_progressbar=False)
    crystal_mapping = indexation_results.get_crystallographic_map(show_progressbar=False)
    phases = crystal_mapping.get_phase_map().data.ravel()
    orientations = crystal_mapping.isig[1:4].data[0]  #crystal_mapping.get_orientation_map().data.ravel()
    scores = crystal_mapping.isig[4].data[0]
    for phase, orientation, score in zip(phases, orientations, scores):
        phase = int(phase)
        factor_index = -1
        for i, (key_phase, a, b, c) in enumerate(known_factors):
            # Quite large bounds, since this is just for grouping similar areas
            if key_phase == phase and\
                    abs(orientation[0] - a) < 15 and\
                    abs(orientation[1] - b) < 15 and\
                    abs(orientation[2] - c) < 15:
                        factor_index = i
                        break
        if factor_index >= 0:
            report_progress.write('    Matched phase {}, ori: {}, {}, {}, score: {}'.format(phase, *orientation, score))
        else:
            factor_index = len(known_factors)
            known_factors.append((phase, *orientation))
            report_progress.write('    New phase {}, ori: {}, {}, {}'.format(phase, *orientation))

    return factor_index, ('template', crystal_mapping.data)


def preprocessor_affine_transform(signal, parameters):
    report_progress.write('Applying transform')
    # TODO(simonhog): What is the cost of wrapping in ElectronDiffraction?
    # signal = pxm.ElectronDiffraction(data)
    if 'scale_x' not in parameters:
        print('Missing transformation information in parameters, skipping')
        return signal
    scale_x = parameters['scale_x']
    scale_y = parameters['scale_y']
    offset_x = parameters['offset_x']
    offset_y = parameters['offset_y']
    transform = np.array([
            [scale_x, 0, offset_x],
            [0, scale_y, offset_y],
            [0, 0, 1]
        ])
    # signal.map(affine_transformation,
          # matrix=transform,
          # inplace=True,
          # order=3,
          # ragged=False,
          # parallel=True)
    signal.apply_affine_transformation(transform)
    return signal


def preprocessor_gaussian_difference(signal, parameters):
    # TODO(simonhog): Does this copy the data? Hopefully not
    report_progress.write('Gaussian')
    if 'gaussian_sigma_min' not in parameters:
        print('Missing gaussian information in parameters, skipping')
        return signal
    # signal = pxm.ElectronDiffraction(data)
    sig_width = signal.axes_manager.signal_shape[0]
    sig_height = signal.axes_manager.signal_shape[1]

    signal = signal.remove_background(
            'gaussian_difference',
            sigma_min=parameters['gaussian_sigma_min'],
            sigma_max=parameters['gaussian_sigma_max'])
    signal.data /= signal.data.max()

    return signal


def create_lineplot(parameters, result_directory, merged_factor_infos, combined_loadings, loading_rotation, full_height, method_name, colors, color_mapping):
    shortname = parameters['shortname']
    line_plot_start = 10  # TODO(simonhog): Move to parameters
    line_plot_end = 22  # TODO(simonhog): Move to parameters
    line_loadings = {}
    for factor_index, factor_infos in merged_factor_infos.items():
        for factor, factor_info, loading, loading_info, factor_weight, tile, classify_extra in factor_infos:
            if tile[2] < line_plot_start and line_plot_start < tile[3]:
                tile_width = (tile[1] - tile[0])
                if factor_index not in line_loadings:
                    line_loadings[factor_index] = np.zeros(line_plot_end - line_plot_start)
                line_loadings[factor_index] += loading[line_plot_start:line_plot_end, tile_width // 2]
                line_x = (tile[0] + tile[1]) // 2

    nav_width = combined_loadings.data.shape[1]
    save_figure(
            os.path.join(result_directory, 'lineplot_loading_map_{}_{}.tex'.format(shortname, method_name)),
            TikzImage(combined_loadings.astype('uint8'), loading_rotation),
            TikzScalebar(scalebar_nm, parameters['nav_scale_x']*nav_width, r'\SI{{{}}}{{\nm}}'.format(scalebar_nm)),
            TikzRectangle(line_x, full_height - line_plot_start, line_x + 1, full_height - line_plot_end, r'black, line width=0.1em'))

    # TODO(simonhog): Utility function for easier multi-line plots
    axis_styles = {
        'legend_pos': 'north west',
        'axis_x_line': 'bottom',
        'axis_y_line': 'left',
        'xmin': 0,
        'ymin': 0,
        'width': r'\textwidth',
        'enlargelimits': 'upper',
    }
    line_styles = {
        'solid': 'true',
        'mark': '*',
        'line_width': '1.5pt'
    }
    line_elements = []
    x_axis_labels = ['{:.1f}'.format(1.28*i) for i in range(line_plot_end - line_plot_start)]
    for factor_index, loading_line in line_loadings.items():
        if np.count_nonzero(loading_line) == 0:
            continue
        color = colors[color_mapping[factor_index % len(colors)]][0]
        styles = {
            **line_styles,
            'color': color,
            'mark_options': '{{fill={}, scale=0.75}}'.format(color)}
        line_elements.append(TikzTablePlot(
            x_axis_labels, loading_line, **styles))

    save_figure(
            os.path.join(result_directory, 'lineplot_{}_{}.tex'.format(shortname, method_name)),
            TikzAxis(
                *line_elements,
                xlabel=r'Position/\si{\nm}',
                ylabel='Loading',
                **axis_styles))


def create_combined_loading_map(parameters, merged_factor_infos, full_width, full_height, colors, color_mapping, method_name, rotation):
    shortname = parameters['shortname']
    combined_loadings = np.zeros((full_height, full_width, 3))
    for factor_index, factor_infos in merged_factor_infos.items():
        for factor, factor_info, loading, loading_info, factor_weight, tile, classify_extra in factor_infos:
            x_slice = slice(factor_info['x_start'], factor_info['x_stop'])
            y_slice = slice(factor_info['y_start'], factor_info['y_stop'])
            color = colors[color_mapping[factor_index % len(colors)]][1]
            combined_loadings[y_slice, x_slice] += np.outer(loading.ravel(), color).reshape(loading.shape[0], loading.shape[1], 3)

    nav_width = combined_loadings.data.shape[1 if rotation == 0 else 0]
    combined_loadings *= 255 / combined_loadings.max()
    save_figure(
            os.path.join(result_directory, 'loading_map_{}_{}.tex'.format(shortname, method_name)),
            TikzImage(combined_loadings.astype('uint8'), rotation),
            TikzScalebar(scalebar_nm, parameters['nav_scale_x']*nav_width, r'\SI{{{}}}{{\nm}}'.format(scalebar_nm)))

    return combined_loadings


def calculate_error(parameters, experimental, reconstruction):
    experimental_data = preprocessor_gaussian_difference(
        preprocessor_affine_transform(
            pxm.ElectronDiffraction(experimental), parameters),
        parameters).data
    return np.sum(np.abs(experimental_data - reconstruction), axis=(2, 3))


def create_reconstruction(parameters, result_directory, method_name,
        experimental, factor_infos, loading_infos, full_width, full_height):
    last_tile = None
    error = np.zeros((full_height, full_width))
    for factor_info, loading_info in report_progress(zip(factor_infos, loading_infos), total=len(factor_infos)):
        tile = (factor_info['x_start'], factor_info['x_stop'],
                factor_info['y_start'], factor_info['y_stop'])
        tile_width = tile[1] - tile[0]
        tile_height = tile[3] - tile[2]
        if last_tile is None or tile != last_tile:
            report_progress.write('Tile {}:{}  {}:{} (of {} {})'.format(*tile, full_width, full_height))
            if last_tile is not None:
                slice_x = slice(last_tile[0], last_tile[1])
                slice_y = slice(last_tile[2], last_tile[3])
                # Done with a tile, calculate the error
                error[slice_y, slice_x] = calculate_error(parameters, experimental.inav[slice_x, slice_y], reconstruction)
            # Clear the reconstruction, ready for new tile
            reconstruction = np.zeros((tile_height, tile_width, 144, 144))
            last_tile = tile

        factor = load_factorization_data(factor_info['filename'])
        loading = load_factorization_data(loading_info['filename'])
        reconstruction += np.outer(
            loading.ravel(),
            factor.ravel()).reshape(*loading.shape, *factor.shape)

    slice_x = slice(last_tile[0], last_tile[1])
    slice_y = slice(last_tile[2], last_tile[3])
    error[slice_y, slice_x] = calculate_error(parameters, experimental.inav[slice_x, slice_y], reconstruction)

    # NOTE: To get the same scale on all the error plots
    error_min = 140
    error_max = 280
    print('Min error', error.min(), 'colorbar', error_min)
    print('Max error', error.max(), 'colorbar', error_max)

    shortname = parameters['shortname']
    # TODO: Parameterize
    if 'three' in shortname:
        scalebar_nm = 20
    elif '110' in shortname:
        scalebar_nm = 100
    else:
        scalebar_nm = 500
    fig, ax = plt.subplots()
    error_image = ax.imshow(error, cmap='viridis')
    error_colors = 255*np.array(error_image.cmap(error_image.norm(error)))[:, :, 0:3]
    save_figure(
            os.path.join(result_directory, 'reconstruction_error_{}_colorbar.tex'.format(shortname)),
            TikzColorbar(error_min, error_max, None, 'viridis', '4cm'))
    save_figure(
            os.path.join(result_directory, 'reconstruction_error_{}_{}.tex'.format(shortname, method_name)),
            TikzImage(error_colors.astype('uint8')),
            TikzScalebar(scalebar_nm, parameters['nav_scale_x']*full_width, r'\SI{{{}}}{{\nm}}'.format(scalebar_nm)))


def image_l2_norm(image_a, image_b):
    """Compute L2 norm difference of two images.

    Parameters
    ----------
    image_a, image_b : 2D numpy.ndarray
        Image data.

    Returns
    -------
    l2_norm : float
        L2 norm of the difference between `image_a` and `image_b`.
    """
    a_max = image_a.max() or 1
    b_max = image_b.max() or 1
    image_a *= 1/a_max
    image_b *= 1/b_max
    return np.linalg.norm(image_a - image_b, ord='fro')


def image_l2_norm_fft(image_a, image_b):
    """Compute L2 norm difference of the Fourier transform of two images.

    Parameters
    ----------
    image_a, image_b : 2D numpy.ndarray
        Image data.

    Returns
    -------
    l2_norm : float
        L2 norm of the difference between Fourier(`image_a`)
        and Fourier(`image_b`).
    """
    image_a = np.fft.fft2(image_a)
    image_b = np.fft.fft2(image_b)
    return image_l2_norm(image_a, image_b)


def classify_l2_norm_normal(parameters, factor, known_factors):
    """Classify `factor` by a factor index using L2 norm difference.
    See classify_norm for description."""
    return classify_norm(parameters['classify_l2_norm_threshold'], factor, known_factors, image_l2_norm)


def classify_l2_norm_fourier(parameters, factor, known_factors):
    """Classify `factor` by a factor index using L2 norm Fourier difference.
    See classify_norm for description."""
    return classify_norm(parameters['classify_l2_norm_threshold'], factor, known_factors, image_l2_norm_fft)


def classify_norm(norm_threshold, factor, known_factors, norm_func):
    """Classify `factor` by an index, either merging it with a similar,
    known factor or adding it as a new, known factor and assigning a new index.

    Parameters
    ----------
    norm_threshold : float
        Threshold for separating new and known factors based on difference
        calculated by `norm_func`.
    factor : 2D numpy.ndarray
        Factor to classify
    known_factor : list
        List of already known factors. May be updated
    norm_func : function
        Callable function taking two 2D numpy.ndarrays and returning the
        difference between them. One of the `image_*` functions above.

    Returns
    -------
    factor_index : int
        Index of the `factor`, either matched or new.
    classification_results
        Any extra results returned from the classification method.
    """
    if len(known_factors) > 0:
        # If we have seen any factors before, calculate the difference between
        # them and the new factor.
        diffs = [norm_func(factor, known_factor) for known_factor in known_factors]
        # And find the best matching
        best_diff_index = np.argmin(diffs)
        best_diff = diffs[best_diff_index]
        if (best_diff < norm_threshold):
            # This factor matched a known factor, return its index
            factor_index = best_diff_index
            report_progress.write('    Matched phase {} (difference {})'.format(factor_index, best_diff))
        else:
            # No factor had a difference below the threshold, assign a new index
            # and record the new factor as a known one.
            factor_index = len(known_factors)
            known_factors.append(factor)
            report_progress.write('    New phase {} (difference {})'.format(factor_index, best_diff))
    else:
        # If no factors have been seen before, this must be a new one.
        factor_index = len(known_factors)
        known_factors.append(factor)

    return factor_index, None

def load_factorization_data(filename):
    """Load a component signal or loading map from tiff or png, but
    first trying to find the original numpy array file for better bit depth.
    """
    numpy_filename = filename.replace('tiff', 'npy').replace('png', 'npy')
    if os.path.exists(numpy_filename):
        # Numpy file exists, load it
        data = np.load(numpy_filename)
    else:
        # Fall back to loading the image, normalized to [0, 1]
        data = np.asarray(Image.open(filename)).astype('float')
        data *= 1/255.0
    return data


def merge_factors(factor_infos, loading_infos, classify):
    """Merge factors and corresponding loading maps based on information
    from the metadata stored during processing.

    Parameters
    ----------
    factor_infors, loading_infos : dict
        Dictionary containing the metadata stored during processing.
    classify : function
        Function for assigning an index to each factor.
    """
    merged_factor_infos = defaultdict(list)
    last_tile = None
    known_factors = []
    # Iterate all factor and loading infos, assumed to be sorted by which section
    for factor_info, loading_info in report_progress(zip(factor_infos, loading_infos), total=len(factor_infos)):
        tile = (factor_info['x_start'], factor_info['x_stop'],
                factor_info['y_start'], factor_info['y_stop'])
        # If we are starting on a new tile, report it
        if last_tile is None or tile != last_tile:
            report_progress.write('Tile {}:{}  {}:{}'.format(*tile))
            last_tile = tile

        # Load component factor and loading map data
        factor = load_factorization_data(factor_info['filename'])
        loading = load_factorization_data(loading_info['filename'])

        # Classify the (normalised) factor
        report_progress.write('    Factor index: {} ({})'.format(factor_index, os.path.basename(factor_info['filename'])))
        factor_max = factor.max() or 1
        factor_index, classify_extra = classify(parameters, factor.copy() * (1/factor_max), known_factors)

        # Measure the density of the loading map
        pixel_count = np.count_nonzero(loading[loading > 0.04])

        # Append the factor and loading information to the list of tiles for this factor
        merged_factor_infos[factor_index].append((factor, factor_info, loading, loading_info, pixel_count, tile, classify_extra))
    return merged_factor_infos


def create_average_factors(parameters, result_directory, merged_factor_infos, method_name):
    dp_rotation = 41  # TODO(simonhog): Move to parameters
    sig_width = 144
    shortname = parameters['shortname']
    for factor_index, factor_infos in merged_factor_infos.items():
        factors = [info[0] for info in factor_infos]
        weights = [info[4] for info in factor_infos]

        if np.sum(weights) == 0:
            factor_average = factors[0]
        else:
            factor_average = np.average(factors, weights=weights, axis=0)
            factor_average *= 255.0 / factor_average.max()
            factor_colors = sk_rotate(factor_average, dp_rotation, resize=False, preserve_range=True)
        if 'cepstrum' in method_name:
            factor_average[factor_average > 10] = 10
            fig, ax = plt.subplots()
            factor_image = ax.imshow(factor_average, cmap='viridis')
            factor_colors = 255*np.array(factor_image.cmap(factor_image.norm(factor_average)))[:, :, 0:3]
        save_figure(
                os.path.join(result_directory, 'factor_average_{}_{}_{}.tex'.format(shortname, method_name, factor_index)),
                TikzImage(factor_colors.astype('uint8')),
                TikzScalebar(1, 0.032*sig_width, r'\SI{1}{\per\angstrom}'))


def create_crystallographic_map(result_directory, merged_factor_infos, full_width, full_height, method_name):
    crystallographic_map = np.zeros((full_height, full_width, 7))
    for factor_index, factor_infos in merged_factor_infos.items():
        for factor, factor_infos, loading, loading_info, factor_weight, tile, classify_extra in factor_infos:
            slice_x = slice(tile[0], tile[1])
            slice_y = slice(tile[2], tile[3])
            if classify_extra is not None and classify_extra[0] == 'template':
                # TODO(simonhog): This only keeps the crystallographic map of the last when overlapping
                crystallographic_map[slice_y, slice_x][loading > 0.2] = classify_extra[1]

    if np.count_nonzero(crystallographic_map) > 0:
        save_crystallographic_map(parameters, result_directory, CrystallographicMap(crystallographic_map), method_name)


def create_umap_projection(parameters, result_directory, merged_factor_infos, colors, color_mapping, full_width, full_height):
    shortname = parameters['shortname']

    embedding_maps = {
        '110_full_umap_a': [
            (slice(0,   145), slice(0,   205)),
            (slice(145, 290), slice(0,   205)),
            (slice(0,   145), slice(205, 410)),
            (slice(145, 290), slice(205, 410))],
        '112_c_full_umap': [],
        '112_d_full_umap': [],
        '112_e_full_umap': [],
        '110_three_phase_no_split': [(slice(0, 50), slice(0, 50))],
        '110_three_phase_no_split_umap_figure': [(slice(0, 50), slice(0, 50))],
    }
    for i in range(0, full_height, 150):
        embedding_maps['112_c_full_umap'].append((slice(0, 200), slice(i, min(i + 150, full_height))))
        embedding_maps['112_d_full_umap'].append((slice(0, 180), slice(i, min(i + 150, full_height))))
        embedding_maps['112_e_full_umap'].append((slice(0, 210), slice(i, min(i + 150, full_height))))

    embedding_map = embedding_maps[shortname]
    n_components = 2 if 'three_phase' in shortname else 3
    embedding = np.zeros((full_height, full_width, n_components))
    for pos, embedding_filename in zip(embedding_map, glob.iglob(os.path.join(result_directory, 'embedding_*.npy'))):
        tile_width = pos[0].stop - pos[0].start
        tile_height = pos[1].stop - pos[1].start
        embedding[pos[1], pos[0]] = np.load(embedding_filename).reshape(tile_height, tile_width, n_components)

    if False:
        clusterer = hdbscan.HDBSCAN(
            min_samples=parameters['umap_cluster_min_samples'],
            # min_cluster_size=parameters['umap_cluster_size'],
            min_cluster_size=2000,
        ).fit(embedding.reshape(full_height*full_width, n_components))

        for i in range(np.max(clusterer.labels_)):
            print(i, np.count_nonzero(clusterer.labels_ == i))
        cluster_colors = [colors[color_mapping[l % len(colors)]][1] if l >= 0
                          else (0.3, 0.3, 0.3)
                          for l in clusterer.labels_]
        cluster_member_colors = [sns.desaturate(x, p) for x, p in
                                 zip(cluster_colors, clusterer.probabilities_)]

    else:
        cluster_colors = np.zeros((full_height, full_width, 3))
        for factor_index, factor_infos in merged_factor_infos.items():
            color = colors[color_mapping[factor_index % len(colors)]][1]
            for factor, factor_infos, loading, loading_info, factor_weight, tile, classify_extra in factor_infos:
                slice_x = slice(tile[0], tile[1])
                slice_y = slice(tile[2], tile[3])
                cluster_colors[slice_y, slice_x, :] += np.outer(loading.ravel(), color).reshape(loading.shape[0], loading.shape[1], 3)
        cluster_member_colors = cluster_colors.reshape(full_height*full_width, 3)

    fig, ax_scatter = plt.subplots()
    if n_components == 3:
        x, y = embedding.reshape(full_width*full_height, n_components)[:, 1:3].T
    else:
        x, y = embedding.T
    ax_scatter.scatter(x, y, s=15, c=cluster_member_colors, alpha=0.25)
    ax_scatter.tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelleft=False,
        labelright=False,
        labelbottom=False)

    plt.savefig(
        os.path.join(result_directory, 'embedding.png'),
        dpi=300,
        frameon=False,
        bbox_inches='tight',
        pad_inches=0)


def process_method(parameters, result_directory, method, classify, factor_infos, loading_infos, loading_rotation):
    allfactors = {}
    allfactor_weights = {}

    experimental = hs.load(parameters['sample_file'], lazy=True)
    full_width = experimental.data.shape[1]
    full_height = experimental.data.shape[0]

    merged_factor_infos = merge_factors(factor_infos, loading_infos, classify)

    colors = [
        ('Red', [1, 0, 0]),
        ('Green', [0, 1, 0]),
        ('Blue', [0, 0, 1]),
        ('Yellow', [1, 1, 0]),
        ('Magenta', [1, 0, 1]),
        ('Cyan', [0, 1, 1]),
    ]
    shortname = parameters['shortname']
    if method == 'umap' or '112' in shortname:
        colors = material_color_palette
    color_mapping = list(range(len(colors)))
    # TODO(simonhog): Parameterize
    if method == 'umap' and '110' in shortname:
        color_mapping[1], color_mapping[2] = color_mapping[2], color_mapping[1]
        color_mapping[1], color_mapping[6] = color_mapping[6], color_mapping[1]
        color_mapping[3], color_mapping[5] = color_mapping[5], color_mapping[3]
        color_mapping[7], color_mapping[3] = color_mapping[3], color_mapping[7]
    elif method == 'nmf_cepstrum' and '110' in shortname:
        color_mapping[2], color_mapping[3] = color_mapping[3], color_mapping[2]

    print('Combining loading map')
    combined_loadings = create_combined_loading_map(parameters, merged_factor_infos, full_width, full_height, colors, color_mapping, method, loading_rotation)

    print('Creating line plot')
    create_lineplot(parameters, result_directory, merged_factor_infos, combined_loadings, loading_rotation, full_height, method, colors, color_mapping)
    print('Creating components')
    create_average_factors(parameters, result_directory, merged_factor_infos, method)
    print('Reconstructing')
    create_reconstruction(parameters, result_directory, method, experimental, factor_infos, loading_infos, full_width, full_height)
    print('Creating crystallographic map (if template matching)')
    create_crystallographic_map(result_directory, merged_factor_infos, full_width, full_height, method)
    if method == 'umap':
        print('Creating UMAP embedding visualisation')
        create_umap_projection(parameters, result_directory, merged_factor_infos, colors, color_mapping, full_width, full_height)


def combine_loading_maps(parameters, result_directory, classification_method, scalebar_nm, rotation):
    shortname = parameters['shortname']
    methods = [
        method.strip() for method in parameters['methods'].split(',')
        if parameters['__save_method_{}'.format(method.strip())] == 'decomposition']
    factor_infos = result_image_file_info(result_directory, 'factors')
    loading_infos = result_image_file_info(result_directory, 'loadings')

    classify = {
        'l1_norm': classify_l1_norm_normal,
        'l1_norm_fourier': classify_l1_norm_fourier,
        'l2_norm': classify_l2_norm_normal,
        'l2_norm_fourier': classify_l2_norm_fourier,
        'l2_norm_log': classify_l2_norm_log,
        'l2_norm_compare': classify_compare_l2_norm,
        'template_match': classify_template_match,
    }[classification_method]

    for (method_name, factor_infos_for_method), loading_infos_for_method in zip(factor_infos.items(), loading_infos.values()):
        process_method(parameters, result_directory, method_name, classify,
                factor_infos_for_method, loading_infos_for_method, rotation)


if __name__ == '__main__':
    hs.preferences.General.nb_progressbar = False
    hs.preferences.General.show_progressbar = False
    result_directory = sys.argv[1]
    classification_method = sys.argv[2] if len(sys.argv) > 2 else 'l2_norm'
    parameters = parameters_parse(os.path.join(result_directory, 'metadata.txt'))
    parameters['classify_l2_norm_threshold'] = float(sys.argv[3]) if len(sys.argv) > 3 else 5
    scalebar_nm = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    rotation = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    combine_loading_maps(parameters, result_directory, classification_method, scalebar_nm, rotation)
