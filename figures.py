import os
import re
import shutil
import sys

import numpy as np
from PIL import Image

from parameters import parameters_parse


parameters = parameters_parse(sys.argv[1])

project_path    = parameters['project_path']
data_path       = parameters['data_path']
gen_output_path = parameters['gen_output_path']
gen_code_path   = parameters['gen_code_path']
constants_path  = parameters['constants_path']

run_dir_three_phase_no_split    = parameters['run_dir_three_phase_no_split']
run_dir_vdf                     = parameters['run_dir_vdf']
run_dir_vdf_line                = parameters['run_dir_vdf_line']
run_dir_full_110_nmf            = parameters['run_dir_full_110_nmf']
run_dir_full_110_nmf_many_phase = parameters['run_dir_full_110_nmf_many_phase']
run_dir_full_110_nmf_cepstrum   = parameters['run_dir_full_110_nmf_cepstrum']
run_dir_full_110_umap           = parameters['run_dir_full_110_umap']
run_dir_full_110_template       = parameters['run_dir_full_110_template']

run_dir_performance_correlate  = parameters['run_dir_performance_correlate']
run_dir_performance_split_nmf  = parameters['run_dir_performance_split_nmf']
run_dir_performance_split_umap = parameters['run_dir_performance_split_umap']
run_dir_scree                  = parameters['run_dir_scree']
run_dir_svg_crystal            = parameters['run_dir_svg_crystal']
run_dir_rotation_list          = parameters['run_dir_rotation_list']
run_dir_diffraction_pattern    = parameters['run_dir_diffraction_pattern']
run_dir_code                   = parameters['run_dir_code']

run_dir_full_112_c_nmf          = parameters['run_dir_full_112_c_nmf']
run_dir_full_112_d_nmf          = parameters['run_dir_full_112_d_nmf']
run_dir_full_112_e_nmf          = parameters['run_dir_full_112_e_nmf']
run_dir_full_112_c_nmf_cepstrum = parameters['run_dir_full_112_c_nmf_cepstrum']
run_dir_full_112_d_nmf_cepstrum = parameters['run_dir_full_112_d_nmf_cepstrum']
run_dir_full_112_e_nmf_cepstrum = parameters['run_dir_full_112_e_nmf_cepstrum']
run_dir_full_112_c_umap         = parameters['run_dir_full_112_c_umap']
run_dir_full_112_d_umap         = parameters['run_dir_full_112_d_umap']
run_dir_full_112_e_umap         = parameters['run_dir_full_112_e_umap']



process_log = []
constants = []


def copy(src, dest):
    shutil.copyfile(src, dest)
    process_log.append(('copy', src, dest))


def copy_pgf(src, dest):
    with open(src) as pgf_file:
        content = pgf_file.read()

    dir_src = os.path.dirname(src)
    dir_dest = os.path.dirname(dest)

    included_file_regex = re.compile(r"""(?P<command>\\pgfimage\[)(?P<pre>.*)(?P<interp>interpolate=true)(?P<post>.*)\]{(?P<filename>[^<>:;,?"*|\/}]+)}""", re.X)
    for match in re.finditer(included_file_regex, content):
        copy(
            os.path.join(dir_src, match.group('filename')),
            os.path.join(dir_dest, match.group('filename')))

    dest_content = re.sub(included_file_regex, r'\g<command>\g<pre>interpolate=false\g<post>]{fig/gen/\g<filename>}', content)
    with open(dest, 'w') as dest_file:
        dest_file.write(dest_content)
    process_log.append(('pgf_rewrite', src, dest))


def copy_tikz(src, dest):
    with open(src) as tikz_file:
        content = tikz_file.read()

    dir_src = os.path.dirname(src)
    dir_dest = os.path.dirname(dest)
    included_file_regex = re.compile(r"""includegraphics\[(?P<param>[^\]]*)\]{(?P<filename>[^<>:;,?"*|\/}]+)}""", re.X)
    for match in re.finditer(included_file_regex, content):
        copy(
            os.path.join(dir_src, match.group('filename')),
            os.path.join(dir_dest, match.group('filename')))
    dest_content = re.sub(included_file_regex, r'includegraphics[\g<param>]{fig/gen/\g<filename>}', content)

    table_plot_regex = re.compile(r"""addplot table \[(?P<param>[^\]]*)\]\s*\{(?P<content>[^\}]*)\};""", re.X)
    table_contents = []
    for i, match in enumerate(re.finditer(table_plot_regex, dest_content)):
        if match.group('content').count('\n') > 200:  # Large table (pgfplots manual), move to file
            table_filename = '_table_data_{}.csv'.format(i)
            table_contents.append(table_filename)
            with open(dest.replace('.tex', table_filename)) as table_file:
                table_file.write(match.group('content'))
        else:
            table_contents.append(match.group('content'))

    match_index = 0
    def table_replace(match):
        nonlocal match_index
        match_index += 1
        return 'addplot table [{}] {{{}}};'.format(match.group('param'), table_contents[match_index - 1])
    dest_content = re.sub(table_plot_regex, table_replace, dest_content)

    with open(dest, 'w') as dest_file:
        dest_file.write(dest_content)
    process_log.append(('tikz_rewrite', src, dest))


def tiff_to_png(src, dest):
    src_file = Image.open(src)
    src_file.save(dest)
    process_log.append(('tiff_to_png', src, dest))


def tiff_to_png_intenstity_scale(src, dest):
    src_data = np.asarray(Image.open(src)).astype('float')
    src_data *= 255.0/(src_data.max() or 1)
    Image.fromarray(src_data.astype('uint8')).save(dest)
    process_log.append(('tiff_to_png_intensity_scale', src, dest))


def tiff_combine_rgb_to_png(src_r, src_g, src_b, dest):
    src_r_file = Image.open(src_r)
    src_g_file = Image.open(src_g)
    src_b_file = Image.open(src_b)
    Image.merge('RGB', (src_r_file, src_g_file, src_b_file)).save(dest)
    process_log.append(('tiff_combine_rgb_to_png', '|'.join((src_r, src_g, src_b)), dest))


def parameter_to_tex(parameters_path, parameter_name, const_name, format, scale=1):
    parameters = parameters_parse(parameters_path)
    value = parameters[parameter_name]
    if scale != 1:
        value *= scale
    constants.append((const_name, value, format))
    process_log.append(('parameter_to_tex', '{}:{}'.format(parameters_path, parameter_name), const_name))


def constant_to_tex(constant, const_name, format):
    constants.append((const_name, constant, format))
    process_log.append(('constant_to_tex', 'constant_value:{}'.format(constant), const_name))


if os.path.exists(gen_output_path):
    shutil.rmtree(gen_output_path)
os.makedirs(gen_output_path)



#
# Constants
#
# TODO: Get these from the actual dataset
constant_to_tex(1.28,     'OneOneZeroSpatialResolution', '.2f')
constant_to_tex(290,      'OneOneZeroPixelWidth', 'd')
constant_to_tex(410,      'OneOneZeroPixelHeight', 'd')
constant_to_tex(290*410,  'OneOneZeroPixelCount', 'd')
constant_to_tex(410*1.28, 'OneOneZeroRealHeight', '.1f')
constant_to_tex(1.28,     'OneOneZeroSimpleSpatialResolution', '.2f')
constant_to_tex(50,      'OneOneZeroSimplePixelWidth', 'd')
constant_to_tex(50,      'OneOneZeroSimplePixelHeight', 'd')
constant_to_tex(50*50,  'OneOneZeroSimplePixelCount', 'd')
constant_to_tex(50*1.28, 'OneOneZeroSimpleRealWidth', '.1f')
constant_to_tex(50*1.28, 'OneOneZeroSimpleRealHeight', '.1f')

constant_to_tex(1.96, 'OneOneTwoSpatialResolution', '.2f')
constant_to_tex(200, 'OneOneTwoPixelWidthC', 'd')
constant_to_tex(1810, 'OneOneTwoPixelHeightC', 'd')
constant_to_tex(180, 'OneOneTwoPixelWidthD', 'd')
constant_to_tex(1830, 'OneOneTwoPixelHeightD', 'd')
constant_to_tex(210, 'OneOneTwoPixelWidthE', 'd')
constant_to_tex(1830, 'OneOneTwoPixelHeightE', 'd')
constant_to_tex(200*1810 + 180*1830 + 210*1830, 'OneOneTwoPixelCount', 'd')


#
# Three phase no split NMF
#
copy_tikz(
        os.path.join(data_path, run_dir_three_phase_no_split, 'loading_map_110_three_phase_no_split_nmf.tex'),
        os.path.join(gen_output_path, 'three_phase_no_split_loading_map_nmf.tex'))
copy_tikz(
        os.path.join(data_path, run_dir_three_phase_no_split, 'factor_average_110_three_phase_no_split_nmf_0.tex'),
        os.path.join(gen_output_path, 'three_phase_no_split_factors_nmf_0.tex'))
copy_tikz(
        os.path.join(data_path, run_dir_three_phase_no_split, 'factor_average_110_three_phase_no_split_nmf_1.tex'),
        os.path.join(gen_output_path, 'three_phase_no_split_factors_nmf_1.tex'))
copy_tikz(
        os.path.join(data_path, run_dir_three_phase_no_split, 'factor_average_110_three_phase_no_split_nmf_2.tex'),
        os.path.join(gen_output_path, 'three_phase_no_split_factors_nmf_2.tex'))

copy_tikz(
        os.path.join(data_path, run_dir_three_phase_no_split, 'reconstruction_error_110_three_phase_no_split_nmf.tex'),
        os.path.join(gen_output_path, 'three_phase_no_split_reconstruction_error_nmf.tex'))
copy_tikz(
        os.path.join(data_path, run_dir_three_phase_no_split, 'reconstruction_error_110_three_phase_no_split_colorbar.tex'),
        os.path.join(gen_output_path, 'three_phase_no_split_reconstruction_error_colorbar.tex'))

copy_tikz(
        os.path.join(data_path, run_dir_three_phase_no_split, 'lineplot_110_three_phase_no_split_nmf.tex'),
        os.path.join(gen_output_path, 'three_phase_no_split_line_plot_nmf.tex'))
copy_tikz(
        os.path.join(data_path, run_dir_three_phase_no_split, 'lineplot_loading_map_110_three_phase_no_split_nmf.tex'),
        os.path.join(gen_output_path, 'three_phase_no_split_line_plot_loading_map_nmf.tex'))

parameter_to_tex(
        os.path.join(data_path, run_dir_three_phase_no_split, 'metadata.txt'),
        '__elapsed_time_nmf', 'ThreePhaseNoSplitNMFTime', '.1f', 1/3600)


#
# Three phase no split UMAP
#
copy_tikz(
        os.path.join(data_path, run_dir_three_phase_no_split, 'loading_map_110_three_phase_no_split_umap.tex'),
        os.path.join(gen_output_path, 'three_phase_no_split_loading_map_umap.tex'))
for i in range(5):
    copy_tikz(
            os.path.join(data_path, run_dir_three_phase_no_split, 'factor_average_110_three_phase_no_split_umap_{}.tex'.format(i)),
            os.path.join(gen_output_path, 'three_phase_no_split_factors_umap_{}.tex'.format(i)))
copy_tikz(
        os.path.join(data_path, run_dir_three_phase_no_split, 'reconstruction_error_110_three_phase_no_split_umap.tex'),
        os.path.join(gen_output_path, 'three_phase_no_split_reconstruction_error_umap.tex'))

copy_tikz(
        os.path.join(data_path, run_dir_three_phase_no_split, 'lineplot_110_three_phase_no_split_umap.tex'),
        os.path.join(gen_output_path, 'three_phase_no_split_line_plot_umap.tex'))
copy_tikz(
        os.path.join(data_path, run_dir_three_phase_no_split, 'lineplot_loading_map_110_three_phase_no_split_umap.tex'),
        os.path.join(gen_output_path, 'three_phase_no_split_line_plot_loading_map_umap.tex'))

parameter_to_tex(
        os.path.join(data_path, run_dir_three_phase_no_split, 'metadata.txt'),
        '__elapsed_time_umap', 'ThreePhaseNoSplitUMAPTime', '.1f', 1/3600)


#
# Three phase no split template matching
#
# copy_pgf(os.path.join(data_path, run_dir_three_phase_no_split, 'phase_map.pgf'),
        # os.path.join(gen_output_path, 'three_phase_no_split_template_match_phase_map.pgf'))
# copy_pgf(os.path.join(data_path, run_dir_three_phase_no_split, 'orientation_map.pgf'),
        # os.path.join(gen_output_path, 'three_phase_no_split_template_match_orientation_map.pgf'))


#
# Three phase no split VDF (line plot)
#
copy_tikz(
        os.path.join(data_path, run_dir_vdf_line, 'lineplot_vdf_vdf.tex'),
        os.path.join(gen_output_path, 'three_phase_no_split_line_plot_vdf.tex'))
copy_tikz(
        os.path.join(data_path, run_dir_vdf_line, 'lineplot_loading_map_vdf_vdf.tex'),
        os.path.join(gen_output_path, 'three_phase_no_split_line_plot_loading_map_vdf.tex'))



#
# 110 full VDF
#
copy_tikz(
        os.path.join(data_path, run_dir_vdf, 'vdf_0-290_0-410_loadings_110_zb_1_0.tex'),
        os.path.join(gen_output_path, 'full_110_vdf_zb_1.tex'))
copy_tikz(
        os.path.join(data_path, run_dir_vdf, 'vdf_0-290_0-410_loadings_110_zb_2_1.tex'),
        os.path.join(gen_output_path, 'full_110_vdf_zb_2.tex'))
copy_tikz(
        os.path.join(data_path, run_dir_vdf, 'vdf_0-290_0-410_loadings_110_wz_2.tex'),
        os.path.join(gen_output_path, 'full_110_vdf_wz.tex'))
copy_tikz(
        os.path.join(data_path, run_dir_vdf, 'vdf_0-290_0-410_factors_110_zb_1_0.tex'),
        os.path.join(gen_output_path, 'full_110_vdf_dp_zb_1.tex'))
copy_tikz(
        os.path.join(data_path, run_dir_vdf, 'vdf_0-290_0-410_factors_110_zb_2_1.tex'),
        os.path.join(gen_output_path, 'full_110_vdf_dp_zb_2.tex'))
copy_tikz(
        os.path.join(data_path, run_dir_vdf, 'vdf_0-290_0-410_factors_110_wz_2.tex'),
        os.path.join(gen_output_path, 'full_110_vdf_dp_wz.tex'))
tiff_combine_rgb_to_png(
        os.path.join(data_path, run_dir_vdf, 'vdf_0-290_0-410_loadings_110_zb_1_0.png'),
        os.path.join(data_path, run_dir_vdf, 'vdf_0-290_0-410_loadings_110_zb_2_1.png'),
        os.path.join(data_path, run_dir_vdf, 'vdf_0-290_0-410_loadings_110_wz_2.png'),
        os.path.join(gen_output_path, 'full_110_vdf.png'))



#
# 110 full NMF
#
copy_tikz(
        os.path.join(data_path, run_dir_full_110_nmf, 'loading_map_110_full_nmf_nmf.tex'),
        os.path.join(gen_output_path, 'full_110_nmf_loading_map.tex'))
for i in range(4):
    copy_tikz(
        os.path.join(data_path, run_dir_full_110_nmf, 'factor_average_110_full_nmf_nmf_{}.tex'.format(i)),
        os.path.join(gen_output_path, 'full_110_nmf_factor_{}.tex'.format(i)))
tiff_to_png_intenstity_scale(
        os.path.join(data_path, run_dir_full_110_nmf, 'nmf_145-290_205-410_loadings_0.tiff'),
        os.path.join(gen_output_path, 'full_110_nmf_loading_map_0.png'))
tiff_to_png_intenstity_scale(
        os.path.join(data_path, run_dir_full_110_nmf, 'nmf_0-145_0-205_loadings_3.tiff'),
        os.path.join(gen_output_path, 'full_110_nmf_loading_map_1.png'))

copy_tikz(
        os.path.join(data_path, run_dir_full_110_nmf, 'reconstruction_error_110_full_nmf_nmf.tex'),
        os.path.join(gen_output_path, 'full_110_reconstruction_error_nmf.tex'))
copy_tikz(
        os.path.join(data_path, run_dir_full_110_nmf, 'reconstruction_error_110_full_nmf_colorbar.tex'),
        os.path.join(gen_output_path, 'full_110_reconstruction_error_colorbar_nmf.tex'))

copy_tikz(
        os.path.join(data_path, run_dir_full_110_nmf, 'phase_map_110_full_nmf_nmf.tex'),
        os.path.join(gen_output_path, 'full_110_nmf_phase_map.tex'))
copy_tikz(
        os.path.join(data_path, run_dir_full_110_nmf, 'reliability_orientation_map_110_full_nmf_nmf.tex'),
        os.path.join(gen_output_path, 'full_110_nmf_reliability_orientation_map.tex'))
copy_tikz(
        os.path.join(data_path, run_dir_full_110_nmf, 'reliability_phase_map_110_full_nmf_nmf.tex'),
        os.path.join(gen_output_path, 'full_110_nmf_reliability_phase_map.tex'))
for orientation_phase in ['xvec_zb', 'yvec_zb', 'zvec_zb', 'xvec_wz', 'yvec_wz', 'zvec_wz']:
    copy(
        os.path.join(data_path, run_dir_full_110_nmf, 'orientation_map_{}.png'.format(orientation_phase)),
        os.path.join(gen_output_path, 'full_110_nmf_template_match_orientation_map_{}.png'.format(orientation_phase)))
    copy(
        os.path.join(data_path, run_dir_full_110_nmf, 'orientation_map_color_{}.png'.format(orientation_phase)),
        os.path.join(gen_output_path, 'full_110_nmf_template_match_orientation_map_color_{}.png'.format(orientation_phase)))

parameter_to_tex(
        os.path.join(data_path, run_dir_full_110_nmf, 'metadata.txt'),
        '__elapsed_time_nmf', 'OneOneZeroNMFTime', '.2f', 1/3600)

for tiff_name, png_name in [
    ('nmf_0-145_0-205_factors_{}.tiff'.format(i), 'full_110_nmf_many_phase_factor_{}.png'.format(i))
        for i in range(10)] + [
    ('nmf_0-145_0-205_loadings_{}.tiff'.format(i), 'full_110_nmf_many_phase_loading_map_{}.png'.format(i))
        for i in range(10)]:
    tiff_to_png(
            os.path.join(data_path, run_dir_full_110_nmf_many_phase, tiff_name),
            os.path.join(gen_output_path, png_name))

copy_tikz(
        os.path.join(data_path, run_dir_full_110_nmf_cepstrum, 'loading_map_110_full_nmf_cepstrum_nmf_cepstrum.tex'),
        os.path.join(gen_output_path, 'full_110_nmf_cepstrum_loading_map.tex'))
for i in range(0, 4):
    copy_tikz(
            os.path.join(data_path, run_dir_full_110_nmf_cepstrum, 'factor_average_110_full_nmf_cepstrum_nmf_cepstrum_{}.tex'.format(i)),
            os.path.join(gen_output_path, 'full_110_nmf_cepstrum_factor_{}.tex'.format(i)))
tiff_to_png_intenstity_scale(
        os.path.join(data_path, run_dir_full_110_nmf_cepstrum, 'nmf_cepstrum_145-290_205-410_loadings_1.tiff'),
        os.path.join(gen_output_path, 'full_110_nmf_cepstrum_loading_map_0.png'))
tiff_to_png_intenstity_scale(
        os.path.join(data_path, run_dir_full_110_nmf_cepstrum, 'nmf_cepstrum_0-145_0-205_loadings_2.tiff'),
        os.path.join(gen_output_path, 'full_110_nmf_cepstrum_loading_map_1.png'))



#
# 110 full UMAP
#
copy_tikz(
        os.path.join(data_path, run_dir_full_110_umap, 'loading_map_110_full_umap_a_umap.tex'),
        os.path.join(gen_output_path, 'full_110_umap_loading_map.tex'))
copy(
        os.path.join(data_path, run_dir_full_110_umap, 'embedding.png'),
        os.path.join(gen_output_path, 'full_110_umap_embedding.png'))
copy_tikz(
        os.path.join(data_path, run_dir_full_110_umap, 'phase_map_110_full_umap_a_umap.tex'),
        os.path.join(gen_output_path, 'full_110_umap_phase_map.tex'))
copy_tikz(
        os.path.join(data_path, run_dir_full_110_umap, 'reliability_orientation_map_110_full_umap_a_umap.tex'),
        os.path.join(gen_output_path, 'full_110_umap_reliability_orientation_map.tex'))
copy_tikz(
        os.path.join(data_path, run_dir_full_110_umap, 'reliability_phase_map_110_full_umap_a_umap.tex'),
        os.path.join(gen_output_path, 'full_110_umap_reliability_phase_map.tex'))
for i in range(12):
    copy_tikz(
        os.path.join(data_path, run_dir_full_110_umap, 'factor_average_110_full_umap_a_umap_{}.tex'.format(i)),
        os.path.join(gen_output_path, 'full_110_umap_factor_{}.tex'.format(i)))
for orientation_phase in ['xvec_zb', 'yvec_zb', 'zvec_zb', 'xvec_wz', 'yvec_wz', 'zvec_wz']:
    copy(
        os.path.join(data_path, run_dir_full_110_umap, 'orientation_map_{}.png'.format(orientation_phase)),
        os.path.join(gen_output_path, 'full_110_umap_template_match_orientation_map_{}.png'.format(orientation_phase)))
    copy(
        os.path.join(data_path, run_dir_full_110_umap, 'orientation_map_color_{}.png'.format(orientation_phase)),
        os.path.join(gen_output_path, 'full_110_umap_template_match_orientation_map_color_{}.png'.format(orientation_phase)))
copy_tikz(
        os.path.join(data_path, run_dir_full_110_umap, 'reconstruction_error_110_full_umap_a_umap.tex'),
        os.path.join(gen_output_path, 'full_110_reconstruction_error_umap.tex'))
copy_tikz(
        os.path.join(data_path, run_dir_full_110_umap, 'reconstruction_error_110_full_umap_a_colorbar.tex'),
        os.path.join(gen_output_path, 'full_110_reconstruction_error_colorbar_umap.tex'))
parameter_to_tex(
        os.path.join(data_path, run_dir_full_110_umap, 'metadata.txt'),
        '__elapsed_time_umap', 'OneOneZeroUMAPTime', '.1f', 1/3600)


#
# 110 full template
#
copy_tikz(
        os.path.join(data_path, run_dir_full_110_template, 'phase_map.tex'),
        os.path.join(gen_output_path, 'full_110_template_match_phase_map.tex'))
copy_tikz(
        os.path.join(data_path, run_dir_full_110_template, 'reliability_orientation_map.tex'),
        os.path.join(gen_output_path, 'full_110_template_match_reliability_orientation_map.tex'))
copy_tikz(
        os.path.join(data_path, run_dir_full_110_template, 'reliability_phase_map.tex'),
        os.path.join(gen_output_path, 'full_110_template_match_reliability_phase_map.tex'))
for orientation_phase in ['xvec_zb', 'yvec_zb', 'zvec_zb', 'xvec_wz', 'yvec_wz', 'zvec_wz']:
    copy(
        os.path.join(data_path, run_dir_full_110_template, 'orientation_map_{}.png'.format(orientation_phase)),
        os.path.join(gen_output_path, 'full_110_template_match_orientation_map_{}.png'.format(orientation_phase)))
    copy(
        os.path.join(data_path, run_dir_full_110_template, 'orientation_map_color_{}.png'.format(orientation_phase)),
        os.path.join(gen_output_path, 'full_110_template_match_orientation_map_color_{}.png'.format(orientation_phase)))

parameter_to_tex(
        os.path.join(data_path, run_dir_full_110_template, 'metadata.txt'),
        '__elapsed_time_template_match', 'OneOneZeroTemplateMatchTime', '.1f', 1/3600)



#
# 112 full NMF
#
copy_tikz(
        os.path.join(data_path, run_dir_full_112_c_nmf, 'loading_map_112_c_full_nmf_nmf.tex'),
        os.path.join(gen_output_path, 'full_112_c_nmf_loading_map.tex'))
copy_tikz(
        os.path.join(data_path, run_dir_full_112_d_nmf, 'loading_map_112_d_full_nmf_nmf.tex'),
        os.path.join(gen_output_path, 'full_112_d_nmf_loading_map.tex'))
copy_tikz(
        os.path.join(data_path, run_dir_full_112_e_nmf, 'loading_map_112_e_full_nmf_nmf.tex'),
        os.path.join(gen_output_path, 'full_112_e_nmf_loading_map.tex'))
for i in [0, 1, 3, 8, 16, 17]:
    copy_tikz(
        os.path.join(data_path, run_dir_full_112_c_nmf, 'factor_average_112_c_full_nmf_nmf_{}.tex'.format(i)),
        os.path.join(gen_output_path, 'full_112_c_nmf_factor_{}.tex'.format(i)))
for i in [0, 3, 9, 6, 16, 17]:
    copy_tikz(
        os.path.join(data_path, run_dir_full_112_d_nmf, 'factor_average_112_d_full_nmf_nmf_{}.tex'.format(i)),
        os.path.join(gen_output_path, 'full_112_d_nmf_factor_{}.tex'.format(i)))
for i in [1, 19, 3, 7, 8, 14]:
    copy_tikz(
        os.path.join(data_path, run_dir_full_112_e_nmf, 'factor_average_112_e_full_nmf_nmf_{}.tex'.format(i)),
        os.path.join(gen_output_path, 'full_112_e_nmf_factor_{}.tex'.format(i)))

copy_tikz(
        os.path.join(data_path, run_dir_full_112_c_nmf, 'reconstruction_error_112_c_full_nmf_nmf_rot.tex'),
        os.path.join(gen_output_path, 'full_112_c_reconstruction_error_nmf.tex'))
copy_tikz(
        os.path.join(data_path, run_dir_full_112_c_nmf, 'reconstruction_error_112_c_full_nmf_colorbar.tex'),
        os.path.join(gen_output_path, 'full_112_c_reconstruction_error_colorbar_nmf.tex'))
copy_tikz(
        os.path.join(data_path, run_dir_full_112_d_nmf, 'reconstruction_error_112_d_full_nmf_nmf_rot.tex'),
        os.path.join(gen_output_path, 'full_112_d_reconstruction_error_nmf.tex'))
copy_tikz(
        os.path.join(data_path, run_dir_full_112_d_nmf, 'reconstruction_error_112_d_full_nmf_colorbar.tex'),
        os.path.join(gen_output_path, 'full_112_d_reconstruction_error_colorbar_nmf.tex'))
copy_tikz(
        os.path.join(data_path, run_dir_full_112_e_nmf, 'reconstruction_error_112_e_full_nmf_nmf_rot.tex'),
        os.path.join(gen_output_path, 'full_112_e_reconstruction_error_nmf.tex'))
copy_tikz(
        os.path.join(data_path, run_dir_full_112_e_nmf, 'reconstruction_error_112_e_full_nmf_colorbar.tex'),
        os.path.join(gen_output_path, 'full_112_e_reconstruction_error_colorbar_nmf.tex'))

copy_tikz(
        os.path.join(data_path, run_dir_full_112_c_nmf_cepstrum, 'loading_map_112_c_full_nmf_cepstrum_nmf_cepstrum.tex'),
        os.path.join(gen_output_path, 'full_112_c_nmf_cepstrum_loading_map.tex'))
for i in range(0):
    copy_tikz(
        os.path.join(data_path, run_dir_full_112_c_nmf_cepstrum, 'factor_average_112_c_full_nmf_cepstrum_nmf_cepstrum_{}.tex'.format(i)),
        os.path.join(gen_output_path, 'full_112_c_nmf_cepstrum_factor_{}.tex'.format(i)))
copy_tikz(
        os.path.join(data_path, run_dir_full_112_d_nmf_cepstrum, 'loading_map_112_d_full_nmf_cepstrum_nmf_cepstrum.tex'),
        os.path.join(gen_output_path, 'full_112_d_nmf_cepstrum_loading_map.tex'))
for i in range(0):
    copy_tikz(
        os.path.join(data_path, run_dir_full_112_d_nmf_cepstrum, 'factor_average_112_d_full_nmf_cepstrum_nmf_cepstrum_{}.tex'.format(i)),
        os.path.join(gen_output_path, 'full_112_d_nmf_cepstrum_factor_{}.tex'.format(i)))
copy_tikz(
        os.path.join(data_path, run_dir_full_112_e_nmf_cepstrum, 'loading_map_112_e_full_nmf_cepstrum_nmf_cepstrum.tex'),
        os.path.join(gen_output_path, 'full_112_e_nmf_cepstrum_loading_map.tex'))
for i in range(0):
    copy_tikz(
        os.path.join(data_path, run_dir_full_112_e_nmf_cepstrum, 'factor_average_112_e_full_nmf_cepstrum_nmf_cepstrum_{}.tex'.format(i)),
        os.path.join(gen_output_path, 'full_112_e_nmf_cepstrum_factor_{}.tex'.format(i)))



#
# 112 full UMAP
#
copy_tikz(
        os.path.join(data_path, run_dir_full_112_c_umap, 'loading_map_112_c_full_umap_umap.tex'),
        os.path.join(gen_output_path, 'full_112_c_umap_loading_map.tex'))
copy_tikz(
        os.path.join(data_path, run_dir_full_112_d_umap, 'loading_map_112_d_full_umap_umap.tex'),
        os.path.join(gen_output_path, 'full_112_d_umap_loading_map.tex'))
copy_tikz(
        os.path.join(data_path, run_dir_full_112_e_umap, 'loading_map_112_e_full_umap_umap.tex'),
        os.path.join(gen_output_path, 'full_112_e_umap_loading_map.tex'))
for i in [3, 29, 65, 102, 144, 34]:
    copy_tikz(
        os.path.join(data_path, run_dir_full_112_c_umap, 'factor_average_112_c_full_umap_umap_{}.tex'.format(i)),
        os.path.join(gen_output_path, 'full_112_c_umap_factor_{}.tex'.format(i)))
for i in [3, 43, 68, 85, 23, 47]:
    copy_tikz(
        os.path.join(data_path, run_dir_full_112_d_umap, 'factor_average_112_d_full_umap_umap_{}.tex'.format(i)),
        os.path.join(gen_output_path, 'full_112_d_umap_factor_{}.tex'.format(i)))
for i in [1, 91, 12, 30, 54, 60]:
    copy_tikz(
        os.path.join(data_path, run_dir_full_112_e_umap, 'factor_average_112_e_full_umap_umap_{}.tex'.format(i)),
        os.path.join(gen_output_path, 'full_112_e_umap_factor_{}.tex'.format(i)))

copy_tikz(
        os.path.join(data_path, run_dir_full_112_c_umap, 'reconstruction_error_112_c_full_umap_umap_rot.tex'),
        os.path.join(gen_output_path, 'full_112_c_reconstruction_error_umap.tex'))
copy_tikz(
        os.path.join(data_path, run_dir_full_112_c_umap, 'reconstruction_error_112_c_full_umap_colorbar.tex'),
        os.path.join(gen_output_path, 'full_112_c_reconstruction_error_colorbar_umap.tex'))
copy_tikz(
        os.path.join(data_path, run_dir_full_112_d_umap, 'reconstruction_error_112_d_full_umap_umap_rot.tex'),
        os.path.join(gen_output_path, 'full_112_d_reconstruction_error_umap.tex'))
copy_tikz(
        os.path.join(data_path, run_dir_full_112_d_umap, 'reconstruction_error_112_d_full_umap_colorbar.tex'),
        os.path.join(gen_output_path, 'full_112_d_reconstruction_error_colorbar_umap.tex'))
copy_tikz(
        os.path.join(data_path, run_dir_full_112_e_umap, 'reconstruction_error_112_e_full_umap_umap_rot.tex'),
        os.path.join(gen_output_path, 'full_112_e_reconstruction_error_umap.tex'))
copy_tikz(
        os.path.join(data_path, run_dir_full_112_e_umap, 'reconstruction_error_112_e_full_umap_colorbar.tex'),
        os.path.join(gen_output_path, 'full_112_e_reconstruction_error_colorbar_umap.tex'))




#
# 112 full VDF
#
for i in range(6):
    copy_tikz(
            os.path.join(data_path, run_dir_vdf, 'vdf_0-200_0-1810_loadings_112_c_{}_{}.tex'.format(i + 1, i)),
            os.path.join(gen_output_path, 'full_112_c_vdf_{}.tex'.format(i+1)))
    copy_tikz(
            os.path.join(data_path, run_dir_vdf, 'vdf_0-200_0-1810_factors_112_c_{}_{}.tex'.format(i + 1, i)),
            os.path.join(gen_output_path, 'full_112_c_vdf_dp_{}.tex'.format(i+1)))


for i in range(8):
    copy_tikz(
            os.path.join(data_path, run_dir_vdf, 'vdf_0-180_0-1830_loadings_112_d_{}_{}.tex'.format(i + 1, i)),
            os.path.join(gen_output_path, 'full_112_d_vdf_{}.tex'.format(i+1)))
    copy_tikz(
            os.path.join(data_path, run_dir_vdf, 'vdf_0-180_0-1830_factors_112_d_{}_{}.tex'.format(i + 1, i)),
            os.path.join(gen_output_path, 'full_112_d_vdf_dp_{}.tex'.format(i+1)))

for i in range(8):
    copy_tikz(
            os.path.join(data_path, run_dir_vdf, 'vdf_0-210_0-1830_loadings_112_e_{}_{}.tex'.format(i + 1, i)),
            os.path.join(gen_output_path, 'full_112_e_vdf_{}.tex'.format(i+1)))
    copy_tikz(
            os.path.join(data_path, run_dir_vdf, 'vdf_0-210_0-1830_factors_112_e_{}_{}.tex'.format(i + 1, i)),
            os.path.join(gen_output_path, 'full_112_e_vdf_dp_{}.tex'.format(i+1)))


#
# Performance graphs
#
copy_tikz(
        os.path.join(data_path, run_dir_performance_correlate, 'performance_time.tex'),
        os.path.join(gen_output_path, 'correlate_performance_time.tex'))
copy_tikz(
        os.path.join(data_path, run_dir_performance_split_nmf, 'performance_time.tex'),
        os.path.join(gen_output_path, 'split_nmf_performance_time.tex'))
copy_tikz(
        os.path.join(data_path, run_dir_performance_split_umap, 'performance_time.tex'),
        os.path.join(gen_output_path, 'split_umap_performance_time.tex'))



#
# Scree plots
#
copy_tikz(
        os.path.join(data_path, run_dir_scree, 'explained_variance_ratio_full_110.tex'),
        os.path.join(gen_output_path, 'full_110_explained_variance_ratio.tex'))


#
# Crystal structures
#
for structure in ['zb', 'zb_110', 'zb_112', 'wz', 'wz_1120', 'wz_1010', 'sc', 'fcc', 'bcc', 'hcp']:
    copy(
        os.path.join(data_path, run_dir_svg_crystal, 'crystal_structure_{}.pdf'.format(structure)),
        os.path.join(gen_output_path, 'crystal_structure_{}.pdf'.format(structure)))


#
# Rotation lists
#
for system in ['cubic', 'hexagonal']:
    copy(
        os.path.join(data_path, run_dir_rotation_list, '{}.tex'.format(system)),
        os.path.join(gen_output_path, 'rotation_list_scatter_{}.tex'.format(system)))



#
# Diffraction patterns
#
for pattern in ['zb_110', 'zb_112', 'wz_1120', 'wz_1010']:
    copy_tikz(
        os.path.join(data_path, run_dir_diffraction_pattern, 'diffraction_pattern_{}.tex'.format(pattern)),
        os.path.join(gen_output_path, 'diffraction_pattern_sim_{}.tex'.format(pattern)))



#
# Code
#
for filename in [
        'pyxem_fork/pyxem/utils/sim_utils.py',
        'compare/factorize.py',
        'compare/methods/nmf.py',
        'compare/methods/umap.py',
        'compare/methods/template_match.py',
        'compare/utils/preprocess.py',
        'compare/runs/run_110_three_phases_no_split_factorization.txt',
        'compare/runs/run_110_full_base.txt',
        'compare/runs/run_112_c_full_base.txt',
        'compare/runs/run_110_full_nmf.txt',
        'compare/runs/run_110_full_nmf_cepstrum.txt',
        'compare/runs/run_112_c_full_nmf.txt',
        'compare/runs/run_112_c_full_nmf_cepstrum.txt',
        'compare/runs/run_110_full_umap_a.txt',
        'compare/runs/run_112_c_full_umap.txt',
        'compare/runs/run_110_full_template_match.txt',
        'figures/combined_loading_map.py',
        ]:
    copy(
        os.path.join(run_dir_code, filename),
        os.path.join(gen_code_path, os.path.basename(filename)))



#
# Actual moving
#
with open(os.path.join(gen_output_path, 'process_log.txt'), 'w') as f:
    for action, src, dest in process_log:
        src = src.replace(data_path, '')
        dest = dest.replace(data_path, '')
        f.write('{}\t{}\t{}\n'.format(action, src, dest))
        # print('{:10s}\t{}\t{}\n'.format(action, src, dest))



with open(os.path.join(constants_path), 'w') as f:
    for const_name, const_value, format in constants:
        f.write(('\\def\\constData{}{{{:' + format + '}}}\n').format(const_name, const_value))
