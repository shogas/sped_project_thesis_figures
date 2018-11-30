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
constants_path  = parameters['constants_path']

run_dir_three_phase_no_split = parameters['run_dir_three_phase_no_split']
run_dir_vdf                  = parameters['run_dir_vdf']
run_dir_full_110_nmf         = parameters['run_dir_full_110_nmf']
run_dir_full_110_umap        = parameters['run_dir_full_110_umap']
run_dir_full_110_template    = parameters['run_dir_full_110_template']

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
    with open(dest, 'w') as dest_file:
        dest_file.write(dest_content)
    process_log.append(('tikz_rewrite', src, dest))


def tiff_to_png(src, dest):
    src_file = Image.open(src)
    src_file.save(dest)
    process_log.append(('tiff_to_png', src, dest))


def tiff_combine_rgb_to_png(src_r, src_g, src_b, dest):
    src_r_file = Image.open(src_r)
    src_g_file = Image.open(src_g)
    src_b_file = Image.open(src_b)
    Image.merge('RGB', (src_r_file, src_g_file, src_b_file)).save(dest)
    process_log.append(('tiff_combine_rgb_to_png', '|'.join((src_r, src_g, src_b)), dest))


def parameter_to_tex(parameters_path, parameter_name, const_name, format):
    parameters = parameters_parse(parameters_path)
    constants.append((const_name, parameters[parameter_name], format))
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



#
# Three phase no split NMF
#
tiff_to_png(
        os.path.join(data_path, run_dir_three_phase_no_split, 'loading_map_nmf.tiff'),
        os.path.join(gen_output_path, 'three_phase_no_split_loading_map_nmf.png'))
tiff_to_png(
        os.path.join(data_path, run_dir_three_phase_no_split, 'nmf_0-50_0-50_factors_0.tiff'),
        os.path.join(gen_output_path, 'three_phase_no_split_factors_nmf_0.png'))
tiff_to_png(
        os.path.join(data_path, run_dir_three_phase_no_split, 'nmf_0-50_0-50_factors_1.tiff'),
        os.path.join(gen_output_path, 'three_phase_no_split_factors_nmf_1.png'))
tiff_to_png(
        os.path.join(data_path, run_dir_three_phase_no_split, 'nmf_0-50_0-50_factors_2.tiff'),
        os.path.join(gen_output_path, 'three_phase_no_split_factors_nmf_2.png'))

parameter_to_tex(
        os.path.join(data_path, run_dir_three_phase_no_split, 'metadata.txt'),
        '__elapsed_time_nmf', 'ThreePhaseNoSplitNMFTime', '.2f')


#
# Three phase no split UMAP
#
tiff_to_png(
        os.path.join(data_path, run_dir_three_phase_no_split, 'loading_map_umap.tiff'),
        os.path.join(gen_output_path, 'three_phase_no_split_loading_map_umap.png'))
tiff_to_png(
        os.path.join(data_path, run_dir_three_phase_no_split, 'umap_0-50_0-50_factors_0.tiff'),
        os.path.join(gen_output_path, 'three_phase_no_split_factors_umap_0.png'))
tiff_to_png(
        os.path.join(data_path, run_dir_three_phase_no_split, 'umap_0-50_0-50_factors_1.tiff'),
        os.path.join(gen_output_path, 'three_phase_no_split_factors_umap_2.png'))
tiff_to_png(
        os.path.join(data_path, run_dir_three_phase_no_split, 'umap_0-50_0-50_factors_3.tiff'),
        os.path.join(gen_output_path, 'three_phase_no_split_factors_umap_3.png'))
tiff_to_png(
        os.path.join(data_path, run_dir_three_phase_no_split, 'umap_0-50_0-50_factors_4.tiff'),
        os.path.join(gen_output_path, 'three_phase_no_split_factors_umap_4.png'))
tiff_to_png(
        os.path.join(data_path, run_dir_three_phase_no_split, 'umap_0-50_0-50_factors_5.tiff'),
        os.path.join(gen_output_path, 'three_phase_no_split_factors_umap_5.png'))
# tiff_to_png(
        # os.path.join(data_path, run_dir_three_phase_no_split, 'umap_0-50_0-50_factors_6.tiff'),
        # os.path.join(gen_output_path, 'three_phase_no_split_factors_umap_6.png'))

parameter_to_tex(
        os.path.join(data_path, run_dir_three_phase_no_split, 'metadata.txt'),
        '__elapsed_time_umap', 'ThreePhaseNoSplitUMAPTime', '.2f')


#
# Three phase no split template matching
#
# copy_pgf(os.path.join(data_path, run_dir_three_phase_no_split, 'phase_map.pgf'),
        # os.path.join(gen_output_path, 'three_phase_no_split_template_match_phase_map.pgf'))
# copy_pgf(os.path.join(data_path, run_dir_three_phase_no_split, 'orientation_map.pgf'),
        # os.path.join(gen_output_path, 'three_phase_no_split_template_match_orientation_map.pgf'))


#
# 110 full VDF
#
copy_tikz(
        os.path.join(data_path, run_dir_vdf, 'vdf_110_zb_1.tex'),
        os.path.join(gen_output_path, 'full_110_vdf_zb_1.tex'))
copy_tikz(
        os.path.join(data_path, run_dir_vdf, 'vdf_110_zb_2.tex'),
        os.path.join(gen_output_path, 'full_110_vdf_zb_2.tex'))
copy_tikz(
        os.path.join(data_path, run_dir_vdf, 'vdf_110_wz.tex'),
        os.path.join(gen_output_path, 'full_110_vdf_wz.tex'))
copy_tikz(
        os.path.join(data_path, run_dir_vdf, 'vdf_110_zb_1_dp.tex'),
        os.path.join(gen_output_path, 'full_110_vdf_dp_zb_1.tex'))
copy_tikz(
        os.path.join(data_path, run_dir_vdf, 'vdf_110_zb_2_dp.tex'),
        os.path.join(gen_output_path, 'full_110_vdf_dp_zb_2.tex'))
copy_tikz(
        os.path.join(data_path, run_dir_vdf, 'vdf_110_wz_dp.tex'),
        os.path.join(gen_output_path, 'full_110_vdf_dp_wz.tex'))
tiff_combine_rgb_to_png(
        os.path.join(data_path, run_dir_vdf, 'vdf_110_zb_1.png'),
        os.path.join(data_path, run_dir_vdf, 'vdf_110_zb_2.png'),
        os.path.join(data_path, run_dir_vdf, 'vdf_110_wz.png'),
        os.path.join(gen_output_path, 'full_110_vdf.png'))



#
# 110 full NMF
#
copy_tikz(
        os.path.join(data_path, run_dir_full_110_nmf, 'loading_map_nmf.tex'),
        os.path.join(gen_output_path, 'full_110_nmf_loading_map.tex'))
tiff_to_png(
        os.path.join(data_path, run_dir_full_110_nmf, 'nmf_0-145_0-205_factors_0.tiff'),
        os.path.join(gen_output_path, 'full_110_nmf_factor_zb_1.png'))
tiff_to_png(
        os.path.join(data_path, run_dir_full_110_nmf, 'nmf_0-145_0-205_factors_1.tiff'),
        os.path.join(gen_output_path, 'full_110_nmf_factor_zb_2.png'))
tiff_to_png(
        os.path.join(data_path, run_dir_full_110_nmf, 'nmf_0-145_0-205_factors_2.tiff'),
        os.path.join(gen_output_path, 'full_110_nmf_factor_wz.png'))
tiff_to_png(
        os.path.join(data_path, run_dir_full_110_nmf, 'nmf_0-145_0-205_factors_3.tiff'),
        os.path.join(gen_output_path, 'full_110_nmf_factor_vacuum.png'))



#
# 110 full UMAP
#
copy_tikz(
        os.path.join(data_path, run_dir_full_110_umap, 'loading_map_umap.tex'),
        os.path.join(gen_output_path, 'full_110_umap_loading_map.tex'))
copy_tikz(
        os.path.join(data_path, run_dir_full_110_umap, 'factor_average_umap_0.tex'),
        os.path.join(gen_output_path, 'full_110_umap_factor_average_zb_1.tex'))
copy_tikz(
        os.path.join(data_path, run_dir_full_110_umap, 'factor_average_umap_1.tex'),
        os.path.join(gen_output_path, 'full_110_umap_factor_average_zb_2.tex'))
copy_tikz(
        os.path.join(data_path, run_dir_full_110_umap, 'factor_average_umap_2.tex'),
        os.path.join(gen_output_path, 'full_110_umap_factor_average_wz.tex'))
copy_tikz(
        os.path.join(data_path, run_dir_full_110_umap, 'factor_average_umap_3.tex'),
        os.path.join(gen_output_path, 'full_110_umap_factor_average_vacuum.tex'))



#
# 110 full template
#
copy_tikz(
        os.path.join(data_path, run_dir_full_110_template, 'phase_map.tex'),
        os.path.join(gen_output_path, 'full_110_template_match_phase_map.tex'))
copy_tikz(
        os.path.join(data_path, run_dir_full_110_template, 'orientation_map_zb.tex'),
        os.path.join(gen_output_path, 'full_110_template_match_orientation_map_zb.tex'))
copy_tikz(
        os.path.join(data_path, run_dir_full_110_template, 'orientation_map_color_zb.tex'),
        os.path.join(gen_output_path, 'full_110_template_match_orientation_map_color_zb.tex'))
copy_tikz(
        os.path.join(data_path, run_dir_full_110_template, 'orientation_map_wz.tex'),
        os.path.join(gen_output_path, 'full_110_template_match_orientation_map_wz.tex'))
copy_tikz(
        os.path.join(data_path, run_dir_full_110_template, 'orientation_map_color_wz.tex'),
        os.path.join(gen_output_path, 'full_110_template_match_orientation_map_color_wz.tex'))



with open(os.path.join(gen_output_path, 'process_log.txt'), 'w') as f:
    for action, src, dest in process_log:
        src = src.replace(data_path, '')
        dest = dest.replace(data_path, '')
        f.write('{}\t{}\t{}\n'.format(action, src, dest))
        print('{:10s}\t{}\t{}\n'.format(action, src, dest))



with open(os.path.join(constants_path), 'w') as f:
    for const_name, const_value, format in constants:
        f.write(('\\def\\constData{}{{{:' + format + '}}}\n').format(const_name, const_value))
