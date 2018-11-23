import os
import re
import shutil
import sys

import numpy as np
from PIL import Image

from parameters import parameters_parse

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


def tiff_to_png(src, dest):
    src_file = Image.open(src)
    src_file.save(dest)
    process_log.append(('tiff_to_png', src, dest))


def constant_to_tex(parameters_path, parameter_name, const_name, format):
    parameters = parameters_parse(parameters_path)
    constants.append((const_name, parameters[parameter_name], format))
    process_log.append(('constant_to_tex', '{}:{}'.format(parameters_path, parameter_name), const_name))


project_path    = 'C:/Users/simho/OneDrive/Skydok/MTNANO/Prosjektoppgave'
data_path       = os.path.join(project_path, 'Data')
gen_output_path = os.path.join(project_path, 'Tekst/fig/gen')
constants_path  = os.path.join(project_path, 'Tekst/constants.tex')
process_log = []
constants = []

if os.path.exists(gen_output_path):
    shutil.rmtree(gen_output_path)
os.makedirs(gen_output_path)

run_dir_three_phase_no_split = 'Runs/run_110_three_phase_no_split_20181121_16_38_54_565809'
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

constant_to_tex(
        os.path.join(data_path, run_dir_three_phase_no_split, 'metadata.txt'),
        '__elapsed_time_nmf', 'ThreePhaseNoSplitNMFTime', '.2f')
constant_to_tex(
        os.path.join(data_path, run_dir_three_phase_no_split, 'metadata.txt'),
        '__elapsed_time_umap', 'ThreePhaseNoSplitUMAPTime', '.2f')


copy_pgf(os.path.join(data_path, run_dir_three_phase_no_split, 'phase_map.pgf'),
        os.path.join(gen_output_path, 'three_phase_no_split_template_match_phase_map.pgf'))
copy_pgf(os.path.join(data_path, run_dir_three_phase_no_split, 'orientation_map.pgf'),
        os.path.join(gen_output_path, 'three_phase_no_split_template_match_orientation_map.pgf'))


with open(os.path.join(gen_output_path, 'process_log.txt'), 'w') as f:
    for action, src, dest in process_log:
        f.write('{}\t{}\t{}\n'.format(action, src, dest))
        print('{}\t{}\t{}\n'.format(action, src, dest))



with open(os.path.join(constants_path), 'w') as f:
    for const_name, const_value, format in constants:
        f.write(('\\def\\constData{}{{{:' + format + '}}}').format(const_name, const_value))
