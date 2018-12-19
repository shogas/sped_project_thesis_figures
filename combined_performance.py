from collections import defaultdict
from datetime import datetime
from datetime import timedelta
import csv
import glob
import math
import os
import sys

import numpy as np
import pandas

from figure import save_figure
from figure import TikzAxis
from figure import TikzLegend
from figure import TikzTablePlot
from figure import material_color_palette
from parameters import parameters_parse


def extract_info(infos, info_type):
    elapsed_and_memory = np.array([info[1:3] for info in infos if info[0] == info_type])
    std = np.std(elapsed_and_memory, axis=0)
    avg = np.average(elapsed_and_memory, axis=0)
    return elapsed_and_memory, std, avg


def propagate_std_div(a_avg, a_std, b_avg, b_std):
    # NOTE(simonhog): From https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Example_formulas
    # outside -> elapsed_covariance = np.cov(new[:, 0], old[:, 0])[0, 1] if len(new) > 1 else 0
    # res = abs(a_avg / b_avg) * math.sqrt(
        # (a_std / a_avg)**2 +
        # (b_std / b_avg)**2 -
        # 2*covariance/(a_avg*b_avg))
    # NOTE(simonhog): From https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3387884/
    res = np.abs(a_avg / b_avg) * np.sqrt(
        (a_std / a_avg)**2 +
        (b_std / b_avg)**2)
    return res


def load_data(dir_glob, key_columns):
    def parse_timestamp(t):
        return datetime.strptime(t, '%Y%m%d_%H_%M_%S_%f')


    def parse_timedelta_s(t):
        return timedelta(seconds=float(t))

    perf_dfs = []
    for run_dir in glob.iglob(dir_glob):
        print('Loading ', run_dir)
        time_filename = os.path.join(run_dir, 'time.txt')
        mem_filename = os.path.join(run_dir, 'mem.txt')
        time_df = pandas.read_csv(
                time_filename,
                sep='\t',
                names=['timestamp', *key_columns, 'elapsed'],
                converters={'timestamp': parse_timestamp})
        mem_df = pandas.read_csv(
                mem_filename,
                sep='\t',
                names=['timestamp', 'bytes'],
                converters={'timestamp': parse_timestamp})

        time_df = time_df.assign(elapsed_delta=time_df.elapsed.apply(parse_timedelta_s))
        print(' ', np.unique(time_df[time_df.columns.values[1]]))
        print(' ', np.unique(time_df.split_size))

        mem_peaks = np.empty(len(time_df))
        for i, row in time_df.iterrows():
            mem_peaks[i] = mem_df.where(
                    mem_df.timestamp >= (row.timestamp - row.elapsed_delta)
                ).where(mem_df.timestamp <= row.timestamp).max().bytes
            if mem_peaks[i] < 0 and i > 0:
                mem_peaks[i] = mem_peaks[i - 1]

        time_df = time_df.assign(mem_peak=mem_peaks)
        perf_dfs.append(time_df)

    return pandas.concat(perf_dfs)


def source_aggregate_correlate(mean, std):
    mean_old = mean.xs('old')
    mean_new = mean.xs('new')
    std_old = std.xs('old')
    std_new = std.xs('new')

    elapsed_means = mean_old.elapsed / mean_new.elapsed
    elapsed_stds = propagate_std_div(mean_old.elapsed, std_old.elapsed, mean_new.elapsed, std_new.elapsed)
    memory_means = mean_new.mem_peak
    memory_stds = std_new.mem_peak
    return np.array([elapsed_means]),\
            np.array([elapsed_stds]),\
            np.array([memory_means]),\
            np.array([memory_stds]),\
            ['']


def source_aggregate_split(mean, std):
    elapsed_means = []
    elapsed_stds = []
    memory_means = []
    memory_stds = []
    legends = []
    print(mean.index)
    for component_count in mean.index.levels[0]:
        print(component_count)
        elapsed_means.append(mean.elapsed[component_count])
        elapsed_stds.append(std.elapsed[component_count])
        memory_means.append(mean.mem_peak[component_count])
        memory_stds.append(std.mem_peak[component_count])
        legends.append('{}'.format(component_count))
    return np.array(elapsed_means),\
            np.array(elapsed_stds),\
            np.array(memory_means),\
            np.array(memory_stds),\
            legends


def combine_performance(source, result_directory):
    source_info = {
        'correlate': (
            'run_perf_correlate_*',
            ['method', 'library_size'],
            source_aggregate_correlate,
            '{Library size}',
            '{Speed-up}',
            'Speed-up'),
        'split_nmf': (
            'run_perf_split_*',
            ['component_count', 'split_size'],
            source_aggregate_split,
            '{Split size}',
            r'{Time/\si{\s}}',
            ' components'),
        'split_umap': (
            'run_perf_split_umap_*',
            ['n_neighbours', 'split_size'],
            source_aggregate_split,
            '{Split size}',
            r'{Time/\si{\s}}',
            ' neighbours'),
    }
    subdir_glob, key_columns, source_aggregate, x_label, y_label, legend_postfix = source_info[source]

    perf_df = load_data(os.path.join(result_directory, subdir_glob), key_columns)

    perf_grouped = perf_df.groupby(key_columns)
    mean = perf_grouped.mean()
    std = perf_grouped.std()

    elapsed_means, elapsed_stds, memory_means, memory_stds, legends = source_aggregate(mean, std)
    x_axis_labels = mean.index.levels[1]
    for i in range(len(legends)):
        legends[i] += legend_postfix

    axis_styles = {
        'legend_pos': 'north west',
        'axis_x_line': 'bottom',
        'axis_y_line': 'left',
        'xmin': 0,
        'ymin': 0,
        'width': r'\textwidth',
        'enlargelimits': 'upper',
        'grid': 'both'
    }
    line_styles = {
        'mark': '*',
        'line_width': '1.5pt'
    }
    colors = material_color_palette

    memory_means /= 1000000000
    memory_stds /=  1000000000
    elapsed_elements = []
    mem_elements = []
    for i, (elapsed_mean, elapsed_std, memory_mean, memory_std, legend) in enumerate(zip(
        elapsed_means, elapsed_stds, memory_means, memory_stds, legends)):
        color = colors[i % len(colors)]
        styles = {
            **line_styles,
            'color': color[0],
            'mark_options': '{{fill={}, scale=0.75}}'.format(color[0])}
        elapsed_elements.append(TikzTablePlot(
            x_axis_labels, elapsed_mean, elapsed_std, **styles))
        elapsed_elements.append(TikzLegend(legend))
        mem_elements.append(TikzTablePlot(
            x_axis_labels, memory_mean, memory_std, **styles))
        mem_elements.append(TikzLegend(legend))

    save_figure(
            os.path.join(result_directory, 'performance_time.tex'),
            TikzAxis(
                *elapsed_elements,
                xlabel=x_label,
                ylabel=y_label,
                **axis_styles))

    save_figure(
            os.path.join(result_directory, 'performance_memory.tex'),
            TikzAxis(
                *mem_elements,
                xlabel=x_label,
                ylabel='{Memory/GB}',
                **axis_styles,
                ymax=4))


if __name__ == '__main__':
    performance_source = sys.argv[1]
    result_directory = sys.argv[2]
    combine_performance(performance_source, result_directory)
