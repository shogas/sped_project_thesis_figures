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
        print(run_dir)
        time_filename = os.path.join(run_dir, 'time.txt')
        mem_filename = os.path.join(run_dir, 'mem.txt')
        time_df = pandas.read_csv(
                time_filename,
                sep='\t',
                names=['timestamp', 'method', 'library_size', 'elapsed'],
                converters={'timestamp': parse_timestamp})
        mem_df = pandas.read_csv(
                mem_filename,
                sep='\t',
                names=['timestamp', 'bytes'],
                converters={'timestamp': parse_timestamp})

        time_df = time_df.assign(elapsed_delta=time_df.elapsed.apply(parse_timedelta_s))

        mem_peaks = np.empty(len(time_df))
        for i, row in time_df.iterrows():
            mem_peaks[i] = mem_df.where(
                    mem_df.timestamp >= (row.timestamp - row.elapsed_delta)
                ).where(mem_df.timestamp <= row.timestamp).max().bytes

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
    return [elapsed_means], [elapsed_stds], memory_means, memory_stds, ['Relative time']


def source_aggregate_split_nmf(mean, std):
    print(mean.index)
    exit(0)
    return mean.elapsed, std.elapsed, mean.mem_peak, std.mem_peak, ['{} components'.format(a) for a in range(1)]


def combine_performance(source, result_directory):
    source_info = {
        'correlate': (
            'run_perf_correlate_*',
            ['method', 'library_size'],
            source_aggregate_correlate,
            '{Library size}',
            '{Relative time}'),
        'split_nmf': (
            'run_perf_split_*',
            ['component_count', 'split_size'],
            source_aggregate_split_nmf,
            '{Split size}',
            r'{Time/\si{\s}}'),
    }
    subdir_glob, key_columns, source_aggregate, x_label, y_label = source_info[source]

    perf_df = load_data(os.path.join(result_directory, subdir_glob), key_columns)

    perf_grouped = perf_df.groupby(key_columns)
    mean = perf_grouped.mean()
    std = perf_grouped.std()

    x_axis_labels = perf_df[key_columns[1]].unique()
    elapsed_means, elapsed_stds, memory_means, memory_stds, legends = source_aggregate(mean, std)

    axis_styles = {
        'legend_pos': 'north west',
        'axis_x_line': 'bottom',
        'axis_y_line': 'left',
        'xmin': 0,
        'enlargelimits': 'upper',
        'grid': 'both'
    }
    line_styles = {
        'color': 'MaterialBlue',
        'mark': '*',
        'mark_options': '{fill=MaterialBlue, scale=0.75}',
        'line_width': '1.5pt'
    }

    memory_stds /= 1000000
    perf_elements = []
    for y_mean, y_std, legend in zip(elapsed_means, elapsed_stds, legends):
        # print(y_mean, y_std, legend)
        perf_elements.append(TikzTablePlot(x_axis_labels, y_mean, y_std, **line_styles))
        perf_elements.append(TikzLegend(legend))

    save_figure(
            os.path.join(result_directory, 'performance_time.tex'),
            TikzAxis(
                *perf_elements,
                xlabel=x_label,
                ylabel=y_label,
                **axis_styles))

    save_figure(
            os.path.join(result_directory, 'performance_memory.tex'),
            TikzAxis(
                TikzTablePlot(x_axis_labels, memory_means, memory_stds, **line_styles),
                TikzLegend('Memory'),
                xlabel=x_label,
                ylabel='{Memory/MB}',
                **axis_styles))


if __name__ == '__main__':
    performance_source = sys.argv[1]
    result_directory = sys.argv[2]
    combine_performance(performance_source, result_directory)
