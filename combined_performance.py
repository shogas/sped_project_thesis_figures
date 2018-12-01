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


def combine_performance(result_directory):
    def parse_timestamp(t):
        return datetime.strptime(t, '%Y%m%d_%H_%M_%S_%f')

    def parse_timedelta_s(t):
        return timedelta(seconds=float(t))

    perf_dfs = []
    for run_dir in glob.iglob(os.path.join(result_directory, 'run_perf_correlate_*')):
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

        mem_peaks = np.empty((len(time_df),))
        for i, row in time_df.iterrows():
            mem_peaks[i] = mem_df.where(
                    mem_df.timestamp >= (row.timestamp - row.elapsed_delta)
                ).where(mem_df.timestamp <= row.timestamp).max().bytes

        time_df = time_df.assign(mem_peak=mem_peaks)
        perf_dfs.append(time_df)

    perf_df = pandas.concat(perf_dfs)
    perf_grouped = perf_df.groupby(['method', 'library_size'])
    mean = perf_grouped.mean()
    std = perf_grouped.std()

    library_sizes = perf_df.library_size.unique()
    mean_old = mean.xs('old')
    mean_new = mean.xs('new')
    std_old = std.xs('old')
    std_new = std.xs('new')
    elapsed_means = mean_old.elapsed / mean_new.elapsed
    elapsed_stds = propagate_std_div(mean_old.elapsed, std_old.elapsed, mean_new.elapsed, std_new.elapsed)
    memory_means = mean_new.mem_peak
    memory_stds = std_new.mem_peak

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
    save_figure(
            os.path.join(result_directory, 'performance_time.tex'),
            TikzAxis(
                TikzTablePlot(library_sizes, elapsed_means, elapsed_stds, **line_styles),
                TikzLegend('Relative time'),
                xlabel='{Library size}',
                ylabel='{Relative time}',
                **axis_styles))

    save_figure(
            os.path.join(result_directory, 'performance_memory.tex'),
            TikzAxis(
                TikzTablePlot(library_sizes, memory_means, memory_stds, **line_styles),
                TikzLegend('Relative memory'),
                xlabel='{Library size}',
                ylabel='{Relative memory}',
                **axis_styles))


if __name__ == '__main__':
    result_directory = sys.argv[1]
    combine_performance(result_directory)
