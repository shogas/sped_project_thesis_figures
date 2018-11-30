from collections import defaultdict
from datetime import datetime
from datetime import timedelta
import csv
import glob
import math
import os
import sys

import numpy as np

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


def propagate_std(a_avg, a_std, b_avg, b_std):
    # NOTE(simonhog): From https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Example_formulas
    # outside -> elapsed_covariance = np.cov(new[:, 0], old[:, 0])[0, 1] if len(new) > 1 else 0
    # res = abs(a_avg / b_avg) * math.sqrt(
        # (a_std / a_avg)**2 +
        # (b_std / b_avg)**2 -
        # 2*covariance/(a_avg*b_avg))
    # NOTE(simonhog): From https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3387884/
    res = abs(a_avg / b_avg) * math.sqrt(
        (a_std / a_avg)**2 +
        (b_std / b_avg)**2)
    return res


def combine_performance(result_directory):
    results = defaultdict(list)
    for run_dir in glob.iglob(os.path.join(result_directory, 'run_perf_correlate_*')):
        time_filename = os.path.join(run_dir, 'time.txt')
        mem_filename = os.path.join(run_dir, 'mem.txt')

        memory_entries = []
        with open(mem_filename) as mem_file:
            mem_reader = csv.reader(mem_file, delimiter='\t')
            for timestamp, bytes in mem_reader:
                memory_entries.append((datetime.strptime(timestamp, '%Y%m%d_%H_%M_%S_%f'), int(bytes)))

        with open(time_filename) as time_file:
            time_reader = csv.reader(time_file, delimiter='\t')
            for timestamp, name, library_size, elapsed in time_reader:
                library_size = int(library_size)
                elapsed = float(elapsed)
                end_time = datetime.strptime(timestamp, '%Y%m%d_%H_%M_%S_%f')
                start_time = end_time - timedelta(seconds=elapsed)
                peak_memory = max((bytes for time, bytes in memory_entries if time > start_time and time < end_time))
                results[library_size].append((name, elapsed, peak_memory))


    library_sizes = []
    elapsed_avgs = []
    elapsed_stds = []
    memory_avgs = []
    memory_stds = []
    for i, (library_size, infos) in enumerate(results.items()):
        new, new_std, new_avg = extract_info(infos, 'new')
        old, old_std, old_avg = extract_info(infos, 'old')

        print('n', new)
        print('o', old)
        library_sizes.append(library_size)
        elapsed_avgs.append(old_avg[0] / new_avg[0])
        elapsed_stds.append(propagate_std(old_avg[0], old_std[0], new_avg[0], new_std[0]))

        memory_avgs.append(old_avg[1] / new_avg[1])
        memory_stds.append(propagate_std(old_avg[1], old_std[1], new_avg[1], new_std[1]))

    save_figure(
            os.path.join(result_directory, 'performance_time.tex'),
            TikzAxis(
                TikzTablePlot(library_sizes, elapsed_avgs, elapsed_stds), # black, mark options={black, scale=0.75}, smooth, 
                TikzLegend('Relative time'),
                xlabel='{Library size}',
                ylabel='{Relative time}',
                legend_pos='north west'))

    save_figure(
            os.path.join(result_directory, 'performance_memory.tex'),
            TikzAxis(
                TikzTablePlot(library_sizes, memory_avgs, memory_stds), # black, mark options={black, scale=0.75}, smooth, 
                TikzLegend('Relative memory'),
                xlabel='{Library size}',
                ylabel='{Relative memory}',
                legend_pos='north west'))


if __name__ == '__main__':
    result_directory = sys.argv[1]
    combine_performance(result_directory)
