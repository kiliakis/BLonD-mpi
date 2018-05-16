#!/usr/bin/python
import matplotlib.pyplot as plt
import numpy as np
import os

from plot.plotting_utilities import *

project_dir = './'
res_dir = project_dir + 'results/'
images_dir = res_dir + 'plots/'

if not os.path.exists(images_dir):
    os.makedirs(images_dir)

# csv_file = res_dir + 'csv/interp-kick1/all_results2.csv'

plots_config = {
    'plot1': {
        'files': {
            res_dir+'raw/strong_scale_mpi_single_node/comm-comp-report.csv': {
                'lines': {'parts': ['10000000'],
                          'type': ['comm', 'comp'],
                          'N': ['1']},
                'labels': {'10000000-total-1': '10M-strong-scale-1node'}
            },

            res_dir+'raw/strong_scale_mpi_dual_node/comm-comp-report.csv': {
                    'lines': {'parts': ['20000000'],
                              'type': ['comm', 'comp'],
                              'N': ['2']},
                    'labels': {'20000000-total-2': '20M-strong-scale-2nodes'}

            },
            res_dir+'raw/weak_scale_mpi_single_node/comm-comp-report.csv': {
                    'lines': {'type': ['comm', 'comp'],
                              'N': ['1']},
                    'labels': {'total-1': '1M-weak-scale-1node'}

            },
            res_dir+'raw/weak_scale_mpi_dual_node/comm-comp-report.csv': {
                    'lines': {'type': ['comm', 'comp'],
                              'N': ['2']},
                    'labels': {'total-2': '1M-weak-scale-2nodes'}

            }
        },
        'labels': {'10000000-comm-1': '10M-strong-N1',
                   '10000000-comp-1': '10M-strong-N1',
                   '20000000-comm-2': '20M-strong-N2',
                   '20000000-comp-2': '20M-strong-N2',
                   'comm-1': '1M-weak-N1',
                   'comp-1': '1M-weak-N1',
                   'comm-2': '1M-weak-N2',
                   'comp-2': '1M-weak-N2'
                   },
        'colors': {'10000000-comm-1': 'blue',
                   '10000000-comp-1': 'blue',
                   '20000000-comm-2': 'red',
                   '20000000-comp-2': 'red',
                   'comm-1': 'orange',
                   'comp-1': 'orange',
                   'comm-2': 'green',
                   'comp-2': 'green'
                   },
        # 'exclude': [['v1', 'notcm'], ['v2', 'notcm'], ['v4', 'notcm']],
        'x_name': 'n',
        'y_name': 'avg_percent',
        'y_err_name': 'std',
        'xlabel': 'MPI Tasks',
        'ylabel': 'Run-time percent',
        'title': '',
        'ylim': [0, 100],
        'image_name': images_dir + 'time-breakdown.pdf'

    }
    # 'plot2': {'lines': {'version': ['v4'],
    #                     'vec': ['vec'],
    #                     'tcm': ['tcm'],
    #                     'cc': ['g++', 'icc']},
    #           'exclude': [],

    #           'x_name': 'threads',
    #           'y_name': 'time(ms)',
    #           'y_err_name': 'std(%)',
    #           'xlabel': 'Threads (500k points/thread)',
    #           'ylabel': 'Run-time (ms)',
    #           'title': 'icc VS gcc',
    #           # 'ylim': [0, 16000],
    #           'image_name': images_dir + 'iccVSgcc.pdf'
    #           },

    # 'plot3': {'lines': {'version': ['v5', 'v6'],
    #                     'vec': ['vec'],
    #                     'tcm': ['tcm'],
    #                     'cc': ['g++']},
    #           'exclude': [],

    #           'x_name': 'threads',
    #           'y_name': 'time(ms)',
    #           'y_err_name': 'std(%)',
    #           'xlabel': 'Threads (500k points/thread)',
    #           'ylabel': 'Run-time (ms)',
    #           'title': 'float VS double precision',
    #           'image_name': images_dir + 'float_vs_double.pdf'
    #           },

    # 'plot4': {'lines': {'version': ['v4'],
    #                     'vec': ['vec', 'novec'],
    #                     'tcm': ['tcm', 'notcm'],
    #                     'cc': ['g++']},
    #           'exclude': [],

    #           'x_name': 'threads',
    #           'y_name': 'time(ms)',
    #           'y_err_name': 'std(%)',
    #           'xlabel': 'Threads (500k points/thread)',
    #           'ylabel': 'Run-time (ms)',
    #           'title': 'tcm and vec effects',
    #           'image_name': images_dir + 'tcm_and_vec_effects.pdf'
    #           },
    # 'plot5': {'lines': {'version': ['v7', 'v8', 'v9', 'v10',
    #                                 'v7-p100', 'v8-p100', 'v9-p100', 'v10-p100'],
    #                     'cc': ['nvcc']},
    #           'exclude': [],
    #           'x_name': 'points',
    #           'y_name': 'time(ms)',
    #           'y_err_name': 'std(%)',
    #           'xlabel': 'Points',
    #           'ylabel': 'Run-time (ms)',
    #           'title': 'All GPU versions',
    #           'extra': ['plt.xscale(\'log\', basex=2)'],
    #           'image_name': images_dir + 'all_gpu_versions.pdf'
    #           },
    # 'plot6': {'lines': {'version': ['v9', 'v4', 'v9-p100'],
    #                     'cc': ['nvcc', 'g++'],
    #                     'tcm': ['tcm', 'na'],
    #                     'vec': ['vec', 'na']},
    #           'exclude': [],
    #           'x_name': 'points',
    #           'y_name': 'time(ms)',
    #           'y_err_name': 'std(%)',
    #           'xlabel': 'Points',
    #           'ylabel': 'Run-time (ms)',
    #           'title': 'GPU vs CPU',
    #           'extra': ['plt.xscale(\'log\', basex=2)'],
    #           'image_name': images_dir + 'gpu_vs_cpu.pdf'
    #           }

}

if __name__ == '__main__':
    for plot_key, config in plots_config.items():
        plots_dir = {}
        for file in config['files'].keys():
            print(file)
            data = np.genfromtxt(file, delimiter='\t', dtype=str)
            header = list(data[0])
            data = data[1:]
            plots_dir.update(get_plots(header, data, config['files'][file]['lines'],
                                       exclude=config['files'][file].get('exclude', [])))
        print(plots_dir)
        fig = plt.figure(figsize=(6, 3.5))
        plt.grid(True, which='major', alpha=0.5)
        plt.title(config['title'])
        plt.xlabel(config['xlabel'])
        plt.ylabel(config['ylabel'])
        # plt.yscale('log', basex=2)
        if 'ylim' in config:
            plt.ylim(config['ylim'])

        for label, values in plots_dir.items():
            # print(values)
            x = np.array(values[:, header.index(config['x_name'])], float)
            y = np.array(values[:, header.index(config['y_name'])], float)
            # parts = np.array(values[:, header.index('parts')], float)
            # turns = np.array(values[:, header.index('turns')], float)
            # y = parts * turns / y
            y_err = np.array(
                values[:, header.index(config['y_err_name'])], float)
            # y_err = y_err * y / 100.
            print(label, x, y)
            # label = config['labels'][label]
            if config['labels'][label] in plt.gca().get_legend_handles_labels()[1]:
                plt.errorbar(x, y, yerr=y_err,
                         capsize=1, marker='', linewidth=1.5, elinewidth=1,
                         color=config['colors'][label])
            else:
                plt.errorbar(x, y, yerr=y_err, label=config['labels'][label],
                         capsize=1, marker='', linewidth=1.5,  elinewidth=1,
                         color=config['colors'][label])
        if 'extra' in config:
            for c in config['extra']:
                exec(c)
        # if plot_key == 'plot6':
        #     plt.gca().get_lines()
        #     for p in plt.gca().get_lines()[::3]:
        #         annotate(plt.gca(), p.get_xdata(),
        #                  p.get_ydata(), fontsize='8')
        # from collections import OrderedDict
        # handles, labels = plt.gca().get_legend_handles_labels()
        # by_label = OrderedDict(zip(labels, handles))

        plt.legend(loc='best', fancybox=True, fontsize=9,
                   labelspacing=0.2, borderpad=0.5, framealpha=0.5,
                   handletextpad=0.5, handlelength=2, borderaxespad=0)
        plt.tight_layout()
        save_and_crop(fig, config['image_name'], dpi=600, bbox_inches='tight')
        # plt.savefig(config['image_name'], dpi=600, bbox_inches='tight')
        # subprocess.call
        plt.show()
        plt.close()

    # plt.legend(loc='best', fancybox=True, fontsize='11')
    # plt.axvline(700.0, color='k', linestyle='--', linewidth=1.5)
    # plt.axvline(1350.0, color='k', linestyle='--', linewidth=1.5)
    # plt.annotate('Light\nCombine\nWorkload', xy=(
    #     200, 6.3), textcoords='data', size='16')
    # plt.annotate('Moderate\nCombine\nWorkload', xy=(
    #     800, 6.3), textcoords='data', size='16')
    # plt.annotate('Heavy\nCombine\nWorkload', xy=(
    #     1400, 8.2), textcoords='data', size='16')
