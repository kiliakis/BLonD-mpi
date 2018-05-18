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
            # res_dir+'raw/strong_scale_mpi_single_node/comm-comp-report.csv': {
            #     'lines': {'parts': ['10000000'],
            #               'type': ['total'],
            #               'N': ['1']}
            # },
            res_dir+'raw/strong_scale_mpi_dual_node/comm-comp-report.csv': {
                'lines': {'parts': ['20000000'],
                          'type': ['total'],
                          'N': ['2']}
            },
            res_dir+'raw/strong_scale_mpi_four_node/comm-comp-report.csv': {
                'lines': {'parts': ['20000000'],
                          'type': ['total'],
                          'N': ['4']}
            },
            # res_dir+'raw/weak_scale_mpi_single_node/comm-comp-report.csv': {
            #     'lines': {'type': ['total'],
            #               'N': ['1']}
            # },
            res_dir+'raw/weak_scale_mpi_dual_node/comm-comp-report.csv': {
                'lines': {'type': ['total'],
                          'N': ['2']}
            },
            res_dir+'raw/weak_scale_mpi_four_node/comm-comp-report.csv': {
                'lines': {'type': ['total'],
                          'N': ['4']}
            }
        },
        'labels': {'10000000-total-1': '10M-strong-N1',
                   '20000000-total-2': '20M-strong-N2',
                   '20000000-total-4': '20M-strong-N4',
                   'total-1': '1M-weak-N1',
                   'total-4': '1M-weak-N4',
                   'total-2': '1M-weak-N2'},
        # 'exclude': [['v1', 'notcm'], ['v2', 'notcm'], ['v4', 'notcm']],
        'ideal': 'total-4',
        'x_name': 'n',
        'y_name': 'avg_time(sec)',
        # 'y_err_name': 'std',
        'xlabel': 'MPI Tasks',
        'ylabel': 'Throughput (Particles/sec)',
        'title': '',
        # 'ylim': [0, 16000],
        'figsize': (6, 2.5),
        'image_name': images_dir + 'mpi-multi-node-throughput.pdf'

    },
    'plot2': {
        'files': {
            # res_dir+'raw/strong_scale_mpi_single_node/comm-comp-report.csv': {
            #     'lines': {'parts': ['10000000'],
            #               'type': ['total'],
            #               'N': ['1']}
            # },
            # res_dir+'raw/strong_scale_mpi_dual_node/comm-comp-report.csv': {
            #     'lines': {'parts': ['20000000'],
            #               'type': ['total'],
            #               'N': ['2']}
            # },
            res_dir+'raw/weak_scale_mpi_single_node/comm-comp-report.csv': {
                'lines': {'type': ['total'],
                          'N': ['1']}
            },
            res_dir+'raw/weak_scale_mpi_dual_node/comm-comp-report.csv': {
                'lines': {'type': ['total'],
                          'N': ['2']}
            },
            project_dir+'../BLonD-kiliakis/results/raw/weak_scale_omp_single_node/comm-comp-report.csv': {
                'lines': {'type': ['total'],
                          'N': ['1'],
                          'n': ['1']}
            },
            project_dir+'../BLonD-kiliakis/results/raw/weak_scale_omp_local/comm-comp-report.csv': {
                'lines': {'type': ['total'],
                          'N': ['1'],
                          'n': ['1'],
                          'turns': ['2000']}
            }
            # project_dir+'../BLonD-kiliakis/results/raw/strong_scale_omp_single_node/comm-comp-report.csv': {
            #     'lines': {'parts': ['10000000'],
            #               'type': ['total'],
            #               'N': ['1'],
            #               'n': ['1']}
            # }
        },
        'labels': {'10000000-total-1': '10M-strong-N1',
                   '10000000-total-1-1': '10M-strong-omp',
                   '20000000-total-2': '20M-strong-N2',
                   'total-1': '1M-weak-N1',
                   'total-1-1': '1M-weak-omp',
                   'total-1-1-2000': '1M-weak-omp-haswell',
                   'total-2': '1M-weak-N2'},
        # 'exclude': [['v1', 'notcm'], ['v2', 'notcm'], ['v4', 'notcm']],
        'ideal': 'total-2',
        'x_name': 'n',
        'y_name': 'avg_time(sec)',
        # 'y_err_name': 'std',
        'xlabel': 'MPI Tasks/OMP Threads',
        'ylabel': 'Throughput (Particles/sec)',
        'title': '',
        # 'ylim': [0, 16000],
        'figsize': (6, 2.5),
        'image_name': images_dir + 'mpi-vs-omp-throughput.pdf'

    }

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
            # print(file)
            data = np.genfromtxt(file, delimiter='\t', dtype=str)
            header = list(data[0])
            data = data[1:]
            plots_dir.update(get_plots(header, data, config['files'][file]['lines'],
                                       exclude=config['files'][file].get('exclude', [])))
        # print(plots_dir)
        fig = plt.figure(figsize=config['figsize'])
        plt.grid(True, which='major', alpha=0.6)
        plt.grid(True, which='minor', alpha=0.6, linestyle=':')
        plt.minorticks_on()
        plt.title(config['title'])
        plt.xlabel(config['xlabel'])
        plt.ylabel(config['ylabel'])
        # plt.yscale('log', basex=2)
        if 'ylim' in config:
            plt.ylim(config['ylim'])

        for key, values in plots_dir.items():
            # print(values)
            label = config['labels'][key]
            if 'omp' in label:
                x = np.array(values[:, header.index('omp')], float)
            else:
                x = np.array(values[:, header.index(config['x_name'])], float)

            y = np.array(values[:, header.index(config['y_name'])], float)
            parts = np.array(values[:, header.index('parts')], float)
            turns = np.array(values[:, header.index('turns')], float)
            y = parts * turns / y
            # y_err = np.array(
            #     values[:, header.index(config['y_err_name'])], float)
            # y_err = y_err * y / 100.
            # print(label, x, y)
            plt.errorbar(x, y, yerr=None, label=label,
                         capsize=2, marker='', linewidth=1.5)
        if 'extra' in config:
            for c in config['extra']:
                exec(c)

        # Ideal line
        ylims = plt.gca().get_ylim()

        x = np.array(plots_dir[config['ideal']]
                     [:, header.index(config['x_name'])], float)
        y = float(plots_dir[config['ideal']][0, header.index(config['y_name'])])
        parts = float(plots_dir[config['ideal']][0, header.index('parts')])
        turns = float(plots_dir[config['ideal']][0, header.index('turns')])
        y = x * (parts * turns) / y
        # print(y)
        plt.plot(x, y, color='black', linestyle='--')

        plt.ylim(ylims)
        # plt.yticks(np.linspace(ylims[0], ylims[1], 5))

        # if plot_key == 'plot6':
        #     plt.gca().get_lines()
        #     for p in plt.gca().get_lines()[::3]:
        #         annotate(plt.gca(), p.get_xdata(),
        #                  p.get_ydata(), fontsize='8')
        plt.legend(loc='best', fancybox=True, fontsize=9.5,
                   labelspacing=0, borderpad=0.5, framealpha=0.4,
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
