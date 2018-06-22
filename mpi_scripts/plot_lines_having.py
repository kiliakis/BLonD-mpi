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


    # 'plot3': {
    #     'files': {
    #         # res_dir+'raw/strong_scale_mpi_single_node/comm-comp-report.csv': {
    #         #     'lines': {'parts': ['10000000'],
    #         #               'type': ['total'],
    #         #               'N': ['1']}
    #         # },
    #         # res_dir+'raw/strong_scale_mpi_dual_node/comm-comp-report.csv': {
    #         #     'lines': {'parts': ['20000000'],
    #         #               'type': ['total'],
    #         #               'N': ['2']}
    #         # },
    #         # res_dir+'raw/strong_scale_hybrid_four_node-2/comm-comp-report.csv': {
    #         #     'lines': {'parts': ['20000000'],
    #         #               'omp': ['2', '4', '5', '10', '20'],
    #         #               'type': ['total']}
    #         # },
    #         res_dir+'raw-hpcbatch/strong_scale_hybrid_four_node-4/comm-comp-report.csv': {
    #             'lines': {'parts': ['20000000'],
    #                       'omp': ['2', '4', '5', '10'],
    #                       'type': ['total']}
    #         }
    #         # ,
    #         # res_dir+'raw/strong_scale_mpi_four_node/comm-comp-report.csv': {
    #         #     'lines': {'parts': ['20000000'],
    #         #               'type': ['total'],
    #         #               'N': ['4']}
    #         # }

    #     },
    #     'labels': {'20000000-total-4': '20M-strong-N2',
    #                '20000000-2-total': '20M-hybrid-T2',
    #                '20000000-4-total': '20M-hybrid-T4',
    #                '20000000-5-total': '20M-hybrid-T5',
    #                '20000000-10-total': '20M-hybrid-T10',
    #                '20000000-20-total': '20M-hybrid-T20'
    #                },
    #     # 'exclude': [['v1', 'notcm'], ['v2', 'notcm'], ['v4', 'notcm']],
    #     'ideal': '20000000-2-total',
    #     'x_name': 'n',
    #     'omp_name': 'omp',
    #     'y_name': 'avg_time(sec)',
    #     # 'y_err_name': 'std',
    #     'xlabel': 'MPI Tasks/OMP Threads',
    #     'ylabel': 'Throughput (Particles/sec)',
    #     'title': '',
    #     # 'ylim': [0, 16000],
    #     'figsize': (6, 3),
    #     'image_name': images_dir + 'mpi-hybrid-throughtput-4.pdf'

    # }

    'plot4': {
        'files': {
            res_dir+'raw/LHC-hybrid-4nodes/comm-comp-report.csv': {
                'lines': {
                          'omp': ['1', '2', '4', '5', '10'],
                          'type': ['total']}
            }

        },
        'labels': {
                   '1-total': 'hybrid-T1',
                   '2-total': 'hybrid-T2',
                   '4-total': 'hybrid-T4',
                   '5-total': 'hybrid-T5',
                   '10-total': 'hybrid-T10',
                   '20-total': 'hybrid-T20'
                   },
        # 'exclude': [['v1', 'notcm'], ['v2', 'notcm'], ['v4', 'notcm']],
        'ideal': '2-total',
        'x_name': 'n',
        'omp_name': 'omp',
        'y_name': 'avg_time(sec)',
        # 'y_err_name': 'std',
        'xlabel': 'MPI Tasks/OMP Threads',
        'ylabel': 'Throughput (Particles/sec)',
        'title': '',
        # 'ylim': [0, 16000],
        'figsize': (6, 3),
        'image_name': images_dir + 'LHC-hybrid-throughtput.pdf'

    }


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
        # plt.minorticks_on()
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
                omp = np.array(
                    values[:, header.index(config['omp_name'])], float)
                x = (x-1) * omp

            y = np.array(values[:, header.index(config['y_name'])], float)
            parts = np.array(values[:, header.index('parts')], float)
            turns = np.array(values[:, header.index('turns')], float)
            y = parts * turns / y
            # y_err = np.array(
            #     values[:, header.index(config['y_err_name'])], float)
            # y_err = y_err * y / 100.
            # print(label, x, y)
            plt.errorbar(x, y, yerr=None, label=label,
                         capsize=2, marker='.', markersize=5, linewidth=1.5)
        if 'extra' in config:
            for c in config['extra']:
                exec(c)

        if config.get('ideal', ''):
            # Ideal line
            ylims = plt.gca().get_ylim()
            xlims = plt.gca().get_xlim()

            x0 = np.array(plots_dir[config['ideal']]
                          [:, header.index(config['x_name'])], float)[0]
            omp0 = np.array(plots_dir[config['ideal']]
                            [:, header.index(config['omp_name'])], float)[0]
            x0 = (x0-1) * omp0
            y0 = float(plots_dir[config['ideal']]
                       [0, header.index(config['y_name'])])
            print(x0)
            print(y0)

            parts0 = float(plots_dir[config['ideal']]
                           [0, header.index('parts')])
            turns0 = float(plots_dir[config['ideal']]
                           [0, header.index('turns')])
            print(parts0)
            print(turns0)
            x = np.arange(x0, xlims[1], 1)
            y = x * (parts0 * turns0) / (y0 * x0)
            print(y)
            plt.plot(x, y, color='black', linestyle='--', label='ideal')
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
