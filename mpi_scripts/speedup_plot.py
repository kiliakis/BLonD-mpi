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


plots_config = {

  
    #   'plot3': {
    #     'files': {
    #         res_dir+'raw/LHC-4n-96B-lt-lb-nogat-int-op-knd-r5-10kt/comm-comp-report.csv': {
    #             'lines': {
    #                       'omp': ['4', '5', '10'],
    #                       'type': ['total']}
    #         }

    #     },
    #     'labels': {
    #                '1-total': 'hybrid-T1',
    #                '2-total': 'hybrid-T2',
    #                '4-total': 'hybrid-T4',
    #                '5-total': 'hybrid-T5',
    #                '10-total': 'hybrid-T10',
    #                '20-total': 'hybrid-T20'
    #                },
    #     'reference': { 'time': 8213. , 'parts': 1000000, 'turns':10000},
    #     # 'exclude': [['v1', 'notcm'], ['v2', 'notcm'], ['v4', 'notcm']],
    #     'ideal': '2-total',
    #     'x_name': 'n',
    #     'omp_name': 'omp',
    #     'y_name': 'avg_time(sec)',
    #     # 'y_err_name': 'std',
    #     'xlabel': 'MPI Tasks/OMP Threads',
    #     'ylabel': 'Speedup',
    #     'title': '',
    #     # 'ylim': [0, 16000],
    #     'figsize': (6, 3),
    #     'image_name': images_dir + 'LHC-4n-speedup-96B-lt-lb-int-op-knd-r5-10kt.pdf'

    # },

'plot4': {
        'files': {
            res_dir+'raw/SPS-8n-72B-imped/comm-comp-report.csv': {
                'lines': {
                          'omp': ['2', '5', '10'],
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
        'reference': { 'time': 430. , 'parts': 4000000, 'turns':100},
        # 'exclude': [['v1', 'notcm'], ['v2', 'notcm'], ['v4', 'notcm']],
        'ideal': '2-total',
        'x_name': 'n',
        'omp_name': 'omp',
        'y_name': 'avg_time(sec)',
        # 'y_err_name': 'std',
        'xlabel': 'MPI Tasks/OMP Threads',
        'ylabel': 'Speedup',
        'title': '',
        # 'ylim': [0, 16000],
        'figsize': (6, 3),
        'image_name': images_dir + 'SPS-8n-72B-imped-speedup.pdf'

    },


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
            x = np.array(values[:, header.index(config['x_name'])], float)
            omp = np.array(
                values[:, header.index(config['omp_name'])], float)
            # sub 1 due to the master
            if (plot_key != 'plot1'):
                x = (x-1) * omp

            y = np.array(values[:, header.index(config['y_name'])], float)
            parts = np.array(values[:, header.index('parts')], float)
            turns = np.array(values[:, header.index('turns')], float)
            # This is the throughput
            y = parts * turns / y

            # Now the reference, 1thread
            yref = config['reference']['time']
            partsref = config['reference']['parts']
            turnsref = config['reference']['turns']
            yref = partsref * turnsref / yref

            speedup = y / yref

            # We want speedup, compared to 1 worker with 1 thread
            plt.errorbar(x, speedup, yerr=None, label=label,
                         capsize=2, marker='.', markersize=5, linewidth=1.5)
        if 'extra' in config:
            for c in config['extra']:
                exec(c)

        # if config.get('ideal', ''):
        #     # Ideal line
        #     ylims = plt.gca().get_ylim()
        #     xlims = plt.gca().get_xlim()

        #     x0 = np.array(plots_dir[config['ideal']]
        #                   [:, header.index(config['x_name'])], float)[0]
        #     omp0 = np.array(plots_dir[config['ideal']]
        #                     [:, header.index(config['omp_name'])], float)[0]
        #     x0 = (x0-1) * omp0
        #     y0 = float(plots_dir[config['ideal']]
        #                [0, header.index(config['y_name'])])
        #     print(x0)
        #     print(y0)

        #     parts0 = float(plots_dir[config['ideal']]
        #                    [0, header.index('parts')])
        #     turns0 = float(plots_dir[config['ideal']]
        #                    [0, header.index('turns')])
        #     print(parts0)
        #     print(turns0)
        #     x = np.arange(x0, xlims[1], 1)
        #     y = x * (parts0 * turns0) / (y0 * x0)
        #     print(y)
        #     plt.plot(x, y, color='black', linestyle='--', label='ideal')
        #     plt.ylim(ylims)

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
