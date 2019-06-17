import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

import matplotlib.ticker
import sys
from plot.plotting_utilities import *

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"
this_filename = sys.argv[0].split('/')[-1]

project_dir = this_directory + '../../'
res_dir = project_dir + 'results/'
images_dir = res_dir + 'plots/redistribute/'

if not os.path.exists(images_dir):
    os.makedirs(images_dir)

case = 'lhc'

config = {
    'files': {
    #     '{}/raw/{}/mvapich2/comm-comp-report.csv'.format(res_dir, case.upper()): {
    #         'key': '{}'.format(case),
    #         'lines': {
    #             'mpi': ['mpich3', 'mvapich2'],
    #             'lb': ['interval', 'reportonly'],
    #             'approx': ['0', '2'],
    #             'lba': ['100', '500'],
    #             'type': ['total'],
    #         }
    #     },
    #     '{}/raw/{}/mpich3/comm-comp-report.csv'.format(res_dir, case.upper()): {
    #         'key': '{}'.format(case),
    #         'lines': {
    #             'mpi': ['mpich3', 'mvapich2'],
    #             'lb': ['interval', 'reportonly'],
    #             'approx': ['0', '2'],
    #             'lba': ['100', '500'],
    #             'type': ['total'],
    #         }
    #     },
    #     '{}/raw/{}/lb-mvapich2/comm-comp-report.csv'.format(res_dir, case.upper()): {
    #         'key': '{}'.format(case),
    #         'lines': {
    #             'mpi': ['mpich3', 'mvapich2'],
    #             'lb': ['interval', 'reportonly'],
    #             'approx': ['0', '2'],
    #             'lba': ['100', '500'],
    #             'type': ['total'],
    #         }
    #     },
    #     '{}/raw/{}/lb-mpich3/comm-comp-report.csv'.format(res_dir, case.upper()): {
    #         'key': '{}'.format(case),
    #         'lines': {
    #             'mpi': ['mpich3', 'mvapich2'],
    #             'lb': ['interval', 'reportonly'],
    #             'approx': ['0', '2'],
    #             'lba': ['100', '500'],
    #             'type': ['total'],
    #         }
    #     },


        '{}/raw/{}/tp-mvapich2/comm-comp-report.csv'.format(res_dir, case.upper()): {
            'key': '{}'.format(case),
            'lines': {
                'mpi': ['mpich3', 'mvapich2'],
                'lb': ['interval', 'reportonly'],
                'approx': ['0', '2'],
                'lba': ['100', '500'],
                'type': ['total'],
            }
        },
        '{}/raw/{}/tp-mpich3/comm-comp-report.csv'.format(res_dir, case.upper()): {
            'key': '{}'.format(case),
            'lines': {
                'mpi': ['mpich3', 'mvapich2'],
                'lb': ['interval', 'reportonly'],
                'approx': ['0', '2'],
                'lba': ['100', '500'],
                'type': ['total'],
            }
        },
        '{}/raw/{}/lb-tp-mvapich2/comm-comp-report.csv'.format(res_dir, case.upper()): {
            'key': '{}'.format(case),
            'lines': {
                'mpi': ['mpich3', 'mvapich2'],
                'lb': ['interval', 'reportonly'],
                'approx': ['0', '2'],
                'lba': ['100', '500'],
                'type': ['total'],
            }
        },
        '{}/raw/{}/lb-tp-mpich3/comm-comp-report.csv'.format(res_dir, case.upper()): {
            'key': '{}'.format(case),
            'lines': {
                'mpi': ['mpich3', 'mvapich2'],
                'lb': ['interval', 'reportonly'],
                'approx': ['0', '2'],
                'lba': ['100', '500'],
                'type': ['total'],
            }
        },
    },
    'markers': {
        'ex01': 'd',
        'lhc': 'o',
        'sps': 's',
        'ps': 'x'
    },
    'ls': {
        'ex01': '-:',
        'lhc': '-',
        'sps': ':',
        'ps': '--'
    },
    'colors': {
        'mpich3-wlb': 'xkcd:light green',
        'mpich3-wolb': 'xkcd:green',

        'mvapich2-wlb': 'xkcd:light orange',
        'mvapich2-wolb': 'xkcd:orange',
    },
    'hatches': {
        'mpich3': 'x',
        'openmpi3': '-',
        'mvapich2': 'o',
    },
    'reference': {
        'ex01': {'time': 21.4, 'ppb': 1000000, 'turns': 2000},
        'sps': {'time': 430., 'ppb': 4000000, 'turns': 100},
        # 'lhc': {'time': 2120., 'ppb': 2000000, 'turns': 1000},
        'lhc': {'time': 350.831, 'ppb': 2000000, 'b': 48, 'turns': 500},
        'ps': {'time': 1623.7, 'ppb': 4000000, 'turns': 2000},
    },
    # 'sequence': ['mpich3']

    # 'exclude': [['v1', 'notcm'], ['v2', 'notcm'], ['v4', 'notcm']],
    'x_name': 'n',
    'x_to_keep': [4, 8, 12, 16],
    'omp_name': 'omp',
    'y_name': 'avg_time(sec)',
    # 'y_err_name': 'std',
    'xlabel': 'Cores (x10)',
    'ylabel': 'Speedup',
    'title': '{} Load-balance'.format(case.upper()),
    'figsize': (8, 6),
    'fontsize': 8,
    'legend': {
        'loc': 'upper left', 'ncol': 3, 'handlelength': 1, 'fancybox': True,
        'framealpha': 0., 'fontsize': 9, 'labelspacing': 0, 'borderpad': 0.5,
        'handletextpad': 0.5, 'borderaxespad': 0, 'columnspacing': 0.5,
    },
    'subplots_adjust': {
        'wspace': 0.05, 'hspace': 0.16, 'top': 0.93
    },
    'tick_params': {
        'pad': 1, 'top': 1, 'bottom': 1, 'left': 1,
        'direction': 'inout', 'length': 3, 'width': 0.5,
    },
    'image_name': '{}/{}-tp.pdf'.format(images_dir, case),

}

if __name__ == '__main__':
    plots_dir = {}
    for file, conf in config['files'].items():
        # print(file)
        data = np.genfromtxt(file, delimiter='\t', dtype=str)
        header, data = list(data[0]), data[1:]
        temp = get_plots(header, data, conf['lines'],
                         exclude=conf.get('exclude', []),
                         prefix=True)
        for key in temp.keys():
            plots_dir['{}-{}'.format(conf['key'], key)] = temp[key].copy()

    fig = plt.figure(figsize=config['figsize'])

    plt.grid(True, which='major', alpha=0.5)
    plt.grid(False, which='major', axis='x')
    plt.title(config['title'])
    plt.xlabel(config['xlabel'], fontsize=config['fontsize'])
    plt.ylabel(config['ylabel'], fontsize=config['fontsize'])

    pos = 0
    step = 0.1
    width = 1. / (len(plots_dir.keys())+1)

    for k in sorted(plots_dir.keys()):
        values = plots_dir[k]
        case = k.split('-')[0]
        mpiv = k.split('-mpi')[1].split('_')[0]
        lb = k.split('lb')[1].split('_')[0]
        lba = k.split('lba')[1].split('_')[0]
        approx = k.split('approx')[1].split('_')[0]

        # key = '{}-{}-{}'.format(case, mpiv, lb)

        label = '{}-{}-{}-{}'.format(mpiv, lb, lba, approx)
        # color = config['colors']['{}-{}'.format(mpiv, lb)]
        # hatch = config['hatches'][mpiv]
        # marker = config['markers'][case]
        # ls = config['ls'][case]

        x = get_values(values, header, config['x_name'])
        omp = get_values(values, header, config['omp_name'])
        y = get_values(values, header, config['y_name'])
        parts = get_values(values, header, 'ppb')
        bunches = get_values(values, header, 'b')
        turns = get_values(values, header, 't')

        # This is the throughput
        y = parts * bunches * turns / y

        # Now the reference, 1thread
        yref = config['reference'][case]['time']
        partsref = config['reference'][case]['ppb']
        bunchesref = config['reference'][case]['b']
        turnsref = config['reference'][case]['turns']
        yref = partsref * bunchesref * turnsref / yref

        speedup = y / yref

        if len(config['x_to_keep']) < len(x):
            x_new = []
            speedup_new = []
            omp_new = []
            for i in range(len(x)):
                if x[i] in config['x_to_keep']:
                    x_new.append(x[i])
                    speedup_new.append(speedup[i])
                    omp_new.append(omp[i])
            x = np.array(x_new)
            speedup = np.array(speedup_new)
            omp = np.array(omp_new)

        x = x * omp

        # efficiency = 100 * speedup / x
        plt.bar(np.arange(len(x)) + pos, speedup, width=width,
                edgecolor='0.3', label=label)
        # hatch=hatch)
        pos += width
    pos += width * step

    plt.xticks(np.arange(len(x)) + pos/2.2, np.array(x//10, int))

    # handles = []
    # for k, v in config['colors'].items():
    #     patch = mpatches.Patch(label=k, edgecolor='black', facecolor=v,
    #                            linewidth=.5, alpha=0.9)
    #     handles.append(patch)

    # for k, v in config['hatches'].items():
    #     patch = mpatches.Patch(label=k, edgecolor='black',
    #                            facecolor='0.8', hatch=v, linewidth=.5,)
    #     handles.append(patch)

    # plt.legend(handles=handles, **config['legend'])
    plt.legend(**config['legend'])
    plt.gca().tick_params(**config['tick_params'])

    plt.subplots_adjust(**config['subplots_adjust'])
    plt.xticks(fontsize=config['fontsize'])
    plt.yticks(fontsize=config['fontsize'])
    plt.tight_layout()
    save_and_crop(fig, config['image_name'], dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()
