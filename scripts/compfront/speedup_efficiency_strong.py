import matplotlib.pyplot as plt
import numpy as np

import os
from matplotlib import cm
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from itertools import cycle
import matplotlib.ticker
import sys
from plot.plotting_utilities import *
import argparse

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"
this_filename = sys.argv[0].split('/')[-1]

parser = argparse.ArgumentParser(description='Run MPI jobs locally.',
                                 usage='python local_scan_mpi.py -i in.yml')

parser.add_argument('-c', '--cases', type=str, nargs='+',
                    choices=['lhc', 'sps', 'ps', 'ex01'],
                    help='The test-case to plot.')

parser.add_argument('-k', '--keysuffix', type=str, default='strong',
                    help='A key suffix to use.')

parser.add_argument('-e', '--errorbars', action='store_true',
                    help='Add errorbars.')

parser.add_argument('-s', '--show', action='store_true',
                    help='Show the plots.')

args = parser.parse_args()

project_dir = this_directory + '../../'
res_dir = project_dir + 'results/'
images_dir = res_dir + 'plots/compfront/'

if not os.path.exists(images_dir):
    os.makedirs(images_dir)

gconfig = {
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
    'approx': {
        '0': 'exact',
        '1': 'SRP',
        '2': 'RDS',
    },
    # 'colors': {
    # 'mvapich2': cycle(['xkcd:pastel green', 'xkcd:green', 'xkcd:olive green', 'xkcd:blue green']),
    # 'mvapich2': cycle([cm.Greens(x) for x in np.linspace(0.2, 0.8, 3)]),
    # 'mpich3-NoLB': cycle(['xkcd:pastel green']),

    # 'mvapich2': cycle(['xkcd:orange', 'xkcd:rust']),
    # 'mvapich2-NoLB': cycle(['xkcd:apricot']),
    # },
    'hatches': ['', '', 'xx'],
    'markers': ['x', 'o', '^'],
    # 'colors': ['0.', '0.4', '0.65'],
    'colors': ['xkcd:red', 'xkcd:green', 'xkcd:blue'],

    # 'colors': ['0.2', '0.6', '0.9'],
    # 'hatches': {
    #     'LB': 'x',
    #     'NoLB': '',
    # },
    # 'hatches': ['', '//', '--', 'xx'],
    'reference': {
        # 'sps': {'ppb': 4000000, 'b': 72, 'turns': 1000, 'w': 1,
        #         'omp': 20, 'time': 225.85},
        # 'sps': {'ppb': 6000000, 'b': 288, 'turns': 5000, 'w': 1,
        #         'omp': 20, 'time': 7732.82},
        'sps': {'ppb': 4000000, 'b': 288, 'turns': 5000, 'w': 1,
                'omp': 20, 'time': 5873.38},
        # 'omp': 20, 'time': 4295.51},

        # 'lhc': {'ppb': 2000000, 'b': 96, 'turns': 1000, 'w': 1,
        #         'omp': 20, 'time': 103.04},
        # 'lhc': {'ppb': 6000000, 'b': 192, 'turns': 5000, 'w': 1,
        #         'omp': 20, 'time': 4772.91},
        'lhc': {'ppb': 4000000, 'b': 192, 'turns': 5000, 'w': 1,
                'omp': 20, 'time': 3391.31},

        # 'ps': {'ppb': 8000000, 'b': 21, 'turns': 1000, 'w': 1,
        #        'omp': 20, 'time': 96.0},
        # 'ps': {'ppb': 32000000, 'b': 21, 'turns': 5000, 'w': 1,
        #        'omp': 20, 'time': 3402.4},
        'ps': {'ppb': 16000000, 'b': 21, 'turns': 5000, 'w': 1,
               'omp': 20, 'time': 1332.8},
    },
    # 'sequence': ['mpich3']

    # 'exclude': [['v1', 'notcm'], ['v2', 'notcm'], ['v4', 'notcm']],
    'x_name': 'n',
    'x_to_keep': [4, 8, 16, 32, 64],
    # 'x_to_keep': [8, 16],
    'omp_name': 'omp',
    'y_name': 'avg_time(sec)',
    # 'y_err_name': 'std',
    'xlabel': 'Nodes (x20 Cores)',
    'ylabel': 'Speedup',
    'ylabel2': 'Efficiency',
    'title': {
                # 's': '{}'.format(case.upper()),
                'fontsize': 10,
                'y': .85,
                # 'x': 0.55,
                'fontweight': 'bold',
    },
    'figsize': [5, 2.2],
    'annotate': {
        'fontsize': 9,
        'textcoords': 'data',
        'va': 'bottom',
        'ha': 'center'
    },
    'ticks': {'fontsize': 10},
    'fontsize': 10,
    'legend': {
        'loc': 'upper left', 'ncol': 1, 'handlelength': 1.5, 'fancybox': False,
        'framealpha': .7, 'fontsize': 10, 'labelspacing': 0, 'borderpad': 0.5,
        'handletextpad': 0.5, 'borderaxespad': 0, 'columnspacing': 0.8,
        'bbox_to_anchor': (0., 0.85)
    },
    'subplots_adjust': {
        'wspace': 0.0, 'hspace': 0.1, 'top': 0.93
    },
    'tick_params_left': {
        'pad': 1, 'top': 0, 'bottom': 1, 'left': 1,
        'direction': 'out', 'length': 3, 'width': 1,
    },
    'tick_params_center_right': {
        'pad': 1, 'top': 0, 'bottom': 1, 'left': 0,
        'direction': 'out', 'length': 3, 'width': 1,
    },
    'fontname': 'DejaVu Sans Mono',

    # 'ylim': [0, 28],
    'ylim': [0, 36],
    'xlim': [1.6, 36],
    # 'yticks': [4, 8, 12, 16, 20, 24],
    # 'yticks': [2, 4, 8, 12, 16, 20, 24, 28, 32],
    'yticks': [4, 8, 12, 16, 20, 24, 28, 32],
    'outfiles': ['{}/{}-{}-speedup-{}.pdf',
                 '{}/{}-{}-speedup-{}.jpg']
}


lconfig = {
    'errorfile': 'comm-comp-std-report.csv',
    'datafile': 'comm-comp-report.csv',
    'figures': {
        'strong': {
            'files': [
                '{}/compfront/{}/lb-tp-approx0-mvapich2-strong-scaling/{}',
                '{}/compfront/{}/lb-tp-approx2-mvapich2-strong-scaling/{}',
                '{}/compfront/{}/lb-tp-approx1-mvapich2-strong-scaling/{}',
            ],
            'lines': {
                'mpi': ['mpich3', 'mvapich2', 'openmpi3'],
                'lb': ['interval', 'reportonly'],
                'approx': ['0', '1', '2'],
                'lba': ['500'],
                'b': ['6', '12', '24', '96', '192',
                      '48', '21', '9', '18', '36',
                      '72', '144', '288'],
                't': ['5000'],
                'type': ['total'],
            }
        },
    },

}
plt.rcParams['font.family'] = gconfig['fontname']

if __name__ == '__main__':
    for title, figconf in lconfig['figures'].items():
        fig, ax_arr = plt.subplots(ncols=len(args.cases), nrows=1,
                                   sharex=True, sharey=True,
                                   figsize=gconfig['figsize'])
        ax_arr = np.atleast_1d(ax_arr)
        labels = set()
        for col, case in enumerate(args.cases):
            ax = ax_arr[col]
            # ax2 = ax.twinx()
            plt.sca(ax)
            # ax.set_yscale('log', basey=2)
            ax.set_xscale('log', basex=2)
            plots_dir = {}
            errors_dir = {}
            for file in figconf['files']:
                # file = file.format(res_dir, case.upper())
                # print(file)
                data = np.genfromtxt(file.format(res_dir, case.upper(), lconfig['datafile']),
                                     delimiter='\t', dtype=str)
                header, data = list(data[0]), data[1:]
                temp = get_plots(header, data, figconf['lines'],
                                 exclude=figconf.get('exclude', []),
                                 prefix=True)
                for key in temp.keys():
                    plots_dir['_{}_{}'.format(
                        key, args.keysuffix)] = temp[key].copy()

                if args.errorbars:
                    data = np.genfromtxt(file.format(res_dir, case.upper(), lconfig['errorfile']),
                                         delimiter='\t', dtype=str)
                    header, data = list(data[0]), data[1:]
                    temp = get_plots(header, data, figconf['lines'],
                                     exclude=figconf.get('exclude', []),
                                     prefix=True)
                    for key in temp.keys():
                        errors_dir['_{}_{}'.format(
                            key, args.keysuffix)] = temp[key].copy()

            plt.grid(True, which='both', axis='y', alpha=0.5)
            # plt.grid(True, which='minor', alpha=0.5, zorder=1)
            plt.grid(False, which='major', axis='x')
            plt.title('{}'.format(case.upper()), **gconfig['title'])
            if col == 1:
                plt.xlabel(gconfig['xlabel'], labelpad=3,
                           fontweight='bold',
                           fontsize=gconfig['fontsize'])
            if col == 0:
                plt.ylabel(gconfig['ylabel'], labelpad=3,
                           fontweight='bold',
                           fontsize=gconfig['fontsize'])

            pos = 0
            step = 0.1
            width = 1. / (1*len(plots_dir.keys())+0.4)

            for idx, k in enumerate(plots_dir.keys()):
                values = plots_dir[k]
                mpiv = k.split('_mpi')[1].split('_')[0]
                lb = k.split('lb')[1].split('_')[0]
                lba = k.split('lba')[1].split('_')[0]
                approx = k.split('approx')[1].split('_')[0]
                if 'tp' in k:
                    tp = '1'
                else:
                    tp = '0'
                experiment = k.split('_')[-1]
                # tp = k.split('tp')[1].split('_')[0]
                if lb == 'interval':
                    lb = 'LB'
                elif lb == 'reportonly':
                    lb = 'NoLB'
                if tp == '1':
                    tp = 'TP'
                elif tp == '0':
                    tp = 'NoTP'
                approx = gconfig['approx'][approx]

                label = '{}'.format(approx)

                x = get_values(values, header, gconfig['x_name'])
                omp = get_values(values, header, gconfig['omp_name'])
                y = get_values(values, header, gconfig['y_name'])
                parts = get_values(values, header, 'ppb')
                bunches = get_values(values, header, 'b')
                turns = get_values(values, header, 't')
                if args.errorbars:
                    # yerr is normalized to y
                    yerr = get_values(errors_dir[k], header, gconfig['y_name'])
                    yerr = yerr/y
                else:
                    yerr = np.zeros(len(y))

                # This is the throughput
                y = parts * bunches * turns / y
                # Now the reference, 1thread
                yref = gconfig['reference'][case]['time']
                partsref = gconfig['reference'][case]['ppb']
                bunchesref = gconfig['reference'][case]['b']
                turnsref = gconfig['reference'][case]['turns']
                yref = partsref * bunchesref * turnsref / yref
                ompref = gconfig['reference'][case]['omp']

                speedup = y / yref
                # yerr = yerr / yref

                x_new = []
                sp_new = []
                yerr_new = []
                for i, xi in enumerate(gconfig['x_to_keep']):
                    if xi in x:
                        x_new.append(xi)
                        sp_new.append(speedup[list(x).index(xi)])
                        yerr_new.append(yerr[list(x).index(xi)])
                    # else:
                    #     sp_new.append(0)
                x = np.array(x_new)
                speedup = np.array(sp_new)
                yerr = np.array(yerr_new)
                # yerr is denormalized again
                yerr = yerr * speedup
                # efficiency = 100 * speedup / (x * omp[0] / ompref)
                x = x * omp[0]

                plt.errorbar(x//20, speedup,
                             label=label, marker=gconfig['markers'][idx],
                             color=gconfig['colors'][idx],
                             yerr=yerr,
                             capsize=2)
                print("{}:{}:".format(case, label), end='\t')
                for xi, yi, yeri in zip(x//20, speedup, yerr):
                    print('N:{:.0f} {:.2f}±{:.2f}'.format(xi, yi, yeri), end=' ')
                print('')
                # print("{}:{}:".format(case, label), speedup)
                pos += 1 * width
            # pos += width * step
            plt.ylim(gconfig['ylim'])
            # plt.xticks(np.arange(len(x)), np.array(x, int)//20)
            plt.xlim(gconfig['xlim'])
            plt.xticks(x//20, np.array(x, int)//20, **gconfig['ticks'])

            if col == 0:
                ax.tick_params(**gconfig['tick_params_left'])
            else:
                ax.tick_params(**gconfig['tick_params_center_right'])

            # if col == 0:
                # handles, labels = ax.get_legend_handles_labels()
                # print(labels)
            ax.legend(**gconfig['legend'])

            plt.xticks(**gconfig['ticks'])
            plt.yticks(gconfig['yticks'], gconfig['yticks'], **gconfig['ticks'])

        # plt.legend(**gconfig['legend'])
        plt.tight_layout()
        plt.subplots_adjust(**gconfig['subplots_adjust'])
        for file in gconfig['outfiles']:
            file = file.format(images_dir, title, '-'.join(args.cases),
                               args.keysuffix)
            save_and_crop(fig, file, dpi=600, bbox_inches='tight')
        if args.show:
            plt.show()
        plt.close()
