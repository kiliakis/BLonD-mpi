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
    'hatches': ['//', '\\', ''],
    'markers': ['x', 'o', '^'],
    'colors': ['0.85', '0.4'],
    'edgecolors': ['xkcd:red', 'xkcd:blue'],
    # 'colors': ['xkcd:red', 'xkcd:green', 'xkcd:blue'],
    # 'colors': [[cm.Reds(x) for x in np.linspace(0.2, 0.8, 4)],
    #            [cm.Greens(x) for x in np.linspace(0.2, 0.8, 4)],
    #            [cm.Blues(x) for x in np.linspace(0.2, 0.8, 4)]
    #            ],

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
    'y_name': 'percent',
    # 'y_err_name': 'std',
    'xlabel': 'Nodes (x20 Cores)',
    'ylabel': 'Runtime(\%)',
    'ylabel2': 'Efficiency',
    'title': {
                # 's': '{}'.format(case.upper()),
                'fontsize': 10,
                'y': .82,
                # 'x': 0.55,
                'fontweight': 'bold',
    },
    'figsize': [5, 2.],
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
        'framealpha': 0.8, 'fontsize': 10, 'labelspacing': 0, 'borderpad': 0.5,
        'handletextpad': 0.5, 'borderaxespad': 0.1, 'columnspacing': 0.8,
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
    'phases': ['comm', 'serial'],
    # 'ylim': [0, 28],
    'ylim': [0, 100],
    'xlim': [1.6, 36],
    'yticks': [0, 20, 40, 60, 80, 100],
    # 'yticks': [2, 4, 8, 12, 16, 20, 24, 28, 32],
    # 'yticks': [2, 4, 8, 16, 32],
    'outfiles': ['{}/{}-{}-breakdown-{}.pdf',
                 '{}/{}-{}-breakdown-{}.png']
}


lconfig = {
    'figures': {
        'strong': {
            'files': [
                '{}/compfront/{}/lb-tp-approx0-mvapich2-strong-scaling/comm-comp-report.csv',
                # '{}/compfront/{}/lb-tp-approx2-mvapich2-strong-scaling/comm-comp-report.csv',
                '{}/compfront/{}/lb-tp-approx1-mvapich2-strong-scaling/comm-comp-report.csv',
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
                'type': ['comm', 'comp', 'serial', 'other'],
            }
        },
    },

}
plt.rcParams['font.family'] = gconfig['fontname']
plt.rcParams['text.usetex'] = True

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
            # ax.set_yscale('log', basey= 2)
            # ax.set_xscale('log', basex= 2)
            plots_dir = {}
            for file in figconf['files']:
                file = file.format(res_dir, case.upper())
                # print(file)
                data = np.genfromtxt(file, delimiter='\t', dtype=str)
                header, data = list(data[0]), data[1:]
                temp = get_plots(header, data, figconf['lines'],
                                 exclude=figconf.get('exclude', []),
                                 prefix=True)
                for key in temp.keys():
                    plots_dir['_{}'.format(key)] = temp[key].copy()

            plt.grid(True, which='major', alpha=0.5, zorder=1)
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

            final_dir = {}
            for key in plots_dir.keys():
                phase = key.split('_type')[1].split('_')[0]
                k = key.split('_type')[0]
                if k not in final_dir:
                    final_dir[k] = {}
                if phase not in final_dir[k]:
                    final_dir[k][phase] = plots_dir[key].copy()

            pos = 0
            step = 1
            width = 0.85 * step / (len(final_dir.keys()))
            for idx, k in enumerate(final_dir.keys()):
                mpiv = k.split('_mpi')[1].split('_')[0]
                lb = k.split('lb')[1].split('_')[0]
                lba = k.split('lba')[1].split('_')[0]
                approx = k.split('approx')[1].split('_')[0]
                approx = gconfig['approx'][approx]
                label = '{}'.format(approx)
                labels.add(label)
                bottom = []
                # colors = gconfig['colors'][idx]
                j = 0
                for phase in gconfig['phases']:

                    values = final_dir[k][phase]
                # for phase, values in final_dir[k].items():
                    y = get_values(values, header, gconfig['y_name'])
                    x = get_values(values, header, gconfig['x_name'])
                    omp = get_values(values, header, gconfig['omp_name'])
                    if phase == 'serial':
                        y += get_values(final_dir[k]['other'], header, gconfig['y_name'])

                    x_new = []
                    y_new = []
                    for i, xi in enumerate(gconfig['x_to_keep']):
                        # if xi in x:
                        x_new.append(xi)
                        y_new.append(y[list(x).index(xi)])
                    x = np.array(x_new)
                    y = np.array(y_new)
                    x = x * omp[0] // 20
                    if len(bottom) == 0:
                        bottom = np.zeros(len(y))

                    plt.bar(np.arange(len(x)) + pos, y, bottom=bottom, width=0.9*width,
                            label=None, 
                            linewidth=1.5,
                            edgecolor=gconfig['edgecolors'][idx],
                            hatch=gconfig['hatches'][idx],
                            color=gconfig['colors'][j],
                            zorder=2)
                    j += 1
                    bottom += y
                    # plt.plot(x, speedup,
                    #         label=label, marker=gconfig['markers'][idx],
                    #         color=gconfig['colors'][idx])
                    # print("{}:{}:".format(case, label), speedup)
                pos += width
            # pos += width * step
            # plt.xticks(np.arange(len(x)), np.array(x, int)//20)
            plt.xticks(np.arange(len(x))+step/4, np.array(x, int), **gconfig['ticks'])

            plt.ylim(gconfig['ylim'])
            plt.xlim(0-width, len(x))
            if col == 0:
                ax.tick_params(**gconfig['tick_params_left'])
            else:
                ax.tick_params(**gconfig['tick_params_center_right'])

            # if col == 0:
                # handles, labels = ax.get_legend_handles_labels()
                # print(labels)
            # ax.legend(**gconfig['legend'])

            plt.xticks(**gconfig['ticks'])
            plt.yticks(gconfig['yticks'], gconfig['yticks'], **gconfig['ticks'])

            if col == 0:
                handles = []
                for c, p in zip(gconfig['colors'], ['comm', 'intra', 'other']):
                    patch = mpatches.Patch(label=p, edgecolor='black', facecolor=c,
                                           linewidth=1.,)
                    handles.append(patch)


                for h, tc, e in zip(gconfig['hatches'], labels, gconfig['edgecolors']):
                    patch = mpatches.Patch(label=tc, edgecolor=e,
                                           facecolor='1', hatch=h, linewidth=1.5,)
                    handles.append(patch)
                plt.legend(handles=handles, **gconfig['legend'])
        plt.tight_layout()
        plt.subplots_adjust(**gconfig['subplots_adjust'])
        for file in gconfig['outfiles']:
            file = file.format(images_dir, title, '-'.join(args.cases),
                               args.keysuffix)
            save_and_crop(fig, file, dpi=600, bbox_inches='tight')
        if args.show:
            plt.show()
        plt.close()
