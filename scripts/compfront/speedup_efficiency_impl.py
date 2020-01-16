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

parser.add_argument('-k', '--keysuffix', type=str, default='impl',
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
        '1': 'SMD',
        '2': 'RDS',
    },
    # 'colors': {
    # 'mvapich2': cycle(['xkcd:pastel green', 'xkcd:green', 'xkcd:olive green', 'xkcd:blue green']),
    # 'mvapich2': cycle([cm.Greens(x) for x in np.linspace(0.2, 0.8, 3)]),
    # 'mpich3-NoLB': cycle(['xkcd:pastel green']),

    # 'mvapich2': cycle(['xkcd:orange', 'xkcd:rust']),
    # 'mvapich2-NoLB': cycle(['xkcd:apricot']),
    # },
    # 'hatches': {
    #     'LB': 'x',
    #     'NoLB': '',
    # },
    'hatches': ['', '', ''],
    'colors': ['0.1', '0.5', '0.8'],
    'reference': {
        # 'sps': {'ppb': 4000000, 'b': 72, 'turns': 500, 'w': 1,
        #         'omp': 1, 'time': 1497.8},
        # 'sps': {'ppb': 4000000, 'b': 72, 'turns': 1000, 'w': 1,
        #         'omp': 10, 'time': 415.4},
        'sps': {'ppb': 4000000, 'b': 72, 'turns': 1000, 'w': 1,
                'omp': 20, 'time': 225.85},

        # 'lhc': {'ppb': 2000000, 'b': 96, 'turns': 500, 'w': 1,
        #         'omp': 1, 'time': 681.59},
        # 'lhc': {'ppb': 2000000, 'b': 96, 'turns': 1000, 'w': 1,
        #         'omp': 10, 'time': 177.585},
        'lhc': {'ppb': 2000000, 'b': 96, 'turns': 1000, 'w': 1,
                'omp': 20, 'time': 103.04},

        # 'ps': {'time': 466.085, 'ppb': 8000000, 'b': 21, 'turns': 500},
        # 'ps': {'ppb': 8000000, 'b': 21, 'turns': 500, 'w': 1,
        #        'omp': 1, 'time': 502.88},
        # 'ps': {'ppb': 8000000, 'b': 21, 'turns': 1000, 'w': 1,
        #        'omp': 10, 'time': 142.066},
        'ps': {'ppb': 8000000, 'b': 21, 'turns': 1000, 'w': 1,
               'omp': 20, 'time': 96.0},
    },
    # 'sequence': ['mpich3']

    # 'exclude': [['v1', 'notcm'], ['v2', 'notcm'], ['v4', 'notcm']],
    'x_name': 'n',
    # 'x_to_keep': [2, 4, 8, 16, 32, 64],
    'x_to_keep': [8],
    'omp_name': 'omp',
    'y_name': 'avg_time(sec)',
    # 'y_err_name': 'std',
    'xlabel': '',
    'ylabel': 'Norm. Runtime',
    'ylabel2': 'Efficiency',
    'title': {
                's': '',
                'fontsize': 10,
                'y': 0.74,
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
        'loc': 'upper left', 'ncol': 3, 'handlelength': 1.5, 'fancybox': True,
        'framealpha': 0., 'fontsize': 10, 'labelspacing': 0, 'borderpad': 0.5,
        'handletextpad': 0.5, 'borderaxespad': 0, 'columnspacing': 0.8,
        # 'bbox_to_anchor': (0, 1.25)
    },
    'subplots_adjust': {
        'wspace': 0.0, 'hspace': 0.1, 'top': 0.93
    },
    'tick_params': {
        'pad': 1, 'top': 0, 'bottom': 0, 'left': 1,
        'direction': 'out', 'length': 3, 'width': 1,
    },
    'fontname': 'DejaVu Sans Mono',
    'ylim': [.9, 1.4],
    # 'ylim2': [10, 90],
    'yticks': [.9, 1., 1.1, 1.2, 1.3, 1.4],
    # 'yticks2': [0, 20, 40, 60, 80, 100],
    'outfiles': ['{}/{}-{}-normtime-{}.pdf',
                 '{}/{}-{}-normtime-{}.jpg']
}


lconfig = {
    'figures': {
        'plot': {
            'files': [
                '{}/compfront/{}/approx0-mvapich2-impl/comm-comp-report.csv',
                '{}/compfront/{}/approx0-mpich3-impl/comm-comp-report.csv',
                '{}/compfront/{}/approx0-openmpi3-impl/comm-comp-report.csv',
            ],
            'lines': {
                'mpi': ['mpich3', 'mvapich2', 'openmpi3'],
                'lb': ['interval', 'reportonly'],
                'approx': ['0', '1', '2'],
                'lba': ['500'],
                'b': ['96', '48', '72', '21'],
                't': ['5000'],
                'type': ['total'],
            }
        },
    },

}

plt.rcParams['font.family'] = gconfig['fontname']


if __name__ == '__main__':
    for title, figconf in lconfig['figures'].items():
        fig, ax = plt.subplots(ncols=1, nrows=1,
                               sharex=True, sharey=True,
                               figsize=gconfig['figsize'])
        plt.sca(ax)
        plt.title(**gconfig['title'])
        plt.xlabel(gconfig['xlabel'], labelpad=3,
                   fontweight='bold',
                   fontsize=gconfig['fontsize'])
        plt.ylabel(gconfig['ylabel'], labelpad=3, color='xkcd:black',
                   fontweight='bold',
                   fontsize=gconfig['fontsize'])
        # plt.setp(ax.get_yticklabels(), color="xkcd:green")

        pos = 0
        step = 1.
        # colors1 = [cm.Blacks(x)
        #            for x in np.linspace(0.1, 0.9, len(args.cases))]
        colors1 = ['0.2', '0.5', '0.8']
        labels = set()
        avg = {}

        for col, case in enumerate(args.cases):
            # ax2 = ax.twinx()
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
                    plots_dir['_{}_{}'.format(
                        key, args.keysuffix)] = temp[key].copy()

            # First the reference value
            keyref = ''
            for k in plots_dir.keys():
                if 'mvapich2' in k:
                    keyref = k
                    break
            if keyref == '':
                print('ERROR: mvapich2 not found')
                exit(-1)
            refvals = plots_dir[keyref]

            x = get_values(refvals, header, gconfig['x_name'])
            omp = get_values(refvals, header, gconfig['omp_name'])
            y = get_values(refvals, header, gconfig['y_name'])
            parts = get_values(refvals, header, 'ppb')
            bunches = get_values(refvals, header, 'b')
            turns = get_values(refvals, header, 't')
            yref = parts * bunches * turns / y

            width = 0.9 * step / (len(plots_dir.keys()))

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
                if approx == '2':
                    approx = 'AC'
                else:
                    approx = 'NoAC'
                # key = '{}-{}-{}'.format(case, mpiv, lb)

                # label = '{}-{}-{}-{}'.format(lb, tp, approx, experiment)
                label = '{}'.format(mpiv)
                if label in labels:
                    label = None
                else:
                    labels.add(label)
                # label = '{}-{}'.format(tp, approx)
                # color = gconfig['colors']['{}'.format(mpiv)].__next__()
                # hatch = gconfig['hatches'][lb]
                # marker = config['markers'][case]
                # ls = config['ls'][case]

                x = get_values(values, header, gconfig['x_name'])
                omp = get_values(values, header, gconfig['omp_name'])
                y = get_values(values, header, gconfig['y_name'])
                parts = get_values(values, header, 'ppb')
                bunches = get_values(values, header, 'b')
                turns = get_values(values, header, 't')

                # This is the throughput
                y = parts * bunches * turns / y

                normtime = yref / y
                x_new = []
                sp_new = []
                for i, xi in enumerate(gconfig['x_to_keep']):
                    if xi in x:
                        x_new.append(xi)
                        sp_new.append(normtime[list(x).index(xi)])
                    # else:
                        # sp_new.append(0)
                x = np.array(x_new)
                normtime = np.array(sp_new)
                x = x * omp[0]

                if mpiv not in avg:
                    avg[mpiv] = []
                avg[mpiv].append(normtime)

                # efficiency = 100 * speedup / x
                plt.bar(pos + width * idx, normtime, width=0.9*width,
                        edgecolor='0.', label=label, hatch=gconfig['hatches'][idx],
                        color=gconfig['colors'][idx])
                # ax.annotate('{:.2f}'.format(normtime[0]), xy=(pos + idx*width, normtime[0]),
                #         **gconfig['annotate'])
                # if True or idx != 1:
                #     for i, s in zip(np.arange(len(x)) + pos + width * col, speedup):
                #         ax.annotate('{:.1f}'.format(s), xy=(i, s),
                #                     **gconfig['annotate'])
            pos += step
            # pos += width * step
        # I plot the averages here

        for idx, key in enumerate(avg.keys()):
            val = np.mean(avg[key])
            plt.bar(pos + idx*width, val, width=0.9*width,
                    edgecolor='0.', label=None, hatch=gconfig['hatches'][idx],
                    color=gconfig['colors'][idx])
            ax.annotate('{:.2f}'.format(val), xy=(pos + idx*width, val),
                        **gconfig['annotate'])
        pos += step

        plt.xticks(np.arange(pos) + width, [c.upper()
                                            for c in args.cases] + ['AVG'])

        plt.ylim(gconfig['ylim'])
        # plt.xlim(0-.8*width, len(x)-.7*width)

        # handles = []
        # for h, c
        #     handles.append(mpatches.Patch(label='Speedup', edgecolor='black',
        #                                   facecolor=colors1[0]))
        # handles.append(mpatches.Patch(label='Efficiency', edgecolor='black',
        #                               facecolor=colors2[0]))

        # plt.legend(handles=handles, **gconfig['legend'])
        plt.legend(**gconfig['legend'])
        # plt.legend()
        ax.tick_params(**gconfig['tick_params'])
        plt.xticks(**gconfig['ticks'], fontweight='bold')
        plt.yticks(gconfig['yticks'], **gconfig['ticks'])

        plt.tight_layout()
        plt.subplots_adjust(**gconfig['subplots_adjust'])
        for file in gconfig['outfiles']:
            file = file.format(images_dir, title, '-'.join(args.cases),
                               args.keysuffix)
            save_and_crop(fig, file, dpi=600, bbox_inches='tight')
        if args.show:
            plt.show()
        plt.close()
