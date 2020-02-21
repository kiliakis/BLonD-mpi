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

parser.add_argument('-k', '--keysuffix', type=str, default='workers',
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
    # 'colors': {
    # 'mvapich2': cycle(['xkcd:pastel green', 'xkcd:green', 'xkcd:olive green', 'xkcd:blue green']),
    # 'mvapich2': cycle([cm.Greens(x) for x in np.linspace(0.2, 0.8, 3)]),
    # 'mpich3-NoLB': cycle(['xkcd:pastel green']),

    # 'mvapich2': cycle(['xkcd:orange', 'xkcd:rust']),
    # 'mvapich2-NoLB': cycle(['xkcd:apricot']),
    # },
    'hatches': ['', '', 'xx', '', 'xx'],
    'colors': ['0.3', '0.6', '0.6', '0.9', '0.9'],
    # 'hatches': {
    #     'LB': 'x',
    #     'NoLB': '',
    # },
    # 'hatches': ['', '//', '--', 'xx'],
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
    'x_name': 'omp',
    'x_to_keep': [2, 5, 10, 20],
    # 'x_to_keep': [8, 16],
    'omp_name': 'n',
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
    'figsize': [5, 2.1],
    'annotate': {
        'fontsize': 9,
        'textcoords': 'data',
        'va': 'bottom',
        'ha': 'center'
    },
    'title_annotate': {
        'label': 'Workers-Per-Node:',
        # 'xytext': (0.19, .95),
        # 'xy': (0.05, .96),
        # 'fontsize': 10,
        # 'textcoords': 'axes fraction',
        # 'va': 'top',
        # 'ha': 'left'
    },
    'ticks': {'fontsize': 10},
    'xticks': {'fontsize': 10, 'rotation': '0', 'fontweight': 'bold'},
    'fontsize': 10,
    'legend': {
        'loc': 'upper right', 'ncol': 5, 'handlelength': 1.5, 'fancybox': True,
        'framealpha': 0., 'fontsize': 10, 'labelspacing': 0, 'borderpad': 0.5,
        'handletextpad': 0.5, 'borderaxespad': 0.1, 'columnspacing': 0.8,
        # 'title': 'Worker-Per-Node',
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
    'ylim': [.5, 1.1],
    # 'ylim2': [10, 90],
    # 'yticks': [.6, .8, 1., 1.2, 1.4, 1.6],
    'yticks': [.5, .6, .7, .8, .9, 1.],

    # 'yticks': [0, 2, 4, 6, 8],
    # 'yticks2': [0, 20, 40, 60, 80, 100],
    'outfiles': ['{}/{}-{}-normtime-{}.pdf',
                 '{}/{}-{}-normtime-{}.png']
}


lconfig = {
    'figures': {

        'workers': {
            'files': [
                '{}/compfront/{}/approx0-mvapich2-workers/comm-comp-report.csv',
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
plt.rcParams['text.usetex'] = True


if __name__ == '__main__':
    for title, figconf in lconfig['figures'].items():
        fig, ax = plt.subplots(ncols=1, nrows=1,
                               sharex=True, sharey=True,
                               figsize=gconfig['figsize'])
        plt.sca(ax)
        plt.plot([], [], ls=' ', **gconfig['title_annotate'])
        # ax.annotate(**gconfig['title_annotate'])
        # plt.title(**gconfig['title'])
        plt.xlabel(gconfig['xlabel'], labelpad=3,
                   fontsize=gconfig['fontsize'])
        plt.ylabel(gconfig['ylabel'], labelpad=3, color='xkcd:black',
                   fontweight='bold',
                   fontsize=gconfig['fontsize'])
        # ax_arr = np.atleast_1d(ax_arr)
        pos = 0
        step = 1.
        labels = set()
        avg = []
        xticks = []
        xtickspos = []
        for col, case in enumerate(args.cases):
            plots_dir = {}
            # ax.annotate(case.upper(), xy=(pos + step/2., 1.7),
            #             **gconfig['annotate'])
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
            # refvals = plots_dir[keyref]

            # x = get_values(refvals, header, gconfig['x_name'])
            # omp = get_values(refvals, header, gconfig['omp_name'])
            # y = get_values(refvals, header, gconfig['y_name'])
            # parts = get_values(refvals, header, 'ppb')
            # bunches = get_values(refvals, header, 'b')
            # turns = get_values(refvals, header, 't')
            # yref = parts * bunches * turns / y

            for idx, k in enumerate(plots_dir.keys()):
                values = plots_dir[k]
                mpiv = k.split('_mpi')[1].split('_')[0]
                lb = k.split('lb')[1].split('_')[0]
                lba = k.split('lba')[1].split('_')[0]
                approx = k.split('approx')[1].split('_')[0]
                # workers = k.split('_w')[1].split('_')[0]
                # nodes = k.split('_N')[1].split('_')[0]
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
                # label = None
                # label = '{}'.format(mpiv)

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
                speedup = y
                # speedup = y[0] / y
                # Now the reference, 1thread
                # yref = gconfig['reference'][case]['time']
                # partsref = gconfig['reference'][case]['ppb']
                # bunchesref = gconfig['reference'][case]['b']
                # turnsref = gconfig['reference'][case]['turns']
                # yref = partsref * bunchesref * turnsref / yref
                # ompref = gconfig['reference'][case]['omp']

                # speedup = y / yref
                x_new = []
                sp_new = []
                omp_new = []
                for i, xi in enumerate(gconfig['x_to_keep']):
                    x_new.append(xi)
                    if xi in x:
                        sp_new.append(speedup[list(x).index(xi)])
                        omp_new.append(omp[list(x).index(xi)])
                    else:
                        sp_new.append(0)
                        omp_new.append(0)
                x = np.array(x_new)
                omp = np.array(omp_new)
                speedup = np.array(sp_new)

                # efficiency = 100 * speedup / (x * omp / ompref)
                # x = x * omp
                speedup = speedup[0] / speedup

                width = .9 * step / (len(x))
                avg.append(speedup)
                # efficiency = 100 * speedup / x
                for ii, sp in enumerate(speedup):
                    # label = str(x[ii])
                    # xticks.append(label)
                    # xtickspos.append(pos + width*ii)
                    # if label in labels:
                    #     label = None
                    # else:
                    #     labels.add(label)
                    plt.bar(pos + width*ii, sp, width=0.9*width,
                            edgecolor='0.', label=None, hatch=gconfig['hatches'][ii],
                            color=gconfig['colors'][ii])
                # ax2.bar(np.arange(len(x)) + pos + width, efficiency, width=0.9*width,
                #         edgecolor='0.', label=label, hatch=gconfig['hatches'][idx],
                #         color=colors2[idx])
                # if True or idx != 1:
                #     for i, s, e in zip(np.arange(len(x)) + pos, speedup, efficiency):
                #         ax.annotate('{:.1f}'.format(s), xy=(i, s),
                #                     **gconfig['annotate'])
                #         ax2.annotate('{:.0f}'.format(e), xy=(i+width, e),
                #                      **gconfig['annotate'])
            pos += step
            # pos += width * step

            # ax.tick_params(axis='y', color=colors1[0])

            # I plot the averages here

        # for idx, key in enumerate(avg.keys()):
        vals = np.mean(avg, axis=0)
        for idx, val in enumerate(vals):
            # xticks.append(x[idx])
            # xtickspos.append(pos + idx*width)
            plt.bar(pos + idx*width, val, width=0.9*width,
                    edgecolor='0.', label=str(20//x[idx]), hatch=gconfig['hatches'][idx],
                    color=gconfig['colors'][idx])
            text = '{:.2f}'.format(val)
            if idx == 0:
                text = ''
            else:
                text = text[1:]
            ax.annotate(text, xy=(pos + idx*width, val),
                        **gconfig['annotate'])

        plt.ylim(gconfig['ylim'])
        pos += step
        plt.xlim(0-step/6, pos-step/7)
        # plt.xticks(np.arange(len(x)) + width/2, np.array(x/omp, int))
        plt.xticks(np.arange(pos) + step/2,
                   [c.upper() for c in args.cases] + ['AVG'], **gconfig['xticks'])
        # plt.xticks(xtickspos, xticks, **gconfig['xticks'])

        plt.legend(**gconfig['legend'])
        # plt.xticks(**gconfig['ticks'])
        plt.yticks(gconfig['yticks'], **gconfig['ticks'])
        ax.tick_params(**gconfig['tick_params'])

        plt.tight_layout()
        plt.subplots_adjust(**gconfig['subplots_adjust'])
        for file in gconfig['outfiles']:
            file = file.format(images_dir, title, '-'.join(args.cases),
                               args.keysuffix)
            save_and_crop(fig, file, dpi=600, bbox_inches='tight')
        if args.show:
            plt.show()
        plt.close()
