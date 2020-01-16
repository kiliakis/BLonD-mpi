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

parser.add_argument('-k', '--keysuffix', type=str, default='weak',
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
    'hatches': ['', '', 'xx'],
    'markers': ['x', 'o', '^'],
    # 'colors': ['0.', '0.4', '0.65'],
    'colors': ['xkcd:red', 'xkcd:green', 'xkcd:blue'],

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
    'ylabel': 'Norm. Throughput',
    'ylabel2': 'Efficiency',
    'title': {
                # 's': '{}'.format(case.upper()),
                'fontsize': 10,
                'y': 0.83,
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
        'loc': 'lower left', 'ncol': 1, 'handlelength': 1., 'fancybox': True,
        'framealpha': 0., 'fontsize': 10, 'labelspacing': 0, 'borderpad': 0.5,
        'handletextpad': 0.5, 'borderaxespad': 0, 'columnspacing': 0.8,
        # 'bbox_to_anchor': (0, 1.25)
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

    'ylim': [0., 1.1],
    # 'ylim2': [0, 110],
    'yticks': [0, 0.2, 0.4, 0.6, 0.8, 1.0],
    # 'yticks2': [0, 20, 40, 60, 80, 100],
    'outfiles': ['{}/{}-{}-normthroughput-{}.pdf',
                 '{}/{}-{}-normthroughput-{}.jpg']
}


lconfig = {
    'figures': {

        'weak': {
            'files': [
                '{}/compfront/{}/lb-tp-approx0-mvapich2-weak-scaling/comm-comp-report.csv',
                '{}/compfront/{}/lb-tp-approx2-mvapich2-weak-scaling/comm-comp-report.csv',
                '{}/compfront/{}/lb-tp-approx1-mvapich2-weak-scaling/comm-comp-report.csv',
            ],
            'lines': {
                'mpi': ['mpich3', 'mvapich2', 'openmpi3'],
                'lb': ['interval', 'reportonly'],
                'approx': ['0', '1', '2'],
                'lba': ['500'],
                # 'b': ['6', '12', '24', '96', '192',
                #       '48', '21', '9', '18', '36',
                #       '72', '144', '288'],
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
                    # if 'tp-' in file:
                    #     plots_dir['_{}_tp1'.format(key)] = temp[key].copy()
                    # else:
                    #     plots_dir['_{}_tp0'.format(key)] = temp[key].copy()
            # fig = plt.figure(figsize=config['figsize'])

            plt.grid(True, which='major', alpha=0.5)
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

            keyref = ''
            for k in plots_dir.keys():
                if 'approx0' in k:
                    keyref = k
                    break
            if keyref == '':
                print('ERROR: reference key not found')
                exit(-1)
            refvals = plots_dir[keyref]
            x = get_values(refvals, header, gconfig['x_name'])
            omp = get_values(refvals, header, gconfig['omp_name'])
            y = get_values(refvals, header, gconfig['y_name'])
            parts = get_values(refvals, header, 'ppb')
            bunches = get_values(refvals, header, 'b')
            turns = get_values(refvals, header, 't')
            # This the reference throughput per node
            yref = parts * bunches * turns / y
            yref /= (x * omp // 20)
            yref = yref[list(x).index(4)]

            # plt.sca(ax2)
            # if col == 1:
            #     plt.ylabel(gconfig['ylabel2'], labelpad=3,
            #                fontweight='bold', color='xkcd:blue',
            #                fontsize=gconfig['fontsize'])
            # plt.setp(ax2.get_yticklabels(), color="xkcd:blue")
            # plt.yticks(gconfig['yticks2'], **gconfig['ticks'])
            # plt.ylim(gconfig['ylim2'])
            # plt.sca(ax)

            pos = 0
            step = 0.1
            width = 1. / (1*len(plots_dir.keys())+0.4)

            # colors = [cm.Greens(x) for x in np.linspace(0.2, 0.8, len(plots_dir))]
            # colors1 = [cm.Greens(x)
            #            for x in np.linspace(0.5, 0.8, len(plots_dir))]
            # colors2 = [cm.Blues(x)
            #            for x in np.linspace(0.5, 0.8, len(plots_dir))]
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
                # if approx == '2':
                #     approx = 'AC'
                # else:
                #     approx = 'NoAC'
                # key = '{}-{}-{}'.format(case, mpiv, lb)

                # label = '{}-{}-{}-{}'.format(lb, tp, approx, experiment)
                label = '{}'.format(approx)
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

                # This is the throughput per node
                y = parts * bunches * turns / y
                y /= (x * omp//20)

                # Now the reference, 1thread
                # yref = gconfig['reference'][case]['time']
                # partsref = gconfig['reference'][case]['ppb']
                # bunchesref = gconfig['reference'][case]['b']
                # turnsref = gconfig['reference'][case]['turns']
                # yref = partsref * bunchesref * turnsref / yref
                # ompref = gconfig['reference'][case]['omp']

                speedup = y
                x_new = []
                sp_new = []
                for i, xi in enumerate(gconfig['x_to_keep']):
                    x_new.append(xi)
                    if xi in x:
                        sp_new.append(speedup[list(x).index(xi)])
                    else:
                        sp_new.append(0)
                x = np.array(x_new)
                speedup = np.array(sp_new)
                # efficiency = 100 * speedup / (x * omp[0] / ompref)
                x = x * omp[0]
                # speedup = speedup / yref
                speedup = speedup / speedup[0]

                # efficiency = 100 * speedup / x
                # plt.bar(np.arange(len(x)) + pos, speedup, width=0.9*width,
                #         edgecolor='0.', label=label, hatch=gconfig['hatches'][idx],
                #         color=gconfig['colors'][idx])
                plt.plot(np.arange(len(x)), speedup,
                         label=label, marker=gconfig['markers'][idx],
                         color=gconfig['colors'][idx])
                print("{}:{}:".format(case, label), speedup)
                # ax2.bar(np.arange(len(x)) + pos + width, efficiency, width=0.9*width,
                #         edgecolor='0.', label=label, hatch=gconfig['hatches'][idx],
                #         color=colors2[idx])
                # if True or idx != 1:
                #     for i, s, e in zip(np.arange(len(x)) + pos, speedup, efficiency):
                #         ax.annotate('{:.1f}'.format(s), xy=(i, s),
                #                     **gconfig['annotate'])
                #         # ax2.annotate('{:.0f}'.format(e), xy=(i+width, e),
                #         #              **gconfig['annotate'])
                pos += 1 * width
            # pos += width * step
            plt.ylim(gconfig['ylim'])
            plt.xlim(0-.8*width, len(x)-1.5*width)
            plt.xticks(np.arange(len(x)), np.array(x, int)//20)
            if col == 0:
                ax.tick_params(**gconfig['tick_params_left'])
            else:
                ax.tick_params(**gconfig['tick_params_center_right'])

            plt.legend(**gconfig['legend'])

            plt.xticks(**gconfig['ticks'])
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
