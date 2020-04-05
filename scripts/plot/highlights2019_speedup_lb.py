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
                    default=['sps', 'lhc', 'ps'],
                    choices=['lhc', 'sps', 'ps', 'ex01'],
                    help='The test-case to plot.')

parser.add_argument('-s', '--show', action='store_true',
                    help='Show the plots.')

args = parser.parse_args()

project_dir = this_directory + '../../'
res_dir = project_dir + 'results/'
images_dir = res_dir + 'plots/redistribute/'
axis_color = '0.9'

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
    'hatches': ['', '//', '--', 'xx'],
    'reference': {
        'sps': {'ppb': 4000000, 'b': 72, 'turns': 1000, 'w': 1,
                'omp': 20, 'time': 225.85},

        'lhc': {'ppb': 2000000, 'b': 96, 'turns': 1000, 'w': 1,
                'omp': 20, 'time': 103.04},

        'ps': {'ppb': 8000000, 'b': 21, 'turns': 1000, 'w': 1,
               'omp': 20, 'time': 96.0},
    },
    'colors': {
        'lhc': 'xkcd:kelly green',
        'sps': 'xkcd:sky blue',
        'ps': 'xkcd:orange',
    },
    'labels': {
        'lhc': '192M Prtcls\nLHC',
        'sps': '288M Prtcls\nSPS',
        'ps': '168M Prtcls\nPS',
    },
    'hatches': {
        'lhc': '',
        'sps': '',
        'ps': '',
    },
    'x_name': 'n',
    'x_to_keep': [4, 8, 12, 16],
    'omp_name': 'omp',
    'y_name': 'avg_time(sec)',
    # 'y_err_name': 'std',
    'xlabel': 'Nodes (x20 Cores)',
    'ylabel': 'Speedup',
    # 'ylabel2': 'Efficiency',
    'title': {
        # 's': '{}'.format(case.upper()),
        'fontsize': 10,
        # 'y': 0.74,
        # 'x': 0.55,
        'fontweight': 'bold',
        'textcoords': 'data',
        'va': 'top',
        'ha': 'center'
    },
    'figsize': [5, 3.3],
    'annotate': {
        'fontsize': 11,
        'fontweight': 'bold',
        'textcoords': 'data',
        'va': 'bottom',
        'ha': 'center'
    },
    'ticks': {'fontsize': 10, 'color': axis_color},
    'fontsize': 10,
    'legend': {
        'loc': 'upper left', 'ncol': 4, 'handlelength': 1.5, 'fancybox': True,
        'framealpha': 0., 'fontsize': 10, 'labelspacing': 0, 'borderpad': 0.5,
        'handletextpad': 0.5, 'borderaxespad': 0, 'columnspacing': 0.8,
        'bbox_to_anchor': (0, 1.25)
    },
    'subplots_adjust': {
        'wspace': 0.0, 'hspace': 0.1, 'top': 0.93
    },
    'tick_params': {
        'pad': 1, 'top': 0, 'bottom': 1, 'left': 1,
        'direction': 'inout', 'length': 5, 'width': 0.5,
        'color': axis_color,
    },
    'ylim': [0, 8],
    # 'ylim2': [40, 120],
    'yticks': [0, 2, 4, 6, 8],
    # 'yticks2': [40, 60, 80, 100],
    'outfiles': [
                 # '{}/highlights2019-speedup-trans.pdf',
                 '{}/highlights2019-speedup-trans.png',
                 '{}/highlights2019-speedup-trans.jpg']
}


lconfig = {
    'figures': {
        'DLB': {
            'files': [
                # '{}/raw/{}/mvapich2/comm-comp-report.csv',
                # '{}/raw/{}/lb-mvapich2/comm-comp-report.csv',
                # '{}/raw/{}/lb-tp-approx0-mvapich2/comm-comp-report.csv',
                # '{}/raw/{}/lb-approx2-mvapich2/comm-comp-report.csv',
                '{}/raw/{}/lb-tp-mvapich2/comm-comp-report.csv',
            ],
            'lines': {
                'mpi': ['mpich3', 'mvapich2'],
                'lb': ['interval', 'reportonly'],
                'approx': ['0', '2'],
                'lba': ['500'],
                'b': ['96', '72', '21'],
                't': ['5000'],
                'type': ['total'],
            }
        },
    },

}


if __name__ == '__main__':
    for title, figconf in lconfig['figures'].items():

        fig, ax_arr = plt.subplots(ncols=1, nrows=1,
                                   sharex=True, sharey=True,
                                   figsize=gconfig['figsize'])
        ax = ax_arr
        plt.sca(ax)
        plt.xlabel(gconfig['xlabel'], labelpad=3,
                   fontweight='bold', color=axis_color,
                   fontsize=gconfig['fontsize'])
        plt.ylabel(gconfig['ylabel'], labelpad=3,
                   fontweight='bold', color=axis_color,
                   fontsize=gconfig['fontsize'])
        # plt.setp(ax.get_yticklabels(), color="xkcd:green")
        pos = 0
        step = 1
        width = 0.95 * step
        xtickspos = []
        xtickslbl = []
        plt.title('Distributed BLonD VS Prior State-Of-Art',
                  fontsize=10, fontweight='bold', fontname='calibri',
                  color=axis_color)
        for col, case in enumerate(args.cases):
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
                    if 'tp-' in file:
                        plots_dir['_{}_tp1'.format(key)] = temp[key].copy()
                    else:
                        plots_dir['_{}_tp0'.format(key)] = temp[key].copy()

            for idx, k in enumerate(plots_dir.keys()):
                values = plots_dir[k]
                mpiv = k.split('_mpi')[1].split('_')[0]
                lb = k.split('lb')[1].split('_')[0]
                lba = k.split('lba')[1].split('_')[0]
                approx = k.split('approx')[1].split('_')[0]
                tp = k.split('tp')[1].split('_')[0]
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

                label = '{}-{}-{}'.format(lb, tp, approx)

                x = get_values(values, header, gconfig['x_name'])
                omp = get_values(values, header, gconfig['omp_name'])
                y = get_values(values, header, gconfig['y_name'])
                parts = get_values(values, header, 'ppb')
                bunches = get_values(values, header, 'b')
                turns = get_values(values, header, 't')

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
                efficiency = 100 * speedup / (x * omp[0] / ompref)
                x = x * omp[0]

                # efficiency = 100 * speedup / x
                plt.bar(np.arange(len(x)) + pos, speedup, width=width,
                        edgecolor='0', label=label, hatch=gconfig['hatches'][case],
                        color=gconfig['colors'][case],
                        linewidth='1.5', alpha=1)
                xtickspos += list(np.arange(len(x)) + pos)
                xtickslbl += list(np.array(x, int)//20)
                # ax2.bar(np.arange(len(x)) + pos + width, efficiency, width=0.9*width,
                #         edgecolor='0.', label=label, hatch=gconfig['hatches'][idx],
                #         color=colors2[idx])
                if idx != 1:
                    # for i, s in zip(np.arange(len(x))[-1] + pos, speedup[-1]):
                    i, s = np.arange(len(x))[-1] + pos, speedup[-1]
                    ax.annotate('{:.1f}'.format(s), xy=(i, s),
                                color=gconfig['colors'][case],
                                **gconfig['annotate'])
                # pos += width
            # plt.title('{}'.format(case.upper()), **gconfig['title'])
            ax.annotate(gconfig['labels'][case], xy=(pos+0.9*step, 7.9),
                        color=gconfig['colors'][case],
                        **gconfig['title'])
            pos += len(x) + step/2

        plt.ylim(gconfig['ylim'])
        plt.xlim(0-width, xtickspos[-1]+width)
        plt.xticks(xtickspos, xtickslbl, **gconfig['ticks'])
        plt.yticks(gconfig['yticks'], **gconfig['ticks'])
        ax.tick_params(**gconfig['tick_params'])
        ax.spines['bottom'].set_color(axis_color)
        ax.spines['top'].set_color(axis_color) 
        ax.spines['right'].set_color(axis_color)
        ax.spines['left'].set_color(axis_color)
        # ax.set_facecolor('xkcd:grey')
        # handles = []
        # handles.append(mpatches.Patch(label='Speedup', edgecolor='black',
        #                               facecolor=colors1[0]))

        # plt.legend(handles=handles, **gconfig['legend'])

        plt.tight_layout()
        plt.subplots_adjust(**gconfig['subplots_adjust'])
        for file in gconfig['outfiles']:
            file = file.format(images_dir)
            save_and_crop(fig, file, dpi=600, bbox_inches='tight',
                          transparent=True)
        if args.show:
            plt.show()
        plt.close()
