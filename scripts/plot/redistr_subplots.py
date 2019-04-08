#!/usr/bin/python
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import os
import matplotlib.patches as mpatches
import sys
from plot.plotting_utilities import *
from cycler import cycle

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"
this_filename = sys.argv[0].split('/')[-1]

project_dir = this_directory + '../../'
res_dir = project_dir + 'results/'
images_dir = res_dir + 'plots/'

if not os.path.exists(images_dir):
    os.makedirs(images_dir)

conf = {
    'files': {
        res_dir+'raw/LHC-lb-mpich3-approx2/particles-report.csv': {
            'key': 'lhc-mpich3-approx2',
            'dic_rows': ['n'],
            'dic_cols': ['turn_num', 'parts_avg', 'parts_min',
                         'parts_max', 'tcomp_avg', 'tcomp_min',
                         'tcomp_max', 'tcomm_avg', 'tcomm_min',
                         'tcomm_max', 'tconst_avg', 'tconst_min',
                         'tconst_max', 'tpp_avg',
                         'tpp_min', 'tpp_max'],
        },
    },
    'subplots_args': {
        'nrows': 3,
        'ncols': 2
    },
    'subplots': [
        {'title': 'tconst',
            'x_name': 'turn_num',
            'y_name': ['tconst_avg'],
            'y_min_name': ['tconst_min'],
            'y_max_name': ['tconst_max']
         },
        {'title': 'tcomp',
            'x_name': 'turn_num',
            'y_name': ['tcomp_avg'],
            'y_min_name': ['tcomp_min'],
            'y_max_name': ['tcomp_max']
         },
        {'title': 'tcomm',
            'x_name': 'turn_num',
            'y_name': ['tcomm_avg'],
            'y_min_name': ['tcomm_min'],
            'y_max_name': ['tcomm_max']
         },
        {'title': 'ttotal',
            'x_name': 'turn_num',
            'y_name': ['tconst_avg', 'tcomm_avg', 'tcomp_avg'],
            'y_min_name': ['tconst_min', 'tcomm_min', 'tcomp_min'],
            'y_max_name': ['tconst_max', 'tcomm_max', 'tcomp_max']
         },
        {'title': 'parts',
            'x_name': 'turn_num',
            'y_name': ['parts_avg'],
            'y_min_name': ['parts_min'],
            'y_max_name': ['parts_max']
         },
        {'title': 'tpp',
            'x_name': 'turn_num',
            'y_name': ['tpp_avg'],
            'y_min_name': ['tpp_min'],
            'y_max_name': ['tpp_max']
         }
    ],
    'colors': {'2': 'xkcd:blue',
               '4': 'xkcd:orange',
               '8': 'xkcd:green',
               '12': 'xkcd:red',
               '16': 'xkcd:purple'
               },
    'legend': {
        'loc': 'upper left', 'ncol': 2, 'handlelength': 0.5, 'fancybox': True,
        'framealpha': 0.3, 'fontsize': 8, 'labelspacing': 0, 'borderpad': 0.5,
        'handletextpad': 0.5, 'borderaxespad': 0
    },
    'subplots_adjust': {
        'wspace': 0.16, 'hspace': 0.16, 'top': 0.91
    },
    'tick_params': {
        'pad': 1, 'top': 1, 'bottom': 1, 'left': 1,
        'direction': 'inout', 'length': 3, 'width': 0.5,
    },
    'figsize': (5, 3),
    'outfiles': [images_dir + 'lhc-lb-mpich-subplots.pdf'],
}

if __name__ == '__main__':
    plots_dir = {}
    for file, fconfig in conf['files'].items():
        print(file)
        data = np.genfromtxt(file, delimiter='\t', dtype=str)
        header, data = list(data[0]), data[1:]
        temp = dictify(
            header, data, fconfig['dic_rows'], fconfig['dic_cols'])
        plots_dir.update(temp)

    fig, ax_arr = plt.subplots(**conf['subplots_args'], sharex=True)
    fig.suptitle('')
    i = 0
    for pltconf in conf['subplots']:
        ax = ax_arr[i//conf['subplots_args']['ncols'],
                    i % conf['subplots_args']['ncols']]
        plt.sca(ax)
        plt.title(pltconf['title'], fontsize=7)
        for nw, data in plots_dir.items():
            x = np.array(data[pltconf['x_name']], float)
            y = np.sum([np.array(data[y_name], float)
                        for y_name in pltconf['y_name']], axis=0)
            ymin = np.sum([np.array(data[y_name], float)
                           for y_name in pltconf['y_min_name']], axis=0)
            ymax = np.sum([np.array(data[y_name], float)
                           for y_name in pltconf['y_max_name']], axis=0)
            if pltconf['title'] in ['tconst', 'tcomm', 'tcomp', 'ttotal', 'tpp']:
                y = np.cumsum(y) / x / (y[0]/x[0])
                ymin = np.cumsum(ymin) / x / (ymin[0]/x[0])
                ymax = np.cumsum(ymax) / x / (ymax[0]/x[0])
                plt.axhline(1, color='k', ls='dotted', lw=1)
            plt.errorbar(x, y, yerr=[ymin, ymax], label='{}'.format(nw),
                         lw=1, capsize=1, color=conf['colors'][nw])
        ax.tick_params(**conf['tick_params'])
        plt.legend(**conf['legend'])
        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)
        plt.tight_layout()
        i += 1

    plt.subplots_adjust(**conf['subplots_adjust'])
    for outfile in conf['outfiles']:
        save_and_crop(fig, outfile, dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()
