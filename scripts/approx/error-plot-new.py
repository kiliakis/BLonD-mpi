import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
import h5py
import argparse
import sys
import os
import matplotlib.lines as mlines
from plot.plotting_utilities import *
from scipy import stats
from cycler import cycle
import bisect

parser = argparse.ArgumentParser(
    description='Approximation error plots.')


parser.add_argument('-i', '--indir', type=str, default=None,
                    help='Input directory name. Contains .h5 files only, one for' + 
                    ' each TS interval.')

parser.add_argument('-o', '--outdir', type=str, default=None,
                    help='Directory to store the results.')

# parser.add_argument('-ymin', '--ymin', type=float, default=None,
#                     help='Min value for y axis.')

# parser.add_argument('-ymax', '--ymax', type=float, default=None,
#                     help='Max value for y axis.')

parser.add_argument('-reduce', '--reduce', type=int, default=[], nargs='+',
                    help='Plot lines for these reduce intervals. \n' +
                    'Default: Use all the available')

parser.add_argument('-b', '--bunch', type=str, default=['1'], nargs='+',
                    help='Plot only the lines with so many bunches.\n' + 
                    'Default: 1')

parser.add_argument('-t', '--turns', type=int, default=None,
                    help='Last turn to plot (default: plot all the turns).')

parser.add_argument('-s', '--show', action='store_true',
                    help='Show the plot or save only. Default: save only')


parser.add_argument('-p', '--points', type=int, default=100,
                    help='Num of points in the plot. Default: 100')


errors = ['profile', 'mean_dt', 'mean_dE', 'std_dE',
          'std_dt', 'bunch_position', 'bunch_length']

if __name__ == '__main__':
    args = parser.parse_args()
    args = vars(args)

    last_t = args['turns']
    indir = args['indir']
    outdir = args['outdir']
    points = args['points']
    bunches = args['bunch']
    bunches = ['bunch_{}'.format(b) for b in bunches]
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for file in os.listdir(indir):
        fullfile = indir + '/' + file
        ts = file.split('ts')[1].split('.h5')[0]
        inh5file = h5py.File(fullfile, 'r')

        for error in errors:
            outfiles = [#outdir + '/ts' + ts + '/' + error + '_' + file.split('.h5')[0] + '.pdf',
                        outdir + '/ts' + ts + '/' + error + '_' + file.split('.h5')[0] + '.jpeg']

            if not os.path.exists(outdir + '/ts' + ts + '/'):
                os.makedirs(outdir + '/ts' + ts + '/')

            fig = plt.figure(figsize=(6, 4))
            lines = 0
            plt.grid()
            if args.get('ymin', None):
                plt.ylim(ymin=args['ymin'])
            if args.get('ymax', None):
                plt.ylim(ymax=args['ymax'])

            plt.title('Ts: {}, Error: {}'.format(ts, error))
            plt.xlabel('#Turn')
            plt.ylabel('Relative error')

            markers = cycle(['+', 'x', 'v'])

            for bunchkey in inh5file.keys():
                if bunchkey not in bunches:
                    continue
                inh5 = inh5file[bunchkey]
                plt_data = {}
                marker = next(markers)
                colors = cycle(['tab:blue', 'tab:orange', 'tab:green'])

                for i in range(len(inh5[error])):
                    if inh5['reduce'][i][0] == 1 and inh5['reduce'][i][1] == 1:
                        if 'base_error' not in plt_data:
                            plt_data['base_error'] = []
                        plt_data['base_error'].append(
                            inh5[error][i])

                        x = inh5['turns'][i]
                    elif inh5['reduce'][i][1] != 1 and \
                            (len(args['reduce']) == 0 or
                                inh5['reduce'][i][1] in args['reduce']):
                        key = '{}-r_{}'.format(bunchkey, inh5['reduce'][i][1])
                        plt_data[key] = inh5[error][i]
                        x = inh5['turns'][i]

                avg_base_error = np.mean(plt_data['base_error'], axis=0)
                sem_base_error = stats.sem(plt_data['base_error'], axis=0)
                sem_base_error = np.abs(sem_base_error / avg_base_error)

                del plt_data['base_error']

                # print('Base error std', 100 * std_base_error/ avg_base_error)
                if last_t is None:
                    idx = len(x)
                else:
                    idx = bisect.bisect(x, last_t)
                x = x[:idx]

                intv = int(np.ceil(len(x)/points))
                displ_step = (x[1] - x[0]) * intv/2. 
                avg_base_error = avg_base_error[:idx]
                avg_base_error = avg_base_error[::intv]
                displs = 0.

                for k, v in plt_data.items():
                    #  x = np.arange(len(v))
                    y = v[:idx]
                    # err = y * sem_base_error
                    plt.errorbar(x[::intv]+displs, y[::intv],
                                 yerr=[np.minimum(avg_base_error, y[::intv]),
                                       avg_base_error],
                                 label=k, linestyle='',
                                 marker=marker, markersize=3, color=next(colors),
                                 capsize=2, elinewidth=1)
                    displs += displ_step
                    lines += 1

            plt.legend(loc='upper left', fancybox=True, fontsize=10,
                           ncol=(lines+1)//2, columnspacing=1,
                           labelspacing=0.1, borderpad=0.2, framealpha=0.5,
                           handletextpad=0.2, handlelength=1.5, borderaxespad=0)

            plt.tight_layout()
            for outfile in outfiles:
                save_and_crop(fig, outfile, dpi=900, bbox_inches='tight')
            if args['show'] is True:
                plt.show()
            plt.close()
        inh5file.close()
