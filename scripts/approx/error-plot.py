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


parser = argparse.ArgumentParser(
    description='Plot the error in the histogram.')


parser.add_argument('-i', '--indir', type=str, default=None,
                    help='Input directory name.')

parser.add_argument('-o', '--outdir', type=str, default=None,
                    help='Directory to store the results.')

parser.add_argument('-ymin', '--ymin', type=float, default=None,
                    help='Min value for y axis.')

parser.add_argument('-ymax', '--ymax', type=float, default=None,
                    help='Max value for y axis.')

parser.add_argument('-reduce', '--reduce', type=int, default=[], nargs='+',
                    help='Max value for y axis.')

parser.add_argument('-b', '--bunch', type=str, default=['1'], nargs='+',
                    help='Plot only the lines with so many bunches.')


parser.add_argument('-points', '--points', type=int, default=100,
                    help='Num of points in the plot.')


errors = ['profile', 'mean_dt', 'mean_dE', 'std_dE', 'std_dt']

if __name__ == '__main__':
    args = parser.parse_args()
    args = vars(args)

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
            outfiles = [outdir + '/ts' + ts + '/' + error + '_' + file.split('.h5')[0] + '.pdf',
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
                intv = int(np.ceil(len(x)/points))

                for k, v in plt_data.items():
                    #  x = np.arange(len(v))
                    y = v / avg_base_error
                    err = y * sem_base_error
                    # plt.errorbar(x, y, yerr=err, label=k, linestyle='',
                    #              marker=marker, markersize=5, color=next(colors))
                    plt.errorbar(x[::intv], y[::intv], yerr=None, label=k, linestyle='',
                                 marker=marker, markersize=4, color=next(colors))
                    lines += 1

            plt.legend(loc='upper left', fancybox=True, fontsize=10,
                           ncol=(lines+1)//2, columnspacing=1,
                           labelspacing=0.1, borderpad=0.2, framealpha=0.5,
                           handletextpad=0.2, handlelength=1.5, borderaxespad=0)

            plt.tight_layout()
            for outfile in outfiles:
                save_and_crop(fig, outfile, dpi=900, bbox_inches='tight')
            # plt.show()
            plt.close()
        inh5file.close()
