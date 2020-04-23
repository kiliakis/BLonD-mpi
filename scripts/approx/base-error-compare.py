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


parser.add_argument('-i', '--infiles', type=str, default=[], nargs='+',
                    help='Input file names.')

#  parser.add_argument('-s', '--slices', type=int, default=[], nargs='+',
#                      help='Slices used in each input file.')

parser.add_argument('-n', '--names', type=str, default=[], nargs='+',
                    help='Names for the plot lines.')

parser.add_argument('-o', '--outdir', type=str, default=None,
                    help='Directory to store the results.')

parser.add_argument('-ymin', '--ymin', type=float, default=None,
                    help='Min value for y axis.')

parser.add_argument('-ymax', '--ymax', type=float, default=None,
                    help='Max value for y axis.')

parser.add_argument('-t', '--ts', type=str, default=['1'], nargs='+',
                    help='Ts values for which the error will be plotted.')

parser.add_argument('-reduce', '--reduce', type=int, default=1,
                    help='Reduce value for which the error will be calculated.')

parser.add_argument('-points', '--points', type=int, default=1000,
                    help='Number of points in the plot.')

errors = ['n_macroparticles', 'mean_dt', 'mean_dE', 'std_dE', 'std_dt']


config = {
    'LHC': {
        'slices': 144,
        'dE_ref': 450.e9,
        'dt_ref': 2.5e-9,
        'n_macroparticles': 2e6
    },
    'SPS': {
        'slices': 10496,
        'dE_ref': 25.9e9,
        'dt_ref': 5.e-9,
        'n_macroparticles': 2e6

    }
}

if __name__ == '__main__':
    args = parser.parse_args()
    args = vars(args)

    infiles = args['infiles']
    outdir = args['outdir']
    tss = args['ts']
    names = cycle(args['names'])
    points = int(args['points'])
    #  slicess = cycle(args['slices'])
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for ts in tss:
        for error in errors:
            outfiles = [outdir + '/ts' + ts + '/' +
                        error + '_' + error + '.jpeg']

            fig = plt.figure(figsize=(6, 4))
            plt.yscale('log')
            lines = 0
            plt.grid()
            if args.get('ymin', None):
                plt.ylim(ymin=args['ymin'])
            if args.get('ymax', None):
                plt.ylim(ymax=args['ymax'])

            plt.title('Ts: {}, Error: {}'.format(ts, error))
            plt.xlabel('#Turn')
            plt.ylabel('Absolute error value')

            markers = cycle(['+', 'x', 'v'])

            colors = cycle(['tab:blue', 'tab:orange', 'tab:green'])

            for file in infiles:
                # fullfile = indir + '/' + file
                name = next(names)

                ts_file = file.split('ts')[1].split('.h5')[0]
                if ts_file != ts:
                    continue

                if not os.path.exists(outdir + '/ts' + ts + '/'):
                    os.makedirs(outdir + '/ts' + ts + '/')

                inh5file = h5py.File(file, 'r')

                for bunchkey in ['bunch_1']:
                    inh5 = inh5file[bunchkey]
                    plt_data = {}
                    marker = next(markers)

                    for i in range(len(inh5[error])):
                        if inh5['reduce'][i][0] == 1 and inh5['reduce'][i][1] == 1:
                            if 'base_error' not in plt_data:
                                plt_data['base_error'] = []
                            plt_data['base_error'].append(
                                inh5[error][i])
                            x = inh5['turns'][i]
                        else:
                            continue

                    avg_base_error = np.mean(plt_data['base_error'], axis=0)
                    sem_base_error = stats.sem(plt_data['base_error'], axis=0)

                    if 'dt' in error:
                        avg_base_error /= (config[name]['dt_ref'])**2
                        sem_base_error /= (config[name]['dt_ref'])**2
                    if 'dE' in error:
                        avg_base_error /= (config[name]['dE_ref')**2
                        sem_base_error /= (config[name]['dE_ref')**2
                    if 'n_macroparticles' in error:
                        avg_base_error /= (config[name]['slices']
                                           * config[name]['n_macroparticles'])**2
                        sem_base_error /= (config[name]['slices']
                                           * config[name]['n_macroparticles'])**2

                    intv = (len(x) + points - 1) // points
                    plt.errorbar(x[::intv], avg_base_error[::intv], yerr=sem_base_error[::intv],
                                 label=name, linestyle='', marker=marker,
                                 markersize=4, color=next(colors))
                    annotate_max(plt.gca(), x, avg_base_error,
                                 size='medium', ha='right', clip_on=True)

                    lines += 1

                inh5file.close()

            plt.legend(loc='upper left', fancybox=True, fontsize=10,
                           ncol=(lines+1)//2, columnspacing=1,
                           labelspacing=0.1, borderpad=0.2, framealpha=0.5,
                           handletextpad=0.2, handlelength=1.5, borderaxespad=0)

            plt.tight_layout()
            for outfile in outfiles:
                save_and_crop(fig, outfile, dpi=600, bbox_inches='tight')
            plt.show()
            plt.close()
