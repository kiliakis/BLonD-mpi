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

# Same as the approx2-raw but reads from different file organization. 

parser = argparse.ArgumentParser(
    description='Evaluate approx2 raw data.')


parser.add_argument('-i', '--indir', type=str, default=None,
                    help='Input directory, contains only .h5 files.')

parser.add_argument('-o', '--outdir', type=str, default=None,
                    help='Directory to store the results.')

# parser.add_argument('-ymin', '--ymin', type=float, default=None,
#                     help='Min value for y axis.')

# parser.add_argument('-ymax', '--ymax', type=float, default=None,
#                     help='Max value for y axis.')

# parser.add_argument('-reduce', '--reduce', type=int, default=[], nargs='+',
#                     help='Plot lines for these reduce intervals. \n' +
#                     'Default: Use all the available')

parser.add_argument('-turns', '--turns', type=int, default=None,
                    help='Last turn to plot (default: plot all the turns).')

parser.add_argument('-p', '--points', type=int, default=100,
                    help='Num of points in the plot. Default: 100')

parser.add_argument('-t', '--ts', type=str, default=['1'], nargs='+',
                    help='Running mean window. Default: [1]')

parser.add_argument('-s', '--show', action='store_true',
                    help='Show the plot or save only. Default: save only')


# errors = ['n_particles', 'profile', 'profile', 'mean_dt', 'mean_dE', 'std_dE',
#           'std_dt', 'bunch_position', 'bunch_length']

# to_plot = []

not_plot = ['dE_norm', 'dt_norm', 'turns']

markers = {'r1': '+',
           'r2': 's',
           'r3': 'v'}


def running_mean(x, N, axis=None):
    cumsum = np.cumsum(np.insert(x, 0, 0, axis=0), axis=0)
    return (cumsum[N:] - cumsum[:-N])/N


if __name__ == '__main__':
    args = parser.parse_args()
    args = vars(args)

    last_t = args['turns']
    indir = args['indir']
    outdir = args['outdir']
    points = args['points']
    tss = args['ts']

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    plot_dir = {}
    for dirpath, dirnames, filenames in os.walk(indir):
        if 'monitor.h5' not in filenames:
            continue


        particles = dirpath.split('_p')[1].split('_')[0]
        bunches = dirpath.split('_b')[1].split('_')[0]
        seed = dirpath.split('_seed')[1].split('_')[0]
        monitor_intv = dirpath.split('_m')[1].split('_')[0]
        red = dirpath.split('_r')[1].split('_')[0]
        workers = dirpath.split('_w')[1].split('_')[0]

        fullfile = dirpath + '/monitor.h5'
        inh5file = h5py.File(fullfile, 'r')
        x = inh5file['default']['turns'].value
        # intv = int(np.ceil(len(x)/points))
        for key in inh5file['default'].keys():
            if (key in not_plot):
                continue
            val = inh5file['default'][key].value
            if (key == 'profile'):
                val = val[-1]
            val = val.reshape(len(val))
            if (key not in plot_dir):
                plot_dir[key] = {}
            if (workers not in plot_dir[key]):
                plot_dir[key][workers] = {'num': 0}
            if ('sum' not in plot_dir[key][workers]):
                plot_dir[key][workers]['sum'] = np.zeros_like(val)
                plot_dir[key][workers]['min'] = val
                plot_dir[key][workers]['max'] = val
            plot_dir[key][workers]['num'] += 1
            plot_dir[key][workers]['sum'] += val
            plot_dir[key][workers]['min'] = np.minimum(
                plot_dir[key][workers]['min'], val)
            plot_dir[key][workers]['max'] = np.maximum(
                plot_dir[key][workers]['max'], val)
            plot_dir[key][workers]['turns'] = x
        inh5file.close()

    # continue here, I need to iterate over the errors, create a figure for each
    # iterate over the reduce values, add an error plot line for each acording to the intv etc

    for ts in tss:
        # filename = outfile + '/ts' + str(ts) + '.h5'
        if not os.path.exists(os.path.dirname(outdir + '/ts' + ts+'/')):
            os.makedirs(os.path.dirname(outdir + '/ts' + ts+'/'))
        lines = 0
        for error in plot_dir.keys():
            fig = plt.figure(figsize=(4, 4))
            outfiles = [
                # '{}/ts{}/{}.pdf'.format(outdir, ts, error),
                '{}/ts{}/{}.jpeg'.format(outdir, ts, error)]

            plt.grid()
            if args.get('ymin', None):
                plt.ylim(ymin=args['ymin'])
            if args.get('ymax', None):
                plt.ylim(ymax=args['ymax'])

            plt.title('Ts: {}, Variable: {}'.format(ts, error))
            plt.xlabel('#Turn')
            plt.ylabel('Raw value')

            for workers, data in plot_dir[error].items():
                x = data['turns']
                y = data['sum'] / data['num']
                ymin = data['min']
                ymax = data['max']
                intv = int(np.ceil(len(x)/points))
                label = 'r{}'.format(workers)
                marker = markers.get('r{}'.format(workers), None)
                if (error == 'profile'):
                    # y = y[-1]
                    nonzero = np.flatnonzero(ymax)
                    y = y[nonzero[0]:nonzero[-1]]
                    ymin = ymin[nonzero[0]:nonzero[-1]]
                    ymax = ymax[nonzero[0]:nonzero[-1]]
                    x = np.arange(len(y))
                    plt.xlabel('#Bin')
                else:
                    y = running_mean(y, int(ts))
                    ymin = running_mean(ymin, int(ts))
                    ymax = running_mean(ymax, int(ts))
                    x = x[:len(y):intv]
                    y = y[::intv]
                    ymin = ymin[::intv]
                    ymax = ymax[::intv]
                if (workers == '1'):
                    plt.fill_between(
                        x, ymin, ymax, facecolor='0.6', interpolate=True)
                    plt.plot(x, ymax, color='black', linewidth=1)
                    plt.plot(x, ymin, color='black',  linewidth=1, label=label)
                else:
                    plt.errorbar(x, y,
                             yerr=[y-ymin, ymax-y],
                             label=label, linestyle='--',
                             marker=marker, markersize=0,
                             # color=next(colors),
                             # alpha=0.5,
                             capsize=1, elinewidth=1)
                lines += 1
            plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            plt.legend(loc='upper left', fancybox=True, fontsize=9,
                           ncol=(lines+2)//3, columnspacing=1,
                           labelspacing=0.1, borderpad=0.2, framealpha=0.5,
                           handletextpad=0.2, handlelength=1.5, borderaxespad=0)

            plt.tight_layout()
            for outfile in outfiles:
                save_and_crop(fig, outfile, dpi=900, bbox_inches='tight')
                # fig.savefig(outfile, dpi=900, bbox_inches='tight')
            if args['show'] is True:
                plt.show()
            plt.close()
