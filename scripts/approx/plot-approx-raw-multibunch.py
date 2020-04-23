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
    description='Evaluate approx raw data.')


parser.add_argument('-i', '--indir', type=str, default=None,
                    help='Input directory, contains only .h5 files.')

parser.add_argument('-o', '--outdir', type=str, default=None,
                    help='Directory to store the results.')

parser.add_argument('-turns', '--turns', type=int, default=None,
                    help='Last turn to plot (default: plot all the turns).')

parser.add_argument('-p', '--points', type=int, default=100,
                    help='Num of points in the plot. Default: 100')

parser.add_argument('-m', '--minparts', type=float, default=0.01,
                    help='Percent of particles in a bin to be considered a new bunch. Default: 0.01')

parser.add_argument('-t', '--ts', type=str, default=['1'], nargs='+',
                    help='Running mean window. Default: [1]')

parser.add_argument('-s', '--show', action='store_true',
                    help='Show the plot or save only. Default: save only')


# errors = ['n_particles', 'profile', 'profile', 'mean_dt', 'mean_dE', 'std_dE',
#           'std_dt', 'bunch_position', 'bunch_length']

# to_plot = []

not_plot = ['dE_norm', 'dt_norm', 'turns']

markers = {'apprx1_1': '+',
           'apprx1_2': 's',
           'r3': 'v'}


def running_mean(x, N, axis=None):
    cumsum = np.cumsum(np.insert(x, 0, 0, axis=0), axis=0)
    return (cumsum[N:] - cumsum[:-N])/N


def remove_nan(x):
    i = 0
    while np.isnan(x[i]):
        i += 1
    x[0:i] = x[i]
    val = x[i]
    while i < len(x):
        if np.isnan(x[i]):
            x[i] = val
        else:
            val = x[i]
        i += 1


def detect_bunches(y, min_parts):
    bunches = []
    phase = 'start'
    i = 0
    while i < len(y):
        # First we look for a slice with more that 1% parts for 10
        # slices in a row
        if phase == 'start':
            if (y[i] >= min_parts) and (np.min(y[i:i+10]) > min_parts):
                bunches.append([i, i])
                phase = 'end'
        # Finally we look for the end of the bunches, a region
        # with very few particles per slice for some slices in a row
        elif phase == 'end':
            if (y[i] < min_parts) and (np.max(y[i:i+10]) < min_parts):
                bunches[-1][1] = i
                phase = 'start'
        i += 1
    return bunches


if __name__ == '__main__':
    args = parser.parse_args()
    # args = vars(args)

    outdir = args.outdir

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    plot_dir = {}
    for dirpath, dirnames, filenames in os.walk(args.indir):
        if 'monitor.h5' not in filenames:
            continue

        particles = dirpath.split('_p')[1].split('_')[0]
        num_bunches = dirpath.split('_b')[1].split('_')[0]
        seed = dirpath.split('_seed')[1].split('_')[0]
        monitor_intv = dirpath.split('_m')[1].split('_')[0]
        red = dirpath.split('_r')[1].split('_')[0]
        workers = dirpath.split('_w')[1].split('_')[0]
        approx = dirpath.split('_approx')[1].split('_')[0]

        if approx == '1':
            approx_knob = red
        elif approx == '2':
            approx_knob = workers
        else:
            print('Warning there is no known approx ', approx)
            continue

        fullfile = dirpath + '/monitor.h5'
        try:
            inh5file = h5py.File(fullfile, 'r')
        except OSError as ose:
            print('OSERROR with file ', fullfile)
            continue

        x = inh5file['default']['turns'].value
        for key in inh5file['default'].keys():
            if (key in not_plot):
                continue
            val = inh5file['default'][key].value
            # print(key)

            if len(val.shape) == 1:
                pass
            elif val.shape[1] == 1:
                val = val.reshape(len(val))
            elif key == 'profile':
                # TODO: Only keep the last turn profile
                val = val[-1]
            else:
                # TODO: Take the mean across the values for each bunch
                # TODO: How about the deviation?
                val = np.nanmean(val, axis=1)
                # TODO: What about the NAN values?
                remove_nan(val)

            if (key not in plot_dir):
                plot_dir[key] = {}
            if (approx_knob not in plot_dir[key]):
                plot_dir[key][approx_knob] = {'num': 0}
            if ('sum' not in plot_dir[key][approx_knob]):
                plot_dir[key][approx_knob]['sum'] = np.zeros_like(val)
                plot_dir[key][approx_knob]['min'] = val
                plot_dir[key][approx_knob]['max'] = val
            plot_dir[key][approx_knob]['num'] += 1
            plot_dir[key][approx_knob]['sum'] += val
            plot_dir[key][approx_knob]['min'] = np.minimum(
                plot_dir[key][approx_knob]['min'], val)
            plot_dir[key][approx_knob]['max'] = np.maximum(
                plot_dir[key][approx_knob]['max'], val)
            plot_dir[key][approx_knob]['turns'] = x
        inh5file.close()

    # continue here, I need to iterate over the errors, create a figure for each
    # iterate over the reduce values, add an error plot line for each acording to the intv etc

    for ts in args.ts:
        # filename = outfile + '/ts' + str(ts) + '.h5'
        if not os.path.exists(os.path.dirname(outdir + '/ts' + ts+'/')):
            os.makedirs(os.path.dirname(outdir + '/ts' + ts+'/'))
        for error in plot_dir.keys():
            outfiles = [
                # '{}/ts{}/{}.pdf'.format(outdir, ts, error),
                '{}/ts{}/{}.jpeg'.format(outdir, ts, error)]
            if error == 'profile':
                fig, ax_arr = plt.subplots(
                    nrows=1, ncols=2, figsize=(4, 4), sharey=True)
                fig.suptitle('Ts: {}, Variable: {}'.format(
                    ts, error), fontsize=8)

                for approx_knob, data in plot_dir[error].items():
                    x = data['turns']
                    y = data['sum'] / data['num']
                    ymin = data['min']
                    ymax = data['max']
                    intv = int(np.ceil(len(x)/args.points))
                    label = 'apprx{}_{}'.format(approx, approx_knob)
                    marker = markers.get(
                        'apprx{}_{}'.format(approx, approx_knob), None)
                    # Here I need a way to isolate the first and the last bunch only
                    # A tuple of three indices, start, max, end

                    left = 0
                    right = 100
                    max_parts = np.max(y)
                    bunches = []
                    percent = args.minparts
                    while left < right:
                        mid = int((left+right)//2)
                        min_parts = mid/1000. * max_parts
                        temp = detect_bunches(y, min_parts)
                        if (len(temp) < int(num_bunches)):
                            right = mid - 1
                        elif (len(temp) > int(num_bunches)):
                            left = mid + 1
                        else:
                            bunches = temp
                            right = mid - 1
                            percent = mid / 10.

                    if len(bunches) == 0:
                        bunches = detect_bunches(y, max_parts * args.minparts)
                        print('WARNING: Knob {}, could not detect {} bunches. Using {}% {} bunches' +
                              ' were detected.'.format(approx_knob, num_bunches, args.minparts, len(bunches)))
                    else:
                        print('Knob {}, detected {} bunches with {}%'.format(approx_knob, len(bunches), percent))
                    label += '_{}%'.format(percent)
                    bunch_idx=[0, -1]
                    for idx in bunch_idx:
                        plt.sca(ax_arr[idx])
                        b=bunches[idx]
                        x1=np.arange(b[0], b[1])
                        y1=y[x1]
                        ymin1=ymin[x1]
                        ymax1=ymax[x1]
                        # x1 = np.arange(len(x1))
                        if (approx_knob == '1'):
                            plt.fill_between(
                                x1, ymin1, ymax1, facecolor='0.6', interpolate=True)
                            plt.plot(x1, ymax1, color='black', linewidth=1)
                            plt.plot(x1, ymin1, color='black',
                                     linewidth=1, label=label)
                        else:
                            # print(label, y)
                            plt.errorbar(x1, y1,
                                         yerr=[y1-ymin1, ymax1-y1],
                                         label=label, linestyle='--',
                                         marker=marker, markersize=0,
                                         linewidth=1,
                                         # color=next(colors),
                                         # alpha=0.5,
                                         capsize=1, elinewidth=1)

                        plt.grid()
                        plt.xlabel('#bin', fontsize=8)
                        plt.xticks(fontsize=8)
                        plt.yticks(fontsize=8)
                        plt.ticklabel_format(
                            axis='y', style='sci', scilimits=(0, 0))
                        plt.gca().tick_params(pad=1, top=1, bottom=1, left=1,
                                              direction='inout', length=3, width=0.5)
                        if idx == 0:
                            plt.ylabel('#particles', fontsize=8)
                            plt.title('First bunch', fontsize=8, pad=1)
                        elif idx == -1:
                            plt.title('Last bunch', fontsize=8, pad=1)
                            plt.legend(loc='best', fancybox=True, fontsize=8,
                                           ncol=(lines+2)//3, columnspacing=1,
                                           labelspacing=0.1, borderpad=0.2, framealpha=0.5,
                                           handletextpad=0.2, handlelength=1.5, borderaxespad=0)

            else:

                lines=0
                fig=plt.figure(figsize=(4, 4))

                plt.grid()
                plt.title('Ts: {}, Variable: {}'.format(ts, error), fontsize=8)
                plt.xlabel('#Turn', fontsize=8)
                plt.ylabel('Raw value', fontsize=8)

                for approx_knob, data in plot_dir[error].items():
                    x=data['turns']
                    y=data['sum'] / data['num']
                    ymin=data['min']
                    ymax=data['max']
                    intv=int(np.ceil(len(x)/args.points))
                    label='apprx{}_{}'.format(approx, approx_knob)
                    marker=markers.get(
                        'apprx{}_{}'.format(approx, approx_knob), None)
                    y=running_mean(y, int(ts))
                    ymin=running_mean(ymin, int(ts))
                    ymax=running_mean(ymax, int(ts))
                    x=x[:len(y):intv]
                    y=y[::intv]
                    ymin=ymin[::intv]
                    ymax=ymax[::intv]
                    if (approx_knob == '1'):
                        plt.fill_between(
                            x, ymin, ymax, facecolor='0.6', interpolate=True)
                        plt.plot(x, ymax, color='black', linewidth=1)
                        plt.plot(x, ymin, color='black',
                                 linewidth=1, label=label)
                    else:
                        # print(label, y)
                        plt.errorbar(x, y,
                                     yerr=[y-ymin, ymax-y],
                                     label=label, linestyle='--',
                                     marker=marker, markersize=0,
                                     linewidth=1,
                                     # color=next(colors),
                                     # alpha=0.5,
                                     capsize=1, elinewidth=1)
                    lines += 1
                plt.xticks(fontsize=8)
                plt.yticks(fontsize=8)
                plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
                plt.gca().tick_params(pad=1, top=1, bottom=1, left=1,
                                      direction='inout', length=3, width=0.5)
                plt.legend(loc='upper left', fancybox=True, fontsize=8,
                               ncol=(lines+2)//3, columnspacing=1,
                               labelspacing=0.1, borderpad=0.2, framealpha=0.5,
                               handletextpad=0.2, handlelength=1.5, borderaxespad=0)

            plt.tight_layout()
            plt.subplots_adjust(wspace=0.0)

            for outfile in outfiles:
                save_and_crop(fig, outfile, dpi=900, bbox_inches='tight')
                # fig.savefig(outfile, dpi=900, bbox_inches='tight')
            if args.show is True:
                plt.show()
            plt.close()
