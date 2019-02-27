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

# parser.add_argument('-a', '--approx', type=int, default=1, choices=[1,2],
#                     help='Approximation method: 1 --> Global reduce, 2 --> Scale histo')

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

markers = {'apprx1_1': '+',
           'apprx1_2': 's',
           'r3': 'v'}


def running_mean(x, N, axis=None):
    cumsum = np.cumsum(np.insert(x, 0, 0, axis=0), axis=0)
    return (cumsum[N:] - cumsum[:-N])/N

def remove_nan(x):
    i = 0
    while np.isnan(x[i]):
        i+=1
    x[0:i] = x[i]
    val = x[i]
    while i < len(x):
        if np.isnan(x[i]):
            x[i] = val
        else:
            val = x[i]
        i+=1

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
        # intv = int(np.ceil(len(x)/points))
        for key in inh5file['default'].keys():
            if (key in not_plot):
                continue
            val = inh5file['default'][key].value
            # print(key)

            if len(val.shape) == 1:
                continue
                pass
            elif val.shape[1] == 1:
                continue
                val = val.reshape(len(val))
            elif key == 'profile':
                # TODO: Only keep the last turn profile
                val = val[-1]
            else:
                continue
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

    for ts in tss:
        # filename = outfile + '/ts' + str(ts) + '.h5'
        if not os.path.exists(os.path.dirname(outdir + '/ts' + ts+'/')):
            os.makedirs(os.path.dirname(outdir + '/ts' + ts+'/'))
        for error in plot_dir.keys():
            lines = 0
            #fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(6, 4))
            #fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(6, 4))
            fig = plt.figure(figsize=(4, 4))
            outfiles = [
                '{}/ts{}/{}.pdf'.format(outdir, ts, error),
                '{}/ts{}/{}.jpeg'.format(outdir, ts, error)]
            ax1 = plt.gca()
            
            ax2 = plt.axes([0.75, 0.72, 0.2, 0.2])
            ax2.set_xlim((5190, 5240))
            ax2.set_ylim((34000, 39400))
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax2.set_facecolor('#ffffe0') 
            
            ax3 = plt.axes([0.18, 0.38, 0.15, 0.25])
            ax3.set_xlim((5145, 5170))
            ax3.set_ylim((11400, 19400))
            ax3.set_xticks([])
            ax3.set_yticks([])
            ax3.set_facecolor('#ffffe0') 
            
            ax4 = plt.axes([0.34, 0.15, 0.15, 0.15])
            ax4.set_xlim((5110, 5125))
            ax4.set_ylim((0, 1500))
            ax4.set_xticks([])
            ax4.set_yticks([])
            ax4.set_facecolor('#ffffe0') 
            
            plt.sca(ax1)

            plt.grid()
            # if args.get('ymin', None):
            #     plt.ylim(ymin=args['ymin'])
            # if args.get('ymax', None):
            #     plt.ylim(ymax=args['ymax'])

            # ax1.set_xlim((5000, 19500))

            # plt.title('Ts: {}, Variable: {}'.format(ts, error))
            plt.xlabel('#Turn')
            plt.ylabel('particles/bin')

            for approx_knob, data in plot_dir[error].items():
                x = data['turns']
                y = data['sum'] / data['num']
                ymin = data['min']
                ymax = data['max']
                intv = int(np.ceil(len(x)/points))
                label = 'apprx{}_{}'.format(approx, approx_knob)
                marker = markers.get('apprx{}_{}'.format(approx, approx_knob), None)
                if (error == 'profile'):
                    # y = y[-1]
                    x1 = np.arange(5100, 5330)
                    #x2 = np.arange(19150, 19400)
                    # nonzero = np.flatnonzero(ymax)
                    y1 = y[x1]
                    y1min = ymin[x1]
                    y1max = ymax[x1]

                    #  y2 = y[x2]
                    #  y2min = ymin[x2]
                    #  y2max = ymax[x2]
                    #  y2 = np.zeros(400, float)
                    #  y2min = np.zeros(400, float)
                    #  y2max = np.zeros(400, float)
                    #  
                    #  for i in range(int(bunches)):
                    #      s = 5000 + i*1208
                    #      e = s + 400
                    #      y2 += y[s:e]
                    #      y2min += ymin[s:e]
                    #      y2max += ymax[s:e]
                    #  y2 /= int(bunches)
                    #  y2min /= int(bunches)
                    #  y2max /= int(bunches)

                    # x = np.arange(len(y))
                    plt.xlabel('bin')
                else:
                    y = running_mean(y, int(ts))
                    ymin = running_mean(ymin, int(ts))
                    ymax = running_mean(ymax, int(ts))
                    x = x[:len(y):intv]
                    y = y[::intv]
                    ymin = ymin[::intv]
                    ymax = ymax[::intv]
                if (approx_knob == '1'):
                    plt.fill_between(
                        x1, y1min, y1max, facecolor='0.6', interpolate=True)
                    plt.plot(x1, y1max, color='black', linewidth=1)
                    plt.plot(x1, y1min, color='black',  linewidth=1, label='exact')
                    
                    
                    ax2.fill_between(
                        x1, y1min, y1max, facecolor='0.6', interpolate=True)
                    ax2.plot(x1, y1max, color='black', linewidth=1)
                    ax2.plot(x1, y1min, color='black',  linewidth=1, label=None)
                    
                    ax3.fill_between(
                        x1, y1min, y1max, facecolor='0.6', interpolate=True)
                    ax3.plot(x1, y1max, color='black', linewidth=1)
                    ax3.plot(x1, y1min, color='black',  linewidth=1, label=None)
                    
                    ax4.fill_between(
                        x1, y1min, y1max, facecolor='0.6', interpolate=True)
                    ax4.plot(x1, y1max, color='black', linewidth=1)
                    ax4.plot(x1, y1min, color='black',  linewidth=1, label=None)
                else:
                    # print(label, y)
                    red = label.split('_')[1]
                    plt.errorbar(x1, y1,
                             yerr=[y1-y1min, y1max-y1],
                             label=red+ 'turns', linestyle='--',
                             marker=marker, markersize=0,
                             # color=next(colors),
                             # alpha=0.5,
                             capsize=1, elinewidth=1)
                    ax2.errorbar(x1, y1,
                             yerr=[y1-y1min, y1max-y1],
                             label=None, linestyle='--',
                             marker=marker, markersize=0,
                             # color=next(colors),
                             # alpha=0.5,
                             capsize=1, elinewidth=1)
                    ax3.errorbar(x1, y1,
                             yerr=[y1-y1min, y1max-y1],
                             label=None, linestyle='--',
                             marker=marker, markersize=0,
                             # color=next(colors),
                             # alpha=0.5,
                             capsize=1, elinewidth=1)
                    ax4.errorbar(x1, y1,
                             yerr=[y1-y1min, y1max-y1],
                             label=None, linestyle='--',
                             marker=marker, markersize=0,
                             # color=next(colors),
                             # alpha=0.5,
                             capsize=1, elinewidth=1)

                lines += 1


            plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            plt.legend(loc='upper left', fancybox=True, fontsize=10,
                           ncol=1, columnspacing=1,
                           labelspacing=0.1, borderpad=0.2, framealpha=0.5,
                           handletextpad=0.2, handlelength=1.5, borderaxespad=0)

            plt.tight_layout()
            for outfile in outfiles:
                save_and_crop(fig, outfile, dpi=900, bbox_inches='tight')
                # fig.savefig(outfile, dpi=900, bbox_inches='tight')
            if args['show'] is True:
                plt.show()
            plt.close()
