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


this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"
this_filename = sys.argv[0].split('/')[-1]

project_dir = this_directory + '../../'

parser = argparse.ArgumentParser(
    description='Evaluate single precision raw data.')

# parser.add_argument('-i', '--infile', type=str, default=None,
#                     help='Input .h5 file.')

# parser.add_argument('-b', '--basefile', type=str, default=None,
#                     help='Base .h5 files.')

# parser.add_argument('-i', '--inputkey', type=str, default='2kT-acc',
#                     choices=['2kT-acc', '1mT-acc', '1mT-noacc',
#                              '1mT-acc-seed'],
#                     help='Key of the input config.')


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

parser.add_argument('-p', '--points', type=int, default=-1,
                    help='Num of points in the plot. Default: all')

# parser.add_argument('-t', '--ts', type=str, default=['1'], nargs='+',
# help='Running mean window. Default: [1]')

parser.add_argument('-s', '--show', action='store_true',
                    help='Show the plot or save only. Default: save only')


args = parser.parse_args()

res_dir = args.outdir
images_dir = os.path.join(res_dir, 'plots')

if not os.path.exists(images_dir):
    os.makedirs(images_dir)


gconfig = {
    'hatches': ['', '', 'xx'],
    'markers': ['x', 'o', '^'],
    'colors': ['xkcd:red', 'xkcd:green', 'xkcd:blue'],
    'group': '/default',
    'labels': {'std_profile': r'$s_{profile}$',
               'std_dE': r'$s_{dE}$',
               'std_dt': r'$s_{dt}$'},
    'x_name': 'turns',
    'y_names': [
        'std_profile',
        'std_dE',
        'std_dt',
        # 'mean_profile',
        # 'mean_dE',
        # 'mean_dt'
    ],
    # 'y_err_name': 'std',
    'xlabel': {'xlabel': 'Turn', 'labelpad': 3, 'fontsize': 10},
    'ylabel': {'ylabel': r'Relative Error (\%)', 'labelpad': 3, 'fontsize': 10},
    'title': {
        # 's': '{}'.format(case.upper()),
        'fontsize': 10,
        'y': .85,
        # 'x': 0.45,
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
        'loc': 'upper left', 'ncol': 1, 'handlelength': 1.5, 'fancybox': False,
        'framealpha': .7, 'fontsize': 10, 'labelspacing': 0, 'borderpad': 0.5,
        'handletextpad': 0.5, 'borderaxespad': 0.1, 'columnspacing': 0.8,
        # 'bbox_to_anchor': (0., 0.85)
    },
    'subplots_adjust': {
        'wspace': 0.0, 'hspace': 0.1, 'top': 0.93
    },
    'tick_params': {
        'pad': 1, 'top': 0, 'bottom': 1, 'left': 1,
        'direction': 'out', 'length': 3, 'width': 1,
    },
    'fontname': 'DejaVu Sans Mono',
    'ylim': [0.00001, 1],
    # 'xlim': [1.6, 36],
    # 'yticks': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
    'outfiles': ['{}/{}-{}.png', '{}/{}-{}.pdf'],
    # 'cases': ['ex01'],
    'inputkeys': [
        # 'lhc-40kt-seed', 'sps-40kt-seed', 'ps-40kt-seed',
        'lhc-40kt', 'sps-40kt', 'ps-40kt'
    ],
    'infiles': {
        'ex01-2kt-acc': {'': 'results/precision-analysis/ex01/precision-monitor/_p2000000_b1_s1000_t2000_w1_o4_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly_lba500_monitor100_tp0_precsingle_/23Apr20.18-46-53-10/monitor.h5'},
        'ex01-1mt-acc': {'': 'results/precision-analysis/ex01/precision-monitor/_p1000000_b1_s1000_t1000000_w1_o14_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly_lba500_monitor1000_tp0_precsingle_/24Apr20.15-11-52-92/monitor.h5'},
        'ex01-1mt-noacc': {'': 'results/precision-analysis/ex01/precision-monitor/_p1000000_b1_s1000_t1000000_w1_o14_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly_lba500_monitor1000_tp0_precsingle_/24Apr20.19-29-44-0/monitor.h5'},
        'ex01-1mt-acc-seed': {
            'seed1-': 'results/precision-analysis/ex01/precision-seed/_p1000000_b1_s1000_t1000000_w1_o14_N1_red1_mtw0_seed1_approx0_mpimpich3_lbreportonly_lba500_monitor1000_tp0_precdouble_/25Apr20.00-58-27-36/monitor.h5',
            'seed2-': 'results/precision-analysis/ex01/precision-seed/_p1000000_b1_s1000_t1000000_w1_o14_N1_red1_mtw0_seed2_approx0_mpimpich3_lbreportonly_lba500_monitor1000_tp0_precdouble_/25Apr20.03-38-30-55/monitor.h5',
            # 'seed3-': 'results/precision-analysis/ex01/precision-seed/_p1000000_b1_s1000_t1000000_w1_o14_N1_red1_mtw0_seed3_approx0_mpimpich3_lbreportonly_lba500_monitor1000_tp0_precdouble_/25Apr20.06-16-34-10/monitor.h5',
        },
        'lhc-40kt-seed': {
            'seed1-': 'results/precision-analysis/lhc/precision-seed/_p1000000_b1_s1000_t40000_w1_o14_N1_red1_mtw0_seed1_approx0_mpimpich3_lbreportonly_lba500_monitor1000_tp0_precdouble_/29Apr20.11-33-28-67/monitor.h5',
            'seed2-': 'results/precision-analysis/lhc/precision-seed/_p1000000_b1_s1000_t40000_w1_o14_N1_red1_mtw0_seed2_approx0_mpimpich3_lbreportonly_lba500_monitor1000_tp0_precdouble_/29Apr20.11-37-42-71/monitor.h5',
            # 'seed3-': 'results/precision-analysis/lhc/precision-seed/_p1000000_b1_s1000_t40000_w1_o14_N1_red1_mtw0_seed3_approx0_mpimpich3_lbreportonly_lba500_monitor1000_tp0_precdouble_/29Apr20.11-41-57-38/monitor.h5',
        },
        'lhc-40kt': {
            '': 'results/precision-analysis/lhc/precision-monitor/_p1000000_b1_s1000_t40000_w1_o14_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly_lba500_monitor1000_tp0_precsingle_/29Apr20.11-25-16-92/monitor.h5'
        },
        'sps-40kt-seed': {
            'seed1-': 'results/precision-analysis/sps/precision-seed/_p1000000_b1_s1408_t40000_w1_o14_N1_red1_mtw0_seed1_approx0_mpimpich3_lbreportonly_lba500_monitor1000_tp0_precdouble_/28Apr20.21-08-19-16/monitor.h5',
            'seed2-': 'results/precision-analysis/sps/precision-seed/_p1000000_b1_s1408_t40000_w1_o14_N1_red1_mtw0_seed2_approx0_mpimpich3_lbreportonly_lba500_monitor1000_tp0_precdouble_/28Apr20.21-29-28-95/monitor.h5',
            # 'seed3-': 'results/precision-analysis/sps/precision-seed/_p1000000_b1_s1408_t40000_w1_o14_N1_red1_mtw0_seed3_approx0_mpimpich3_lbreportonly_lba500_monitor1000_tp0_precdouble_/28Apr20.21-52-41-4/monitor.h5',
        },
        'sps-40kt': {
            '': 'results/precision-analysis/sps/precision-monitor/_p1000000_b1_s1408_t40000_w1_o14_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly_lba500_monitor1000_tp0_precsingle_/28Apr20.20-26-09-34/monitor.h5'
        },
        'ps-40kt-seed': {
            'seed1-': 'results/precision-analysis/ps/precision-seed/_p1000000_b1_s256_t40000_w1_o14_N1_red1_mtw50_seed1_approx0_mpimpich3_lbreportonly_lba500_monitor100_tp0_precdouble_/29Apr20.13-22-35-2/monitor.h5',
            'seed2-': 'results/precision-analysis/ps/precision-seed/_p1000000_b1_s256_t40000_w1_o14_N1_red1_mtw50_seed2_approx0_mpimpich3_lbreportonly_lba500_monitor100_tp0_precdouble_/29Apr20.13-27-53-16/monitor.h5',
            # 'seed3-': 'results/precision-analysis/ps/precision-seed/_p1000000_b1_s256_t40000_w1_o14_N1_red1_mtw50_seed3_approx0_mpimpich3_lbreportonly_lba500_monitor100_tp0_precdouble_/29Apr20.13-33-05-20/monitor.h5',
        },
        'ps-40kt': {
            '': 'results/precision-analysis/ps/precision-monitor/_p1000000_b1_s256_t40000_w1_o14_N1_red1_mtw50_seed0_approx0_mpimpich3_lbreportonly_lba500_monitor100_tp0_precsingle_/29Apr20.13-12-14-53/monitor.h5'
        }

    },
    'basefile': {
        'ex01-2kt-acc': 'results/precision-analysis/ex01/precision-monitor/_p2000000_b1_s1000_t2000_w1_o4_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly_lba500_monitor100_tp0_precdouble_/23Apr20.18-47-57-19/monitor.h5',
        'ex01-1mt-acc': 'results/precision-analysis/ex01/precision-monitor/_p1000000_b1_s1000_t1000000_w1_o14_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly_lba500_monitor1000_tp0_precdouble_/24Apr20.16-51-30-23/monitor.h5',
        'ex01-1mt-noacc': 'results/precision-analysis/ex01/precision-monitor/_p1000000_b1_s1000_t1000000_w1_o14_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly_lba500_monitor1000_tp0_precdouble_/24Apr20.21-08-20-88/monitor.h5',
        'ex01-1mt-acc-seed': 'results/precision-analysis/ex01/precision-monitor/_p1000000_b1_s1000_t1000000_w1_o14_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly_lba500_monitor1000_tp0_precdouble_/24Apr20.16-51-30-23/monitor.h5',
        'ps-40kt': 'results/precision-analysis/ps/precision-monitor/_p1000000_b1_s256_t40000_w1_o14_N1_red1_mtw50_seed0_approx0_mpimpich3_lbreportonly_lba500_monitor100_tp0_precdouble_/29Apr20.13-17-14-89/monitor.h5',
        'sps-40kt': 'results/precision-analysis/sps/precision-monitor/_p1000000_b1_s1408_t40000_w1_o14_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly_lba500_monitor1000_tp0_precdouble_/28Apr20.20-45-00-98/monitor.h5',
        'lhc-40kt': 'results/precision-analysis/lhc/precision-monitor/_p1000000_b1_s1000_t40000_w1_o14_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly_lba500_monitor1000_tp0_precdouble_/29Apr20.11-29-09-77/monitor.h5',
        'ps-40kt-seed': 'results/precision-analysis/ps/precision-monitor/_p1000000_b1_s256_t40000_w1_o14_N1_red1_mtw50_seed0_approx0_mpimpich3_lbreportonly_lba500_monitor100_tp0_precdouble_/29Apr20.13-17-14-89/monitor.h5',
        'sps-40kt-seed': 'results/precision-analysis/sps/precision-monitor/_p1000000_b1_s1408_t40000_w1_o14_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly_lba500_monitor1000_tp0_precdouble_/28Apr20.20-45-00-98/monitor.h5',
        'lhc-40kt-seed': 'results/precision-analysis/lhc/precision-monitor/_p1000000_b1_s1000_t40000_w1_o14_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly_lba500_monitor1000_tp0_precdouble_/29Apr20.11-29-09-77/monitor.h5',

    },

}

plt.rcParams['ps.useafm'] = True
plt.rcParams['pdf.use14corefonts'] = True
plt.rcParams['text.usetex'] = True  # Let TeX do the typsetting
# Force sans-serif math mode (for axes labels)
plt.rcParams['text.latex.preamble'] = [r'\usepackage{sansmath}', r'\sansmath']
plt.rcParams['font.family'] = 'sans-serif'  # ... for regular text
plt.rcParams['font.sans-serif'] = 'Helvetica'


def running_mean(x, N, axis=None):
    cumsum = np.cumsum(np.insert(x, 0, 0, axis=0), axis=0)
    return (cumsum[N:] - cumsum[:-N])/N


if __name__ == '__main__':

    last_t = args.turns
    outdir = args.outdir
    points = args.points
    # tss = args.ts

    for inputkey in gconfig['inputkeys']:
        # for case in gconfig['cases']:
        # inputkey = args.inputkey
        infiles = gconfig['infiles'][inputkey]
        basefile = gconfig['basefile'][inputkey]
        based = {}
        ind = {}

        # Read basefile
        fullfile = os.path.join(project_dir, basefile)
        h5file = h5py.File(fullfile, 'r')
        for key in h5file[gconfig['group']]:
            val = h5file[gconfig['group']][key][()]
            if key not in based:
                based[key] = val.reshape(len(val))
        h5file.close()
        turns = based[gconfig['x_name']]
        del based[gconfig['x_name']]

        # Read infile
        for keyf, infile in infiles.items():
            fullfile = os.path.join(project_dir, infile)
            h5file = h5py.File(fullfile, 'r')
            if keyf not in ind:
                ind[keyf] = {}
            for key in h5file[gconfig['group']]:
                val = h5file[gconfig['group']][key][()]
                if key not in ind:
                    ind[keyf][key] = val.reshape(len(val))
            h5file.close()
            assert np.array_equal(turns, ind[keyf][gconfig['x_name']])
            del ind[keyf][gconfig['x_name']]

        points = min(len(turns), points) if points > 0 else len(turns)
        intv = int(np.ceil(len(turns)/points))

        fig, ax = plt.subplots(ncols=1, nrows=1,
                               sharex=True, sharey=True,
                               figsize=gconfig['figsize'])
        for keyf in ind.keys():
            for key in (set(based.keys()) & set(ind[keyf].keys())):
                if key not in gconfig['y_names']:
                    continue
                basevals = based[key]
                indvals = ind[keyf][key]
                assert len(basevals) == len(
                    indvals) and len(turns) == len(basevals)

                error = 100 * np.abs(1 - indvals / basevals)
                plt.plot(turns[::intv], error[::intv],
                         label='{}{}'.format(keyf, gconfig['labels'][key]),
                         )
                # marker=gconfig['markers'][idx],
                # color=gconfig['colors'][idx],
                # yerr=yerr,
                # capsize=2)
        plt.yscale('log')

        plt.grid(True, which='both', axis='y', alpha=0.5)
        plt.grid(False, which='major', axis='x')
        plt.title('{}'.format('Accuracy'), **gconfig['title'])
        plt.xlabel(**gconfig['xlabel'])
        plt.ylabel(**gconfig['ylabel'])
        plt.ylim(gconfig['ylim'])
        # plt.xlim(gconfig['xlim'])
        # plt.xticks(x//20, np.array(x, int)//20, **gconfig['ticks'])
        ax.tick_params(**gconfig['tick_params'])
        ax.legend(**gconfig['legend'])

        plt.xticks(**gconfig['ticks'])
        yticks = [10**i for i in range(int(np.log10(gconfig['ylim'][0])),
                                       int(np.log10(gconfig['ylim'][1]))+1)]
        plt.yticks(yticks, yticks, **gconfig['ticks'])

        plt.tight_layout()
        plt.subplots_adjust(**gconfig['subplots_adjust'])
        for file in gconfig['outfiles']:
            file = file.format(images_dir, this_filename[:-3], inputkey)
            print('[{}] {}: {}'.format(this_filename[:-3], 'Saving figure', file))

            save_and_crop(fig, file, dpi=600, bbox_inches='tight')
        if args.show:
            plt.show()
        plt.close()

    # for dirpath, dirnames, filenames in os.walk(indir):
    #     if 'monitor.h5' not in filenames:
    #         continue

    #     particles = dirpath.split('_p')[1].split('_')[0]
    #     bunches = dirpath.split('_b')[1].split('_')[0]
    #     seed = dirpath.split('_seed')[1].split('_')[0]
    #     monitor_intv = dirpath.split('_m')[1].split('_')[0]
    #     red = dirpath.split('_r')[1].split('_')[0]
    #     workers = dirpath.split('_w')[1].split('_')[0]

    #     fullfile = dirpath + '/monitor.h5'
    #     inh5file = h5py.File(fullfile, 'r')
    #     x = inh5file['default']['turns'].value
    #     # intv = int(np.ceil(len(x)/points))
    #     for key in inh5file['default'].keys():
    #         if (key in not_plot):
    #             continue
    #         val = inh5file['default'][key].value
    #         if (key == 'profile'):
    #             val = val[-1]
    #         val = val.reshape(len(val))
    #         if (key not in plot_dir):
    #             plot_dir[key] = {}
    #         if (workers not in plot_dir[key]):
    #             plot_dir[key][workers] = {'num': 0}
    #         if ('sum' not in plot_dir[key][workers]):
    #             plot_dir[key][workers]['sum'] = np.zeros_like(val)
    #             plot_dir[key][workers]['min'] = val
    #             plot_dir[key][workers]['max'] = val
    #         plot_dir[key][workers]['num'] += 1
    #         plot_dir[key][workers]['sum'] += val
    #         plot_dir[key][workers]['min'] = np.minimum(
    #             plot_dir[key][workers]['min'], val)
    #         plot_dir[key][workers]['max'] = np.maximum(
    #             plot_dir[key][workers]['max'], val)
    #         plot_dir[key][workers]['turns'] = x
    #     inh5file.close()

    # # continue here, I need to iterate over the errors, create a figure for each
    # # iterate over the reduce values, add an error plot line for each acording to the intv etc

    # for ts in tss:
    #     # filename = outfile + '/ts' + str(ts) + '.h5'
    #     if not os.path.exists(os.path.dirname(outdir + '/ts' + ts+'/')):
    #         os.makedirs(os.path.dirname(outdir + '/ts' + ts+'/'))
    #     lines = 0
    #     for error in plot_dir.keys():
    #         fig = plt.figure(figsize=(4, 4))
    #         outfiles = [
    #             # '{}/ts{}/{}.pdf'.format(outdir, ts, error),
    #             '{}/ts{}/{}.jpeg'.format(outdir, ts, error)]

    #         plt.grid()
    #         if args.get('ymin', None):
    #             plt.ylim(ymin=args.ymin)
    #         if args.get('ymax', None):
    #             plt.ylim(ymax=args.ymax)

    #         plt.title('Ts: {}, Variable: {}'.format(ts, error))
    #         plt.xlabel('#Turn')
    #         plt.ylabel('Raw value')

    #         for workers, data in plot_dir[error].items():
    #             x = data['turns']
    #             y = data['sum'] / data['num']
    #             ymin = data['min']
    #             ymax = data['max']
    #             intv = int(np.ceil(len(x)/points))
    #             label = 'r{}'.format(workers)
    #             marker = markers.get('r{}'.format(workers), None)
    #             if (error == 'profile'):
    #                 # y = y[-1]
    #                 nonzero = np.flatnonzero(ymax)
    #                 y = y[nonzero[0]:nonzero[-1]]
    #                 ymin = ymin[nonzero[0]:nonzero[-1]]
    #                 ymax = ymax[nonzero[0]:nonzero[-1]]
    #                 x = np.arange(len(y))
    #                 plt.xlabel('#Bin')
    #             else:
    #                 y = running_mean(y, int(ts))
    #                 ymin = running_mean(ymin, int(ts))
    #                 ymax = running_mean(ymax, int(ts))
    #                 x = x[:len(y):intv]
    #                 y = y[::intv]
    #                 ymin = ymin[::intv]
    #                 ymax = ymax[::intv]
    #             if (workers == '1'):
    #                 plt.fill_between(
    #                     x, ymin, ymax, facecolor='0.6', interpolate=True)
    #                 plt.plot(x, ymax, color='black', linewidth=1)
    #                 plt.plot(x, ymin, color='black',  linewidth=1, label=label)
    #             else:
    #                 plt.errorbar(x, y,
    #                              yerr=[y-ymin, ymax-y],
    #                              label=label, linestyle='--',
    #                              marker=marker, markersize=0,
    #                              # color=next(colors),
    #                              # alpha=0.5,
    #                              capsize=1, elinewidth=1)
    #             lines += 1
    #         plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    #         plt.legend(loc='upper left', fancybox=True, fontsize=9,
    #                        ncol=(lines+2)//3, columnspacing=1,
    #                        labelspacing=0.1, borderpad=0.2, framealpha=0.5,
    #                        handletextpad=0.2, handlelength=1.5, borderaxespad=0)

    #         plt.tight_layout()
    #         for outfile in outfiles:
    #             save_and_crop(fig, outfile, dpi=900, bbox_inches='tight')
    #             # fig.savefig(outfile, dpi=900, bbox_inches='tight')
    #         if args.show is True:
    #             plt.show()
    #         plt.close()
