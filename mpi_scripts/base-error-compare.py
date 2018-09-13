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

errors = ['n_macroparticles', 'mean_dt', 'mean_dE', 'std_dE', 'std_dt']

if __name__ == '__main__':
    args = parser.parse_args()
    args = vars(args)

    infiles = args['infiles']
    outdir = args['outdir']
    tss = args['ts']
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for ts in tss:
        for error in errors:
            outfiles = [outdir + '/ts' + ts + '/' +
                        error + '_' + error + '.jpeg']

            fig = plt.figure(figsize=(6, 4))
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
                ts_file = file.split('ts')[1].split('.h5')[0]
                if ts_file != ts:
                    continue

                if not os.path.exists(outdir + '/ts' + ts + '/'):
                    os.makedirs(outdir + '/ts' + ts + '/')

                inh5file = h5py.File(file, 'r')

                # if 'SPS' in file:

                # elif 'LHC' in file:

                # else:
                #     sys.exit('Not an LHC or SPS testcase: {}'.format(file))

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
                        else:
                            continue
                        # elif inh5['reduce'][i][1] != 1 and \
                        #         (len(args['reduce']) == 0 or
                        #             inh5['reduce'][i][1] in args['reduce']):

                        #     key = '{}-r_{}'.format(bunchkey,
                        #                            inh5['reduce'][i][1])
                        #     plt_data[key] = inh5[error][i]
                        # elif inh5['seed'][i][1] == 1980:
                        #     key = 'r-{}'.format(inh5['reduce'][i][1])
                        #     plt_data[key] = inh5['errors'][i]

                    avg_base_error = np.mean(plt_data['base_error'], axis=0)
                    # std_base_error = np.std(plt_data['base_error'], axis=0)
                    sem_base_error = stats.sem(plt_data['base_error'], axis=0)
                    # sem_base_error = np.abs(sem_base_error / avg_base_error)

                    if 'SPS' in file:
                        label = 'SPS' + '-' + file.split('/')[-2]
                    elif 'LHC' in file:
                        label = 'LHC' + '-' + file.split('/')[-2]
                    else:
                        sys.exit('Not an LHC or SPS testcase: {}'.format(file))

                    x = np.arange(len(avg_base_error))
                    plt.errorbar(x[::50], avg_base_error[::50], yerr=sem_base_error[::50],
                                 label=label, linestyle='', marker=marker,
                                 markersize=4, color=next(colors))
                    if label == 'SPS-indvolt_no_ploop':
                        if error == 'mean_dt':
                            annotate_max(plt.gca(), x, avg_base_error,
                                         size='medium', ha='right')
                    else:
                        annotate_max(plt.gca(), x, avg_base_error,
                                     size='medium', ha='right')

                    lines += 1
                    # del plt_data['base_error']

                    # print('Base error std', 100 * std_base_error/ avg_base_error)

                    # for k, v in plt_data.items():
                    #     x = np.arange(len(v))
                    #     y = v / avg_base_error
                    #     err = y * sem_base_error
                    #     # plt.errorbar(x, y, yerr=err, label=k, linestyle='',
                    #     #              marker=marker, markersize=5, color=next(colors))
                    #     plt.errorbar(x[::50], y[::50], yerr=None, label=k, linestyle='',
                    #                  marker=marker, markersize=4, color=next(colors))
                    #     lines += 1

                inh5file.close()

            plt.legend(loc='upper left', fancybox=True, fontsize=10,
                           ncol=(lines+1)//2, columnspacing=1,
                           labelspacing=0.1, borderpad=0.2, framealpha=0.5,
                           handletextpad=0.2, handlelength=1.5, borderaxespad=0)

            plt.tight_layout()
            for outfile in outfiles:
                save_and_crop(fig, outfile, dpi=900, bbox_inches='tight')
            plt.show()
            plt.close()
