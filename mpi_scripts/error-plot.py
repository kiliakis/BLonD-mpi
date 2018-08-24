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

parser = argparse.ArgumentParser(
    description='Calculate the error in the histogram.')


parser.add_argument('-i', '--infile', type=str, default=None,
                    help='Input file name.')

parser.add_argument('-o', '--outfile', type=str, default=None,
                    help='File to store the results.')

parser.add_argument('-ymin', '--ymin', type=float, default=None,
                    help='Min value for y axis.')

parser.add_argument('-ymax', '--ymax', type=float, default=None,
                    help='Max value for y axis.')



if __name__ == '__main__':
    args = parser.parse_args()
    args = vars(args)

    infile = args['infile']
    outfile = args['outfile']
    ts = infile.split('-ts')[1].split('-')[0]
    # ts = '1'
    # if not os.path.exists(outfile):
    #     os.makedirs(outfile)

    inh5file = h5py.File(infile, 'r')
    inh5 = inh5file['default']

    plt_data = {}

    for i in range(len(inh5['errors'])):
        if inh5['reduce'][i][0] == 1 and inh5['reduce'][i][1] == 1:
            if 'base_error' not in plt_data:
                plt_data['base_error'] = []
            plt_data['base_error'].append(inh5['errors'][i])
        elif inh5['reduce'][i][1] != 1:
            key = 'r-{}'.format(inh5['reduce'][i][1])
            plt_data[key] = inh5['errors'][i]
        # elif inh5['seed'][i][1] == 1980:
        #     key = 'r-{}'.format(inh5['reduce'][i][1])
        #     plt_data[key] = inh5['errors'][i]


    avg_base_error = np.mean(plt_data['base_error'], axis=0)
    # std_base_error = np.std(plt_data['base_error'], axis=0)
    sem_base_error = stats.sem(plt_data['base_error'], axis=0)
    sem_base_error = np.abs(sem_base_error / avg_base_error)

    # print('Base error std', 100 * std_base_error/ avg_base_error)

    del plt_data['base_error']

    fig = plt.figure(figsize=(6, 4))
    plt.grid()
    if args.get('ymin', None):
        plt.ylim(ymin=args['ymin'])
    if args.get('ymax', None):
        plt.ylim(ymax=args['ymax'])


    for k, v in plt_data.items():
        x = np.arange(len(v))
        y = v / avg_base_error
        err = y * sem_base_error
        # plt.plot(x, y, label=k)
        plt.errorbar(x, y, yerr=err, label=k)

    plt.title('Ts = '+ts)
    plt.xlabel('#Turn')
    plt.ylabel('Relative error')
    plt.legend(loc='best', fancybox=True, fontsize=9,
                   ncol=2, columnspacing=1,
                   labelspacing=0.1, borderpad=0.2, framealpha=0.5,
                   handletextpad=0.2, handlelength=1.5, borderaxespad=0)

    plt.tight_layout()
    save_and_crop(fig, args['outfile'], dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()

    inh5file.close()
