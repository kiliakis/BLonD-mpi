import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
import h5py
import argparse
import sys
import os

parser = argparse.ArgumentParser(description='Validate histogram.')


parser.add_argument('-b', '--base', type=str, default=None,
                    help='Base file name.')

parser.add_argument('-a', '--approx', type=str, default=None,
                    help='Approximate file name.')

parser.add_argument('-o', '--outdir', type=str, default=None,
                    help='Directory to store the images.')


if __name__ == '__main__':
    args = parser.parse_args()
    args = vars(args)

    basefile = args['base']
    approxfile = args['approx']
    outdir = args['outdir']

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    reduce_turns = approxfile.split('-r')[1].split('-')[0]

    baseh5 = h5py.File(basefile, 'r')
    approxh5 = h5py.File(approxfile, 'r')

    basedata = baseh5['Slices']['n_macroparticles'].value
    approxdata = approxh5['Slices']['n_macroparticles'].value

    bins = np.arange(len(basedata[0]))

    assert basedata.shape == approxdata.shape

    turns = baseh5['Slices']['turns'].value
    # slices = np.range()
    bins = len(basedata[0])
    real_bins = 100

    interval = (bins +real_bins - 1) // real_bins

    for i in range(0, len(turns), 4):
        baserow = basedata[i]
        approxrow = approxdata[i]

        # ids = np.where(baserow > 0)
        # baserow = baserow[ids]
        # approxrow = approxrow[ids]

        diff = baserow - approxrow
        # diff = 100. * diff / baserow
        turn = turns[i]
        fig = plt.figure(figsize=(6, 4))
        gs = gridspec.GridSpec(2, 2)
        
        axbase = plt.subplot(gs[0, 0])
        axbase.set_title('Base, turn {}'.format(turn))
        axbase.set_ylim([0, 40000])

        axapprox = plt.subplot(gs[0, 1])
        axapprox.set_title('Reduce every {} turns'.format(reduce_turns))
        axapprox.set_ylim([0, 40000])

        axdiff = plt.subplot(gs[1, :])
        axdiff.set_title('Diff (base - approx)')
        axdiff.set_ylim([-40000, 40000])

        axbase.plot(baserow, ls='-', marker='')
        # axbase.plot(approxrow, ls='-', marker='')


        axapprox.plot(approxrow, ls='-', marker='')
        # axapprox.plot(baserow, ls='-', marker='')

        # axbase.plot(bins, baserow, ls='', marker='.', ms=0.01, alpha=0.5, color='red')
        # axbase.plot(bins, approxrow, ls='', marker='.', ms=0.01, alpha=0.5, color='green')


        # axdiff.plot(diff, ls='-', marker='')

        # axdiff.bar(bins[::interval], baserow[::interval], width=interval/2)
        # axdiff.bar(bins[::interval]+interval/2, approxrow[::interval], width=interval/2)

        # base_less = [sum(baserow[i*interval: min((i+1)*interval, bins)])/interval for i in range(real_bins)]
        # approx_less = [sum(approxrow[i*interval: min((i+1)*interval, bins)])/interval for i in range(real_bins)]

        # axdiff.bar(np.linspace(0, bins, real_bins), base_less, width=interval/2.)
        # axdiff.bar(np.linspace(0, bins, real_bins) + interval/2., approx_less, width=interval/2.)

        # axdiff.bar(bins[10000:10000+interval], baserow[10000:10000+interval], width=0.5)
        # axdiff.bar(bins[10000:10000+interval]+0.5, approxrow[10000:10000+interval], width=0.5)



        # axapprox.plot(approxrow, ls='-', marker='', alpha=0.5)
        axdiff.plot(diff, ls='-', marker='')



        # axbase.hist(baserow, bins)
        # axapprox.hist(approxrow, bins)
        # axdiff.plot(diff, )


        plt.tight_layout()
        plt.savefig('{}/profile-t{:06}.jpeg'.format(outdir, turn), bbox_inches='tight', dpi=900)
        plt.show()
        plt.close()

