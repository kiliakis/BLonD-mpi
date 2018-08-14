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

# parser.add_argument('-s', '--slices', type=int,
#                     help='Number of slices.')

# parser.add_argument('-b', '--bunches', type=int,
#                     help='Number of bunches.')

# parser.add_argument('--reduce', type=int,
#                     help='Number of turns to reduce.')


# parser.add_argument('-t', '--turns', type=int, default=2000,
#                     help='Number of simulation turns.'
#                     '\nDefault: 2000')



# parser.add_argument('-o', '--omp', type=int, default=1,
#                     help='Number of openmp threads to use.'
#                     '\nDefault: 1')

# parser.add_argument('-l', '--log', type=str, default=None,
#                     nargs='?', const='logs',
#                     help='Directory to store the log files.'
#                     '\nDefault: Do not generate log files.')

# parser.add_argument('-r', '--report', type=str, default='./',
#                     help='Directory to store the timing reports.'
#                     '\nDefault: Do not generate timing reports.')

# parser.add_argument('-m', '--monitor', type=int, default=0,
#                     help='Monitoring interval (0: no monitor).'
#                     '\nDefault: 0')

# parser.add_argument('-time', '--time', action='store_true',
#                     help='Time the specified regions of interest.'
#                     '\nDefault: No timing.')


# parser.add_argument('-trace', '--trace', action='store_true',
#                     help='Trace the specified regions of interest (MPE).'
#                     '\nDefault: No tracing.')

# parser.add_argument('-tracefile', '--tracefile', type=str, default='mpe-trace',
#                     help='The file name to save the MPE trace (without the file type).'
#                     '\nDefault: mpe-trace')


# parser.add_argument('-d', '--debug', action='store_true',
#                     help='Run workers in debug mode.'
#                     '\nDefault: No')


# def parse():
#     args = parser.parse_args()
#     return vars(args)

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

    assert basedata.shape == approxdata.shape

    turns = baseh5['Slices']['turns'].value
    # slices = np.range()

    for i in range(len(turns)):
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
        axbase.set_ylim([0, 4000])

        axapprox = plt.subplot(gs[0, 1])
        axapprox.set_title('Reduce every {} turns'.format(reduce_turns))
        axapprox.set_ylim([0, 4000])

        axdiff = plt.subplot(gs[1, :])
        axdiff.set_title('Diff (base - approx)')
        axdiff.set_ylim([-250, 250])

        axbase.plot(baserow, ls='-', marker='')
        axapprox.plot(approxrow, ls='-', marker='')
        axdiff.plot(diff, ls='-', marker='')

        plt.tight_layout()
        plt.savefig('{}/profile-t{:06}.png'.format(outdir, turn), bbox_inches='tight')
        # plt.show()
        plt.close()

