# import matplotlib.pyplot as plt
# from matplotlib import gridspec
import numpy as np
import h5py
import argparse
import os

parser = argparse.ArgumentParser(description='Patch an error with the turns.')


parser.add_argument('-i', '--indir', type=str, default=None,
                    help='Directory with the input raw files.')

parser.add_argument('-f', '--files', type=str, default=[], nargs='+',
                    help='Directory with the input raw files.')

parser.add_argument('-intv', '--interval', type=int, default=None,
                    help='Turn interval')


def patch_turns(f, intv):
    f = h5py.File(f, 'r+')

    turns = f['Slices']['turns']

    turns[...] = np.arange(0, len(turns)*intv, intv)

    f.close()


if __name__ == '__main__':
    args = parser.parse_args()
    args = vars(args)

    intv = args['interval']
    indir = args['indir']
    files = args['files']
    if indir:

        for file in os.listdir(indir):
            print(indir + '/' + file)
            try:
                patch_turns(indir + '/' + file, intv)
            except:
                print(file)

    if files:
        for file in files:
            print(file)
            try:
                patch_turns(file, intv)
            except:
                print(file)
