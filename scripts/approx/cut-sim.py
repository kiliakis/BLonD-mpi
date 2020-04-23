# import matplotlib.pyplot as plt
# from matplotlib import gridspec
import numpy as np
import h5py
import argparse
import os

parser = argparse.ArgumentParser(
    description='Keep only the first turns of sim.')


parser.add_argument('-i', '--indir', type=str, default=None,
                    help='Directory with the input raw files.')

parser.add_argument('-f', '--files', type=str, default=[], nargs='+',
                    help='Directory with the input raw files.')

parser.add_argument('-t', '--turns', type=int, default=None,
                    help='Turns to keep.')


def cut_sim(f, after):
    f = h5py.File(f, 'r+')

    turns = f['Slices']['turns'].value
    idx = np.where(turns > after)[0][0]

    for key in f['Slices'].keys():
        f['Slices'][key][...] = f['Slices'][key].value[:idx]


    f.close()


if __name__ == '__main__':
    args = parser.parse_args()
    args = vars(args)

    turns = args['turns']
    indir = args['indir']
    files = args['files']
    if indir:

        for file in os.listdir(indir):
            print(indir + '/' + file)
            try:
                cut_sim(indir + '/' + file, turns)
            except:
                print(file)

    if files:
        for file in files:
            print(file)
            try:
                cut_sim(file, turns)
            except:
                print(file)
