import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
import h5py
import argparse
import sys
import os

parser = argparse.ArgumentParser(
    description='Calculate the error in the histogram.')


parser.add_argument('-a', '--approx', type=str, nargs='+',
                    help='Approximate file names.')

parser.add_argument('-o', '--outfile', type=str, default=None,
                    help='File to store the results.')

parser.add_argument('-b', '--base', type=str, nargs='+',
                    help='Base file names')

parser.add_argument('-t', '--ts', type=int, default=100,
                    help='Qs^(-1)')


data = {}


def calc_error(f1, f2):

    if f1 in data:
        f1data = data[f1]
    else:
        f1 = h5py.File(f1, 'r')
        f1data = f1['Slices']['n_macroparticles'].value

    if f2 in data:
        f2data = data[f2]
    else:
        f2 = h5py.File(f2, 'r')
        f2data = f2['Slices']['n_macroparticles'].value

    errors = np.empty(len(f1data)-ts, dtype=float)
    f1sum = np.sum(f1data[0:ts-1], axis=0)
    f2sum = np.sum(f2data[0:ts-1], axis=0)
    for i in range(len(f1data)-ts):
        f1sum += f1data[ts-1]
        f2sum += f2data[ts-1]
        errors[i] = np.sum((f1sum/ts - f2sum/ts)**2)
        f1sum -= f1data[i]
        f2sum -= f2data[i]

        # errors[i] = np.sum((np.sum(f1data[i:i+ts], axis=0) / ts
        #                     - np.sum(f2data[i:i+ts], axis=0)/ts)**2)

    if f1 not in data:
        # data[f1] = f1data
        f1.close()

    if f2 not in data:
        # data[f2] = f1data
        f2.close()

    return errors


def getDim(file):
    if file in data:
        return len(data[file])
    else:
        h5file = h5py.File(file, 'r')
        h5data = h5file['Slices']['n_macroparticles'].value
        data[file] = h5data
        h5file.close()
        return len(h5data)-ts


if __name__ == '__main__':
    args = parser.parse_args()
    args = vars(args)

    basefiles = args['base']
    approxfiles = args['approx']
    outfile = args['outfile']
    ts = args['ts']

    # if not os.path.exists(outfile):
    #     os.makedirs(outfile)

    dims = ((len(basefiles) * (len(basefiles)-1)) // 2
            + len(approxfiles), getDim(basefiles[0]))

    outh5file = h5py.File(outfile + '.h5', 'w')
    outh5file.create_group('default')
    outh5 = outh5file['default']

    outh5.create_dataset('errors', dims, compression='gzip',
                         compression_opts=9, dtype='float64')

    outh5.create_dataset('seeds', (dims[0], 2), compression='gzip',
                         compression_opts=9, dtype='int32')

    outh5.create_dataset('reduce', (dims[0], 2), compression='gzip',
                         compression_opts=9, dtype='int32')

    # outh5.create_dataset('average_base_error', (dims[0],) , compression='gzip',
    #                      compression_opts=9, dtype='float64')

    # outh5.create_dataset('average_base_error', (dims[0],) , compression='gzip',
    #                      compression_opts=9, dtype='float64')


    errors = []
    k = 0
    for i in range(len(basefiles)):
        bf1 = basefiles[i]
        seed1 = int(bf1.split('-se')[1].split('.h5')[0])
        for j in range(i+1, len(basefiles)):
            bf2 = basefiles[j]
            seed2 = int(bf2.split('-se')[1].split('.h5')[0])
            print('\n------------------------')
            print('Calculating the error between the basefiles with seeds {} and {}'.format(
                seed1, seed2))
            err = calc_error(bf1, bf2)
            outh5['errors'][k] = err
            outh5['seeds'][k] = [seed1, seed2]
            outh5['reduce'][k] = [1, 1]
            errors.append(err)
            k += 1

            # print('The error is: {}'.format(errors[-1]))
    avg_base_error = np.mean(errors, axis=0)

    std_base_error = np.std(errors, axis=0)

    seed = int(approxfiles[0].split('-se')[1].split('.h5')[0])

    todelete = []
    for key in data:
        if('se{}.h5'.format(seed) not in key):
            todelete.append(key)

    for key in todelete:
        del data[key]


    for approxfile in approxfiles:
        seed = int(approxfile.split('-se')[1].split('.h5')[0])
        red = int(approxfile.split('-r')[1].split('-')[0])
        for file in basefiles:
            if('se{}.h5'.format(seed) in file):
                bf = file

        err = calc_error(approxfile, bf)

        outh5['errors'][k] = err
        outh5['seeds'][k] = [seed, seed]
        outh5['reduce'][k] = [1, red]
        k += 1
        err = err / avg_base_error

        print('Total error is for approxfile {} is :\n'.format(approxfile), err)

    outh5file.close()
