# import matplotlib.pyplot as plt
# from matplotlib import gridspec
import numpy as np
import h5py
import argparse
import sys
import os

parser = argparse.ArgumentParser(
    description='Calculate the error in the histogram.')


parser.add_argument('-i', '--indir', type=str, default=None,
                    help='Directory with the input raw files.')

# parser.add_argument('-a', '--approx', type=str, nargs='+',
#                     help='Approximate file names.')

parser.add_argument('-o', '--outfile', type=str, default=None,
                    help='File to store the results.')

# parser.add_argument('-b', '--base', type=str, nargs='+',
#                     help='Base file names')

parser.add_argument('-t', '--ts', type=int, default=[1, 100], nargs='+',
                    help='Qs^(-1)')

# parser.add_argument('-q', '--quantities', type=str, default=[], nargs='+',
#                     help='The quantities for which the error will be computed.')

quantities = ['n_macroparticles', 'mean_dE', 'mean_dt', 'std_dt', 'std_dE']


def calc_error(f1, f2, ts):
    f1 = h5py.File(f1, 'r')
    f2 = h5py.File(f2, 'r')

    errors = {}

    for key in quantities:

        f1data = f1['Slices'][key].value
        f2data = f2['Slices'][key].value

        errors[key] = np.empty(len(f1data)-ts, dtype=float)
        f1sum = np.sum(f1data[0:ts-1], axis=0)
        f2sum = np.sum(f2data[0:ts-1], axis=0)
        for i in range(len(f1data)-ts):
            f1sum += f1data[ts-1]
            f2sum += f2data[ts-1]
            errors[key][i] = np.sum((f1sum/ts - f2sum/ts)**2)
            f1sum -= f1data[i]
            f2sum -= f2data[i]

            # errors[i] = np.sum((np.sum(f1data[i:i+ts], axis=0) / ts
            #                     - np.sum(f2data[i:i+ts], axis=0)/ts)**2)
    f1.close()
    f2.close()

    return errors


def getDim(file):
    h5file = h5py.File(file, 'r')
    length = h5file['Slices']['n_macroparticles'].len()
    h5file.close()
    return length


def calc_and_write(indir, basefiles, approxfiles, reffile, bunch, ts, outh5):
    errors = {}
    # idx = 0
    for i in range(len(basefiles)):
        bf1 = basefiles[i]
        seed1 = int(bf1.split('_s')[1].split('_t')[0])
        for j in range(i+1, len(basefiles)):
            bf2 = basefiles[j]
            seed2 = int(bf2.split('_s')[1].split('_t')[0])
            # print('\n------------------------')
            print('Calculating the error between the basefiles with ' +
                  'seeds {} and {}'.format(seed1, seed2))
            err = calc_error(indir + '/' + bf1, indir + '/' + bf2, ts)
            for key in err.keys():
                outh5[key][calc_and_write.idx] = err[key]
                if key not in errors:
                    errors[key] = []
                errors[key].append(err[key])
            outh5['seeds'][calc_and_write.idx] = [seed1, seed2]
            outh5['reduce'][calc_and_write.idx] = [1, 1]

            calc_and_write.idx += 1

            # print('The error is: {}'.format(errors[-1]))
    # avg_base_error = np.mean(errors, axis=0)
    avg_base_error = {}
    for key in errors.keys():
        avg_base_error[key] = np.mean(errors[key], axis=0)

    # std_base_error = np.std(errors, axis=0)

    for approxfile in approxfiles:
        seed = int(approxfile.split('_s')[1].split('_t')[0])
        red = int(approxfile.split('_r')[1].split('.h5')[0])
        print('Calculating the error between the files with ' +
              'reduce {} and {}'.format(red, 1))

        err = calc_error(indir + '/' + approxfile,
                         indir + '/' + reffile, ts)
        for key in err.keys():
            outh5[key][calc_and_write.idx] = err[key]
            # err[key] = err[key] / avg_base_error[key]
            print('{} error for approxfile {} is :{}'.format(
                key, approxfile, err[key] / avg_base_error[key]))
        # outh5['errors'][calc_and_write.idx] = err
        outh5['seeds'][calc_and_write.idx] = [seed, seed]
        outh5['reduce'][calc_and_write.idx] = [1, red]
        calc_and_write.idx += 1


# calc_and_write.idx = 0


if __name__ == '__main__':
    args = parser.parse_args()
    args = vars(args)

    outfile = args['outfile']
    tss = args['ts']
    indir = args['indir']
    # all these per bunch
    # basefiles: files with reduce == 1
    # approxfiles: files with reduce != 1
    # reference file: reduce == 1 and same seed as the approxfiles

    infiles_d = {}
    for file in os.listdir(indir):
        # print(file)
        bunches = file.split('_b')[1].split('_r')[0]
        red = file.split('_r')[1].split('.h5')[0]
        if bunches not in infiles_d:
            infiles_d[bunches] = {'basefiles': [],
                                  'approxfiles': [],
                                  'reffile': '',
                                  'approxseed': '0'}
        if red == '1':
            infiles_d[bunches]['basefiles'].append(file)
        else:
            infiles_d[bunches]['approxfiles'].append(file)
            infiles_d[bunches]['approxseed'] = file.split('_s')[
                1].split('_t')[0]

    for file in os.listdir(indir):
        bunches = file.split('_b')[1].split('_r')[0]
        seed = file.split('_s')[1].split('_t')[0]
        red = file.split('_r')[1].split('.h5')[0]
        if seed == infiles_d[bunches]['approxseed'] and red == '1':
            infiles_d[bunches]['reffile'] = file

    for ts in tss:
        filename = outfile + '/ts' + str(ts) + '.h5'
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

        outh5file = h5py.File(filename, 'w')
        for bunch, data in infiles_d.items():
            calc_and_write.idx = 0
            datasets = len(data['basefiles']) * \
                (len(data['basefiles'])-1) // 2 + len(data['approxfiles'])
            print('Ts: {}, Bunches: {}'.format(ts, bunch))

            turns = getDim(indir + '/' + data['reffile'])
            outh5file.create_group('bunch_{}'.format(bunch))
            outh5 = outh5file['bunch_{}'.format(bunch)]

            outh5.create_dataset('n_macroparticles', (datasets, turns - ts),
                                 compression='gzip', compression_opts=4,
                                 dtype='float64')

            outh5.create_dataset('mean_dE', (datasets, turns - ts),
                                 compression='gzip', compression_opts=4,
                                 dtype='float64')

            outh5.create_dataset('mean_dt', (datasets, turns - ts),
                                 compression='gzip', compression_opts=4,
                                 dtype='float64')

            outh5.create_dataset('std_dE', (datasets, turns - ts),
                                 compression='gzip', compression_opts=4,
                                 dtype='float64')
            outh5.create_dataset('std_dt', (datasets, turns - ts),
                                 compression='gzip', compression_opts=4,
                                 dtype='float64')

            outh5.create_dataset('seeds', (datasets, 2), compression='gzip',
                                 compression_opts=4, dtype='int32')

            outh5.create_dataset('reduce', (datasets, 2), compression='gzip',
                                 compression_opts=4, dtype='int32')

            calc_and_write(indir, data['basefiles'], data['approxfiles'],
                           data['reffile'], bunch, ts, outh5)

        outh5file.close()
