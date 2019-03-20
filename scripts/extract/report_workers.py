#!/usr/bin/python
import matplotlib.pyplot as plt
import os
import numpy as np
import sys
import fnmatch
import csv
import argparse

this_directory = os.path.dirname(os.path.realpath(__file__)) + '/'


parser = argparse.ArgumentParser(description='Report the avg time spent on communication and computation.',
                                 usage='python script.py [-p file_pattern] [-i indir] [-o outfile]')

parser.add_argument('-p', '--pattern', type=str, default='report-worker-*.csv',
                    help='The report file names pattern. '
                    ' Default: report-worker-*.csv')

parser.add_argument('-o', '--outfile', type=argparse.FileType('w'),
                    default=sys.stdout,
                    help='The file(s) to save the report.'
                    ' Default: Print to the stdout')

parser.add_argument('-i', '--indir', type=str, default='./',
                    help='The directory containing the report files.'
                    ' Default: Use the current working directory.')


parser.add_argument('-r', '--report', type=str, choices=['comm-comp', 'avg'],
                    default='comm-comp',
                    help='Choose from 2 report types: comm-comp or avg.'
                    ' Default: comm-comp.')


def report_comm_comp(indir, files, outfile):
    d = {'comm': [], 'comp': [], 'other': [],
         'total': [], 'serial': [], 'overhead': []}
    for f in files:
        data = np.genfromtxt(indir+'/'+f, dtype=str, delimiter='\t')
        header, data = list(data[0]), data[1:]
        percent_idx = header.index('global_percentage')
        time_idx = header.index('total_time(sec)')
        type_idx = header.index('function')

        # All these are tuples in the form (percent, time)
        d['comm'].append(np.sum([(float(r[percent_idx]), float(r[time_idx]))
                       for r in data if 'comm' in r[type_idx]], axis=0))
        d['comp'].append(np.sum([(float(r[percent_idx]), float(r[time_idx]))
                       for r in data if 'comp' in r[type_idx]], axis=0))
        d['overhead'].append(np.sum([(float(r[percent_idx]), float(r[time_idx]))
                           for r in data if 'overhead' in r[type_idx]], axis=0))
        d['serial'].append(np.sum([(float(r[percent_idx]), float(r[time_idx]))
                         for r in data if 'serial' in r[type_idx]], axis=0))
        d['other'].append(np.sum([(float(r[percent_idx]), float(r[time_idx]))
                        for r in data if 'Other' in r[type_idx]], axis=0))
        d['total'].append(np.sum([(float(r[percent_idx]), float(r[time_idx]))
                        for r in data if 'total_time' in r[type_idx]], axis=0))
        
    string = 'type\tavg_time(sec)\tpercent\tmin%\tmax%\tstd\n'
    for k in ['comp', 'comm', 'serial', 'overhead', 'other', 'total']:
        v = d[k]
        try:
            string += ('%s\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n' %
                       (k, np.mean(v, axis=0)[1], np.mean(v, axis=0)[0],
                        np.min(v, axis=0)[0], np.max(v, axis=0)[0], np.std(v, axis=0)[0]))
        except:
            pass

    outfile.write(string)


def report_avg(indir, files, outfile):
    default_funcs = []
    default_header = []
    acc_data = []
    num = 0
    for f in files:
        data = np.genfromtxt(indir+'/'+f, dtype=str, delimiter='\t')
        header, data = data[0], data[1:]
        funcs, data = data[:, 0], np.array(data[:, 1:], float)
        if len(default_funcs) == 0:
            default_funcs = funcs
        elif not np.array_equal(default_funcs, funcs):
            print('Problem with file: ', indir+'/'+f)
            continue

        if len(default_header) == 0:
            default_header = header
        elif not np.array_equal(default_header, header):
            print('Problem with file: ', indir+'/'+f)
            continue

        if len(acc_data) == 0:
            acc_data = data
        else:
            acc_data += data
        num += 1
    try:
        acc_data = np.around(acc_data/num, 2)
    except TypeError as e:
        print('[Error] with dir ', indir)
        return

    writer = csv.writer(outfile, delimiter='\t')
    writer.writerow(default_header)
    for f, r in zip(default_funcs, acc_data):
        writer.writerow([f]+list(r))


if __name__ == '__main__':
    args = parser.parse_args()
    file_pattern = args.pattern
    indir = args.indir
    files = fnmatch.filter(os.listdir(indir), file_pattern)
    outfile = args.outfile
    if outfile == 'sys.stdout':
        outfile = sys.stdout

    if args.report == 'comm-comp':
        report_comm_comp(indir, files, outfile)
    elif args.report == 'avg':
        report_avg(indir, files, outfile)
