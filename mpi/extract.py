#!/usr/bin/python
import os
import csv
import sys
import numpy as np
from extract.extract_utilities import *

import subprocess
import argparse

# usage: python report.py -r [summary | hw-events] -i [indir] -o [outfile]

parser = argparse.ArgumentParser(description='Generate a csv report from the input raw data.',
                                 usage='python extract.py -i [indir] -o [outfile]')

# parser.add_argument('-r', '--report', type=str, default='hw-events',
#                     help='The report type. (summary or hw-events)'
#                     ' Default: hw-events')

parser.add_argument('-o', '--outfile', type=str, default=None,
                    help='The file to save the report.'
                    ' Default: (indir)-report.csv')

parser.add_argument('-i', '--indir', type=str, default=None,
                    help='The directory containing the collected data.')


# header = ['version', 'cc', 'vec', 'tcm', 'turns', 'points', 
#           'slices', 'threads','time(ms)', 'std(%)']

def extract_results(input, outfile):
    outdir = os.path.dirname(outfile)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    records = []
    for dirs, subdirs, files in os.walk(input):
        for file in files:
            if('.txt' not in file):
                continue
            times = []
            print(file)
            turns = string_between(file, 'i', '-')
            points = string_between(file, 'p', '-')
            slices = string_between(file, 's', '-')
            threads = string_between(file, 't', '-')
            cc = file.split('-')[4]
            vec = file.split('-')[5]
            tcm = file.split('-')[6].split('.txt')[0]
            for line in open(os.path.join(dirs, file), 'r'):
                line = get_line_matching(line, [application])
                if not line:
                    continue
                line = line.split('\t')
                app = line[0].split(application)[1][1:]
                time = line[2]
                times.append(float(time))
            if times:
                records.append([app, cc, vec, tcm, turns, points, slices, threads,
                                '%.1lf' % np.mean(times),
                                '%.1lf' % (100 * np.std(times) / np.mean(times))])
    # print(records)
    records.sort(key=lambda a: (a[0], a[1], a[2], a[3],
                                int(a[4]), int(a[5]), int(a[6])))
    out = open(outfile, 'w')
    writer = csv.writer(out, delimiter='\t')
    writer.writerow(header)
    writer.writerows(records)


if __name__ == '__main__':
    args = parser.parse_args()
    if args.indir == None:
        print("You have to specify the input directory.")
        exit(-1)
    if args.outfile == None:
        args.outfile = args.indir.split('/')[-1] + '-report.csv'
    extract_results(args.indir, args.outfile)
