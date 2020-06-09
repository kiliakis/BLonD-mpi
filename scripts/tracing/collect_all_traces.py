#!/usr/bin/python
import os
import numpy as np
import sys
import fnmatch
# import csv
import argparse
import subprocess

this_directory = os.path.dirname(os.path.realpath(__file__)) + '/'
this_filename = sys.argv[0].split('/')[-1]


parser = argparse.ArgumentParser(description='Generate worker trace from pickle files.',
                                 usage='python script.py [-i indir] [-p pattern]')

parser.add_argument('-p', '--pattern', type=str, default='worker-*.*',
                    help='The input file names pattern. '
                    ' Default: worker-*.log')

parser.add_argument('-o', '--outdir', type=str,
                    default=None,
                    help='The directory to store the traces.'
                    ' Default: Same as the input directory')

# parser.add_argument('-f', '--filename', type=str,
#                     default=None,
#                     help='The output filename.'
#                     ' Default: Automatically assigned.')

parser.add_argument('-t', '--tracescript', type=str,
                    default=this_directory+'generate_trace.py',
                    help='The script to use to plot the traces.')


parser.add_argument('-i', '--indir', type=str, default=None,
                    help='The directory to search for trace files.')

parser.add_argument('-skip', '--skip', type=int, default=5,
                    help='How many points to skip'
                    ' Default: 5, plot every 5 points.')

parser.add_argument('-w', '--window', type=int, default=10,
                    help='Width of running mean to smoothen spiky curves.'
                    ' Default: 10.')


# parser.add_argument('-from', '--from', type=str, default='log', choices=['pickle', 'log'],
#                     dest='fromfile',
#                     help='Use log files as input or pickle files.'
#                     ' Default: log.')


# parser.add_argument('-r', '--report', type=str, choices=['comm-comp', 'avg', 'delta'],
#                     default='comm-comp',
#                     help='Choose from 3 report types: comm-comp, avg, delta'
#                     ' Default: comm-comp.')

# parser.add_argument('-s', '--show', action='store_true',
# help='Show the plots.')


if __name__ == '__main__':
    args = parser.parse_args()
    file_pattern = args.pattern
    indir = args.indir
    # files = fnmatch.filter(os.listdir(indir), file_pattern)
    outdir = args.outdir
    if not outdir:
        outdir = os.path.join(indir, '/traces')
    os.makedirs(outdir, exist_ok=True)

    failed = []
    for root, dirs, files in os.walk(indir):
        # we are looking to match the
        last_level = os.path.basename(os.path.normpath(root))
        if last_level == 'log' and len(fnmatch.filter(files, file_pattern)) > 0:
            print(f'\nExtracting traces from dir: {root}')
            experiment = root.split('/')[-4]
            date = root.split('/')[-2]
            lb = root.split('_lb')[1].split('_')[0]
            filename = f'{experiment}-{lb}-{date}'
            output = subprocess.run(['python', args.tracescript,
                                     '--from', 'log',
                                     '--pattern', file_pattern,
                                     '--skip', str(args.skip),
                                     '--window', str(args.window),
                                     # '--show', str(args.show),
                                     '--outdir', outdir,
                                     '--indir', root,
                                     '--filename', filename],
                                    stdout=sys.stdout,
                                    stderr=subprocess.STDOUT)
            if output.returncode != 0:
                failed.append(root)
    if failed:
        print('[{}] The following plots raised an error:'.format(
            this_filename[:-3]))
        for f in failed:
            print(f)
