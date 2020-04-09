import subprocess
import os
import sys
from math import ceil
import yaml
import argparse
import importlib

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"
this_filename = sys.argv[0].split('/')[-1]

parser = argparse.ArgumentParser(description='Sriver script to run experiments, extract the result and generate the figures.',
                                 usage='python {} -s stage'.format(this_filename[:-3]))

parser.add_argument('-e', '--environment', type=str, , choices=['local', 'cluster'], default='local',
                    help='Run the experiments locally or in the cluster. Default: local')

parser.add_argument('-a', '--action', type=str, choices=['scan', 'extract', 'plot'],
                    help='Which action to run, required option')

parser.add_argument('-t', '--testcases', type=str, default=['lhc', 'sps', 'ps'], choices=['lhc', 'sps', 'ps'],
                    help='Which testcases to run. Default: all')

parser.add_argument('-o', '--output', type=str, default='./results',
                    help='Output directory to store the output data. Default: ./results')


scripts = {
    'scan': os.path.join(this_directory, '../scan/scan.py'),
    'extract': os.path.join(this_directory, '../extract/extract_all.py'),
    'plot': os.path.join(this_directory, '../plot/plot_all.py'),
}


if __name__ == '__main__':
    args = parser.parse_args()
    outdir = os.path.abspath(args.output)
    environment = args.environment
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if args.action == 'scan':
        print('Running: {} action: {}'.format(args.action))
        cmd = ['python', scripts[environment]['scan'], '-o',
               os.path.join(outdir, 'raw', environment), '-t'] + args.testcases
        subprocess.run(cmd, stdout=sys.stdout,
                       stderr=subprocess.STDOUT, env=os.environ.copy())

    elif args.action == 'extract':
        print('Running: {} action: {}'.format(args.action))
        cmd = ['python', scripts['extract'], '-i',
               os.path.join(outdir, 'raw', environment), '-t'] + args.testcases
        subprocess.run(cmd, stdout=sys.stdout,
                       stderr=subprocess.STDOUT, env=os.environ.copy())

    elif args.action == 'plot':
        print('Running: {} action: {}'.format(args.action))
        cmd = ['python', scripts['plot'], '-i',
               os.path.join(outdir, 'raw', environment), '-t'] + args.testcases
        subprocess.run(cmd, stdout=sys.stdout,
                       stderr=subprocess.STDOUT, env=os.environ.copy())
