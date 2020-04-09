import subprocess
import os
import sys
import argparse

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"
this_filename = sys.argv[0].split('/')[-1]

parser = argparse.ArgumentParser(description='Sriver script to run experiments, extract the result and generate the figures.',
                                 usage='python {} -s stage'.format(this_filename[:-3]))

parser.add_argument('-e', '--environment', type=str, choices=['local', 'cluster'], default='local',
                    help='Run the experiments locally or in the cluster. Only for the scan action. Default: local')

parser.add_argument('-a', '--action', type=str, choices=['scan', 'extract', 'plot'],
                    help='Which action to run, required option')

parser.add_argument('-t', '--testcases', default='lhc,sps,ps',
                    help='A comma separated list of the testcases to run. Default: lhc,sps,ps')

parser.add_argument('-d', '--dir', type=str, default='./results/local',
                    help='Directory to store the output data (scan) or to read the input (extract, plot). Default: ./results/local')


scripts = {
    'scan': os.path.join(this_directory, 'scan/scan.py'),
    'extract': os.path.join(this_directory, 'extract/extract_all.py'),
    'plot': os.path.join(this_directory, 'plot/plot_all.py'),
}


if __name__ == '__main__':
    args = parser.parse_args()
    environment = args.environment

    if args.action == 'scan':
        print('Running: {} action.'.format(args.action))
        cmd = ['python', scripts['scan'],
               '-e', args.environment,
               '-o', os.path.join(args.dir, environment),
               '-t', args.testcases]
        subprocess.run(cmd, stdout=sys.stdout,
                       stderr=subprocess.STDOUT, env=os.environ.copy())

    elif args.action == 'extract':
        print('Running: {} action.'.format(args.action))
        cmd = ['python', scripts['extract'], '-i', args.dir,
               '-t', args.testcases]
        subprocess.run(cmd, stdout=sys.stdout,
                       stderr=subprocess.STDOUT, env=os.environ.copy())

    elif args.action == 'plot':
        print('Running: {} action.'.format(args.action))
        cmd = ['python', scripts['plot'], '-i', args.dir,
               '-t', args.testcases] 
        subprocess.run(cmd, stdout=sys.stdout,
                       stderr=subprocess.STDOUT, env=os.environ.copy())
