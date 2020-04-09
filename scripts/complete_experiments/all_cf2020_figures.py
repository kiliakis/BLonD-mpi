import subprocess
import os
import sys
import argparse

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"
this_filename = sys.argv[0].split('/')[-1]
project_dir = this_directory + '../../'

parser = argparse.ArgumentParser(description='Run all the CF2020 figure scripts.',
                                 usage='python {} -i reference_results'.format(this_filename))

parser.add_argument('-i', '--inputdir', type=str, default=os.path.join(project_dir, 'reference_results'),
                    help='The directory with the results.')


parser.add_argument('-c', '--cases', type=str, nargs='+', default=['lhc', 'sps', 'ps'],
                    choices=['lhc', 'sps', 'ps', 'ex01'],
                    help='The test-case to plot.')


if __name__ == '__main__':
    args = parser.parse_args()
    plot_scripts = ['intermediate_effect_analysis.py',
                    'strong_scaling_experiment.py',
                    'workers_per_node_evaluation.py',
                    'load_imbalance_spread.py',
                    'time_breakdown_strong_scaling.py',
                    'mpi_implementations_bench.py',
                    'weak_scaling_experiment.py']
    plot_scripts = [os.path.join(this_directory, ps) for ps in plot_scripts]

    for ps in plot_scripts:
        print('Calling script: ', ps)
        command = ['python', ps, '-i', args.inputdir, '-c'] + args.cases
        completedProcess = subprocess.run(command, env=os.environ.copy(),
                                          stdout=sys.stdout, stderr=sys.stderr)
        if completedProcess.returncode == 0:
            print('{} completed successfully'.format(ps))
