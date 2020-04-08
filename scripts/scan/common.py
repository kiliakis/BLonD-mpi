import os

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"

blond_home = os.path.abspath(os.path.join(this_directory, '../../'))
exe_home = os.path.join(blond_home, '__EXAMPLES/main_files')
batch_script = os.path.join(blond_home, 'scripts/scan/batch-simple.sh')

mpirun = 'mpirun'
python = 'python'
batchsubmit = 'sbatch'
batchrun = 'srun'

cores_per_cpu = 20
# Batch flags
nodes = '-N'
workers = '-n'
tasks_per_node = '--ntasks-per-node'
cores = '-c'
time = '-t'
output = '-o'
error = '-e'
jobname = '-J'

# default batch flags
default_batch_args = [
    '--mem', '0',
    '--export', 'ALL',
    '--hint', 'nomultithread',
    '--partition', 'inf-short'
]
