import subprocess
import os
import sys
from cycler import cycle
from math import ceil
from datetime import datetime
import random
import yaml
import argparse
import importlib

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"
this_filename = sys.argv[0].split('/')[-1]

parser = argparse.ArgumentParser(description='Submit the mpi jobs to the batch scheduler.',
                                 usage='python scan_mpi.py -i in.yml')

parser.add_argument('-i', '--input', type=str, default=None, nargs='+',
                    help='The input file(s) containing the configs to run.')

parser.add_argument('-flavor', '--flavor', type=str, default='small', choices=['small', 'medium', 'large'],
                    help='The input size flavor to run.')

parser.add_argument('-o', '--output', type=str, default='./results/raw',
                    help='Output directory to store the output data. Default: ./results/raw')


if __name__ == '__main__':
    args = parser.parse_args()
    blond_home = os.path.join(this_directory, '../../')
    exe_home = os.path.join(blond_home, '__EXAMPLES/main_files')
    batch_script = os.path.join(blond_home, 'scripts/other/batch-simple.sh')
    batch_setup = os.path.join(blond_home, 'scripts/other/batch-setup.sh')
    top_result_dir = args.output

    mpi_versions = {
        'openmpi3': {
            'module_name': 'mpi/openmpi/3.0.0',
        },
        'mvapich2': {
            'module_name': 'mpi/mvapich2/2.3',
        },
        'mpich3': {
            'module_name': 'mpi/mpich/3.2.1',
        }
    }

    yc = yaml.load(open(this_directory + 'config.yml', 'r'))
    for input in args.input:
        module = importlib.import_module(input.split('/')[-1].replace('.py', ''),
                                         package=os.path.dirname(input).replace('/', '.'))
        
        configs = module.configs[args.flavor]
        case = module.case
        run_configs = module.run_configs

        result_dir = top_result_dir + '/{}/{}/{}/{}/{}'
        job_name_form = '_p{}_b{}_s{}_t{}_w{}_o{}_N{}_red{}_mtw{}_seed{}_approx{}_mpi{}_lb{}_lba{}_monitor{}_tp{}_'
        total_sims = 0
        for rc in run_configs:
            total_sims += configs[rc]['repeats'] * len(configs[rc]['w'])

        print("Total runs: ", total_sims)
        current_sim = 0
        for analysis in run_configs:
            config = configs[analysis]
            ps = config['p']
            bs = config['b']
            ss = config['s']
            ts = config['t']
            ws = config['w']
            oss = config['o']
            rs = config['reduce']
            exes = config['exe']
            times = config['time']
            partitions = config['partition']
            mtws = config['mtw']
            ms = config['m']
            seeds = config['seed']
            approxs = config['approx']
            timings = config['timing']
            mpis = config['mpi']
            logs = config['log']
            lbs = config['lb']
            lbas = config['lba']
            repeats = config['repeats']
            tps = config['withtp']
            stdout = open(analysis + '.txt', 'w')

            for (p, b, s, t, r, w, o, time,
                 partition, mtw, m,
                 seed, exe, approx, timing, mpi,
                 log, lb, lba, tp) in zip(ps, bs, ss, ts, rs, ws,
                                          oss, times, partitions,
                                          mtws, ms, seeds,
                                          exes, approxs, timings, mpis,
                                          logs, lbs, lbas, tps):

                N = (w * o + 20-1) // 20

                job_name = job_name_form.format(p, b, s, t, w, o, N,
                                                r, mtw, seed, approx, mpi,
                                                lb, lba, m, tp)
                mpiconf = mpi_versions[mpi]

                for i in range(repeats):
                    timestr = datetime.now().strftime('%d%b%y.%H-%M-%S')
                    timestr = timestr + '-' + str(random.randint(0, 100))
                    output = result_dir.format(
                        case, analysis, job_name, timestr, 'output.txt')
                    error = result_dir.format(
                        case, analysis, job_name, timestr, 'error.txt')
                    monitorfile = result_dir.format(
                        case, analysis, job_name, timestr, 'monitor')
                    log_dir = result_dir.format(
                        case, analysis, job_name, timestr, 'log')
                    report_dir = result_dir.format(
                        case, analysis, job_name, timestr, 'report')
                    for d in [log_dir, report_dir]:
                        if not os.path.exists(d):
                            os.makedirs(d)
                    # exe_args = ['-n', str('python', exe,
                    exe_args = [
                        mpiconf['module_name'],
                        'python', os.path.join(exe_home, exe),
                        '-p', str(int(p)), '-s', str(s),
                        '-b', str(int(b)),
                        '-t', str(t), '-o', str(o), '-seed', str(seed),
                        str(timing), '-timedir', report_dir,
                        '-m', str(m), '-monitorfile', monitorfile,
                        '--reduce', str(r), '-mtw', str(mtw),
                        '-approx', str(approx),
                        '-lb', lb, '-lba', str(lba),
                        '-withtp', tp]
                    if log:
                        exe_args += ['--log', '-logdir', log_dir]

                    print(job_name, timestr)
                    batch_args = ['-N', str(N), '-n', str(w),
                                  '--ntasks-per-node', str(ceil(w/N)),
                                  '-c', str(o),  # str(o),
                                  '-t', str(time), '-p', partition,
                                  '-o', output,
                                  '-e', error,
                                  '-J', case + '-' + analysis + job_name.split('/')[0] + '-' + str(i)]

                    all_args = ['sbatch'] + batch_args + \
                        [batch_script] + exe_args
                    subprocess.call(all_args, stdout=stdout,
                                    stderr=stdout, env=os.environ.copy())
                    current_sim += 1
                    print("%lf %% is completed" % (100.0 * current_sim
                                                   / total_sims))
