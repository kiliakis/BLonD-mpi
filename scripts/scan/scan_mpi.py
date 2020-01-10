import subprocess
import os
import sys
from cycler import cycle
from math import ceil
from datetime import datetime
import numpy as np
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

if __name__ == '__main__':
    args = parser.parse_args()
    yc = yaml.load(open(this_directory + 'config.yml', 'r'))
    for input in args.input:
        module = importlib.import_module(input.split('/')[-1].replace('.py', ''),
                                         package=os.path.dirname(input).replace('/', '.'))
        configs = module.configs
        case = module.case
        run_configs = module.run_configs

        result_dir = yc['result_dir'] + '{}/{}/{}/{}/{}'
        os.environ['PYTHONPATH'] = '{}:{}'.format(
            yc['blond_repos'], os.environ['PYTHONPATH'])
        job_name_form = '_p{}_b{}_s{}_t{}_w{}_o{}_N{}_red{}_mtw{}_seed{}_approx{}_mpi{}_lb{}_lba{}_monitor{}_'

        total_sims = 0
        for rc in run_configs:
            total_sims += configs[rc]['repeats'] * len(configs[rc]['w'])

        print("Total runs: ", total_sims)
        current_sim = 0
        # compile first
        # subprocess.call(['srun', '-t1', '-N1', '-n1', '-p',
        #                  'be-short', 'bash', setup_script])
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
            loads = config['load']
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
            stdout = open(analysis + '.txt', 'w')

            for (p, b, s, t, r, w, o, time,
                 partition, load, mtw, m,
                 seed, exe, approx, timing, mpi,
                 log, lb, lba) in zip(ps, bs, ss, ts, rs, ws,
                                      oss, times, partitions,
                                      loads, mtws, ms, seeds,
                                      exes, approxs, timings, mpis,
                                      logs, lbs, lbas):

                N = (w * o + 20-1) // 20

                job_name = job_name_form.format(p, b, s, t, w, o, N,
                                                r, mtw, seed, approx, mpi,
                                                lb, lba, m)
                mpiconf = yc['mpi_versions'][mpi]

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
                        # '-n', str(w),
                        mpiconf['module_name'],
                        mpiconf['path'],
                        mpiconf['path']+'python', yc['exe_home']+exe,
                        '-p', str(int(p)), '-s', str(s),
                        '-b', str(int(b)), '-addload', str(load),
                        '-t', str(t), '-o', str(o), '-seed', str(seed),
                        str(timing), '-timedir', report_dir,
                        '-m', str(m), '-monitorfile', monitorfile,
                        '--reduce', str(r), '-mtw', str(mtw),
                        '-approx', str(approx),
                        '-lb', lb, '-lba', str(lba)]
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
                        [yc['batch_script']] + exe_args
                    subprocess.call(all_args, stdout=stdout,
                                    stderr=stdout, env=os.environ.copy())
                    # sleep(5)
                    current_sim += 1
                    print("%lf %% is completed" % (100.0 * current_sim
                                                   / total_sims))