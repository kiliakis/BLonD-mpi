import subprocess
import os
import sys
from datetime import datetime
import random
import yaml
import argparse
import numpy as np

import common

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"
this_filename = sys.argv[0].split('/')[-1]

parser = argparse.ArgumentParser(description='Run locally the MPI experiments.',
                                 usage='python {} -t lhc sps ps'.format(this_filename[:-3]))

parser.add_argument('-e', '--environment', type=str, default='local', choices=['local', 'cluster'],
                    help='The environment to run the scan.')

parser.add_argument('-t', '--testcases', type=str, default=['lhc,sps,ps'],
                    help='Which testcases to run. Default: all')

parser.add_argument('-o', '--output', type=str, default='./results/local',
                    help='Output directory to store the output data. Default: ./results/local')


if __name__ == '__main__':
    args = parser.parse_args()
    top_result_dir = args.output

    for tc in args.testcases.split(','):
        yc = yaml.load(open(this_directory + '/{}_configs.yml'.format(tc), 'r'),
                       Loader=yaml.FullLoader)[args.environment]

        result_dir = top_result_dir + '/{}/{}/{}/{}/{}'

        job_name_form = '_p{}_b{}_s{}_t{}_w{}_o{}_N{}_red{}_mtw{}_seed{}_approx{}_mpi{}_lb{}_lba{}_monitor{}_tp{}_prec{}_'

        total_sims = 0
        for rc in yc['run_configs']:
            total_sims += yc['configs'][rc]['repeats'] * \
                len(yc['configs'][rc]['workers'])

        print("Total runs: ", total_sims)
        current_sim = 0

        for analysis in yc['run_configs']:
            config = yc['configs'][analysis]
            ps = config['particles']
            bs = config['bunches']
            ss = config['slices']
            ts = config['turns']
            ws = config['workers']
            oss = config['omp']
            rs = config['reduce']
            exes = config['exe']
            times = config['time']
            # partitions = config['partition']
            mtws = config.get('mtw', ['0']*len(ps))
            ms = config.get('monitor', ['0']*len(ps))
            seeds = config.get('seed', ['0']*len(ps))
            approxs = config['approx']
            timings = config['timing']
            mpis = config['mpi']
            logs = config['log']
            lbs = config['loadbalance']
            lbas = config['loadbalancearg']
            repeats = config['repeats']
            tps = config['withtp']
            precs = config['precision']

            for (p, b, s, t, r, w, o, time,
                 mtw, m, seed, exe, approx,
                 timing, mpi, log, lb, lba,
                 tp, prec) in zip(ps, bs, ss, ts, rs, ws,
                            oss, times, mtws, ms, seeds,
                            exes, approxs, timings, mpis,
                            logs, lbs, lbas, tps, precs):

                N = int(max(np.ceil(w * o / common.cores_per_cpu), 1))

                job_name = job_name_form.format(p, b, s, t, w, o, N,
                                                r, mtw, seed, approx, mpi,
                                                lb, lba, m, tp, prec)

                for i in range(repeats):
                    timestr = datetime.now().strftime('%d%b%y.%H-%M-%S')
                    timestr = timestr + '-' + str(random.randint(0, 100))
                    output = result_dir.format(
                        tc, analysis, job_name, timestr, 'output.txt')
                    error = result_dir.format(
                        tc, analysis, job_name, timestr, 'error.txt')
                    monitorfile = result_dir.format(
                        tc, analysis, job_name, timestr, 'monitor')
                    log_dir = result_dir.format(
                        tc, analysis, job_name, timestr, 'log')
                    report_dir = result_dir.format(
                        tc, analysis, job_name, timestr, 'report')
                    for d in [log_dir, report_dir]:
                        if not os.path.exists(d):
                            os.makedirs(d)

                    # For the extract script
                    open(os.path.join(top_result_dir, tc,
                                      analysis, '.analysis'), 'a').close()

                    os.environ['OMP_NUM_THREADS'] = str(o) 
                    if args.environment == 'local':
                        batch_args = [common.mpirun, '-n', str(w)]
                    elif args.environment == 'cluster':
                        batch_args = [
                            common.batchsubmit,
                            common.nodes, str(N),
                            common.workers, str(w),
                            common.tasks_per_node, str(int(np.ceil(w/N))),
                            common.cores, str(o),  # str(o),
                            common.time, str(time),
                            common.output, output,
                            common.error, error,
                            common.jobname, tc + '-' + analysis + job_name.split('/')[0] + '-' + str(i)]
                        batch_args += common.default_batch_args
                        batch_args += [common.batch_script, common.batchrun]

                    exe_args = [
                        common.python, os.path.join(common.exe_home, exe),
                        '--particles', str(int(p)),
                        '--slices', str(s),
                        '--bunches', str(int(b)),
                        '--turns', str(t),
                        '--omp', str(o),
                        '--seed', str(seed),
                        '--time', str(timing), '--timedir', report_dir,
                        '--monitor', str(m), '--monitorfile', monitorfile,
                        '--reduce', str(r),
                        '--mtw', str(mtw),
                        '--precision', str(prec),
                        '--approx', str(approx),
                        '--loadbalance', lb, '--loadbalancearg', str(lba),
                        '--withtp', str(tp),
                        '--log', str(log), '--logdir', log_dir]

                    print(job_name, timestr)

                    all_args = batch_args + exe_args
                    subprocess.call(all_args, stdout=open(output, 'w'),
                                    stderr=open(error, 'w'), env=os.environ.copy())
                    # sleep(5)
                    current_sim += 1
                    print("%lf %% is completed" % (100.0 * current_sim
                                                   / total_sims))
