import subprocess
import os
import sys
from cycler import cycle
from math import ceil
from datetime import datetime
import numpy as np
import random
import yaml

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"
this_filename = sys.argv[0].split('/')[-1]

yc = yaml.load(open(this_directory + 'config.yml', 'r'))
result_dir = yc['result_dir'] + '{}/{}/{}/{}'
os.environ['PYTHONPATH'] = '{}:{}'.format(
    yc['blond_repos'], os.environ['PYTHONPATH'])
job_name_form = '_p{}_b{}_s{}_t{}_w{}_o{}_N{}_r{}_m{}_seed{}_approx{}_mpi{}'

configs = {



    # 'EX01-mpich3-approx2': {
    #     'exe': cycle([yc['exe_home'] + 'EX_01_Acceleration.py']),
    #     'p': cycle([8000000]),
    #     'b': cycle([1]),  # 96
    #     's': cycle([1000]),
    #     't': cycle([10000]),
    #     'm': cycle([0]),
    #     'seed': cycle([0]),
    #     'reduce': cycle([1]),
    #     'load': cycle([0.0]),
    #     'mtw': cycle([50]),
    #     'approx': cycle([2]),
    #     'log': cycle([True]),
    #     'lb': cycle(['off']),
    #     'lba': cycle([0]),
    #     'timing': cycle(['-time']),  # otherwise pass -time
    #     'w': [] +
    #     list(np.arange(2, 17, 2)),
    #     'o': cycle([10]),
    #     'mpi': cycle(['mpich3']),
    #     'time': cycle([90]),
    #     'partition': cycle(['be-long'])
    # },

    'EX01-lb-mpich3': {
        'exe': cycle([yc['exe_home'] + 'EX_01_Acceleration_lb.py']),
        'p': cycle([8000000]),
        'b': cycle([1]),  # 96
        's': cycle([1000]),
        't': cycle([10000]),
        'm': cycle([0]),
        'seed': cycle([0]),
        'reduce': cycle([1]),
        'load': cycle([0.0]),
        'mtw': cycle([50]),
        'approx': cycle([0]),
        'log': cycle([True]),
        'lb': cycle(['dynamic']),
        'lba': cycle([10]),
        'timing': cycle(['-time']),  # otherwise pass -time
        'w': [] +
        list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mpich3']),
        'time': cycle([10]),
        'partition': cycle(['be-short'])
    },

    'EX01-lb-mpich3-approx2': {
        'exe': cycle([yc['exe_home'] + 'EX_01_Acceleration_lb.py']),
        'p': cycle([8000000]),
        'b': cycle([1]),  # 96
        's': cycle([1000]),
        't': cycle([10000]),
        'm': cycle([0]),
        'seed': cycle([0]),
        'reduce': cycle([1]),
        'load': cycle([0.0]),
        'mtw': cycle([50]),
        'approx': cycle([2]),
        'log': cycle([True]),
        'lb': cycle(['dynamic']),
        'lba': cycle([10]),
        'timing': cycle(['-time']),  # otherwise pass -time
        'w': [] +
        list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mpich3']),
        'time': cycle([10]),
        'partition': cycle(['be-short'])
    },
}

repeats = 5


total_sims = repeats * \
    sum([len(y['w']) for y in configs.values()])

print("Total runs: ", total_sims)
current_sim = 0
os.chdir(yc['blond_home'])


# compile first
# subprocess.call(['srun', '-t1', '-N1', '-n1', '-p',
#                  'be-short', 'bash', setup_script])


for analysis, config in configs.items():
    ps = config['p']
    bs = config['b']
    ss = config['s']
    ts = config['t']
    ws = config['w']
    oss = config['o']
    rs = config['reduce']
    exes = config['exe']
    # Ns = config['N']
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
                                        r, m, seed, approx, mpi)
        mpi_config = yc['mpi_versions'][mpi]

        for i in range(repeats):
            timestr = datetime.now().strftime('%d%b%y.%H-%M-%S')
            timestr = timestr + '-' + str(random.randint(0, 100))
            output = result_dir.format(
                analysis, job_name, timestr, 'output.txt')
            error = result_dir.format(analysis, job_name, timestr, 'error.txt')
            monitorfile = result_dir.format(
                analysis, job_name, timestr, 'monitor')
            log_dir = result_dir.format(analysis, job_name, timestr, 'log')
            report_dir = result_dir.format(
                analysis, job_name, timestr, 'report')
            for d in [log_dir, report_dir]:
                if not os.path.exists(d):
                    os.makedirs(d)
            # exe_args = ['-n', str('python', exe,
            exe_args = [
                # '-n', str(w),
                mpi_config['module_name'],
                mpi_config['path'],
                mpi_config['path']+'python', exe,
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
                          '-J', analysis + job_name.split('/')[0] + '-' + str(i)]

            all_args = ['sbatch'] + batch_args + [yc['batch_script']] + exe_args
            subprocess.call(all_args, stdout=stdout,
                            stderr=stdout, env=os.environ.copy())
            # sleep(5)
            current_sim += 1
            print("%lf %% is completed" % (100.0 * current_sim /
                                           total_sims))
