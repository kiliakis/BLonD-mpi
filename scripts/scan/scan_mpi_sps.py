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

    # 'SPS-rand-72B-4MPPB-approx2': {
    #     'exe': cycle([yc['exe_home'] + 'SPS_main_random.py']),
    #     'p': cycle([4000000]),
    #     'b': cycle([48]),  # 72
    #     's': cycle([1408]),
    #     't': cycle([43349]),  # 4000
    #     'm': cycle([50]),
    #     'reduce': cycle([1]),
    #     'load': cycle([0.0]),
    #     'mtw': cycle([0]),
    #     'approx': cycle([2]),
    #     'timing': cycle(['']),  # otherwise pass -time
    #     'seed': [0] * 4 + [1] * 4 + [2] * 4 + [3] * 4 + [4] * 4 + [5] * 4,
    #     'w': []
    #     + [2, 4, 8, 16] * 6,
    #     'o': cycle([10]),
    #     'time': cycle([1000]),
    #     'mpi': cycle(['mpich3']),
    #     'partition': cycle(['be-short'])
    # },

    # 'SPS-rand-72B-4MPPB-approx1': {
    #     'exe': cycle([yc['exe_home'] + 'SPS_main_random.py']),
    #     'p': cycle([4000000]),
    #     'b': cycle([72]),  # 72
    #     's': cycle([1408]),
    #     't': cycle([43349]),  # 4000
    #     'm': cycle([50]),
    #     'load': cycle([0.0]),
    #     'mtw': cycle([0]),
    #     'approx': cycle([1]),
    #     'timing': cycle(['']),  # otherwise pass -time
    #     # 'seed': [0] * 3 + [1] * 3 + [2] * 3,
    #     # 'seed': [3] * 3 + [4] * 3 + [5] * 3,
    #     'seed': [0] * 3 + [1] * 3 + [2] * 3 + [3] * 3 + [4] * 3 + [5] * 3,
    #     'reduce': []
    #     + [1, 2, 3] * 6,
    #     'w': [16] * 18,
    #     'o': cycle([10]),
    #     'time': cycle([2000]),
    #     'mpi': cycle(['mpich3']),
    #     'partition': cycle(['be-long'])
    # },

    # 'SPS-b72-4MPPB-t10k-mpich3': {
    #     'exe': cycle([yc['exe_home'] + 'SPS_main_random.py']),
    #     'p': cycle([4000000]),
    #     'b': cycle([72]), # 72
    #     's': cycle([1408]),
    #     't': cycle([10000]), # 4000
    #     'm': cycle([0]),
    #     'reduce': cycle([1]),
    #     'load': cycle([0.0]),
    #     'mtw': cycle([0]),
    #     'approx': cycle([0]),
    #     'timing': cycle(['-time']), # otherwise pass -time
    #     'seed': cycle([0]),
    #     'w': [12],
    #     # + list(np.arange(2, 17, 2)),
    #     'o': cycle([10]),
    #     'mpi': cycle(['mpich3']),
    #     'time': cycle([180]),
    #     'partition': cycle(['be-short']),
    # },

    # 'SPS-b72-4MPPB-t10k-openmpi3': {
    #     'exe': cycle([yc['exe_home'] + 'SPS_main_random.py']),
    #     'p': cycle([4000000]),
    #     'b': cycle([72]), # 72
    #     's': cycle([1408]),
    #     't': cycle([10000]), # 4000
    #     'm': cycle([0]),
    #     'reduce': cycle([1]),
    #     'load': cycle([0.0]),
    #     'mtw': cycle([0]),
    #     'approx': cycle([0]),
    #     'timing': cycle(['-time']), # otherwise pass -time
    #     'seed': cycle([0]),
    #     'w': []
    #     + [16, 14],
    #     # + list(np.arange(2, 17, 2)),
    #     'o': cycle([10]),
    #     'mpi': cycle(['openmpi3']),
    #     'time': cycle([180]),
    #     'partition': cycle(['be-long']),
    #     # 'repeats': cycle([5])
    # },

    # 'SPS-b72-4MPPB-t10k-mvapich2': {
    #     'exe': cycle([yc['exe_home'] + 'SPS_main_random.py']),
    #     'p': cycle([4000000]),
    #     'b': cycle([72]), # 72
    #     's': cycle([1408]),
    #     't': cycle([10000]), # 4000
    #     'm': cycle([0]),
    #     'reduce': cycle([1]),
    #     'load': cycle([0.0]),
    #     'mtw': cycle([0]),
    #     'approx': cycle([0]),
    #     'timing': cycle(['-time']), # otherwise pass -time
    #     'seed': cycle([0]),
    #     'w': []
    #     + [14, 16],
    #     # + list(np.arange(2, 17, 2)),
    #     'o': cycle([10]),
    #     'mpi': cycle(['mvapich2']),
    #     'time': cycle([180]),
    #     'partition': cycle(['be-long']),
    #     # 'repeats': cycle([5])
    # },

    'SPS-weak-scale-mpich3': {
        'exe': cycle([yc['exe_home'] + 'SPS_main_random.py']),
        # 'p': cycle([4000000]),
        'p': [1e6, 2e6, 4e6, 3e6, 4e6, 2.5e6, 3e6, 3.5e6, 4e6],
        # 'b': cycle([72]), # 72
        'b': [18, 18, 18, 36, 36, 72, 72, 72, 72], # 72
        's': cycle([1408]),
        't': cycle([10000]), # 4000
        'm': cycle([0]),
        'reduce': cycle([1]),
        'load': cycle([0.0]),
        'mtw': cycle([0]),
        'approx': cycle([0]),
        'timing': cycle(['-time']), # otherwise pass -time
        'seed': cycle([0]),
        'w': [1, 2, 4, 6, 8, 10, 12, 14, 16],
        # + list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mpich3']),
        'time': cycle([90]),
        'partition': cycle(['be-long']),
    },

    # 'SPS-weak-scale-openmpi3': {
    #     'exe': cycle([yc['exe_home'] + 'SPS_main_random.py']),

    #     # 'p': cycle([4000000]),
    #     'p': [1e6, 2e6, 4e6, 3e6, 4e6, 2.5e6, 3e6, 3.5e6, 4e6],
    #     # 'b': cycle([72]), # 72
    #     'b': [18, 18, 18, 36, 36, 72, 72, 72, 72], # 72
    #     's': cycle([1408]),
    #     't': cycle([10000]), # 4000
    #     'm': cycle([0]),
    #     'reduce': cycle([1]),
    #     'load': cycle([0.0]),
    #     'mtw': cycle([0]),
    #     'approx': cycle([0]),
    #     'timing': cycle(['-time']), # otherwise pass -time
    #     'seed': cycle([0]),
    #     'w': [1, 2, 4, 6, 8, 10, 12, 14, 16],
    #     # + list(np.arange(2, 17, 2)),
    #     'o': cycle([10]),
    #     'mpi': cycle(['openmpi3']),
    #     'time': cycle([90]),
    #     'partition': cycle(['be-long']),
    # },

    # 'SPS-weak-scale-mvapich2': {
    #     'exe': cycle([yc['exe_home'] + 'SPS_main_random.py']),

    #     # 'p': cycle([4000000]),
    #     'p': [1e6, 2e6, 4e6, 3e6, 4e6, 2.5e6, 3e6, 3.5e6, 4e6],
    #     # 'b': cycle([72]), # 72
    #     'b': [18, 18, 18, 36, 36, 72, 72, 72, 72], # 72
    #     's': cycle([1408]),
    #     't': cycle([10000]), # 4000
    #     'm': cycle([0]),
    #     'reduce': cycle([1]),
    #     'load': cycle([0.0]),
    #     'mtw': cycle([0]),
    #     'approx': cycle([0]),
    #     'timing': cycle(['-time']), # otherwise pass -time
    #     'seed': cycle([0]),
    #     'w': [1, 2, 4, 6, 8, 10, 12, 14, 16],
    #     # + list(np.arange(2, 17, 2)),
    #     'o': cycle([10]),
    #     'mpi': cycle(['mvapich2']),
    #     'time': cycle([90]),
    #     'partition': cycle(['be-long']),
    # },

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
    stdout = open(analysis + '.txt', 'w')

    for (p, b, s, t, r, w, o, time,
         partition, load, mtw, m,
         seed, exe, approx, timing, mpi) in zip(ps, bs, ss, ts, rs, ws,
                                                oss, times, partitions,
                                                loads, mtws, ms, seeds,
                                                exes, approxs, timings, mpis):
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
                '-approx', str(approx)]

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
            print("%lf %% is completed" % (100.0 * current_sim
                                           / total_sims))
