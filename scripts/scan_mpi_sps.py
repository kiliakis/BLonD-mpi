import subprocess
import os
from functools import reduce
from operator import mul
from cycler import cycle
from math import ceil
from datetime import datetime
import numpy as np
import random
from time import sleep

# home = '/afs/cern.ch/work/k/kiliakis/git/BLonD-mpi'
home = os.environ['HOME'] + '/git/BLonD-mpi'
blond_repos = os.environ['HOME'] + '/git/blond_repos'
result_dir = home + '/results/raw/{}/{}/{}/{}'
os.environ['PYTHONPATH'] = '{}:{}'.format(blond_repos, os.environ['PYTHONPATH'])

# exe = home + '/__EXAMPLES/main_files/SPS_main.py'
batch_script = home + '/scripts/batch-simple.sh'
setup_script = home + '/scripts/batch-setup.sh'
job_name_form = '_p{}_b{}_s{}_t{}_w{}_o{}_N{}_r{}_m{}_seed{}_approx{}_'

configs = {

    # 'SPS-rand-12B-4MPPB-approx2': {
    #     'exe': cycle([home + '/__EXAMPLES/main_files/SPS_main_random.py']),
    #     'p': cycle([4000000]),
    #     'b': cycle([12]),  # 72
    #     's': cycle([1408]),
    #     't': cycle([43349]),  # 4000
    #     'm': cycle([50]),
    #     'reduce': cycle([1]),
    #     'load': cycle([0.0]),
    #     'mtw': cycle([0]),
    #     'approx': cycle([2]),
    #     'timing': cycle(['']),  # otherwise pass -time
    #     # 'seed': [0] * 5 + [1] * 5 + [2] * 5,
    #     'seed': [3] * 5 + [4] * 5 + [5] * 5,
    #     'w': []
    #     + [1, 2, 4, 8, 16]
    #     + [1, 2, 4, 8, 16]
    #     + [1, 2, 4, 8, 16],
    #     #       list(np.arange(2, 17, 1)) +
    #     # list(np.arange(2, 9, 1)),
    #     'o': cycle([10]),
    #     # + [10]*15,
    #     # + [10]*5,
    #     'time': cycle([360]),
    #     'partition': cycle(['be-short'])
    # },

    # 'SPS-rand-12B-4MPPB-approx1': {
    #     'exe': cycle([home + '/__EXAMPLES/main_files/SPS_main_random.py']),
    #     'p': cycle([4000000]),
    #     'b': cycle([12]),  # 72
    #     's': cycle([1408]),
    #     't': cycle([43349]),  # 4000
    #     'm': cycle([50]),
    #     'load': cycle([0.0]),
    #     'mtw': cycle([0]),
    #     'approx': cycle([1]),
    #     'timing': cycle(['']),  # otherwise pass -time
    #     # 'seed': [0] * 3 + [1] * 3 + [2] * 3,
    #     'seed': [3] * 3 + [4] * 3 + [5] * 3,
    #     # 'seed': [0, 1, 2, 3, 4, 5],
    #     'reduce': []
    #     # + [4] * 6,
    #     + [1, 2, 3]
    #     + [1, 2, 3]
    #     + [1, 2, 3],
    #     'w': [16] * 9,
    #     'o': cycle([10]),
    #     'time': cycle([360]),
    #     'partition': cycle(['be-short'])
    # },


    'SPS-b72-4MPPB-t10k-approx-time-3': {
        'exe': cycle([home + '/__EXAMPLES/main_files/SPS_main_random.py']),
        'p': cycle([4000000]),
        'b': cycle([72]), # 72
        's': cycle([1408]),
        't': cycle([10000]), # 4000
        'm': cycle([0]),
        'reduce': cycle([1]),
        'load': cycle([0.0]),
        'mtw': cycle([0]),
        'approx': cycle([2]),
        'timing': cycle(['-time']), # otherwise pass -time
        'seed': cycle([0]),
        'w': []
        + [16],
        # + [1, 2, 4, 8, 16],
        # + list(np.arange(2, 17, 2)),
        # list(np.arange(2, 9, 1)),
        'o': cycle([10]),
        # + [10]*8,
        'time': cycle([180]),
        'partition': cycle(['be-short']),
        # 'repeats': cycle([5])
    },

    # 'SPS-b72-4MPPB-t10k-red2-time-2': {
    #     'exe': cycle([home + '/__EXAMPLES/main_files/SPS_main_random.py']),
    #     'p': cycle([4000000]),
    #     'b': cycle([72]), # 72
    #     's': cycle([1408]),
    #     't': cycle([5000]), # 4000
    #     'm': cycle([0]),
    #     'reduce': cycle([2]),
    #     'load': cycle([0.0]),
    #     'mtw': cycle([0]),
    #     'approx': cycle([1]),
    #     'timing': cycle(['-time']), # otherwise pass -time
    #     'seed': cycle([0]),
    #     'w': []
    #     # + [1, 2, 4, 8, 16],
    #     + list(np.arange(2, 17, 2)),
    #     # list(np.arange(2, 9, 1)),
    #     'o': cycle([10]),
    #     # + [10]*8,
    #     'time': cycle([180]),
    #     'partition': cycle(['be-short']),
    #     # 'repeats': cycle([5])
    # },


    # 'SPS-b72-4MPPB-t10k-red3-time-3': {
    #     'exe': cycle([home + '/__EXAMPLES/main_files/SPS_main_random.py']),
    #     'p': cycle([4000000]),
    #     'b': cycle([72]), # 72
    #     's': cycle([1408]),
    #     't': cycle([5000]), # 4000
    #     'm': cycle([0]),
    #     'reduce': cycle([3]),
    #     'load': cycle([0.0]),
    #     'mtw': cycle([0]),
    #     'approx': cycle([1]),
    #     'timing': cycle(['-time']), # otherwise pass -time
    #     'seed': cycle([0]),
    #     'w': []
    #     # + [1, 2, 4, 8, 16],
    #     + list(np.arange(2, 17, 2)),
    #     # list(np.arange(2, 9, 1)),
    #     'o': cycle([10]),
    #     # + [10]*8,
    #     'time': cycle([180]),
    #     'partition': cycle(['be-short']),
    #     # 'repeats': cycle([5])
    # },


    # 'SPS-b72-4MPPB-t10k-red4-time-3': {
    #     'exe': cycle([home + '/__EXAMPLES/main_files/SPS_main_random.py']),
    #     'p': cycle([4000000]),
    #     'b': cycle([72]), # 72
    #     's': cycle([1408]),
    #     't': cycle([5000]), # 4000
    #     'm': cycle([0]),
    #     'reduce': cycle([4]),
    #     'load': cycle([0.0]),
    #     'mtw': cycle([0]),
    #     'approx': cycle([1]),
    #     'timing': cycle(['-time']), # otherwise pass -time
    #     'seed': cycle([0]),
    #     'w': []
    #     # + [1, 2, 4, 8, 16],
    #     + list(np.arange(2, 17, 2)),
    #     # list(np.arange(2, 9, 1)),
    #     'o': cycle([10]),
    #     # + [10]*8,
    #     'time': cycle([180]),
    #     'partition': cycle(['be-short']),
    #     # 'repeats': cycle([5])
    # },




    # 'SPS-b72-4MPPB-t10k-2': {
    #     'exe': cycle([home + '/__EXAMPLES/main_files/SPS_main_random.py']),
    #     'p': cycle([4000000]),
    #     'b': cycle([72]), # 72
    #     's': cycle([1408]),
    #     't': cycle([5000]), # 4000
    #     'm': cycle([0]),
    #     'reduce': cycle([1]),
    #     'load': cycle([0.0]),
    #     'mtw': cycle([0]),
    #     'approx': cycle([0]),
    #     'timing': cycle(['-time']), # otherwise pass -time
    #     'seed': cycle([0]),
    #     'w': []
    #     # + [16],
    #     + list(np.arange(2, 17, 2)),
    #     # list(np.arange(2, 9, 1)),
    #     'o': cycle([10]),
    #     # + [10]*8,
    #     'time': cycle([180]),
    #     'partition': cycle(['be-short']),
    #     # 'repeats': cycle([5])
    # },

}

repeats = 10


total_sims = repeats * \
    sum([len(y['w']) for y in configs.values()])

print("Total runs: ", total_sims)
current_sim = 0
os.chdir(home)

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
    stdout = open(analysis + '.txt', 'w')

    for (p, b, s, t, r, w, o, time,
         partition, load, mtw, m,
         seed, exe, approx, timing) in zip(ps, bs, ss, ts, rs, ws,
                                           oss, times, partitions,
                                           loads, mtws, ms, seeds,
                                           exes, approxs, timings):
        N = (w * o + 20-1) // 20

        job_name = job_name_form.format(p, b, s, t, w, o, N, r, m, seed, approx)

        # if(N < 5):
        #     partition = 'be-short'
        #     # continue
        # else:
        #     partition = 'be-long'
        # os.environ['OMP_NUM_THREADS'] = str(o)
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
                '-n', str(w), 'python', exe,
                '-p', str(p), '-s', str(s),
                '-b', str(b), '-addload', str(load),
                '-t', str(t), '-o', str(o), '-seed', str(seed),
                str(timing), '-timedir', report_dir,
                '-m', str(m), '-monitorfile', monitorfile,
                '--reduce', str(r), '-mtw', str(mtw),
                '-approx', str(approx)]

            print(job_name, timestr)
            batch_args = ['-N', str(N), '-n', str(w),
                          '--ntasks-per-node', str(ceil(w/N)),
                          '-c', str(o),
                          '-t', str(time), '-p', partition,
                          '-o', output,
                          '-e', error,
                          '-J', job_name.split('/')[0] + '-' + str(i)]

            all_args = ['sbatch'] + batch_args + [batch_script] + exe_args
            subprocess.call(all_args, stdout=stdout,
                            stderr=stdout, env=os.environ.copy())
            # sleep(5)
            current_sim += 1
            print("%lf %% is completed" % (100.0 * current_sim
                                           / total_sims))
