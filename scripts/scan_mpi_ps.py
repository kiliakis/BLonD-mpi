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
result_dir = home + '/results/raw/{}/{}/{}'
os.environ['PYTHONPATH'] = '{}:{}'.format(blond_repos, os.environ['PYTHONPATH'])

exe = home + '/__EXAMPLES/main_files/PS_main.py'
batch_script = home + '/scripts/batch-simple.sh'
setup_script = home + '/scripts/batch-setup.sh'
job_name_form = '{}/_p{}_b{}_s{}_t{}_w{}_o{}_N{}_r{}'

configs = {

    'PS-4MPPB-comb-mtw50-icc': {'p': cycle([4000000]),
                         'b': cycle([21]),
                         's': cycle([128]),
                         't': cycle([10000]),
                         'reduce': cycle([1]),
                         'load': cycle([0.0]),
                         'mtw': cycle([50]),
                         'w': []
                         + [16],
                         # + list(np.arange(2, 17, 1))
                         # + list(np.arange(2, 9, 1)),
                         'o': []
                         + [10],
                         # + [10]*15
                         # + [20]*7,
                         'time': cycle([45]),
                         'partition': cycle(['be-short'])
                         },

    # 'PS-2MPPB-comb1-mtw50-r1-2': {'p': cycle([2000000]),
    #                             'b': cycle([21]),
    #                             's': cycle([128]),
    #                             't': cycle([10000]),
    #                             'reduce': cycle([1]),
    #                             'load': cycle([0.0]),
    #                             'mtw': cycle([50]),
    #                             'w': []
    #                             # + [16],
    #                             + list(np.arange(2, 17, 1))
    #                             + list(np.arange(2, 9, 1)),
    #                             'o': []
    #                             # + [10],
    #                             + [10]*15
    #                             + [20]*7,
    #                             'time': cycle([45]),
    #                             'partition': cycle(['be-short'])
    #                             },

    # 'PS-4MPPB-comb1-mtw50-r1-2': {'p': cycle([4000000]),
    #                             'b': cycle([21]),
    #                             's': cycle([128]),
    #                             't': cycle([10000]),
    #                             'reduce': cycle([1]),
    #                             'load': cycle([0.0]),
    #                             'mtw': cycle([50]),
    #                             'w': []
    #                             # + [16],
    #                             + list(np.arange(2, 17, 1))
    #                             + list(np.arange(2, 9, 1)),
    #                             'o': []
    #                             # + [10],
    #                             + [10]*15
    #                             + [20]*7,
    #                             'time': cycle([45]),
    #                             'partition': cycle(['be-short'])
    #                             },


    # 'PS-2MPPB-interp-r2': {'p': cycle([2000000]),
    #                        'b': cycle([21]),
    #                        's': cycle([128]),
    #                        't': cycle([10000]),
    #                        'reduce': cycle([2]),
    #                        'load': cycle([0.0]),
    #                        'w': []
    #                        # + [16],
    #                        + list(np.arange(4, 17, 1)),
    #                        # + list(np.arange(2, 9, 1)),
    #                        'o': []
    #                        # + [10],
    #                        + [10]*13,
    #                        # + [20]*7,
    #                        'time': cycle([25]),
    #                        'partition': cycle(['be-short'])
    #                        },

    # 'PS-2MPPB-interp-r4': {'p': cycle([2000000]),
    #                        'b': cycle([21]),
    #                        's': cycle([128]),
    #                        't': cycle([10000]),
    #                        'reduce': cycle([4]),
    #                        'load': cycle([0.0]),
    #                        'w': []
    #                        # + [16],
    #                        + list(np.arange(4, 17, 1)),
    #                        # + list(np.arange(2, 9, 1)),
    #                        'o': []
    #                        # + [10],
    #                        + [10]*13,
    #                        # + [20]*7,
    #                        'time': cycle([25]),
    #                        'partition': cycle(['be-short'])
    #                        },

}

repeats = 5


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
    # Ns = config['N']
    times = config['time']
    partitions = config['partition']
    loads = config['load']
    mtws = config['mtw']
    stdout = open(analysis + '.txt', 'w')

    for p, b, s, t, r, w, o, time, partition, load, mtw in zip(ps, bs, ss, ts, rs, ws,
                                                               oss, times, partitions,
                                                               loads, mtws):
        N = (w * o + 20-1) // 20

        job_name = job_name_form.format(analysis, p, b, s, t, w, o, N, r)
        # if(N < 13):
        #     partition = 'be-short'
        # else:
        #     partition = 'be-long'

        # os.environ['OMP_NUM_THREADS'] = str(o)
        for i in range(repeats):
            # if(current_sim % 2 == 0):
            #     partition = 'be-short'
            # else:
            #     partition = 'be-long'
            timestr = datetime.now().strftime('%d%b%y.%H-%M-%S')
            timestr = timestr + '-' + str(random.randint(0, 100))
            output = result_dir.format(job_name, timestr, 'output.txt')
            error = result_dir.format(job_name, timestr, 'error.txt')
            log_dir = result_dir.format(job_name, timestr, 'log')
            report_dir = result_dir.format(job_name, timestr, 'report')
            for d in [log_dir, report_dir]:
                if not os.path.exists(d):
                    os.makedirs(d)
            # exe_args = ['-n', str('python', exe,
            exe_args = ['-n', str(w), 'python', exe,
                        '-p', str(p), '-s', str(s),
                        '-b', str(b), '-addload', str(load),
                        '-t', str(t), '-time',
                        '-o', str(o), '-timedir', report_dir,
                        '--reduce', str(r),
                        '-mtw', str(mtw)]
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
            print("%lf %% is completed" % (100.0 * current_sim /
                                           total_sims))
