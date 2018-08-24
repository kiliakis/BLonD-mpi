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
result_dir = home + '/results/raw/{}/{}/{}'

exe = home + '/__EXAMPLES/mpi_main_files/SPS_main.py'
batch_script = home + '/mpi_scripts/batch-simple.sh'
setup_script = home + '/mpi_scripts/batch-setup.sh'
job_name_form = '{}/_p{}_b{}_s{}_t{}_w{}_o{}_N{}_'

configs = {

    'SPS-8n-72B-packed-mul-uint16-r1': {'p': cycle([4000000]),
                                        'b': cycle([72]),
                                        's': cycle([1408]),
                                        't': cycle([4000]),
                                        'reduce': cycle([1]),
                                        'w': []
                                        # + list(np.arange(2, 81, 2))
                                        # + list(np.arange(2, 41, 2))
                                        # + list(np.arange(2, 33, 1))
                                        + list(np.arange(2, 17, 1))
                                        + list(np.arange(2, 9, 1)),
                                        #  + list([16, 32])
                                        #  + list([8, 16])
                                        #  + list([4, 8]),

                                        'o': []
                                        #  + [5]*2
                                        #  + [10]*2
                                        #  + [20]*2,
                                        # + [5]*31
                                        + [10]*15
                                        + [20]*7,

                                        'time': cycle([40]),
                                        'partition': cycle(['be-long'])
                                        }

    #      'SPS-8n-72B-packed-mul-r2': {'p': cycle([4000000]),
    #                               'b': cycle([72]),
    #                               's': cycle([1408]),
    #                               't': cycle([4000]),
    #                               'reduce': cycle([2]),
    #                               'w': []
    #                               # + list(np.arange(2, 81, 2))
    #                               # + list(np.arange(2, 41, 2))
    #                               + list(np.arange(2, 33, 1))
    #                               + list(np.arange(2, 17, 1))
    #                               + list(np.arange(2, 9, 1)),
    #
    #                               'o': []
    #                               + [5]*31
    #                               + [10]*15
    #                               + [20]*7,
    #
    #                               'time': cycle([40]),
    #                               'partition': cycle(['be-long'])
    #                               },
    #
    #  'SPS-8n-72B-packed-mul-r5': {'p': cycle([4000000]),
    #                               'b': cycle([72]),
    #                               's': cycle([1408]),
    #                               't': cycle([4000]),
    #                               'reduce': cycle([5]),
    #                               'w': []
    #                               # + list(np.arange(2, 81, 2))
    #                               # + list(np.arange(2, 41, 2))
    #                               + list(np.arange(2, 33, 1))
    #                               + list(np.arange(2, 17, 1))
    #                               + list(np.arange(2, 9, 1)),
    #
    #                               'o': []
    #                               + [5]*31
    #                               + [10]*15
    #                               + [20]*7,
    #
    #                               'time': cycle([40]),
    #                               'partition': cycle(['be-long'])
    #                               }
    #

}

repeats = 2


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
    stdout = open(analysis + '.txt', 'w')

    for p, b, s, t, r, w, o, time, partition in zip(ps, bs, ss, ts, rs, ws,
                                                    oss, times, partitions):
        N = (w * o + 20-1) // 20

        job_name = job_name_form.format(analysis, p, b, s, t, w, o, N)
        if(N < 5):
            partition = 'be-short'
            # continue
        else:
            partition = 'be-long'
        # os.environ['OMP_NUM_THREADS'] = str(o)
        for i in range(repeats):
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
                        '-p', str(p),
                        # '-s', str(s),
                        '-b', str(b),
                        '-t', str(t), '-time',
                        '-o', str(o), '-r', report_dir,
                        '--reduce', str(r)]
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
