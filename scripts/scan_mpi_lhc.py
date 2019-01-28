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

exe = home + '/__EXAMPLES/main_files/_LHC_BUP_2017.py'
batch_script = home + '/scripts/batch-simple.sh'
setup_script = home + '/scripts/batch-setup.sh'
job_name_form = '{}/_p{}_b{}_s{}_t{}_w{}_o{}_N{}_r{}_'

configs = {

    # 'LHC-96B-2MPPB-sl512-uint16-nobcast-r1-lb10': {'p': cycle([2000000]),
    #                                                'b': cycle([96]),
    #                                                's': cycle([512]),
    #                                                't': cycle([10000]),
    #                                                'reduce': cycle([1]),
    #                                                'load': cycle([0.1]),
    #                                                'w': []
    #                                                + list(np.arange(16, 17, 1)),
    #                                                # + list(np.arange(4, 9, 1)),
    #                                                # + list([3]),
    #                                                # + list([7]),

    #                                                'o': []
    #                                                + [10]*1,
    #                                                # + [20]*5,

    #                                                'time': cycle([25]),
    #                                                'partition': cycle(['be-long'])
    #                                                },

    # 'LHC-96B-2MPPB-sl512-uint16-nobcast-r1-lb20': {'p': cycle([2000000]),
    #                                                'b': cycle([96]),
    #                                                's': cycle([512]),
    #                                                't': cycle([10000]),
    #                                                'reduce': cycle([1]),
    #                                                'load': cycle([0.2]),
    #                                                'w': []
    #                                                + list(np.arange(16, 17, 1)),
    #                                                # + list(np.arange(4, 9, 1)),
    #                                                # + list([3]),
    #                                                # + list([7]),

    #                                                'o': []
    #                                                + [10]*1,
    #                                                # + [20]*5,

    #                                                'time': cycle([25]),
    #                                                'partition': cycle(['be-long'])
    #                                                },


    # 'LHC-96B-2MPPB-uint16-nobcast-r1-lb30': {'p': cycle([2000000]),
    #                                          'b': cycle([96]),
    #                                          's': cycle([1000]),
    #                                          't': cycle([10000]),
    #                                          'reduce': cycle([1]),
    #                                          'load': cycle([0.3]),
    #                                          'w': []
    #                                          + list(np.arange(16, 17, 1)),
    #                                          # + list(np.arange(4, 9, 1)),
    #                                          # + list([3]),
    #                                          # + list([7]),

    #                                          'o': []
    #                                          + [10]*1,
    #                                          # + [20]*5,

    #                                          'time': cycle([25]),
    #                                          'partition': cycle(['be-long'])
    #                                          },

    # 'LHC-96B-2MPPB-uint16-nobcast-r2-2': {'p': cycle([2000000]),
    #                                          'b': cycle([96]),
    #                                          's': cycle([1000]),
    #                                          't': cycle([10000]),
    #                                          'reduce': cycle([2]),
    #                                          'load': cycle([0]),
    #                                          'w': []
    #                                          + list(np.arange(2, 17, 1))
    #                                          + list(np.arange(2, 9, 1)),
    #                                          # + list([3]),
    #                                          # + list([7]),

    #                                          'o': []
    #                                          + [10]*15
    #                                          + [20]*7,

    #                                          'time': cycle([45]),
    #                                          'partition': cycle(['be-short'])
    #                                          },
    # 'LHC-96B-2MPPB-uint16-nobcast-r3-2': {'p': cycle([2000000]),
    #                                          'b': cycle([96]),
    #                                          's': cycle([1000]),
    #                                          't': cycle([10000]),
    #                                          'reduce': cycle([3]),
    #                                          'load': cycle([0]),
    #                                          'w': []
    #                                          + list(np.arange(2, 17, 1))
    #                                          + list(np.arange(2, 9, 1)),
    #                                          # + list([3]),
    #                                          # + list([7]),

    #                                          'o': []
    #                                          + [10]*15
    #                                          + [20]*7,

    #                                          'time': cycle([45]),
    #                                          'partition': cycle(['be-short'])
    #                                          },
    # 'LHC-96B-2MPPB-uint16-nobcast-r4-2': {'p': cycle([2000000]),
    #                                          'b': cycle([96]),
    #                                          's': cycle([1000]),
    #                                          't': cycle([10000]),
    #                                          'reduce': cycle([4]),
    #                                          'load': cycle([0]),
    #                                          'w': []
    #                                          + list(np.arange(2, 17, 1))
    #                                          + list(np.arange(2, 9, 1)),
    #                                          # + list([3]),
    #                                          # + list([7]),

    #                                          'o': []
    #                                          + [10]*15
    #                                          + [20]*7,

    #                                          'time': cycle([45]),
    #                                          'partition': cycle(['be-short'])
    #                                          },


    'LHC-96B-2MPPB-uint16-r1': {'p': cycle([2000000]),
                                'b': cycle([96]),
                                's': cycle([1000]),
                                't': cycle([10000]),
                                'reduce': cycle([1]),
                                'load': cycle([0]),
                                'w': []
                                # + list(np.arange(2, 17, 1)),
                                + list(np.arange(2, 9, 1)),
                                'o': []
                                # + [10]*15,
                                + [20]*7,

                                'time': cycle([45]),
                                'partition': cycle(['be-short'])
                                }

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
    stdout = open(analysis + '.txt', 'w')

    for p, b, s, t, r, w, o, time, partition, load in zip(ps, bs, ss, ts, rs, ws,
                                                          oss, times, partitions,
                                                          loads):
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
                        '--reduce', str(r)]
            print(job_name, timestr)
            batch_args = ['-N', str(N), '-n', str(w),
                          '--ntasks-per-node', str(ceil(w/N)),
                          '-c', str(o),  # str(o),
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
