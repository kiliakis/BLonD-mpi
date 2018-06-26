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

exe = home + '/__EXAMPLES/mpi_main_files/_LHC_BUP_2017.py'
batch_script = home + '/mpi_scripts/batch-simple.sh'
setup_script = home + '/mpi_scripts/batch-setup.sh'
job_name_form = '{}/_p{}_b{}_s{}_t{}_w{}_o{}_N{}_'

configs = {
    # 'weak_scale_mpi_single_node': {'p': np.arange(1000000, 19000001, 1000000),
    #                                's': np.arange(500, 9501, 500),
    #                                't': cycle([2000]),
    #                                'w': np.arange(1, 20, 1),
    #                                'o': cycle([1]),
    #                                'N': cycle([1]),
    #                                'time': cycle([45]),
    #                                'partition': cycle(['be-short'])
    #                                }

    # 'strong_scale_mpi_single_node-2': {'p': cycle([10000000]),
    #                                  's': cycle([5000]),
    #                                  't': cycle([2000]),
    #                                  'w': np.arange(2, 21, 1),
    #                                  'o': cycle([1]),
    #                                  'N': cycle([1]),
    #                                  'time': cycle([60]),
    #                                  'partition': cycle(['be-short'])
    #                                  },

    # 'weak_scale_mpi_dual_node': {'p': np.arange(1000000, 39000001, 2000000),
    #                              's': np.arange(500, 19501, 1000),
    #                              't': cycle([2000]),
    #                              'w': np.arange(1, 40, 2),
    #                              'o': cycle([1]),
    #                              'N': cycle([2]),
    #                              'time': cycle([45]),
    #                              'partition': cycle(['be-long'])
    #                              },

    # 'strong_scale_mpi_dual_node': {'p': cycle([20000000]),
    #                                's': cycle([10000]),
    #                                't': cycle([2000]),
    #                                'w': np.arange(3, 40, 2),
    #                                'o': cycle([1]),
    #                                'N': cycle([2]),
    #                                'time': cycle([60]),
    #                                'partition': cycle(['be-long'])
    #                                }

    # 'weak_scale_mpi_four_node': {'p': np.arange(1000000, 78000001, 2000000),
    #                              's': np.arange(500, 39501, 1000),
    #                              't': cycle([2000]),
    #                              'w': np.arange(1, 80, 2),
    #                              'o': cycle([1]),
    #                              'N': cycle([4]),
    #                              'time': cycle([45]),
    #                              'partition': cycle(['be-long'])
    #                              },

    # 'weak_scale_hybrid_four_node': {'p': np.arange(1000000, 78000001, 2000000),
    #                              's': np.arange(500, 39501, 1000),
    #                              't': cycle([2000]),
    #                              'w': np.arange(1, 80, 2),
    #                              'o': cycle([10]),
    #                              'N': cycle([4]),
    #                              'time': cycle([45]),
    #                              'partition': cycle(['be-long'])
    #                              },

    # 'strong_scale_mpi_four_node': {'p': cycle([20000000]),
    #                                's': cycle([10000]),
    #                                't': cycle([2000]),
    #                                'w': np.arange(3, 80, 2),
    #                                'o': cycle([1]),
    #                                'N': cycle([4]),
    #                                'time': cycle([60]),
    #                                'partition': cycle(['be-long'])
    #                                }


    # 'weak_scale_hybrid_four_node': {'p': np.arange(20000000, 80000001, 20000000),
    #                                 's': np.arange(10000, 40001, 10000),
    #                                 't': cycle([2000]),
    #                                 'w': np.arange(1, 5, 1),
    #                                 'o': cycle([20]),
    #                                 'N': np.arange(2, 6, 1),
    #                                 'time': cycle([60]),
    #                                 'partition': cycle(['be-long'])
    #                                 }

    # 'strong_scale_hybrid_four_node-4': {'p': cycle([20000000]),
    #                                     's': cycle([10000]),
    #                                     't': cycle([2000]),
    #                                     'w': list(np.arange(2, 21, 1))
    #                                     + list(np.arange(2, 41, 1))
    #                                     + list(np.arange(2, 17, 1)),
    #                                     # list(np.arange(2, 9, 1))
    #                                     # + list(np.arange(3, 20, 2))
    #                                     'o': [4]*19 + [2]*39 + [5]*15,
    #                                     # [10]*7 + [5]*8,
    #                                     # + [4]*9 + [2]*10,
    #                                     'N': [1]*4 + [2]*5 + [3]*5 + [4]*5
    #                                     + [1]*9 + [2]*10 + [3]*10 + [4]*10
    #                                     + [1]*3 + [2]*4 + [3]*4 + [4]*4,
    #                                     # [1, 1, 2, 2, 2, 3, 3, 4, 4, 4],
    #                                     # [1, 2, 2, 3, 3, 4, 4]
    #                                     # + [1, 2, 2, 2, 3, 3, 4, 4, 4]
    #                                     'time': cycle([60]),
    #                                     'partition': cycle(['be-long'])
    #                                     }


    # 'LHC-hybrid-4nodes': {'p': cycle([1000000]),
    #                       's': cycle([1000]),
    #                       't': cycle([10000]),
    #                       'w': list(np.arange(4, 41, 4))
    #                       + list(np.arange(2, 21, 2))
    #                       + list(np.arange(2, 17, 2))
    #                       + list(np.arange(2, 9, 1)),
    #                       # + list(np.arange(3, 20, 2))
    #                       'o': [2]*10 + [4]*10 + [5]*8 + [10]*7,
    #                       'N': [1, 1, 2, 2, 2, 3, 3, 4, 4, 4]
    #                       + [1, 1, 2, 2, 2, 3, 3, 4, 4, 4]
    #                       + [1, 1, 2, 2, 3, 3, 4, 4]
    #                       + [1, 2, 2, 3, 3, 4, 4],
    #                       # + [1]*4 + [2]*5 + [3]*5 + [4]*5
    #                       # + [1]*9 + [2]*10 + [3]*10 + [4]*10
    #                       # + [1]*3 + [2]*4 + [3]*4 + [4]*4,
    #                       'time': cycle([60]),
    #                       'partition': cycle(['be-long'])
    #                       }

    'LHC-4n-96B-lt-nored-nogat-lessbcast': {'p': cycle([1000000]),
                                            'b': cycle([96]),
                                            's': cycle([1000]),
                                            't': cycle([10000]),
                                            'w': []
                                            # + list(np.arange(8, 41, 4))
                                            # + list(np.arange(4, 21, 2))
                                            + list(np.arange(2, 17, 2))
                                            + list(np.arange(2, 9, 1)),
                                            # list(np.arange(2, 81, 2)),
                                            # + list(np.arange(3, 20, 2))
                                            'o': [5]*8 + [10]*7,
                                            # [2]*9 + [4]*9 + [5]*8 + [10]*7,
                                            # [5]*5 + [10]*2,
                                            # [1] * 40,
                                            'N': []
                                            # + [1, 2, 2, 2, 3, 3, 4, 4, 4]
                                            # + [1, 2, 2, 2, 3, 3, 4, 4, 4]
                                            + [1, 1, 2, 2, 3, 3, 4, 4]
                                            + [1, 2, 2, 3, 3, 4, 4],
                                            # [1] * 10 + [2] * 10 + [3] * 10 + [4] * 10,
                                            'time': cycle([180]),
                                            'partition': cycle(['be-long'])
                                            }


    # 'strong_scale_hybrid_four_node-2': {'p': cycle([20000000]),
    #                                     's': cycle([10000]),
    #                                     't': cycle([2000]),
    #                                     'w': [1, 2, 3, 4],
    #                                     'o': cycle([20]),
    #                                     'N': [2, 3, 4, 5],
    #                                     'time': cycle([60]),
    #                                     'partition': cycle(['be-long'])
    #                                     }




}

repeats = 1


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
    Ns = config['N']
    times = config['time']
    partitions = config['partition']
    stdout = open(analysis + '.txt', 'w')

    for p, b, s, t, w, o, N, time, partition in zip(ps, bs, ss, ts, ws,
                                                    oss, Ns, times, partitions):
        job_name = job_name_form.format(analysis, p, b, s, t, w, o, N)
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
                        '-p', str(p), '-s', str(s),
                        '-b', str(b),
                        '-t', str(t), '-time',
                        '-o', str(o), '-r', report_dir]
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
