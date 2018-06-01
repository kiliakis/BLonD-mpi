import subprocess
import os
from functools import reduce
from operator import mul
from cycler import cycle
from math import ceil
from datetime import datetime
import numpy as np
import random

home = '/afs/cern.ch/work/k/kiliakis/git/BLonD-mpi'
# home = os.environ['HOME'] + '/git/BLonD-kiliakis'
result_dir = home + '/results/raw/{}/{}/{}'

exe = home + '/mpi/EX_01_Acceleration-master.py'
# batch_script = home + '/mpi/batch-simple.sh'
# setup_script = home + '/mpi/batch-setup.sh'
job_name_form = '{}/_p{}_s{}_t{}_w{}_o{}_N{}_'

configs = {
    # 'weak_scale_omp_local': {'p': np.arange(21000000, 28000001, 1000000),
    #                          's': np.arange(10500, 14001, 500),
    #                          't': cycle([2000]),
    #                          'w': cycle([1]),
    #                          'o': np.arange(21, 29, 1),
    #                          'N': cycle([1]),
    #                          'time': cycle([45]),
    #                          'partition': cycle(['be-short'])
    #                          }

    'strong_scale_mpi_local': {'p': cycle([1000000]),
                               's': cycle([500]),
                               't': cycle([2000]),
                               # 'w': np.arange(1, 8, 1),
                               'w': np.arange(1, 2, 1),
                               'o': cycle([7]),
                               'N': cycle([1]),
                               'time': cycle([60]),
                               'partition': cycle(['be-short'])
                               }

    # 'strong_scale_omp_local': {'p': cycle([10000000]),
    #                            's': cycle([5000]),
    #                            't': cycle([2000]),
    #                            'o': np.arange(1, 21, 1),
    #                            'w': cycle([1]),
    #                            'N': cycle([1]),
    #                            'time': cycle([60]),
    #                            'partition': cycle(['be-short'])
    #                            }

}

repeats = 1


total_sims = repeats * \
    sum([len(y['w']) for y in configs.values()])

print("Total runs: ", total_sims)
current_sim = 0
os.chdir(home)

# compile first
os.environ['PYTHONPATH'] = home + ':' + os.environ['PYTHONPATH']
subprocess.call(['python', 'setup_cpp.py', '-p'])
for analysis, config in configs.items():
    ps = config['p']
    ss = config['s']
    ts = config['t']
    ws = config['w']
    oss = config['o']
    Ns = config['N']
    times = config['time']
    partitions = config['partition']
    stdout = open(analysis + '.txt', 'w')

    for p, s, t, w, o, N, time, partition in zip(ps, ss, ts, ws,
                                                 oss, Ns, times, partitions):
        job_name = job_name_form.format(analysis, p, s, t, w, o, N)
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
            exe_args = ['-p', str(p), '-s', str(s),
                        '-t', str(t), '-w', str(w),
                        '-o', str(o), '-r', report_dir]
            print(job_name, timestr)
            # batch_args = ['-N', str(N), '-n', str(w),
            #               '--ntasks-per-node', str(ceil((w)/N)),
            #               '-c', str(o),
            #               '-t', str(time), '-p', partition,
            #               '-o', output,
            #               '-e', error,
            #               '-J', job_name + '-' + str(i)]

            all_args = ['mpiexec', '-n', '1',
                        'python', '-m', 'mpi4py', exe] + exe_args
            subprocess.call(all_args, stdout=output,
                            stderr=error, env=os.environ.copy())
            current_sim += 1
            print("%lf %% is completed" % (100.0 * current_sim /
                                           total_sims))
