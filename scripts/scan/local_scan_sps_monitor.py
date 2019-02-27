import subprocess
import os
# from functools import reduce
# from operator import mul
from cycler import cycle
# from math import ceil
# from datetime import datetime
# import numpy as np
# import random
# import multiprocessing


home = '/afs/cern.ch/work/k/kiliakis/git/BLonD-mpi'
eos = '/eos/user/k/kiliakis'
# home = os.environ['HOME'] + '/git/BLonD-kiliakis'
result_dir = eos + '/profiles/SPS/raw/{}/{}/{}'

exe = home + '/__EXAMPLES/mpi_main_files/SPS_main_random.py'
# batch_script = home + '/mpi/batch-simple.sh'
# setup_script = home + '/mpi/batch-setup.sh'
job_name_form = '_p{}_s{}_t{}_w{}_m{}_b{}_r{}_o{}_N{}'

configs = {

    'acc_indvolt': {'p': cycle([1000000]),
                    # 's': cycle([144]),
                    't': cycle([5000]),
                    'w': cycle([2]),
                    'm': cycle([1]),
                    'b': []
                    + [1]*5
                    + [24]*5,
                    # + [72]*5,
                    's': []
                    + [0, 1, 2] + [0, 0]
                    + [0, 1, 2] + [0, 0],
                    # + [0, 1, 2] + [0, 0],
                    'r': []
                    + [1, 1, 1] + [2, 3]
                    + [1, 1, 1] + [2, 3],
                    # + [1, 1, 1] + [2, 3],
                    # 'w': np.arange(1, 2, 1),
                    'o': cycle([3]),
                    'N': cycle([1]),
                    'time': cycle([60]),
                    'partition': cycle(['be-short'])
                    }

}

repeats = 1


total_sims = repeats * \
    sum([len(y['b']) for y in configs.values()])

print("Total runs: ", total_sims)
current_sim = 0
os.chdir(home)

# compile first
os.environ['PYTHONPATH'] = home + ':' + os.environ['PYTHONPATH']
# subprocess.call(['python', 'setup_cpp.py', '-p'])
for analysis, config in configs.items():
    ps = config['p']
    ss = config['s']
    ts = config['t']
    ws = config['w']
    ms = config['m']
    bs = config['b']
    rs = config['r']
    oss = config['o']
    Ns = config['N']
    times = config['time']
    partitions = config['partition']
    stdout = open(analysis + '.txt', 'w')

    for p, s, t, w, m, b, r, o, N, time, partition in zip(ps, ss, ts, ws, ms, bs, rs,
                                                          oss, Ns, times, partitions):
        job_name = job_name_form.format(p, s, t, w, m, b, r, o, N)
        for i in range(repeats):
            # timestr = datetime.now().strftime('%d%b%y.%H-%M-%S')
            # timestr = timestr + '-' + str(random.randint(0, 100))

            output = result_dir.format(analysis, 'output', job_name + '.txt')
            error = result_dir.format(analysis, 'error', job_name + '.txt')
            monitorfile = result_dir.format(analysis, 'h5', job_name)
            for file in [output, error, monitorfile]:
                if not os.path.exists(os.path.dirname(file)):
                    os.makedirs(os.path.dirname(file))

            exe_args = ['-p', str(p), '-seed', str(s),
                        '-t', str(t), '-o', str(o),
                        '-reduce', str(r), '-m', str(m),
                        '--monitorfile='+monitorfile]
            print(job_name)

            all_args = ['mpirun', '-n', str(w),
                        'python', exe] + exe_args
            subprocess.Popen(all_args,
                             stdout=open(output, 'w'),
                             stderr=open(error, 'w'),
                             env=os.environ.copy())
            current_sim += 1
            print("%lf %% is completed" % (100.0 * current_sim /
                                           total_sims))
