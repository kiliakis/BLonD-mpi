import subprocess
import os
from functools import reduce
from operator import mul
from cycler import cycle
from math import ceil
from datetime import datetime

home = '/afs/cern.ch/work/k/kiliakis/git/BLonD-mpi'
result_dir = home + '/results/raw/{}/{}/'

exe = home + '/mpi/EX_01_Acceleration-master.py'
batch_script = home + '/mpi/batch-simple.sh'
setup_script = home + '/mpi/bathc-setup.sh'
job_name_form = '{}_p{}_s{}_t{}_w{}_o{}_N{}_{}'

out_file_name = result_dir + 'i{}-p{}-s{}-t{}-{}-{}-{}.txt'
configs = {
    'weak_scale_mpi_single_node': {'p': np.arange(1000000, 10000001, 1000000),
                                   's': np.arange(500, 5001, 500),
                                   't': cycle([2000]),
                                   'w': np.arange(1, 11, 1),
                                   'o': cycle([1]),
                                   'N': cycle([1]),
                                   'time': cycle([30]),
                                   'partition': cycle(['be-short'])
                                   },

    'strong_scale_mpi_single_node': {'p': cycle([5000000]),
                                     's': cycle([2500]),
                                     't': cycle([2000]),
                                     'w': np.arange(1, 11, 1),
                                     'o': cycle([1]),
                                     'N': cycle([1])
                                     'time': cycle([45]),
                                     'partition': cycle(['be-short'])

                                     }

}

repeats = 1


total_sims = repeats * \
    sum([len(y['p']) for y in configs.values()])

print("Total runs: ", total_sims)
current_sim = 0
os.chdir(home)

compile()
for analysis, config in configs.items():
    ps = config['p']
    ss = config['s']
    ts = config['t']
    ws = config['w']
    os = config['o']
    Ns = config['N']
    times = config['time']
    partitions = config['partition']
    output_dir = result_dir.format(analysis, 'output')
    error_dir = result_dir.format(analysis, 'error')
    log_dir = result_dir.format(analysis, 'log')
    report_dir = result_dir.format(analysis, 'report')
    for d in [output_dir, error_dir, log_dir, report_dir]:
        if not os.path.exists(d):
            os.mkdirs(d)
    stdout = open(analysis + '.txt', 'w')

    for p, s, t, w, o, N, time, partition in zip(ps, ss, ts, ws, os, Ns, times, partitions):

        exe_args = ['-p', str(p), '-s', str(s),
                    '-t', str(t), '-w', str(w),
                    '-o', str(o)]

        for i in range(repeats):
            timestr = datetime.now().strftime('%d%b%y.%H-%M-%S')
            job_name = job_name_form.format(p, s, t, w, o, N, timestr)
            print(job_name)
            batch_args = ['-N', str(N), '-n', str(w+1),
                          '--ntasks-per-node', str(ceil((w+1)/N)),
                          '-c', str(o),
                          '-t', str(time), '-p', partition,
                          '-o', output_dir + job_name + '.txt',
                          '-e', error_dir + job_name + '.txt',
                          '-j', job_name]

            all_args = ['sbatch'] + batch_args + \
                [batch_script, exe, ' '.join(exe_args)]
            subprocess.call(all_args, stdout=stdout,
                            stderr=stdout, env=os.environ.copy())
            current_sim += 1
            print("%lf %% is completed" % (100.0 * current_sim /
                                           total_sims))


def compile():
    subprocess.call(['srun', '-t1', '-N1', '-n1', '-p',
                     'be-short', 'bash', setup_script])
