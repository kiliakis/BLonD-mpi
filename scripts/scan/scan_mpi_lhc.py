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

    # 'LHC-48B-2MPPB-approx2': {
    #     'exe': cycle([yc['exe_home'] + '_LHC_BUP_2017.py']),    
    #     'p': cycle([2000000]),
    #     'b': cycle([48]),  # 96
    #     's': cycle([1000]),
    #     't': cycle([200000]),
    #     'm': cycle([250]),
    #     'reduce': cycle([1]),
    #     'load': cycle([0.0]),
    #     'mtw': cycle([50]),
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


    # 'LHC-48B-2MPPB-approx1': {
    #     'exe': cycle([yc['exe_home'] + '_LHC_BUP_2017.py']),    
    #     'p': cycle([2000000]),
    #     'b': cycle([48]),  # 96
    #     's': cycle([1000]),
    #     't': cycle([200000]),
    #     'm': cycle([250]),
    #     'load': cycle([0.0]),
    #     'mtw': cycle([50]),
    #     'approx': cycle([1]),
    #     'timing': cycle(['']),  # otherwise pass -time
    #     'seed': [0] * 3 + [1] * 3 + [2] * 3 + [3] * 3 + [4] * 3 + [5] * 3,
    #     # 'seed': [1, 2],
    #     'reduce': []
    #     + [1, 2, 3] * 6,
    #     # + [1, 2],
    #     'w': [16] * 18,
    #     'o': cycle([10]),
    #     'time': cycle([1000]),
    #     'mpi': cycle(['mpich3']),
    #     'partition': cycle(['be-short'])
    # }

  

    # 'LHC-96B-2MPPB-t10k-mpich3': {
    #     'exe': cycle([yc['exe_home'] + '_LHC_BUP_2017.py']),
    #     'p': cycle([2000000]),
    #     'b': cycle([96]),  # 96
    #     's': cycle([1000]),
    #     't': cycle([10000]),
    #     'm': cycle([0]),
    #     'seed': cycle([0]),
    #     'reduce': cycle([1]),
    #     'load': cycle([0.0]),
    #     'mtw': cycle([50]),
    #     'approx': cycle([0]),
    #     'timing': cycle(['-time']),  # otherwise pass -time
    #     'w': []
    #      + [12, 14, 16],
    #     # + list(np.arange(2, 17, 2)),
    #     'o': cycle([10]),
    #     'mpi': cycle(['mpich3']),
    #     'time': cycle([180]),
    #     'partition': cycle(['be-short'])
    # },

    # 'LHC-96B-2MPPB-t10k-openmpi3': {
    #     'exe': cycle([yc['exe_home'] + '_LHC_BUP_2017.py']),
    #     'p': cycle([2000000]),
    #     'b': cycle([96]),  # 96
    #     's': cycle([1000]),
    #     't': cycle([10000]),
    #     # 't': cycle([100]),
    #     'm': cycle([0]),
    #     'seed': cycle([0]),
    #     'reduce': cycle([1]),
    #     'load': cycle([0.0]),
    #     'mtw': cycle([50]),
    #     'approx': cycle([0]),
    #     'timing': cycle(['-time']),  # otherwise pass -time
    #     'w': []
    #     + [10, 12, 16],
    #     # + list(np.arange(2, 17, 2)),
    #     'o': cycle([10]),
    #     'mpi': cycle(['openmpi3']),
    #     'time': cycle([180]),
    #     'partition': cycle(['be-long'])
    # },

    # 'LHC-96B-2MPPB-t10k-mvapich2': {
    #     'exe': cycle([yc['exe_home'] + '_LHC_BUP_2017.py']),
    #     'p': cycle([2000000]),
    #     'b': cycle([96]),  # 96
    #     's': cycle([1000]),
    #     't': cycle([10000]),
    #     # 't': cycle([100]),
    #     'm': cycle([0]),
    #     'seed': cycle([0]),
    #     'reduce': cycle([1]),
    #     'load': cycle([0.0]),
    #     'mtw': cycle([50]),
    #     'approx': cycle([0]),
    #     'timing': cycle(['-time']),  # otherwise pass -time
    #     'w': []
    #     + [6, 12, 16],
    #     # + list(np.arange(2, 17, 2)),
    #     'o': cycle([10]),
    #     'mpi': cycle(['mvapich2']),
    #     'time': cycle([180]),
    #     'partition': cycle(['be-long'])
    # },


    # 'LHC-sync-mpich3': {
    #     'exe': cycle([yc['exe_home'] + '_LHC_BUP_2017_test.py']),
    #     'p': cycle([2000000]),
    #     # 'p': [1e6, 2e6, 2e6, 2e6, 2e6, 1.25e6, 1.5e6, 1.75e6, 2e6],
    #     'b': cycle([96]),  # 96
    #     # 'b': [12, 12, 24, 36, 48, 96, 96, 96, 96],  # 96
    #     's': cycle([1000]),
    #     't': cycle([10000]),
    #     'm': cycle([0]),
    #     'seed': cycle([0]),
    #     'reduce': cycle([1]),
    #     'load': cycle([0.0]),
    #     'mtw': cycle([50]),
    #     'approx': cycle([2]),
    #     'timing': cycle(['-time']),  # otherwise pass -time
    #     'w': []
    #     + [2, 4, 6, 8, 10, 12, 14, 16],
    #     # + list(np.arange(2, 17, 2)),
    #     'o': cycle([10]),
    #     'mpi': cycle(['mpich3']),
    #     'time': cycle([180]),
    #     'partition': cycle(['be-short'])
    # },

    # 'LHC-sync-openmpi3': {
    #     'exe': cycle([yc['exe_home'] + '_LHC_BUP_2017_test.py']),
    #     'p': cycle([2000000]),
    #     # 'p': [1e6, 2e6, 2e6, 2e6, 2e6, 1.25e6, 1.5e6, 1.75e6, 2e6],
    #     'b': cycle([96]),  # 96
    #     # 'b': [12, 12, 24, 36, 48, 96, 96, 96, 96],  # 96
    #     's': cycle([1000]),
    #     't': cycle([10000]),
    #     'm': cycle([0]),
    #     'seed': cycle([0]),
    #     'reduce': cycle([1]),
    #     'load': cycle([0.0]),
    #     'mtw': cycle([50]),
    #     'approx': cycle([2]),
    #     'timing': cycle(['-time']),  # otherwise pass -time
    #     'w': []
    #     + [10, 12],
    #     # + list(np.arange(2, 17, 2)),
    #     'o': cycle([10]),
    #     'mpi': cycle(['openmpi3']),
    #     'time': cycle([180]),
    #     'partition': cycle(['be-short'])
    # },
    # 'LHC-sync-mvapich2': {
    #     'exe': cycle([yc['exe_home'] + '_LHC_BUP_2017_test.py']),
    #     'p': cycle([2000000]),
    #     # 'p': [1e6, 2e6, 2e6, 2e6, 2e6, 1.25e6, 1.5e6, 1.75e6, 2e6],
    #     'b': cycle([96]),  # 96
    #     # 'b': [12, 12, 24, 36, 48, 96, 96, 96, 96],  # 96
    #     's': cycle([1000]),
    #     't': cycle([10000]),
    #     'm': cycle([0]),
    #     'seed': cycle([0]),
    #     'reduce': cycle([1]),
    #     'load': cycle([0.0]),
    #     'mtw': cycle([50]),
    #     'approx': cycle([2]),
    #     'timing': cycle(['-time']),  # otherwise pass -time
    #     'w': []
    #     + [2, 4, 6, 8, 10, 12, 14, 16],
    #     # + list(np.arange(2, 17, 2)),
    #     'o': cycle([10]),
    #     'mpi': cycle(['mvapich2']),
    #     'time': cycle([180]),
    #     'partition': cycle(['be-short'])
    # },



    # 'LHC-approx2-mpich3': {
    #     'exe': cycle([yc['exe_home'] + '_LHC_BUP_2017.py']),
    #     'p': cycle([2000000]),
    #     # 'p': [1e6, 2e6, 2e6, 2e6, 2e6, 1.25e6, 1.5e6, 1.75e6, 2e6],
    #     'b': cycle([96]),  # 96
    #     # 'b': [12, 12, 24, 36, 48, 96, 96, 96, 96],  # 96
    #     's': cycle([1000]),
    #     't': cycle([10000]),
    #     'm': cycle([0]),
    #     'seed': cycle([0]),
    #     'reduce': cycle([1]),
    #     'load': cycle([0.0]),
    #     'mtw': cycle([50]),
    #     'approx': cycle([2]),
    #     'timing': cycle(['-time']),  # otherwise pass -time
    #     'w': []
    #     + list(np.arange(2, 17, 2)),
    #     'o': cycle([10]),
    #     'mpi': cycle(['mpich3']),
    #     'time': cycle([180]),
    #     'partition': cycle(['be-short'])
    # },


    # 'LHC-approx2-openmpi3': {
    #     'exe': cycle([yc['exe_home'] + '_LHC_BUP_2017.py']),
    #     'p': cycle([2000000]),
    #     # 'p': [1e6, 2e6, 2e6, 2e6, 2e6, 1.25e6, 1.5e6, 1.75e6, 2e6],
    #     'b': cycle([96]),  # 96
    #     # 'b': [12, 12, 24, 36, 48, 96, 96, 96, 96],  # 96
    #     's': cycle([1000]),
    #     't': cycle([10000]),
    #     'm': cycle([0]),
    #     'seed': cycle([0]),
    #     'reduce': cycle([1]),
    #     'load': cycle([0.0]),
    #     'mtw': cycle([50]),
    #     'approx': cycle([2]),
    #     'timing': cycle(['-time']),  # otherwise pass -time
    #     'w': []
    #     # + [10, 12],
    #     + list(np.arange(2, 17, 2)),
    #     'o': cycle([10]),
    #     'mpi': cycle(['openmpi3']),
    #     'time': cycle([180]),
    #     'partition': cycle(['be-short'])
    # },

    # 'LHC-approx2-mvapich2': {
    #     'exe': cycle([yc['exe_home'] + '_LHC_BUP_2017.py']),
    #     'p': cycle([2000000]),
    #     # 'p': [1e6, 2e6, 2e6, 2e6, 2e6, 1.25e6, 1.5e6, 1.75e6, 2e6],
    #     'b': cycle([96]),  # 96
    #     # 'b': [12, 12, 24, 36, 48, 96, 96, 96, 96],  # 96
    #     's': cycle([1000]),
    #     't': cycle([10000]),
    #     'm': cycle([0]),
    #     'seed': cycle([0]),
    #     'reduce': cycle([1]),
    #     'load': cycle([0.0]),
    #     'mtw': cycle([50]),
    #     'approx': cycle([2]),
    #     'timing': cycle(['-time']),  # otherwise pass -time
    #     'w': []
    #     + list(np.arange(2, 17, 2)),
    #     'o': cycle([10]),
    #     'mpi': cycle(['mvapich2']),
    #     'time': cycle([180]),
    #     'partition': cycle(['be-short'])
    # },

    'LHC-lb-mpich3-approx2': {
        'exe': cycle([yc['exe_home'] + '_LHC_BUP_2017_lb.py']),
        'p': cycle([2000000]),
        'b': cycle([96]),  # 96
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
        'w': []
        # + [2, 4, 6, 8, 10, 12, 14, 16],
        + list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mpich3']),
        'time': cycle([30]),
        'partition': cycle(['be-short'])
    },

    'LHC-lb-mpich3': {
        'exe': cycle([yc['exe_home'] + '_LHC_BUP_2017_lb.py']),
        'p': cycle([2000000]),
        'b': cycle([96]),  # 96
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
        'w': []
        # + [2, 4, 6, 8, 10, 12, 14, 16],
        + list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mpich3']),
        'time': cycle([30]),
        'partition': cycle(['be-short'])
    },


    'LHC-lb-openmpi3-approx2': {
        'exe': cycle([yc['exe_home'] + '_LHC_BUP_2017_lb.py']),
        'p': cycle([2000000]),
        'b': cycle([96]),  # 96
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
        'w': []
        # + [2, 4, 6, 8, 10, 12, 14, 16],
        + list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['openmpi3']),
        'time': cycle([30]),
        'partition': cycle(['be-short'])
    },

    'LHC-lb-openmpi3': {
        'exe': cycle([yc['exe_home'] + '_LHC_BUP_2017_lb.py']),
        'p': cycle([2000000]),
        'b': cycle([96]),  # 96
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
        'w': []
        # + [2, 4, 6, 8, 10, 12, 14, 16],
        + list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['openmpi3']),
        'time': cycle([30]),
        'partition': cycle(['be-short'])
    },


    'LHC-lb-mvapich2-approx2': {
        'exe': cycle([yc['exe_home'] + '_LHC_BUP_2017_lb.py']),
        'p': cycle([2000000]),
        'b': cycle([96]),  # 96
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
        'w': []
        # + [2, 4, 6, 8, 10, 12, 14, 16],
        + list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mvapich2']),
        'time': cycle([30]),
        'partition': cycle(['be-short'])
    },

    'LHC-lb-mvapich2': {
        'exe': cycle([yc['exe_home'] + '_LHC_BUP_2017_lb.py']),
        'p': cycle([2000000]),
        'b': cycle([96]),  # 96
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
        'w': []
        # + [2, 4, 6, 8, 10, 12, 14, 16],
        + list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mvapich2']),
        'time': cycle([30]),
        'partition': cycle(['be-short'])
    },

    # 'LHC-lb-openmpi3': {
    #     'exe': cycle([yc['exe_home'] + '_LHC_BUP_2017_lb.py']),
    #     'p': cycle([2000000]),
    #     # 'p': [1e6, 2e6, 2e6, 2e6, 2e6, 1.25e6, 1.5e6, 1.75e6, 2e6],
    #     'b': cycle([96]),  # 96
    #     # 'b': [12, 12, 24, 36, 48, 96, 96, 96, 96],  # 96
    #     's': cycle([1000]),
    #     't': cycle([10000]),
    #     'm': cycle([0]),
    #     'seed': cycle([0]),
    #     'reduce': cycle([1]),
    #     'load': cycle([0.0]),
    #     'mtw': cycle([50]),
    #     'approx': cycle([2]),
    #     'timing': cycle(['-time']),  # otherwise pass -time
    #     'w': []
    #     + [10, 12],
    #     # + list(np.arange(2, 17, 2)),
    #     'o': cycle([10]),
    #     'mpi': cycle(['openmpi3']),
    #     'time': cycle([180]),
    #     'partition': cycle(['be-long'])
    # },

    # 'LHC-lb-mvapich2': {
    #     'exe': cycle([yc['exe_home'] + '_LHC_BUP_2017_lb.py']),
    #     'p': cycle([2000000]),
    #     # 'p': [1e6, 2e6, 2e6, 2e6, 2e6, 1.25e6, 1.5e6, 1.75e6, 2e6],
    #     'b': cycle([96]),  # 96
    #     # 'b': [12, 12, 24, 36, 48, 96, 96, 96, 96],  # 96
    #     's': cycle([1000]),
    #     't': cycle([10000]),
    #     'm': cycle([0]),
    #     'seed': cycle([0]),
    #     'reduce': cycle([1]),
    #     'load': cycle([0.0]),
    #     'mtw': cycle([50]),
    #     'approx': cycle([2]),
    #     'timing': cycle(['-time']),  # otherwise pass -time
    #     'w': []
    #     + [2, 4, 6, 8, 10, 12, 14, 16],
    #     # + list(np.arange(2, 17, 2)),
    #     'o': cycle([10]),
    #     'mpi': cycle(['mvapich2']),
    #     'time': cycle([180]),
    #     'partition': cycle(['be-short'])
    # },

    # 'LHC-weak-scale-mpich3': {
    #     'exe': cycle([yc['exe_home'] + '_LHC_BUP_2017.py']),
    #     # 'p': cycle([2000000]),
    #     # 'p': [1e6, 2e6, 2e6, 2e6, 2e6, 1.25e6, 1.5e6, 1.75e6, 2e6],
    #     'p': [2e6, 2e6, 2e6, 1.25e6, 1.5e6, 1.75e6, 2e6],
    #     # 'b': cycle([96]),  # 96
    #     # 'b': [12, 12, 24, 36, 48, 96, 96, 96, 96],  # 96
    #     'b': [24, 36, 48, 96, 96, 96, 96],  # 96
    #     's': cycle([1000]),
    #     't': cycle([10000]),
    #     'm': cycle([0]),
    #     'seed': cycle([0]),
    #     'reduce': cycle([1]),
    #     'load': cycle([0.0]),
    #     'mtw': cycle([50]),
    #     'approx': cycle([0]),
    #     'timing': cycle(['-time']),  # otherwise pass -time
    #     'w': []
    #     + [4, 6, 8, 10, 12, 14, 16],
    #     # + list(np.arange(2, 17, 2)),
    #     'o': cycle([10]),
    #     'mpi': cycle(['mpich3']),
    #     'time': cycle([90]),
    #     'partition': cycle(['be-short'])
    # },

    # 'LHC-weak-scale-openmpi3': {
    #     'exe': cycle([yc['exe_home'] + '_LHC_BUP_2017.py']),
    #     # 'p': cycle([2000000]),
    #     # 'p': [1e6, 2e6, 2e6, 2e6, 2e6, 1.25e6, 1.5e6, 1.75e6, 2e6],
    #     'p': [1.75e6],
    #     # 'b': cycle([96]),  # 96
    #     # 'b': [12, 12, 24, 36, 48, 96, 96, 96, 96],  # 96
    #     'b': [96],  # 96
    #     's': cycle([1000]),
    #     't': cycle([10000]),
    #     'm': cycle([0]),
    #     'seed': cycle([0]),
    #     'reduce': cycle([1]),
    #     'load': cycle([0.0]),
    #     'mtw': cycle([50]),
    #     'approx': cycle([0]),
    #     'timing': cycle(['-time']),  # otherwise pass -time
    #     'w': []
    #     + [14],
    #     # + list(np.arange(2, 17, 2)),
    #     'o': cycle([10]),
    #     'mpi': cycle(['openmpi3']),
    #     'time': cycle([90]),
    #     'partition': cycle(['be-long'])
    # },

    # 'LHC-weak-scale-mvapich2': {
    #     'exe': cycle([yc['exe_home'] + '_LHC_BUP_2017.py']),
    #     # 'p': cycle([2000000]),
    #     # 'p': [1e6, 2e6, 2e6, 2e6, 2e6, 1.25e6, 1.5e6, 1.75e6, 2e6],
    #     'p': [1.25e6, 1.75e6],
    #     # 'b': cycle([96]),  # 96
    #     # 'b': [12, 12, 24, 36, 48, 96, 96, 96, 96],  # 96
    #     'b': [96, 96],  # 96
    #     's': cycle([1000]),
    #     't': cycle([10000]),
    #     'm': cycle([0]),
    #     'seed': cycle([0]),
    #     'reduce': cycle([1]),
    #     'load': cycle([0.0]),
    #     'mtw': cycle([50]),
    #     'approx': cycle([0]),
    #     'timing': cycle(['-time']),  # otherwise pass -time
    #     'w': []
    #     + [10, 14],
    #     # + list(np.arange(2, 17, 2)),
    #     'o': cycle([10]),
    #     'mpi': cycle(['mvapich2']),
    #     'time': cycle([90]),
    #     'partition': cycle(['be-long'])
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
