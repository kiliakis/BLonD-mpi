from cycler import cycle
import numpy as np

configs = {

    # 'SPS-rand-72B-4MPPB-approx2': {
    #     'exe': cycle(['SPS_main_random.py']),
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
    #     'exe': cycle(['SPS_main_random.py']),
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
    #     'seed': [1] * 3 + [2] * 3 + [3] * 3 + [4] * 3 + [5] * 3,
    #     'reduce': []
    #     + [1, 2, 3] * 5,
    #     'w': [16] * 18,
    #     'o': cycle([10]),
    #     'time': cycle([2000]),
    #     'mpi': cycle(['mpich3']),
    #     'partition': cycle(['be-long'])
    # },

    # 'SPS-b72-4MPPB-t10k-mpich3': {
    #     'exe': cycle(['SPS_main_random.py']),
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
    #     'exe': cycle(['SPS_main_random.py']),
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
    #     'exe': cycle(['SPS_main_random.py']),
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

    # 'SPS-weak-scale-mpich3': {
    #     'exe': cycle(['SPS_main_random.py']),
    #     # 'p': cycle([4000000]),
    #     # 'p': [1e6, 2e6, 4e6, 3e6, 4e6, 2.5e6, 3e6, 3.5e6, 4e6],
    #     'p': [2.5e6, 3e6, 3.5e6, 4e6],
    #     # 'b': cycle([72]), # 72
    #     'b': [72, 72, 72, 72], # 72
    #     's': cycle([1408]),
    #     't': cycle([10000]), # 4000
    #     'm': cycle([0]),
    #     'reduce': cycle([1]),
    #     'load': cycle([0.0]),
    #     'mtw': cycle([0]),
    #     'approx': cycle([0]),
    #     'timing': cycle(['-time']), # otherwise pass -time
    #     'seed': cycle([0]),
    #     'w': [10, 12, 14, 16],
    #     # + list(np.arange(2, 17, 2)),
    #     'o': cycle([10]),
    #     'mpi': cycle(['mpich3']),
    #     'time': cycle([90]),
    #     'partition': cycle(['be-short']),
    # },

    # 'SPS-weak-scale-openmpi3': {
    #     'exe': cycle(['SPS_main_random.py']),
    #     # 'p': cycle([4000000]),
    #     # 'p': [1e6, 2e6, 4e6, 3e6, 4e6, 2.5e6, 3e6, 3.5e6, 4e6],
    #     'p': [2.5e6, 3e6, 3.5e6, 4e6],
    #     # 'b': cycle([72]), # 72
    #     # 'b': [18, 18, 18, 36, 36, 72, 72, 72, 72], # 72
    #     'b': [72, 72, 72, 72], # 72
    #     's': cycle([1408]),
    #     't': cycle([10000]), # 4000
    #     'm': cycle([0]),
    #     'reduce': cycle([1]),
    #     'load': cycle([0.0]),
    #     'mtw': cycle([0]),
    #     'approx': cycle([0]),
    #     'timing': cycle(['-time']), # otherwise pass -time
    #     'seed': cycle([0]),
    #     'w': [10, 12, 14, 16],
    #     # + list(np.arange(2, 17, 2)),
    #     'o': cycle([10]),
    #     'mpi': cycle(['openmpi3']),
    #     'time': cycle([90]),
    #     'partition': cycle(['be-short']),
    # },

    # 'SPS-weak-scale-mvapich2': {
    #     'exe': cycle(['SPS_main_random.py']),

    #     # 'p': cycle([4000000]),
    #     # 'p': [1e6, 2e6, 4e6, 3e6, 4e6, 2.5e6, 3e6, 3.5e6, 4e6],
    #     'p': [2.5e6, 3e6, 3.5e6, 4e6],
    #     # 'b': cycle([72]), # 72
    #     # 'b': [18, 18, 18, 36, 36, 72, 72, 72, 72], # 72
    #     'b': [72, 72, 72, 72], # 72
    #     's': cycle([1408]),
    #     't': cycle([10000]), # 4000
    #     'm': cycle([0]),
    #     'reduce': cycle([1]),
    #     'load': cycle([0.0]),
    #     'mtw': cycle([0]),
    #     'approx': cycle([0]),
    #     'timing': cycle(['-time']), # otherwise pass -time
    #     'seed': cycle([0]),
    #     'w': [10, 12, 14, 16],
    #     # + list(np.arange(2, 17, 2)),
    #     'o': cycle([10]),
    #     'mpi': cycle(['mvapich2']),
    #     'time': cycle([90]),
    #     'partition': cycle(['be-short']),
    # },


    # 'SPS-sync-mpich3': {
    #     'exe': cycle(['SPS_main_random_test.py']),
    #     'p': cycle([4000000]),
    #     # 'p': [1e6, 2e6, 4e6, 3e6, 4e6, 2.5e6, 3e6, 3.5e6, 4e6],
    #     'b': cycle([72]), # 72
    #     # 'b': [18, 18, 18, 36, 36, 72, 72, 72, 72], # 72
    #     's': cycle([1408]),
    #     't': cycle([10000]), # 4000
    #     'm': cycle([0]),
    #     'reduce': cycle([1]),
    #     'load': cycle([0.0]),
    #     'mtw': cycle([0]),
    #     'approx': cycle([0]),
    #     'timing': cycle(['-time']), # otherwise pass -time
    #     'seed': cycle([0]),
    #     'w': [1, 10, 12, 14, 16],
    #     # + list(np.arange(2, 17, 2)),
    #     'o': cycle([10]),
    #     'mpi': cycle(['mpich3']),
    #     'time': cycle([180]),
    #     'partition': cycle(['be-long']),
    # },
    # 'SPS-sync-openmpi3': {
    #     'exe': cycle(['SPS_main_random_test.py']),
    #     'p': cycle([4000000]),
    #     # 'p': [1e6, 2e6, 4e6, 3e6, 4e6, 2.5e6, 3e6, 3.5e6, 4e6],
    #     'b': cycle([72]), # 72
    #     # 'b': [18, 18, 18, 36, 36, 72, 72, 72, 72], # 72
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
    #     + list(np.arange(2, 17, 2)),
    #     'o': cycle([10]),
    #     'mpi': cycle(['openmpi3']),
    #     'time': cycle([180]),
    #     'partition': cycle(['be-short']),
    # },
    # 'SPS-sync-mvapich2': {
    #     'exe': cycle(['SPS_main_random_test.py']),
    #     'p': cycle([4000000]),
    #     # 'p': [1e6, 2e6, 4e6, 3e6, 4e6, 2.5e6, 3e6, 3.5e6, 4e6],
    #     'b': cycle([72]), # 72
    #     # 'b': [18, 18, 18, 36, 36, 72, 72, 72, 72], # 72
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
    #     + list(np.arange(2, 17, 2)),
    #     'o': cycle([10]),
    #     'mpi': cycle(['mvapich2']),
    #     'time': cycle([180]),
    #     'partition': cycle(['be-short']),
    # },

    'SPS-lb-mpich3-approx2': {
        'exe': cycle(['SPS_main_random.py']),
        'p': cycle([4000000]),
        # 'p': [1e6, 2e6, 4e6, 3e6, 4e6, 2.5e6, 3e6, 3.5e6, 4e6],
        'b': cycle([72]), # 72
        # 'b': [18, 18, 18, 36, 36, 72, 72, 72, 72], # 72
        's': cycle([1408]),
        't': cycle([10000]), # 4000
        'm': cycle([0]),
        'reduce': cycle([1]),
        'load': cycle([0.0]),
        'mtw': cycle([0]),
        'approx': cycle([2]),
        'log': cycle([True]),
        'lb': cycle(['dynamic']),
        'lba': cycle([10]),
        'timing': cycle(['-time']), # otherwise pass -time
        'seed': cycle([0]),
        'w': []
        + [2, 4, 8, 12, 16],
        # + list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mpich3']),
        'time': cycle([60]),
        'partition': cycle(['be-short']),
        'repeats': 5,
    },
    # 'SPS-lb-openmpi3-approx2': {
    #     'exe': cycle(['SPS_main_random.py']),
    #     'p': cycle([4000000]),
    #     # 'p': [1e6, 2e6, 4e6, 3e6, 4e6, 2.5e6, 3e6, 3.5e6, 4e6],
    #     'b': cycle([72]), # 72
    #     # 'b': [18, 18, 18, 36, 36, 72, 72, 72, 72], # 72
    #     's': cycle([1408]),
    #     't': cycle([10000]), # 4000
    #     'm': cycle([0]),
    #     'reduce': cycle([1]),
    #     'load': cycle([0.0]),
    #     'mtw': cycle([0]),
    #     'approx': cycle([2]),
    #     'log': cycle([True]),
    #     'lb': cycle(['dynamic']),
    #     'lba': cycle([10]),
    #     'timing': cycle(['-time']), # otherwise pass -time
    #     'seed': cycle([0]),
    #     'w': []
    #     + list(np.arange(2, 17, 2)),
    #     #+ [2],
    #     'o': cycle([10]),
    #     'mpi': cycle(['openmpi3']),
    #     'time': cycle([60]),
    #     'partition': cycle(['be-short']),
    #     'repeats': 5,

    # },
    # 'SPS-lb-mvapich2-approx2': {
    #     'exe': cycle(['SPS_main_random.py']),
    #     'p': cycle([4000000]),
    #     # 'p': [1e6, 2e6, 4e6, 3e6, 4e6, 2.5e6, 3e6, 3.5e6, 4e6],
    #     'b': cycle([72]), # 72
    #     # 'b': [18, 18, 18, 36, 36, 72, 72, 72, 72], # 72
    #     's': cycle([1408]),
    #     't': cycle([10000]), # 4000
    #     'm': cycle([0]),
    #     'reduce': cycle([1]),
    #     'load': cycle([0.0]),
    #     'mtw': cycle([0]),
    #     'approx': cycle([2]),
    #     'log': cycle([True]),
    #     'lb': cycle(['dynamic']),
    #     'lba': cycle([10]),
    #     'timing': cycle(['-time']), # otherwise pass -time
    #     'seed': cycle([0]),
    #     'w': []
    #    + list(np.arange(2, 17, 2)),
    #     'o': cycle([10]),
    #     'mpi': cycle(['mvapich2']),
    #     'time': cycle([60]),
    #     'partition': cycle(['be-short']),
    #     'repeats': 5,

    # },

    'SPS-lb-mpich3': {
        'exe': cycle(['SPS_main_random_lb.py']),
        'p': cycle([4000000]),
        # 'p': [1e6, 2e6, 4e6, 3e6, 4e6, 2.5e6, 3e6, 3.5e6, 4e6],
        'b': cycle([72]), # 72
        # 'b': [18, 18, 18, 36, 36, 72, 72, 72, 72], # 72
        's': cycle([1408]),
        't': cycle([10000]), # 4000
        'm': cycle([0]),
        'reduce': cycle([1]),
        'load': cycle([0.0]),
        'mtw': cycle([0]),
        'approx': cycle([0]),
        'log': cycle([True]),
        'lb': cycle(['dynamic']),
        'lba': cycle([10]),
        'timing': cycle(['-time']), # otherwise pass -time
        'seed': cycle([0]),
        'w': []
        + [2, 4, 8, 12, 16],
        # + list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mpich3']),
        'time': cycle([60]),
        'partition': cycle(['be-short']),
        'repeats': 5,
    },
    # 'SPS-lb-openmpi3': {
    #     'exe': cycle(['SPS_main_random_lb.py']),
    #     'p': cycle([4000000]),
    #     # 'p': [1e6, 2e6, 4e6, 3e6, 4e6, 2.5e6, 3e6, 3.5e6, 4e6],
    #     'b': cycle([72]), # 72
    #     # 'b': [18, 18, 18, 36, 36, 72, 72, 72, 72], # 72
    #     's': cycle([1408]),
    #     't': cycle([10000]), # 4000
    #     'm': cycle([0]),
    #     'reduce': cycle([1]),
    #     'load': cycle([0.0]),
    #     'mtw': cycle([0]),
    #     'approx': cycle([0]),
    #     'log': cycle([True]),
    #     'lb': cycle(['dynamic']),
    #     'lba': cycle([10]),
    #     'timing': cycle(['-time']), # otherwise pass -time
    #     'seed': cycle([0]),
    #     'w': []
    #     + list(np.arange(2, 17, 2)),
    #     'o': cycle([10]),
    #     'mpi': cycle(['openmpi3']),
    #     'time': cycle([60]),
    #     'partition': cycle(['be-short']),
    #     'repeats': 5,
    # },
    # 'SPS-lb-mvapich2': {
    #     'exe': cycle(['SPS_main_random_lb.py']),
    #     'p': cycle([4000000]),
    #     # 'p': [1e6, 2e6, 4e6, 3e6, 4e6, 2.5e6, 3e6, 3.5e6, 4e6],
    #     'b': cycle([72]), # 72
    #     # 'b': [18, 18, 18, 36, 36, 72, 72, 72, 72], # 72
    #     's': cycle([1408]),
    #     't': cycle([10000]), # 4000
    #     'm': cycle([0]),
    #     'reduce': cycle([1]),
    #     'load': cycle([0.0]),
    #     'mtw': cycle([0]),
    #     'approx': cycle([0]),
    #     'log': cycle([True]),
    #     'lb': cycle(['dynamic']),
    #     'lba': cycle([10]),
    #     'timing': cycle(['-time']), # otherwise pass -time
    #     'seed': cycle([0]),
    #     'w': []
    #     + list(np.arange(2, 17, 2)),
    #     'o': cycle([10]),
    #     'mpi': cycle(['mvapich2']),
    #     'time': cycle([60]),
    #     'partition': cycle(['be-short']),
    #     'repeats': 5,

    # },
}
