from cycler import cycle
import numpy as np

case = 'SPS'

run_configs = [
    'mpich3',
    # 'mvapich2',
    # 'tp-mpich3',
    # 'tp-mvapich2',
    # 'lb-tp-mpich3',
    # 'lb-tp-mvapich2',
    # 'lb-mpich3',
    # 'lb-mvapich2',
    # 'lb-mpich3-intv100',
    # 'lb-mvapich2-intv100',
    # 'lb-mpich3-approx2',
    # 'lb-mvapich2-approx2',
    # 'dynamic-lb-mpich3',
    # 'dynamic-lb-mvapich2',
    # 'dynamic-lb-mpich3-approx2',
    # 'dynamic-lb-mvapich2-approx2',
]


configs = {

     'mpich3': {
        'exe': cycle(['SPS_main_random.py']),
        'p': cycle([4000000]),
        'b': cycle([72]),  # 72
        's': cycle([1408]),
        't': cycle([1000]),  # 4000
        # 't': cycle([500]),  # 4000
        'm': cycle([0]),
        'reduce': cycle([1]),
        'load': cycle([0.0]),
        'mtw': cycle([0]),
        'approx': cycle([0]),
        'log': cycle([False]),
        'lb': cycle(['reportonly']),
        'lba': cycle([500]),
        'timing': cycle(['-time']),  # otherwise pass -time
        'seed': cycle([0]),
        'w': [] +
        [1],
        # [4, 8, 12, 16],
        # + list(np.arange(2, 17, 2)),
        # 'o': cycle([1]),
        'o': cycle([10]),
        'mpi': cycle(['mpich3']),
        'time': cycle([60]),
        'partition': cycle(['inf-short']),
        'repeats': 3,
    },
    
    'mvapich2': {
        'exe': cycle(['SPS_main_random.py']),
        'p': cycle([4000000]),
        'b': cycle([36]),  # 72
        's': cycle([1408]),
        't': cycle([10000]),  # 4000
        'm': cycle([0]),
        'reduce': cycle([1]),
        'load': cycle([0.0]),
        'mtw': cycle([0]),
        'approx': cycle([0]),
        'log': cycle([True]),
        'lb': cycle(['reportonly']),
        'lba': cycle([500]),
        'timing': cycle(['-time']),  # otherwise pass -time
        'seed': cycle([0]),
        'w': [] +
        [4, 8, 12, 16],
        #+ list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mvapich2']),
        'time': cycle([60]),
        'partition': cycle(['inf-short']),
        'repeats': 4,
    },

     'tp-mpich3': {
        'exe': cycle(['SPS_main_random_tp.py']),
        'p': cycle([4000000]),
        'b': cycle([36]),  # 72
        's': cycle([1408]),
        't': cycle([10000]),  # 4000
        'm': cycle([0]),
        'reduce': cycle([1]),
        'load': cycle([0.0]),
        'mtw': cycle([0]),
        'approx': cycle([2]),
        'log': cycle([True]),
        'lb': cycle(['reportonly']),
        'lba': cycle([500]),
        'timing': cycle(['-time']),  # otherwise pass -time
        'seed': cycle([0]),
        'w': [] +
        [4, 8, 12, 16],
        # + list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mpich3']),
        'time': cycle([60]),
        'partition': cycle(['inf-short']),
        'repeats': 4,
    },
    
    'tp-mvapich2': {
        'exe': cycle(['SPS_main_random_tp.py']),
        'p': cycle([4000000]),
        'b': cycle([36]),  # 72
        's': cycle([1408]),
        't': cycle([10000]),  # 4000
        'm': cycle([0]),
        'reduce': cycle([1]),
        'load': cycle([0.0]),
        'mtw': cycle([0]),
        'approx': cycle([2]),
        'log': cycle([True]),
        'lb': cycle(['reportonly']),
        'lba': cycle([500]),
        'timing': cycle(['-time']),  # otherwise pass -time
        'seed': cycle([0]),
        'w': [] +
        [4, 8, 12, 16],
        #+ list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mvapich2']),
        'time': cycle([60]),
        'partition': cycle(['inf-short']),
        'repeats': 4,
    },


     'lb-mpich3': {
        'exe': cycle(['SPS_main_random.py']),
        'p': cycle([4000000]),
        'b': cycle([36]),  # 72
        's': cycle([1408]),
        't': cycle([10000]),  # 4000
        'm': cycle([0]),
        'reduce': cycle([1]),
        'load': cycle([0.0]),
        'mtw': cycle([0]),
        'approx': cycle([0]),
        'log': cycle([True]),
        'lb': cycle(['interval']),
        'lba': cycle([500]),
        'timing': cycle(['-time']),  # otherwise pass -time
        'seed': cycle([0]),
        'w': [] +
        [4, 8, 12, 16],
        # + list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mpich3']),
        'time': cycle([60]),
        'partition': cycle(['inf-short']),
        'repeats': 4,
    },
    
    'lb-mvapich2': {
        'exe': cycle(['SPS_main_random.py']),
        'p': cycle([4000000]),
        'b': cycle([36]),  # 72
        's': cycle([1408]),
        't': cycle([10000]),  # 4000
        'm': cycle([0]),
        'reduce': cycle([1]),
        'load': cycle([0.0]),
        'mtw': cycle([0]),
        'approx': cycle([0]),
        'log': cycle([True]),
        'lb': cycle(['interval']),
        'lba': cycle([500]),
        'timing': cycle(['-time']),  # otherwise pass -time
        'seed': cycle([0]),
        'w': [] +
        [4, 8, 12, 16],
        #+ list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mvapich2']),
        'time': cycle([60]),
        'partition': cycle(['inf-short']),
        'repeats': 4,
    },


     'lb-tp-mpich3': {
        'exe': cycle(['SPS_main_random_tp.py']),
        'p': cycle([4000000]),
        'b': cycle([36]),  # 72
        's': cycle([1408]),
        't': cycle([10000]),  # 4000
        'm': cycle([0]),
        'reduce': cycle([1]),
        'load': cycle([0.0]),
        'mtw': cycle([0]),
        'approx': cycle([2]),
        'log': cycle([True]),
        'lb': cycle(['interval']),
        'lba': cycle([500]),
        'timing': cycle(['-time']),  # otherwise pass -time
        'seed': cycle([0]),
        'w': [] +
        [4, 8, 12, 16],
        # + list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mpich3']),
        'time': cycle([60]),
        'partition': cycle(['inf-short']),
        'repeats': 4,
    },
    
    'lb-tp-mvapich2': {
        'exe': cycle(['SPS_main_random_tp.py']),
        'p': cycle([4000000]),
        'b': cycle([36]),  # 72
        's': cycle([1408]),
        't': cycle([10000]),  # 4000
        'm': cycle([0]),
        'reduce': cycle([1]),
        'load': cycle([0.0]),
        'mtw': cycle([0]),
        'approx': cycle([2]),
        'log': cycle([True]),
        'lb': cycle(['interval']),
        'lba': cycle([500]),
        'timing': cycle(['-time']),  # otherwise pass -time
        'seed': cycle([0]),
        'w': [] +
        [4, 8, 12, 16],
        #+ list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mvapich2']),
        'time': cycle([60]),
        'partition': cycle(['inf-short']),
        'repeats': 4,
    },

    'lb-mpich3-approx2': {
        'exe': cycle(['SPS_main_random_tp.py']),
        'p': cycle([4000000]),
        'b': cycle([72]),  # 72
        's': cycle([1408]),
        't': cycle([10000]),  # 4000
        'm': cycle([0]),
        'reduce': cycle([1]),
        'load': cycle([0.0]),
        'mtw': cycle([0]),
        'approx': cycle([2]),
        'log': cycle([True]),
        'lb': cycle(['interval']),
        'lba': cycle([1000]),
        'timing': cycle(['-time']),  # otherwise pass -time
        'seed': cycle([0]),
        'w': [] +
        [4, 8, 12, 16],
        # + list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mpich3']),
        'time': cycle([60]),
        'partition': cycle(['inf-short']),
        'repeats': 5,
    },

 
    'lb-mvapich2-approx2': {
        'exe': cycle(['SPS_main_random_tp.py']),
        'p': cycle([4000000]),
        'b': cycle([72]),  # 72
        's': cycle([1408]),
        't': cycle([10000]),  # 4000
        'm': cycle([0]),
        'reduce': cycle([1]),
        'load': cycle([0.0]),
        'mtw': cycle([0]),
        'approx': cycle([2]),
        'log': cycle([True]),
        'lb': cycle(['interval']),
        'lba': cycle([1000]),
        'timing': cycle(['-time']),  # otherwise pass -time
        'seed': cycle([0]),
        'w': [] + [4, 8, 12, 16],
        #+ list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mvapich2']),
        'time': cycle([60]),
        'partition': cycle(['inf-short']),
        'repeats': 5,
    },

    'lb-mpich3-intv100': {
        'exe': cycle(['SPS_main_random_tp.py']),
        'p': cycle([4000000]),
        'b': cycle([72]),  # 72
        's': cycle([1408]),
        't': cycle([10000]),  # 4000
        'm': cycle([0]),
        'reduce': cycle([1]),
        'load': cycle([0.0]),
        'mtw': cycle([0]),
        'approx': cycle([0]),
        'log': cycle([True]),
        'lb': cycle(['interval']),
        'lba': cycle([100]),
        'timing': cycle(['-time']),  # otherwise pass -time
        'seed': cycle([0]),
        'w': [] +
        [4, 8, 12, 16],
        # + list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mpich3']),
        'time': cycle([60]),
        'partition': cycle(['inf-short']),
        'repeats': 5,
    },


    'lb-mvapich2-intv100': {
        'exe': cycle(['SPS_main_random_tp.py']),
        'p': cycle([4000000]),
        'b': cycle([72]),  # 72
        's': cycle([1408]),
        't': cycle([10000]),  # 4000
        'm': cycle([0]),
        'reduce': cycle([1]),
        'load': cycle([0.0]),
        'mtw': cycle([0]),
        'approx': cycle([0]),
        'log': cycle([True]),
        'lb': cycle(['interval']),
        'lba': cycle([100]),
        'timing': cycle(['-time']),  # otherwise pass -time
        'seed': cycle([0]),
        'w': [] +
        [4, 8, 12, 16],
        #+ list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mvapich2']),
        'time': cycle([60]),
        'partition': cycle(['inf-short']),
        'repeats': 5,
    },


    'dynamic-lb-mpich3-approx2': {
        'exe': cycle(['SPS_main_random_tp.py']),
        'p': cycle([4000000]),
        'b': cycle([72]),  # 72
        's': cycle([1408]),
        't': cycle([10000]),  # 4000
        'm': cycle([0]),
        'reduce': cycle([1]),
        'load': cycle([0.0]),
        'mtw': cycle([0]),
        'approx': cycle([2]),
        'log': cycle([True]),
        'lb': cycle(['dynamic']),
        'lba': cycle([1000]),
        'timing': cycle(['-time']),  # otherwise pass -time
        'seed': cycle([0]),
        'w': [] +
        [4, 8, 12, 16],
        # + list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mpich3']),
        'time': cycle([60]),
        'partition': cycle(['inf-short']),
        'repeats': 5,
    },

    'dynamic-lb-mvapich2-approx2': {
        'exe': cycle(['SPS_main_random_tp.py']),
        'p': cycle([4000000]),
        'b': cycle([72]),  # 72
        's': cycle([1408]),
        't': cycle([10000]),  # 4000
        'm': cycle([0]),
        'reduce': cycle([1]),
        'load': cycle([0.0]),
        'mtw': cycle([0]),
        'approx': cycle([2]),
        'log': cycle([True]),
        'lb': cycle(['dynamic']),
        'lba': cycle([1000]),
        'timing': cycle(['-time']),  # otherwise pass -time
        'seed': cycle([0]),
        'w': [] + [4, 8, 12, 16],
        #+ list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mvapich2']),
        'time': cycle([60]),
        'partition': cycle(['inf-short']),
        'repeats': 5,
    },

    'dynamic-lb-mpich3': {
        'exe': cycle(['SPS_main_random_tp.py']),
        'p': cycle([4000000]),
        'b': cycle([72]),  # 72
        's': cycle([1408]),
        't': cycle([10000]),  # 4000
        'm': cycle([0]),
        'reduce': cycle([1]),
        'load': cycle([0.0]),
        'mtw': cycle([0]),
        'approx': cycle([0]),
        'log': cycle([True]),
        'lb': cycle(['dynamic']),
        'lba': cycle([1000]),
        'timing': cycle(['-time']),  # otherwise pass -time
        'seed': cycle([0]),
        'w': [] +
        [4, 8, 12, 16],
        # + list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mpich3']),
        'time': cycle([60]),
        'partition': cycle(['inf-short']),
        'repeats': 5,
    },
    'dynamic-lb-openmpi3': {
        'exe': cycle(['SPS_main_random_tp.py']),
        'p': cycle([4000000]),
        'b': cycle([72]),  # 72
        's': cycle([1408]),
        't': cycle([10000]),  # 4000
        'm': cycle([0]),
        'reduce': cycle([1]),
        'load': cycle([0.0]),
        'mtw': cycle([0]),
        'approx': cycle([0]),
        'log': cycle([True]),
        'lb': cycle(['dynamic']),
        'lba': cycle([1000]),
        'timing': cycle(['-time']),  # otherwise pass -time
        'seed': cycle([0]),
        'w': [] +
        [4, 8, 12, 16],
        #+ list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['openmpi3']),
        'time': cycle([60]),
        'partition': cycle(['inf-short']),
        'repeats': 5,
    },
    'dynamic-lb-mvapich2': {
        'exe': cycle(['SPS_main_random_tp.py']),
        'p': cycle([4000000]),
        'b': cycle([72]),  # 72
        's': cycle([1408]),
        't': cycle([10000]),  # 4000
        'm': cycle([0]),
        'reduce': cycle([1]),
        'load': cycle([0.0]),
        'mtw': cycle([0]),
        'approx': cycle([0]),
        'log': cycle([True]),
        'lb': cycle(['dynamic']),
        'lba': cycle([1000]),
        'timing': cycle(['-time']),  # otherwise pass -time
        'seed': cycle([0]),
        'w': [] +
        [4, 8, 12, 16],
        #+ list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mvapich2']),
        'time': cycle([60]),
        'partition': cycle(['inf-short']),
        'repeats': 5,
    },

   

    'SPS-weak-scale-mpich3': {
        'exe': cycle(['SPS_main_random.py']),
        # 'p': cycle([4000000]),
        # 'p': [1e6, 2e6, 4e6, 3e6, 4e6, 2.5e6, 3e6, 3.5e6, 4e6],
        'p': [2.5e6, 3e6, 3.5e6, 4e6],
        # 'b': cycle([72]), # 72
        'b': [72, 72, 72, 72],  # 72
        's': cycle([1408]),
        't': cycle([10000]),  # 4000
        'm': cycle([0]),
        'reduce': cycle([1]),
        'load': cycle([0.0]),
        'mtw': cycle([0]),
        'approx': cycle([0]),
        'timing': cycle(['-time']),  # otherwise pass -time
        'seed': cycle([0]),
        'w': [10, 12, 14, 16],
        # + list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mpich3']),
        'time': cycle([90]),
        'partition': cycle(['inf-short']),
    },

    'SPS-weak-scale-openmpi3': {
        'exe': cycle(['SPS_main_random.py']),
        # 'p': cycle([4000000]),
        # 'p': [1e6, 2e6, 4e6, 3e6, 4e6, 2.5e6, 3e6, 3.5e6, 4e6],
        'p': [2.5e6, 3e6, 3.5e6, 4e6],
        # 'b': cycle([72]), # 72
        # 'b': [18, 18, 18, 36, 36, 72, 72, 72, 72], # 72
        'b': [72, 72, 72, 72],  # 72
        's': cycle([1408]),
        't': cycle([10000]),  # 4000
        'm': cycle([0]),
        'reduce': cycle([1]),
        'load': cycle([0.0]),
        'mtw': cycle([0]),
        'approx': cycle([0]),
        'timing': cycle(['-time']),  # otherwise pass -time
        'seed': cycle([0]),
        'w': [10, 12, 14, 16],
        # + list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['openmpi3']),
        'time': cycle([90]),
        'partition': cycle(['inf-short']),
    },

    'SPS-weak-scale-mvapich2': {
        'exe': cycle(['SPS_main_random.py']),

        # 'p': cycle([4000000]),
        # 'p': [1e6, 2e6, 4e6, 3e6, 4e6, 2.5e6, 3e6, 3.5e6, 4e6],
        'p': [2.5e6, 3e6, 3.5e6, 4e6],
        # 'b': cycle([72]), # 72
        # 'b': [18, 18, 18, 36, 36, 72, 72, 72, 72], # 72
        'b': [72, 72, 72, 72],  # 72
        's': cycle([1408]),
        't': cycle([10000]),  # 4000
        'm': cycle([0]),
        'reduce': cycle([1]),
        'load': cycle([0.0]),
        'mtw': cycle([0]),
        'approx': cycle([0]),
        'timing': cycle(['-time']),  # otherwise pass -time
        'seed': cycle([0]),
        'w': [10, 12, 14, 16],
        # + list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mvapich2']),
        'time': cycle([90]),
        'partition': cycle(['inf-short']),
    },

}
