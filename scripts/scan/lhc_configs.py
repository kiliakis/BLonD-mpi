from cycler import cycle
import numpy as np

case = 'LHC'

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
        'exe': cycle(['_LHC_BUP_2017.py']),
        'p': cycle([2000000]),
        'b': cycle([96]),  # 96
        's': cycle([1000]),
        't': cycle([1000]),
        'm': cycle([0]),
        'seed': cycle([0]),
        'reduce': cycle([1]),
        'load': cycle([0.0]),
        'mtw': cycle([50]),
        'approx': cycle([0]),
        'log': cycle([False]),
        'lb': cycle(['reportonly']),
        'lba': cycle([500]),
        'timing': cycle(['-time']),  # otherwise pass -time
        'w': [] +
        [1],
        # [4, 8, 12, 16],
        #+ list(np.arange(2, 17, 2)),
        # 'o': cycle([10]),
        'o': cycle([10]),
        'mpi': cycle(['mpich3']),
        'time': cycle([30]),
        'partition': cycle(['inf-short']),
        'repeats': 4
    },

    'mvapich2': {
        'exe': cycle(['_LHC_BUP_2017.py']),
        'p': cycle([2000000]),
        'b': cycle([48]),  # 96
        's': cycle([1000]),
        't': cycle([10000]),
        'm': cycle([0]),
        'seed': cycle([0]),
        'reduce': cycle([1]),
        'load': cycle([0.0]),
        'mtw': cycle([50]),
        'approx': cycle([0]),
        'log': cycle([True]),
        'lb': cycle(['reportonly']),
        'lba': cycle([500]),
        'timing': cycle(['-time']),  # otherwise pass -time
        'w': [] +
        [4, 8, 12, 16],
        #+ list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mvapich2']),
        'time': cycle([30]),
        'partition': cycle(['inf-short']),
        'repeats': 4
    },

    'tp-mpich3': {
        'exe': cycle(['_LHC_BUP_2017_tp.py']),
        'p': cycle([2000000]),
        'b': cycle([48]),  # 96
        's': cycle([1000]),
        't': cycle([10000]),
        'm': cycle([0]),
        'seed': cycle([0]),
        'reduce': cycle([1]),
        'load': cycle([0.0]),
        'mtw': cycle([50]),
        'approx': cycle([2]),
        'log': cycle([True]),
        'lb': cycle(['reportonly']),
        'lba': cycle([500]),
        'timing': cycle(['-time']),  # otherwise pass -time
        'w': [] +
        [4, 8, 12, 16],
        #+ list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mpich3']),
        'time': cycle([30]),
        'partition': cycle(['inf-short']),
        'repeats': 4
    },

    'tp-mvapich2': {
        'exe': cycle(['_LHC_BUP_2017_tp.py']),
        'p': cycle([2000000]),
        'b': cycle([48]),  # 96
        's': cycle([1000]),
        't': cycle([10000]),
        'm': cycle([0]),
        'seed': cycle([0]),
        'reduce': cycle([1]),
        'load': cycle([0.0]),
        'mtw': cycle([50]),
        'approx': cycle([2]),
        'log': cycle([True]),
        'lb': cycle(['reportonly']),
        'lba': cycle([500]),
        'timing': cycle(['-time']),  # otherwise pass -time
        'w': [] +
        [4, 8, 12, 16],
        #+ list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mvapich2']),
        'time': cycle([30]),
        'partition': cycle(['inf-short']),
        'repeats': 4
    },

    'lb-mpich3': {
        'exe': cycle(['_LHC_BUP_2017.py']),
        'p': cycle([2000000]),
        'b': cycle([48]),  # 96
        's': cycle([1000]),
        't': cycle([10000]),
        'm': cycle([0]),
        'seed': cycle([0]),
        'reduce': cycle([1]),
        'load': cycle([0.0]),
        'mtw': cycle([50]),
        'approx': cycle([0]),
        'log': cycle([True]),
        'lb': cycle(['interval']),
        'lba': cycle([500]),
        'timing': cycle(['-time']),  # otherwise pass -time
        'w': [] +
        [4, 8, 12, 16],
        #+ list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mpich3']),
        'time': cycle([30]),
        'partition': cycle(['inf-short']),
        'repeats': 4
    },

    'lb-mvapich2': {
        'exe': cycle(['_LHC_BUP_2017.py']),
        'p': cycle([2000000]),
        'b': cycle([48]),  # 96
        's': cycle([1000]),
        't': cycle([10000]),
        'm': cycle([0]),
        'seed': cycle([0]),
        'reduce': cycle([1]),
        'load': cycle([0.0]),
        'mtw': cycle([50]),
        'approx': cycle([0]),
        'log': cycle([True]),
        'lb': cycle(['interval']),
        'lba': cycle([500]),
        'timing': cycle(['-time']),  # otherwise pass -time
        'w': [] +
        [4, 8, 12, 16],
        #+ list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mvapich2']),
        'time': cycle([30]),
        'partition': cycle(['inf-short']),
        'repeats': 4
    },

    'lb-tp-mpich3': {
        'exe': cycle(['_LHC_BUP_2017_tp.py']),
        'p': cycle([2000000]),
        'b': cycle([2]),  # 96
        's': cycle([1000]),
        't': cycle([10000]),
        'm': cycle([0]),
        'seed': cycle([0]),
        'reduce': cycle([1]),
        'load': cycle([0.0]),
        'mtw': cycle([50]),
        'approx': cycle([2]),
        'log': cycle([True]),
        'lb': cycle(['interval']),
        'lba': cycle([500]),
        'timing': cycle(['-time']),  # otherwise pass -time
        'w': [] +
        [4, 8, 12, 16],
        #+ list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mpich3']),
        'time': cycle([30]),
        'partition': cycle(['inf-short']),
        'repeats': 4
    },

    'lb-tp-mvapich2': {
        'exe': cycle(['_LHC_BUP_2017_tp.py']),
        'p': cycle([2000000]),
        'b': cycle([2]),  # 96
        's': cycle([1000]),
        't': cycle([10000]),
        'm': cycle([0]),
        'seed': cycle([0]),
        'reduce': cycle([1]),
        'load': cycle([0.0]),
        'mtw': cycle([50]),
        'approx': cycle([2]),
        'log': cycle([True]),
        'lb': cycle(['interval']),
        'lba': cycle([500]),
        'timing': cycle(['-time']),  # otherwise pass -time
        'w': [] +
        [4, 8, 12, 16],
        #+ list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mvapich2']),
        'time': cycle([30]),
        'partition': cycle(['inf-short']),
        'repeats': 4
    },

    'openmpi3': {
        'exe': cycle(['_LHC_BUP_2017.py']),
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
        'lb': cycle(['off']),
        'lba': cycle([10]),
        'timing': cycle(['-time']),  # otherwise pass -time
        'w': [] +
        [4, 8, 12, 16],
        #+ list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['openmpi3']),
        'time': cycle([30]),
        'partition': cycle(['inf-short']),
        'repeats': 5
    },


    'mpich3-approx2': {
        'exe': cycle(['_LHC_BUP_2017.py']),
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
        'lb': cycle(['off']),
        'lba': cycle([10]),
        'timing': cycle(['-time']),  # otherwise pass -time
        'w': [] +
        [4, 8, 12, 16],
        #+ list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mpich3']),
        'time': cycle([30]),
        'partition': cycle(['inf-short']),
        'repeats': 5
    },

    'openmpi3-approx2': {
        'exe': cycle(['_LHC_BUP_2017.py']),
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
        'lb': cycle(['off']),
        'lba': cycle([10]),
        'timing': cycle(['-time']),  # otherwise pass -time
        'w': [] +
        [4, 8, 12, 16],
        #+ list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['openmpi3']),
        'time': cycle([30]),
        'partition': cycle(['inf-short']),
        'repeats': 5
    },

    'mvapich2-approx2': {
        'exe': cycle(['_LHC_BUP_2017.py']),
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
        'lb': cycle(['off']),
        'lba': cycle([10]),
        'timing': cycle(['-time']),  # otherwise pass -time
        'w': [] +
        [4, 8, 12, 16],
        #+ list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mvapich2']),
        'time': cycle([30]),
        'partition': cycle(['inf-short']),
        'repeats': 5
    },

    'lb-mpich3-approx2': {
        'exe': cycle(['_LHC_BUP_2017_tp.py']),
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
        'lb': cycle(['interval']),
        'lba': cycle([1000]),
        'timing': cycle(['-time']),  # otherwise pass -time
        'w': []
        + [4, 8, 12, 16],
        #        + list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mpich3']),
        'time': cycle([30]),
        'partition': cycle(['inf-short']),
        'repeats': 5
    },


    'lb-openmpi3-approx2': {
        'exe': cycle(['_LHC_BUP_2017_tp.py']),
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
        'lb': cycle(['interval']),
        'lba': cycle([1000]),
        'timing': cycle(['-time']),  # otherwise pass -time
        'w': [] +
        [4, 8, 12, 16],
        #+ list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['openmpi3']),
        'time': cycle([30]),
        'partition': cycle(['inf-short']),
        'repeats': 5
    },

    'lb-mvapich2-approx2': {
        'exe': cycle(['_LHC_BUP_2017_tp.py']),
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
        'lb': cycle(['interval']),
        'lba': cycle([1000]),
        'timing': cycle(['-time']),  # otherwise pass -time
        'w': [] +
        [4, 8, 12, 16],
        #+ list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mvapich2']),
        'time': cycle([30]),
        'partition': cycle(['inf-short']),
        'repeats': 5
    },


    'lb-mpich3-intv100': {
        'exe': cycle(['_LHC_BUP_2017_tp.py']),
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
        'lb': cycle(['interval']),
        'lba': cycle([100]),
        'timing': cycle(['-time']),  # otherwise pass -time
        'w': [] +
        [4, 8, 12, 16],
        #+ list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mpich3']),
        'time': cycle([30]),
        'partition': cycle(['inf-short']),
        'repeats': 5
    },


    'lb-openmpi3': {
        'exe': cycle(['_LHC_BUP_2017_tp.py']),
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
        'lb': cycle(['interval']),
        'lba': cycle([1000]),
        'timing': cycle(['-time']),  # otherwise pass -time
        'w': [] +
        [4, 8, 12, 16],
        #+ list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['openmpi3']),
        'time': cycle([30]),
        'partition': cycle(['inf-short']),
        'repeats': 5
    },

    'lb-mvapich2-intv100': {
        'exe': cycle(['_LHC_BUP_2017_tp.py']),
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
        'lb': cycle(['interval']),
        'lba': cycle([100]),
        'timing': cycle(['-time']),  # otherwise pass -time
        'w': [] +
        [4, 8, 12, 16],
        #+ list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mvapich2']),
        'time': cycle([30]),
        'partition': cycle(['inf-short']),
        'repeats': 5
    },

    'dynamic-lb-mpich3': {
        'exe': cycle(['_LHC_BUP_2017_tp.py']),
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
        'w': [] +
        [4, 8, 12, 16],
        #+ list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mpich3']),
        'time': cycle([30]),
        'partition': cycle(['inf-short']),
        'repeats': 5
    },

    'dynamic-lb-openmpi3': {
        'exe': cycle(['_LHC_BUP_2017_tp.py']),
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
        'w': [] +
        [4, 8, 12, 16],
        #+ list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['openmpi3']),
        'time': cycle([30]),
        'partition': cycle(['inf-short']),
        'repeats': 5
    },

    'dynamic-lb-mvapich2': {
        'exe': cycle(['_LHC_BUP_2017_tp.py']),
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
        'w': [] +
        [4, 8, 12, 16],
        #+ list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mvapich2']),
        'time': cycle([30]),
        'partition': cycle(['inf-short']),
        'repeats': 5
    },


    'dynamic-lb-mpich3-approx2': {
        'exe': cycle(['_LHC_BUP_2017_tp.py']),
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
        'w': [] +
        [4, 8, 12, 16],
        #+ list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mpich3']),
        'time': cycle([30]),
        'partition': cycle(['inf-short']),
        'repeats': 5
    },

    'dynamic-lb-openmpi3-approx2': {
        'exe': cycle(['_LHC_BUP_2017_tp.py']),
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
        'w': [] +
        [4, 8, 12, 16],
        #+ list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['openmpi3']),
        'time': cycle([30]),
        'partition': cycle(['inf-short']),
        'repeats': 5
    },

    'dynamic-lb-mvapich2-approx2': {
        'exe': cycle(['_LHC_BUP_2017_tp.py']),
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
        'w': [] +
        [4, 8, 12, 16],
        #+ list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mvapich2']),
        'time': cycle([30]),
        'partition': cycle(['inf-short']),
        'repeats': 5
    },


    'weak-scale-mpich3': {
        'exe': cycle(['_LHC_BUP_2017.py']),
        # 'p': cycle([2000000]),
        # 'p': [1e6, 2e6, 2e6, 2e6, 2e6, 1.25e6, 1.5e6, 1.75e6, 2e6],
        'p': [2e6, 2e6, 2e6, 1.25e6, 1.5e6, 1.75e6, 2e6],
        # 'b': cycle([96]),  # 96
        # 'b': [12, 12, 24, 36, 48, 96, 96, 96, 96],  # 96
        'b': [24, 36, 48, 96, 96, 96, 96],  # 96
        's': cycle([1000]),
        't': cycle([10000]),
        'm': cycle([0]),
        'seed': cycle([0]),
        'reduce': cycle([1]),
        'load': cycle([0.0]),
        'mtw': cycle([50]),
        'approx': cycle([0]),
        'timing': cycle(['-time']),  # otherwise pass -time
        'w': [] +
        [4, 6, 8, 10, 12, 14, 16],
        # + list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mpich3']),
        'time': cycle([90]),
        'partition': cycle(['inf-short']),
        'repeats': 5
    },

    'weak-scale-openmpi3': {
        'exe': cycle(['_LHC_BUP_2017.py']),
        # 'p': cycle([2000000]),
        # 'p': [1e6, 2e6, 2e6, 2e6, 2e6, 1.25e6, 1.5e6, 1.75e6, 2e6],
        'p': [1.75e6],
        # 'b': cycle([96]),  # 96
        # 'b': [12, 12, 24, 36, 48, 96, 96, 96, 96],  # 96
        'b': [96],  # 96
        's': cycle([1000]),
        't': cycle([10000]),
        'm': cycle([0]),
        'seed': cycle([0]),
        'reduce': cycle([1]),
        'load': cycle([0.0]),
        'mtw': cycle([50]),
        'approx': cycle([0]),
        'timing': cycle(['-time']),  # otherwise pass -time
        'w': [] +
        [14],
        # + list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['openmpi3']),
        'time': cycle([90]),
        'partition': cycle(['inf-long']),
        'repeats': 5
    },

    'weak-scale-mvapich2': {
        'exe': cycle(['_LHC_BUP_2017.py']),
        # 'p': cycle([2000000]),
        # 'p': [1e6, 2e6, 2e6, 2e6, 2e6, 1.25e6, 1.5e6, 1.75e6, 2e6],
        'p': [1.25e6, 1.75e6],
        # 'b': cycle([96]),  # 96
        # 'b': [12, 12, 24, 36, 48, 96, 96, 96, 96],  # 96
        'b': [96, 96],  # 96
        's': cycle([1000]),
        't': cycle([10000]),
        'm': cycle([0]),
        'seed': cycle([0]),
        'reduce': cycle([1]),
        'load': cycle([0.0]),
        'mtw': cycle([50]),
        'approx': cycle([0]),
        'timing': cycle(['-time']),  # otherwise pass -time
        'w': [] +
        [10, 14],
        # + list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mvapich2']),
        'time': cycle([90]),
        'partition': cycle(['inf-long']),
        'repeats': 5
    },


}
