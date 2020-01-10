from cycler import cycle
import numpy as np

case = 'LHC'
repeats = 2

run_configs = [
    # Strong scaling
    # 'lb-tp-approx0-mvapich2-strong-scaling',
    # # Weak Scaling
    # 'lb-tp-approx0-mvapich2-weak-scaling',

    # # Intermediate stages
    # 'approx0-mvapich2-interm',
    # 'approx2-mvapich2-interm',
    # 'tp-approx0-mvapich2-interm',
    # 'tp-approx2-mvapich2-interm',
    # 'lb-approx0-mvapich2-interm',
    # 'lb-approx2-mvapich2-interm',
    # 'lb-tp-approx0-mvapich2-interm',
    # 'lb-tp-approx2-mvapich2-interm',
    
    # Various MPI implementations
    'approx0-mvapich2-impl',
    'approx0-mpich3-impl',
    'approx0-openmpi3-impl',

    # Optimal num of workers per node
    'approx0-mvapich2-workers',

    # 'mpich3',
    # 'mvapich2',
    # 'tp-mpich3',
    # 'tp-mvapich2',
    # 'lb-tp-mpich3',
    # 'lb-tp-mvapich2',
    # 'lb-tp-approx0-mvapich2',
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

    'lb-tp-approx0-mvapich2-strong-scaling': {
        'exe': cycle(['_LHC_BUP_2017_tp.py']),
        'p': cycle([2000000]),
        'b': cycle([192]),  # 96
        's': cycle([1000]),
        't': cycle([5000]),
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
        [2, 4, 8, 16, 32, 64],
        # [12],
        #+ list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mvapich2']),
        'time': cycle([180]),
        'partition': cycle(['inf-short']),
        'repeats': repeats
    },

    'lb-tp-approx0-mvapich2-weak-scaling': {
        'exe': cycle(['_LHC_BUP_2017_tp.py']),
        'p': cycle([4000000]),
        'b': [6, 12, 24, 48, 96, 192],  # 96
        's': cycle([1000]),
        't': cycle([5000]),
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
        [2, 4, 8, 16, 32, 64],
        # [12],
        #+ list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mvapich2']),
        'time': cycle([180]),
        'partition': cycle(['inf-short']),
        'repeats': repeats
    },

    'approx0-mvapich2-interm': {
        'exe': cycle(['_LHC_BUP_2017.py']),
        'p': cycle([2000000]),
        'b': cycle([192]),  # 96
        's': cycle([1000]),
        't': cycle([5000]),
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
        [8, 16],
        #+ list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mvapich2']),
        'time': cycle([180]),
        'partition': cycle(['inf-short']),
        'repeats': repeats
    },


    'approx2-mvapich2-interm': {
        'exe': cycle(['_LHC_BUP_2017.py']),
        'p': cycle([2000000]),
        'b': cycle([192]),  # 96
        's': cycle([1000]),
        't': cycle([5000]),
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
        [8, 16],
        #+ list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mvapich2']),
        'time': cycle([180]),
        'partition': cycle(['inf-short']),
        'repeats': repeats
    },

    'lb-approx0-mvapich2-interm': {
        'exe': cycle(['_LHC_BUP_2017.py']),
        'p': cycle([2000000]),
        'b': cycle([192]),  # 96
        's': cycle([1000]),
        't': cycle([5000]),
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
        [8, 16],
        #+ list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mvapich2']),
        'time': cycle([180]),
        'partition': cycle(['inf-short']),
        'repeats': repeats
    },


    'lb-approx2-mvapich2-interm': {
        'exe': cycle(['_LHC_BUP_2017.py']),
        'p': cycle([2000000]),
        'b': cycle([192]),  # 96
        's': cycle([1000]),
        't': cycle([5000]),
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
        [8, 16],
        #+ list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mvapich2']),
        'time': cycle([180]),
        'partition': cycle(['inf-short']),
        'repeats': repeats
    },


    'tp-approx0-mvapich2-interm': {
        'exe': cycle(['_LHC_BUP_2017_tp.py']),
        'p': cycle([2000000]),
        'b': cycle([192]),  # 96
        's': cycle([1000]),
        't': cycle([5000]),
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
        [8, 16],
        #+ list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mvapich2']),
        'time': cycle([180]),
        'partition': cycle(['inf-short']),
        'repeats': repeats
    },


    'tp-approx2-mvapich2-interm': {
        'exe': cycle(['_LHC_BUP_2017_tp.py']),
        'p': cycle([2000000]),
        'b': cycle([192]),  # 96
        's': cycle([1000]),
        't': cycle([5000]),
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
        [8, 16],
        #+ list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mvapich2']),
        'time': cycle([180]),
        'partition': cycle(['inf-short']),
        'repeats': repeats
    },

    'lb-tp-approx0-mvapich2-interm': {
        'exe': cycle(['_LHC_BUP_2017_tp.py']),
        'p': cycle([2000000]),
        'b': cycle([192]),  # 96
        's': cycle([1000]),
        't': cycle([5000]),
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
        [8, 16],
        #+ list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mvapich2']),
        'time': cycle([180]),
        'partition': cycle(['inf-short']),
        'repeats': repeats
    },


    'lb-tp-approx2-mvapich2-interm': {
        'exe': cycle(['_LHC_BUP_2017_tp.py']),
        'p': cycle([2000000]),
        'b': cycle([192]),  # 96
        's': cycle([1000]),
        't': cycle([5000]),
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
        [8, 16],
        #+ list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mvapich2']),
        'time': cycle([180]),
        'partition': cycle(['inf-short']),
        'repeats': repeats
    },

    'approx0-mvapich2-impl': {
        'exe': cycle(['_LHC_BUP_2017.py']),
        'p': cycle([2000000]),
        'b': cycle([96]),  # 96
        's': cycle([1000]),
        't': cycle([5000]),
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
        [8, 16],
        #+ list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mvapich2']),
        'time': cycle([180]),
        'partition': cycle(['inf-short']),
        'repeats': repeats
    },

    'approx0-mpich3-impl': {
        'exe': cycle(['_LHC_BUP_2017.py']),
        'p': cycle([2000000]),
        'b': cycle([96]),  # 96
        's': cycle([1000]),
        't': cycle([5000]),
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
        [8, 16],
        #+ list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mpich3']),
        'time': cycle([180]),
        'partition': cycle(['inf-short']),
        'repeats': repeats
    },

    'approx0-openmpi3-impl': {
        'exe': cycle(['_LHC_BUP_2017.py']),
        'p': cycle([2000000]),
        'b': cycle([96]),  # 96
        's': cycle([1000]),
        't': cycle([5000]),
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
        [8, 16],
        #+ list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['openmpi3']),
        'time': cycle([180]),
        'partition': cycle(['inf-short']),
        'repeats': repeats
    },

    'approx0-mvapich2-workers': {
        'exe': cycle(['_LHC_BUP_2017.py']),
        'p': cycle([2000000]),
        'b': cycle([96]),  # 96
        's': cycle([1000]),
        't': cycle([5000]),
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
        [80, 40, 32, 16, 8],
        #+ list(np.arange(2, 17, 2)),
        'o': [2, 4, 5, 10, 20],
        'mpi': cycle(['mvapich2']),
        'time': cycle([180]),
        'partition': cycle(['inf-short']),
        'repeats': repeats
    },


    'mpich3': {
        'exe': cycle(['_LHC_BUP_2017.py']),
        'p': cycle([2000000]),
        'b': cycle([96]),  # 96
        's': cycle([1000]),
        # 't': cycle([5000]),
        't': cycle([1000]),
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
        # [4, 8, 12, 16],
        [1],
        #+ list(np.arange(2, 17, 2)),
        # 'o': cycle([10]),
        'o': cycle([20]),
        'mpi': cycle(['mpich3']),
        'time': cycle([30]),
        'partition': cycle(['inf-short']),
        'repeats': repeats
    },

    'mvapich2': {
        'exe': cycle(['_LHC_BUP_2017.py']),
        'p': cycle([2000000]),
        'b': cycle([96]),  # 96
        's': cycle([1000]),
        't': cycle([5000]),
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
        'repeats': repeats
    },

    'tp-mpich3': {
        'exe': cycle(['_LHC_BUP_2017_tp.py']),
        'p': cycle([2000000]),
        'b': cycle([96]),  # 96
        's': cycle([1000]),
        't': cycle([5000]),
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
        'repeats': repeats
    },

    'tp-mvapich2': {
        'exe': cycle(['_LHC_BUP_2017_tp.py']),
        'p': cycle([2000000]),
        'b': cycle([96]),  # 96
        's': cycle([1000]),
        't': cycle([5000]),
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
        [8, 16],
        #+ list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mvapich2']),
        'time': cycle([30]),
        'partition': cycle(['inf-short']),
        'repeats': repeats
    },

    'lb-mpich3': {
        'exe': cycle(['_LHC_BUP_2017.py']),
        'p': cycle([2000000]),
        'b': cycle([96]),  # 96
        's': cycle([1000]),
        't': cycle([5000]),
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
        # [4, 8],
        [4, 8, 12, 16],
        #+ list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mpich3']),
        'time': cycle([30]),
        'partition': cycle(['inf-short']),
        'repeats': repeats
    },

    'lb-mvapich2': {
        'exe': cycle(['_LHC_BUP_2017.py']),
        'p': cycle([2000000]),
        'b': cycle([96]),  # 96
        's': cycle([1000]),
        't': cycle([5000]),
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
        # [4, 8],
        [12],
        #+ list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mvapich2']),
        'time': cycle([30]),
        'partition': cycle(['inf-short']),
        'repeats': repeats
    },

    'lb-tp-mpich3': {
        'exe': cycle(['_LHC_BUP_2017_tp.py']),
        'p': cycle([2000000]),
        'b': cycle([96]),  # 96
        's': cycle([1000]),
        't': cycle([5000]),
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
        # [4, 8],
        [12, 16],
        #+ list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mpich3']),
        'time': cycle([30]),
        'partition': cycle(['inf-short']),
        'repeats': repeats
    },

    'lb-tp-mvapich2': {
        'exe': cycle(['_LHC_BUP_2017_tp.py']),
        'p': cycle([2000000]),
        'b': cycle([96]),  # 96
        's': cycle([1000]),
        't': cycle([5000]),
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
        [8, 12, 16],
        # [12],
        #+ list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mvapich2']),
        'time': cycle([30]),
        'partition': cycle(['inf-short']),
        'repeats': repeats
    },

    'lb-tp-approx0-mvapich2': {
        'exe': cycle(['_LHC_BUP_2017_tp.py']),
        'p': cycle([2000000]),
        'b': cycle([96]),  # 96
        's': cycle([1000]),
        't': cycle([5000]),
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
        # [12],
        #+ list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mvapich2']),
        'time': cycle([30]),
        'partition': cycle(['inf-short']),
        'repeats': repeats
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
        'repeats': repeats
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
        'repeats': repeats
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
        'repeats': repeats
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
        'repeats': repeats
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
        'repeats': repeats
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
        'repeats': repeats
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
        'repeats': repeats
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
        'repeats': repeats
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
        'repeats': repeats
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
        'repeats': repeats
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
        'repeats': repeats
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
        'repeats': repeats
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
        'repeats': repeats
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
        'repeats': repeats
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
        'repeats': repeats
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
        'repeats': repeats
    },


    'weak-scale-mpich3': {
        'exe': cycle(['_LHC_BUP_2017.py']),
        # 'p': cycle([2000000]),
        # 'p': [1e6, 2e6, 2e6, 2e6, 2e6, 1.25e6, 1.5e6, 1.75e6, 2e6],
        'p': [2e6, 2e6, 2e6, 1.25e6, 1.5e6, 1.75e6, 2e6],
        # 'b': cycle([96]),  # 96
        # 'b': [12, 12, 24, 36, 96, 96, 96, 96, 96],  # 96
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
        'repeats': repeats
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
        'repeats': repeats
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
        'repeats': repeats
    },


}
