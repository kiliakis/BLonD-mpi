from cycler import cycle
import numpy as np

case = 'PS'

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
        'exe': cycle(['PS_main.py']),
        'p': cycle([8000000]),
        'b': cycle([21]),  # 21
        's': cycle([128]),
        # 't': cycle([500]),
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
        # + list(np.arange(2, 17, 2)),
        # 'o': cycle([1]),
        'o': cycle([10]),
        'mpi': cycle(['mpich3']),
        'time': cycle([45]),
        'partition': cycle(['inf-short']),
        'repeats': 3
    },

    'mvapich2': {
        'exe': cycle(['PS_main.py']),
        'p': cycle([4000000]),
        'b': cycle([21]),  # 21
        's': cycle([128]),
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
        # + list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mvapich2']),
        'time': cycle([45]),
        'partition': cycle(['inf-short']),
        'repeats': 4
    },


    'tp-mpich3': {
        'exe': cycle(['PS_main_tp.py']),
        'p': cycle([4000000]),
        'b': cycle([21]),  # 21
        's': cycle([128]),
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
        # + list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mpich3']),
        'time': cycle([45]),
        'partition': cycle(['inf-short']),
        'repeats': 4
    },

    'tp-mvapich2': {
        'exe': cycle(['PS_main_tp.py']),
        'p': cycle([4000000]),
        'b': cycle([21]),  # 21
        's': cycle([128]),
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
        # + list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mvapich2']),
        'time': cycle([45]),
        'partition': cycle(['inf-short']),
        'repeats': 4
    },

    'lb-mpich3': {
        'exe': cycle(['PS_main.py']),
        'p': cycle([4000000]),
        'b': cycle([21]),  # 21
        's': cycle([128]),
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
        # + list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mpich3']),
        'time': cycle([45]),
        'partition': cycle(['inf-short']),
        'repeats': 4
    },

    'lb-mvapich2': {
        'exe': cycle(['PS_main.py']),
        'p': cycle([4000000]),
        'b': cycle([21]),  # 21
        's': cycle([128]),
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
        # + list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mvapich2']),
        'time': cycle([45]),
        'partition': cycle(['inf-short']),
        'repeats': 4
    },

    'lb-tp-mpich3': {
        'exe': cycle(['PS_main_tp.py']),
        'p': cycle([4000000]),
        'b': cycle([21]),  # 21
        's': cycle([128]),
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
        # + list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mpich3']),
        'time': cycle([45]),
        'partition': cycle(['inf-short']),
        'repeats': 4
    },

    'lb-tp-mvapich2': {
        'exe': cycle(['PS_main_tp.py']),
        'p': cycle([4000000]),
        'b': cycle([21]),  # 21
        's': cycle([128]),
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
        # + list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mvapich2']),
        'time': cycle([45]),
        'partition': cycle(['inf-short']),
        'repeats': 4
    },


    'lb-mpich3-intv100': {
        'exe': cycle(['PS_main_tp.py']),
        'p': cycle([8000000]),
        'b': cycle([21]),  # 21
        's': cycle([128]),
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
        # + list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mpich3']),
        'time': cycle([45]),
        'partition': cycle(['inf-short']),
        'repeats': 5
    },


    'lb-mvapich2-intv100': {
        'exe': cycle(['PS_main_tp.py']),
        'p': cycle([8000000]),
        'b': cycle([21]),  # 21
        's': cycle([128]),
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
        'time': cycle([45]),
        'partition': cycle(['inf-short']),
        'repeats': 5
    },

    'lb-mpich3-approx2': {
        'exe': cycle(['PS_main_tp.py']),
        'p': cycle([8000000]),
        'b': cycle([21]),  # 21
        's': cycle([128]),
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
        # + list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mpich3']),
        'time': cycle([45]),
        'partition': cycle(['inf-short']),
        'repeats': 5
    },


    'lb-openmpi3-approx2': {
        'exe': cycle(['PS_main_tp.py']),
        'p': cycle([8000000]),
        'b': cycle([21]),  # 21
        's': cycle([128]),
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
        'time': cycle([45]),
        'partition': cycle(['inf-short']),
        'repeats': 5
    },


    'lb-mvapich2-approx2': {
        'exe': cycle(['PS_main_tp.py']),
        'p': cycle([8000000]),
        'b': cycle([21]),  # 21
        's': cycle([128]),
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
        'time': cycle([45]),
        'partition': cycle(['inf-short']),
        'repeats': 5
    },

    'dynamic-lb-mpich3': {
        'exe': cycle(['PS_main_tp.py']),
        'p': cycle([8000000]),
        'b': cycle([21]),  # 21
        's': cycle([128]),
        't': cycle([10000]),
        'm': cycle([0]),
        'seed': cycle([0]),
        'reduce': cycle([1]),
        'load': cycle([0.0]),
        'mtw': cycle([50]),
        'approx': cycle([0]),
        'log': cycle([True]),
        'lb': cycle(['dynamic']),
        'lba': cycle([100]),
        'timing': cycle(['-time']),  # otherwise pass -time
        'w': [] +
        [4, 8, 12, 16],
        # + list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mpich3']),
        'time': cycle([45]),
        'partition': cycle(['inf-short']),
        'repeats': 5
    },


    'dynamic-lb-openmpi3': {
        'exe': cycle(['PS_main_tp.py']),
        'p': cycle([8000000]),
        'b': cycle([21]),  # 21
        's': cycle([128]),
        't': cycle([10000]),
        'm': cycle([0]),
        'seed': cycle([0]),
        'reduce': cycle([1]),
        'load': cycle([0.0]),
        'mtw': cycle([50]),
        'approx': cycle([0]),
        'log': cycle([True]),
        'lb': cycle(['dynamic']),
        'lba': cycle([100]),
        'timing': cycle(['-time']),  # otherwise pass -time
        'w': [] +
        [4, 8, 12, 16],
        #+ list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['openmpi3']),
        'time': cycle([45]),
        'partition': cycle(['inf-short']),
        'repeats': 5
    },


    'dynamic-lb-mvapich2': {
        'exe': cycle(['PS_main_tp.py']),
        'p': cycle([8000000]),
        'b': cycle([21]),  # 21
        's': cycle([128]),
        't': cycle([10000]),
        'm': cycle([0]),
        'seed': cycle([0]),
        'reduce': cycle([1]),
        'load': cycle([0.0]),
        'mtw': cycle([50]),
        'approx': cycle([0]),
        'log': cycle([True]),
        'lb': cycle(['dynamic']),
        'lba': cycle([100]),
        'timing': cycle(['-time']),  # otherwise pass -time
        'w': [] +
        [4, 8, 12, 16],
        #+ list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mvapich2']),
        'time': cycle([45]),
        'partition': cycle(['inf-short']),
        'repeats': 5
    },

    'dynamic-lb-mpich3-approx2': {
        'exe': cycle(['PS_main_tp.py']),
        'p': cycle([8000000]),
        'b': cycle([21]),  # 21
        's': cycle([128]),
        't': cycle([10000]),
        'm': cycle([0]),
        'seed': cycle([0]),
        'reduce': cycle([1]),
        'load': cycle([0.0]),
        'mtw': cycle([50]),
        'approx': cycle([2]),
        'log': cycle([True]),
        'lb': cycle(['dynamic']),
        'lba': cycle([100]),
        'timing': cycle(['-time']),  # otherwise pass -time
        'w': [] +
        [4, 8, 12, 16],
        # + list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mpich3']),
        'time': cycle([45]),
        'partition': cycle(['inf-short']),
        'repeats': 5
    },


    'dynamic-lb-openmpi3-approx2': {
        'exe': cycle(['PS_main_tp.py']),
        'p': cycle([8000000]),
        'b': cycle([21]),  # 21
        's': cycle([128]),
        't': cycle([10000]),
        'm': cycle([0]),
        'seed': cycle([0]),
        'reduce': cycle([1]),
        'load': cycle([0.0]),
        'mtw': cycle([50]),
        'approx': cycle([2]),
        'log': cycle([True]),
        'lb': cycle(['dynamic']),
        'lba': cycle([100]),
        'timing': cycle(['-time']),  # otherwise pass -time
        'w': [] +
        [4, 8, 12, 16],
        #+ list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['openmpi3']),
        'time': cycle([45]),
        'partition': cycle(['inf-short']),
        'repeats': 5
    },


    'dynamic-lb-mvapich2-approx2': {
        'exe': cycle(['PS_main_tp.py']),
        'p': cycle([8000000]),
        'b': cycle([21]),  # 21
        's': cycle([128]),
        't': cycle([10000]),
        'm': cycle([0]),
        'seed': cycle([0]),
        'reduce': cycle([1]),
        'load': cycle([0.0]),
        'mtw': cycle([50]),
        'approx': cycle([2]),
        'log': cycle([True]),
        'lb': cycle(['dynamic']),
        'lba': cycle([100]),
        'timing': cycle(['-time']),  # otherwise pass -time
        'w': [] +
        [4, 8, 12, 16],
        #+ list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mvapich2']),
        'time': cycle([45]),
        'partition': cycle(['inf-short']),
        'repeats': 5
    },


    'weak-scale-mpich3': {
        'exe': cycle(['PS_main.py']),
        'p': [6e6],
        'b': cycle([21]),  # 21
        's': cycle([128]),
        't': cycle([10000]),
        'm': cycle([0]),
        'seed': cycle([0]),
        'reduce': cycle([1]),
        'load': cycle([0.0]),
        'mtw': cycle([50]),
        'approx': cycle([0]),
        'timing': cycle(['-time']),  # otherwise pass -time
        'w': [] +
        [12],
        # + list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mpich3']),
        'time': cycle([90]),
        'partition': cycle(['inf-short']),
        'repeats': 5
    },

    'PS-weak-scale-mvapich2': {
        'exe': cycle(['PS_main.py']),
        # 'p': [0.5e6, 1e6, 2e6, 3e6, 4e6, 5e6, 6e6, 7e6, 8e6],
        'p': [8e6],
        # 'p': cycle([4000000]),
        'b': cycle([21]),  # 21
        's': cycle([128]),
        't': cycle([10000]),
        'm': cycle([0]),
        'seed': cycle([0]),
        'reduce': cycle([1]),
        'load': cycle([0.0]),
        'mtw': cycle([50]),
        'approx': cycle([0]),
        'timing': cycle(['-time']),  # otherwise pass -time
        'w': [] +
        [16],
        # + list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mvapich2']),
        'time': cycle([90]),
        'partition': cycle(['inf-short']),
        'repeats': 5
    },

    'PS-weak-scale-openmpi3': {
        'exe': cycle(['PS_main.py']),
        # 'p': [0.5e6, 1e6, 2e6, 3e6, 4e6, 5e6, 6e6, 7e6, 8e6],
        'p': [8e6],
        # 'p': cycle([4000000]),
        'b': cycle([21]),  # 21
        's': cycle([128]),
        't': cycle([10000]),
        'm': cycle([0]),
        'seed': cycle([0]),
        'reduce': cycle([1]),
        'load': cycle([0.0]),
        'mtw': cycle([50]),
        'approx': cycle([0]),
        'timing': cycle(['-time']),  # otherwise pass -time
        'w': [] +
        [16],
        # + list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['openmpi3']),
        'time': cycle([90]),
        'partition': cycle(['inf-short']),
        'repeats': 5
    },
}
