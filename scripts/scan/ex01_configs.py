from cycler import cycle
import numpy as np

case = 'EX01'

run_configs = [
    'mpich3-approx2',
    'mpich3',
    'lb-mpich3',
    'lb-mpich3-approx2',
]

configs = {

    'mpich3-approx2': {
        'exe': cycle(['EX_01_Acceleration.py']),
        'p': cycle([20000000]),
        'b': cycle([1]),  # 96
        's': cycle([4000]),
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
        # list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mpich3']),
        'time': cycle([30]),
        'partition': cycle(['be-short']),
        'repeats': 5
    },



    'mpich3': {
        'exe': cycle(['EX_01_Acceleration.py']),
        'p': cycle([20000000]),
        'b': cycle([1]),  # 96
        's': cycle([4000]),
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
        # list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mpich3']),
        'time': cycle([30]),
        'partition': cycle(['be-short']),
        'repeats': 5
    },


    'lb-mpich3': {
        'exe': cycle(['EX_01_Acceleration_tp.py']),
        'p': cycle([20000000]),
        'b': cycle([1]),  # 96
        's': cycle([4000]),
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
        # list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mpich3']),
        'time': cycle([30]),
        'partition': cycle(['be-short']),
        'repeats': 5
    },

    'lb-mpich3-approx2': {
        'exe': cycle(['EX_01_Acceleration_tp.py']),
        'p': cycle([20000000]),
        'b': cycle([1]),  # 96
        's': cycle([4000]),
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
        # list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mpich3']),
        'time': cycle([30]),
        'partition': cycle(['be-short']),
        'repeats': 5
    },
}
