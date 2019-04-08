from cycler import cycle
import numpy as np

configs = {

    # 'EX01-mpich3-approx2': {
    #     'exe': cycle(['EX_01_Acceleration.py']),
    #     'p': cycle([8000000]),
    #     'b': cycle([1]),  # 96
    #     's': cycle([1000]),
    #     't': cycle([10000]),
    #     'm': cycle([0]),
    #     'seed': cycle([0]),
    #     'reduce': cycle([1]),
    #     'load': cycle([0.0]),
    #     'mtw': cycle([50]),
    #     'approx': cycle([2]),
    #     'log': cycle([True]),
    #     'lb': cycle(['off']),
    #     'lba': cycle([0]),
    #     'timing': cycle(['-time']),  # otherwise pass -time
    #     'w': [] +
    #     list(np.arange(2, 17, 2)),
    #     'o': cycle([10]),
    #     'mpi': cycle(['mpich3']),
    #     'time': cycle([90]),
    #     'partition': cycle(['be-long']),
        # 'repeats': 5
    # },

    'EX01-lb-mpich3': {
        'exe': cycle(['EX_01_Acceleration_lb.py']),
        'p': cycle([8000000]),
        'b': cycle([1]),  # 96
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
        list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mpich3']),
        'time': cycle([10]),
        'partition': cycle(['be-short']),
        'repeats': 5
    },

    'EX01-lb-mpich3-approx2': {
        'exe': cycle(['EX_01_Acceleration_lb.py']),
        'p': cycle([8000000]),
        'b': cycle([1]),  # 96
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
        list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mpich3']),
        'time': cycle([10]),
        'partition': cycle(['be-short']),
        'repeats': 5
    },
}
