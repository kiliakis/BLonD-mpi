from cycler import cycle
import numpy as np

configs = {

    # 'PS-21B-approx2': {
    #     'exe': cycle(['PS_main.py']),
    #     'p': cycle([4000000]),
    #     'b': cycle([21]),  # 21
    #     's': cycle([128]),
    #     't': cycle([200000]), #378708
    #     'm': cycle([100]),
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
    #     'partition': cycle(['be-short']),
          # 'repeats': 5
    # },



    # 'PS-21B-approx1': {
    #     'exe': cycle([home + '/__EXAMPLES/main_files/PS_main.py']),
    #     'p': cycle([4000000]),
    #     'b': cycle([21]),  # 21
    #     's': cycle([128]),
    #     't': cycle([200000]), #378708
    #     'm': cycle([100]),
    #     'load': cycle([0.0]),
    #     'mtw': cycle([50]),
    #     'approx': cycle([1]),
    #     'timing': cycle(['']),  # otherwise pass -time
    #     # 'seed': [0] * 3 + [1] * 3 + [2] * 3,
    #     'seed': [2, 0],
    #     'reduce': [1, 3],
    #     # + [1, 2, 3]
    #     # + [1, 2, 3]
    #     # + [1, 2, 3],
    #     # 'w': [16] * 9,
    #     'w': [16] * 2,
    #     'o': cycle([10]),
    #     'time': cycle([480]),
    #     'mpi':cycle('mpich3'),
    #     'partition': cycle(['be-short']),
          # 'repeats': 5
    # }


    # 'PS-sync-mpich3': {
    #     'exe': cycle(['PS_main_test.py']),
    #     'p': cycle([4000000]),
    #     'b': cycle([21]), # 21
    #     's': cycle([128]),
    #     't': cycle([10000]),
    #     'm': cycle([0]),
    #     'seed': cycle([0]),
    #     'reduce': cycle([1]),
    #     'load': cycle([0.0]),
    #     'mtw': cycle([50]),
    #     'approx': cycle([0]),
    #     'timing': cycle(['-time']), # otherwise pass -time
    #     'w': []
    #     + list(np.arange(2, 17, 2)),
    #     'o': cycle([10]),
    #     'mpi': cycle(['mpich3']),
    #     'time': cycle([180]),
    #     'partition': cycle(['be-long']),
          # 'repeats': 5
    # },


    # 'PS-sync-mvapich2': {
    #     'exe': cycle(['PS_main_test.py']),
    #     'p': cycle([4000000]),
    #     'b': cycle([21]), # 21
    #     's': cycle([128]),
    #     't': cycle([10000]),
    #     'm': cycle([0]),
    #     'seed': cycle([0]),
    #     'reduce': cycle([1]),
    #     'load': cycle([0.0]),
    #     'mtw': cycle([50]),
    #     'approx': cycle([0]),
    #     'timing': cycle(['-time']), # otherwise pass -time
    #     'w': []
    #     + list(np.arange(2, 17, 2)),
    #     'o': cycle([10]),
    #     'mpi': cycle(['mvapich2']),
    #     'time': cycle([180]),
    #     'partition': cycle(['be-short']),
          # 'repeats': 5
    # },


    # 'PS-sync-openmpi3': {
    #     'exe': cycle(['PS_main_test.py']),
    #     'p': cycle([4000000]),
    #     'b': cycle([21]), # 21
    #     's': cycle([128]),
    #     't': cycle([10000]),
    #     'm': cycle([0]),
    #     'seed': cycle([0]),
    #     'reduce': cycle([1]),
    #     'load': cycle([0.0]),
    #     'mtw': cycle([50]),
    #     'approx': cycle([0]),
    #     'timing': cycle(['-time']), # otherwise pass -time
    #     'w': []
    #     + list(np.arange(2, 17, 2)),
    #     'o': cycle([10]),
    #     'mpi': cycle(['openmpi3']),
    #     'time': cycle([180]),
    #     'partition': cycle(['be-short']),
          # 'repeats': 5
    # },


    # 'PS-approx2-mpich3': {
    #     'exe': cycle(['PS_main.py']),
    #     'p': cycle([4000000]),
    #     'b': cycle([21]), # 21
    #     's': cycle([128]),
    #     't': cycle([10000]),
    #     'm': cycle([0]),
    #     'seed': cycle([0]),
    #     'reduce': cycle([1]),
    #     'load': cycle([0.0]),
    #     'mtw': cycle([50]),
    #     'approx': cycle([2]),
    #     'timing': cycle(['-time']), # otherwise pass -time
    #     'w': []
    #     # + [4, 14],
    #     + list(np.arange(2, 17, 2)),
    #     'o': cycle([10]),
    #     'mpi': cycle(['mpich3']),
    #     'time': cycle([180]),
    #     'partition': cycle(['be-long']),
          # 'repeats': 5
    # },


    # 'PS-approx2-openmpi3': {
    #     'exe': cycle(['PS_main.py']),
    #     'p': cycle([4000000]),
    #     'b': cycle([21]), # 21
    #     's': cycle([128]),
    #     't': cycle([10000]),
    #     'm': cycle([0]),
    #     'seed': cycle([0]),
    #     'reduce': cycle([1]),
    #     'load': cycle([0.0]),
    #     'mtw': cycle([50]),
    #     'approx': cycle([2]),
    #     'timing': cycle(['-time']), # otherwise pass -time
    #     'w': []
    #     # + [14],
    #     + list(np.arange(2, 17, 2)),
    #     'o': cycle([10]),
    #     'mpi': cycle(['openmpi3']),
    #     'time': cycle([180]),
    #     'partition': cycle(['be-long']),
          # 'repeats': 5
    # },


    # 'PS-approx2-mvapich2': {
    #     'exe': cycle(['PS_main.py']),
    #     'p': cycle([4000000]),
    #     'b': cycle([21]), # 21
    #     's': cycle([128]),
    #     't': cycle([10000]),
    #     'm': cycle([0]),
    #     'seed': cycle([0]),
    #     'reduce': cycle([1]),
    #     'load': cycle([0.0]),
    #     'mtw': cycle([50]),
    #     'approx': cycle([2]),
    #     'timing': cycle(['-time']), # otherwise pass -time
    #     'w': []
    #     # + [12, 16],
    #     + list(np.arange(2, 17, 2)),
    #     'o': cycle([10]),
    #     'mpi': cycle(['mvapich2']),
    #     'time': cycle([180]),
    #     'partition': cycle(['be-short']),
          # 'repeats': 5
    # },

    'PS-lb-mpich3': {
        'exe': cycle(['PS_main_lb.py']),
        'p': cycle([4000000]),
        'b': cycle([21]), # 21
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
        'lba': cycle([10]),
        'timing': cycle(['-time']), # otherwise pass -time
        'w': []
        + [2, 4, 8, 12, 16],
        # + list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mpich3']),
        'time': cycle([30]),
        'partition': cycle(['be-short']),
        'repeats': 5
    },


    # 'PS-lb-openmpi3': {
    #     'exe': cycle(['PS_main_lb.py']),
    #     'p': cycle([4000000]),
    #     'b': cycle([21]), # 21
    #     's': cycle([128]),
    #     't': cycle([10000]),
    #     'm': cycle([0]),
    #     'seed': cycle([0]),
    #     'reduce': cycle([1]),
    #     'load': cycle([0.0]),
    #     'mtw': cycle([50]),
    #     'approx': cycle([0]),
    #     'log': cycle([True]),
    #     'lb': cycle(['dynamic']),
    #     'lba': cycle([10]),
    #     'timing': cycle(['-time']), # otherwise pass -time
    #     'w': []
    #     + list(np.arange(2, 17, 2)),
    #     'o': cycle([10]),
    #     'mpi': cycle(['openmpi3']),
    #     'time': cycle([30]),
    #     'partition': cycle(['be-short']),
    #     'repeats': 5
    # },


    # 'PS-lb-mvapich2': {
    #     'exe': cycle(['PS_main_lb.py']),
    #     'p': cycle([4000000]),
    #     'b': cycle([21]), # 21
    #     's': cycle([128]),
    #     't': cycle([10000]),
    #     'm': cycle([0]),
    #     'seed': cycle([0]),
    #     'reduce': cycle([1]),
    #     'load': cycle([0.0]),
    #     'mtw': cycle([50]),
    #     'approx': cycle([0]),
    #     'log': cycle([True]),
    #     'lb': cycle(['dynamic']),
    #     'lba': cycle([10]),
    #     'timing': cycle(['-time']), # otherwise pass -time
    #     'w': []
    #     + list(np.arange(2, 17, 2)),
    #     'o': cycle([10]),
    #     'mpi': cycle(['mvapich2']),
    #     'time': cycle([30]),
    #     'partition': cycle(['be-long']),
    #     'repeats': 5
    # },

    'PS-lb-mpich3-approx2': {
        'exe': cycle(['PS_main_lb.py']),
        'p': cycle([4000000]),
        'b': cycle([21]), # 21
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
        'lba': cycle([10]),
        'timing': cycle(['-time']), # otherwise pass -time
        'w': []
        + [2, 4, 8, 12, 16],
        # + list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mpich3']),
        'time': cycle([30]),
        'partition': cycle(['be-short']),
        'repeats': 5
    },


    # 'PS-lb-openmpi3-approx2': {
    #     'exe': cycle(['PS_main_lb.py']),
    #     'p': cycle([4000000]),
    #     'b': cycle([21]), # 21
    #     's': cycle([128]),
    #     't': cycle([10000]),
    #     'm': cycle([0]),
    #     'seed': cycle([0]),
    #     'reduce': cycle([1]),
    #     'load': cycle([0.0]),
    #     'mtw': cycle([50]),
    #     'approx': cycle([2]),
    #     'log': cycle([True]),
    #     'lb': cycle(['dynamic']),
    #     'lba': cycle([10]),
    #     'timing': cycle(['-time']), # otherwise pass -time
    #     'w': [],
    #     # + [14],
    #     #+ list(np.arange(2, 17, 2)),
    #     'o': cycle([10]),
    #     'mpi': cycle(['openmpi3']),
    #     'time': cycle([30]),
    #     'partition': cycle(['be-short']),
    #           # 'repeats': 5
    # },


    # 'PS-lb-mvapich2-approx2': {
    #     'exe': cycle(['PS_main_lb.py']),
    #     'p': cycle([4000000]),
    #     'b': cycle([21]), # 21
    #     's': cycle([128]),
    #     't': cycle([10000]),
    #     'm': cycle([0]),
    #     'seed': cycle([0]),
    #     'reduce': cycle([1]),
    #     'load': cycle([0.0]),
    #     'mtw': cycle([50]),
    #     'approx': cycle([2]),
    #     'log': cycle([True]),
    #     'lb': cycle(['dynamic']),
    #     'lba': cycle([10]),
    #     'timing': cycle(['-time']), # otherwise pass -time
    #     'w': []
    #     + list(np.arange(2, 17, 2)),
    #     'o': cycle([10]),
    #     'mpi': cycle(['mvapich2']),
    #     'time': cycle([30]),
    #     'partition': cycle(['be-short']),
    #     'repeats': 5
    # },

    # 'PS-weak-scale-mpich3': {
    #     'exe': cycle(['PS_main.py']),
    #     # 'p': [0.5e6, 1e6, 2e6, 3e6, 4e6, 5e6, 6e6, 7e6, 8e6],
    #     'p': [6e6],
    #     # 'p': cycle([4000000]),
    #     'b': cycle([21]), # 21
    #     's': cycle([128]),
    #     't': cycle([10000]),
    #     'm': cycle([0]),
    #     'seed': cycle([0]),
    #     'reduce': cycle([1]),
    #     'load': cycle([0.0]),
    #     'mtw': cycle([50]),
    #     'approx': cycle([0]),
    #     'timing': cycle(['-time']), # otherwise pass -time
    #     'w': []
    #     + [12],
    #     # + list(np.arange(2, 17, 2)),
    #     'o': cycle([10]),
    #     'mpi': cycle(['mpich3']),
    #     'time': cycle([90]),
    #     'partition': cycle(['be-short']),
          # 'repeats': 5
    # },

    # 'PS-weak-scale-mvapich2': {
    #     'exe': cycle(['PS_main.py']),
    #     # 'p': [0.5e6, 1e6, 2e6, 3e6, 4e6, 5e6, 6e6, 7e6, 8e6],
    #     'p': [8e6],
    #     # 'p': cycle([4000000]),
    #     'b': cycle([21]), # 21
    #     's': cycle([128]),
    #     't': cycle([10000]),
    #     'm': cycle([0]),
    #     'seed': cycle([0]),
    #     'reduce': cycle([1]),
    #     'load': cycle([0.0]),
    #     'mtw': cycle([50]),
    #     'approx': cycle([0]),
    #     'timing': cycle(['-time']), # otherwise pass -time
    #     'w': []
    #     + [16],
    #     # + list(np.arange(2, 17, 2)),
    #     'o': cycle([10]),
    #     'mpi': cycle(['mvapich2']),
    #     'time': cycle([90]),
    #     'partition': cycle(['be-short']),
          # 'repeats': 5
    # },

    # 'PS-weak-scale-openmpi3': {
    #     'exe': cycle(['PS_main.py']),
    #     # 'p': [0.5e6, 1e6, 2e6, 3e6, 4e6, 5e6, 6e6, 7e6, 8e6],
    #     'p': [8e6],
    #     # 'p': cycle([4000000]),
    #     'b': cycle([21]), # 21
    #     's': cycle([128]),
    #     't': cycle([10000]),
    #     'm': cycle([0]),
    #     'seed': cycle([0]),
    #     'reduce': cycle([1]),
    #     'load': cycle([0.0]),
    #     'mtw': cycle([50]),
    #     'approx': cycle([0]),
    #     'timing': cycle(['-time']), # otherwise pass -time
    #     'w': []
    #     + [16],
    #     # + list(np.arange(2, 17, 2)),
    #     'o': cycle([10]),
    #     'mpi': cycle(['openmpi3']),
    #     'time': cycle([90]),
    #     'partition': cycle(['be-short']),
          # 'repeats': 5
    # },

    # 'PS-b21-t10k-mpich3': {
    #     'exe': cycle(['PS_main.py']),
    #     'p': cycle([4000000]),
    #     'b': cycle([21]), # 21
    #     's': cycle([128]),
    #     't': cycle([10000]),
    #     'm': cycle([0]),
    #     'seed': cycle([0]),
    #     'reduce': cycle([1]),
    #     'load': cycle([0.0]),
    #     'mtw': cycle([50]),
    #     'approx': cycle([0]),
    #     'timing': cycle(['-time']), # otherwise pass -time
    #     'w': []
    #     + [14],
    #     # + list(np.arange(2, 17, 2)),
    #     'o': cycle([10]),
    #     'mpi': cycle(['mpich3']),
    #     'time': cycle([90]),
    #     'partition': cycle(['be-long']),
          # 'repeats': 5
    # },

    # 'PS-b21-t10k-openmpi3': {
    #     'exe': cycle(['PS_main.py']),
    #     'p': cycle([4000000]),
    #     'b': cycle([21]), # 21
    #     's': cycle([128]),
    #     't': cycle([10000]),
    #     'm': cycle([0]),
    #     'seed': cycle([0]),
    #     'reduce': cycle([1]),
    #     'load': cycle([0.0]),
    #     'mtw': cycle([50]),
    #     'approx': cycle([0]),
    #     'timing': cycle(['-time']), # otherwise pass -time
    #     'w': []
    #     # + [1, 2, 4, 8, 16],
    #     + list(np.arange(2, 17, 2)),
    #     # + list(np.arange(2, 9, 1)),
    #     'o': cycle([10]),
    #     # + [10] * 5,
    #     # + [10]*8,
    #     # + [20]*7,
    #     'mpi': cycle(['openmpi3']),
    #     'time': cycle([90]),
    #     'partition': cycle(['be-short']),
          # 'repeats': 5
    # },
    # 'PS-b21-t10k-mvapich2': {
    #     'exe': cycle(['PS_main.py']),
    #     'p': cycle([4000000]),
    #     'b': cycle([21]), # 21
    #     's': cycle([128]),
    #     't': cycle([10000]),
    #     'm': cycle([0]),
    #     'seed': cycle([0]),
    #     'reduce': cycle([1]),
    #     'load': cycle([0.0]),
    #     'mtw': cycle([50]),
    #     'approx': cycle([0]),
    #     'timing': cycle(['-time']), # otherwise pass -time
    #     'w': []
    #     # + [1, 2, 4, 8, 16],
    #     + list(np.arange(2, 17, 2)),
    #     # + list(np.arange(2, 9, 1)),
    #     'o': cycle([10]),
    #     # + [10] * 5,
    #     # + [10]*8,
    #     # + [20]*7,
    #     'mpi': cycle(['mvapich2']),
    #     'time': cycle([90]),
    #     'partition': cycle(['be-short']),
          # 'repeats': 5
    # }

}
