from cycler import cycle
import numpy as np

configs = {

    # 'LHC-48B-2MPPB-approx2': {
    #     'exe': cycle(['_LHC_BUP_2017.py']),    
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
    #     'partition': cycle(['be-short']),
    #       'repeats': 5
    # },


    # 'LHC-48B-2MPPB-approx1': {
    #     'exe': cycle(['_LHC_BUP_2017.py']),    
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
    #     'partition': cycle(['be-short']),
    #       'repeats': 5
    # }

  

    # 'LHC-96B-2MPPB-t10k-mpich3': {
    #     'exe': cycle(['_LHC_BUP_2017.py']),
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
    #     'partition': cycle(['be-short']),
    #       'repeats': 5
    # },

    # 'LHC-96B-2MPPB-t10k-openmpi3': {
    #     'exe': cycle(['_LHC_BUP_2017.py']),
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
    #     'partition': cycle(['be-long']),
    #       'repeats': 5
    # },

    # 'LHC-96B-2MPPB-t10k-mvapich2': {
    #     'exe': cycle(['_LHC_BUP_2017.py']),
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
    #     'partition': cycle(['be-long']),
    #       'repeats': 5
    # },


    # 'LHC-sync-mpich3': {
    #     'exe': cycle(['_LHC_BUP_2017_test.py']),
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
    #     'partition': cycle(['be-short']),
    #       'repeats': 5
    # },

    # 'LHC-sync-openmpi3': {
    #     'exe': cycle(['_LHC_BUP_2017_test.py']),
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
    #     'partition': cycle(['be-short']),
    #       'repeats': 5
    # },
    # 'LHC-sync-mvapich2': {
    #     'exe': cycle(['_LHC_BUP_2017_test.py']),
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
    #     'partition': cycle(['be-short']),
    #       'repeats': 5
    # },



    # 'LHC-approx2-mpich3': {
    #     'exe': cycle(['_LHC_BUP_2017.py']),
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
    #     'partition': cycle(['be-short']),
    #       'repeats': 5
    # },


    # 'LHC-approx2-openmpi3': {
    #     'exe': cycle(['_LHC_BUP_2017.py']),
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
    #     'partition': cycle(['be-short']),
    #       'repeats': 5
    # },

    # 'LHC-approx2-mvapich2': {
    #     'exe': cycle(['_LHC_BUP_2017.py']),
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
    #     'partition': cycle(['be-short']),
    #       'repeats': 5
    # },

    'LHC-lb-mpich3-approx2': {
        'exe': cycle(['_LHC_BUP_2017_lb.py']),
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
        + [2, 4, 8, 12, 16],
#        + list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mpich3']),
        'time': cycle([30]),
        'partition': cycle(['be-short']),
        'repeats': 5
    },

    'LHC-lb-mpich3': {
        'exe': cycle(['_LHC_BUP_2017_lb.py']),
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
        'w': [],
        # + [2, 4, 6, 8, 10, 12, 14, 16],
        #+ list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mpich3']),
        'time': cycle([30]),
        'partition': cycle(['be-short']),
        'repeats': 5
    },


    'LHC-lb-openmpi3-approx2': {
        'exe': cycle(['_LHC_BUP_2017_lb.py']),
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
        'w': [14],
        # + [2, 4, 6, 8, 10, 12, 14, 16],
        #+ list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['openmpi3']),
        'time': cycle([30]),
        'partition': cycle(['be-short']),
        'repeats': 5
    },

    'LHC-lb-openmpi3': {
        'exe': cycle(['_LHC_BUP_2017_lb.py']),
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
        'w': [16],
        # + [2, 4, 6, 8, 10, 12, 14, 16],
        #+ list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['openmpi3']),
        'time': cycle([30]),
        'partition': cycle(['be-short']),
        'repeats': 5
    },


    'LHC-lb-mvapich2-approx2': {
        'exe': cycle(['_LHC_BUP_2017_lb.py']),
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
        'w': [],
        # + [2, 4, 6, 8, 10, 12, 14, 16],
        #+ list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mvapich2']),
        'time': cycle([30]),
        'partition': cycle(['be-short']),
        'repeats': 5
    },

    'LHC-lb-mvapich2': {
        'exe': cycle(['_LHC_BUP_2017_lb.py']),
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
        'w': [],
        # + [2, 4, 6, 8, 10, 12, 14, 16],
        #+ list(np.arange(2, 17, 2)),
        'o': cycle([10]),
        'mpi': cycle(['mvapich2']),
        'time': cycle([30]),
        'partition': cycle(['be-short']),
        'repeats': 5
    },

    # 'LHC-lb-openmpi3': {
    #     'exe': cycle(['_LHC_BUP_2017_lb.py']),
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
    #     'partition': cycle(['be-long']),
    #       'repeats': 5
    # },

    # 'LHC-lb-mvapich2': {
    #     'exe': cycle(['_LHC_BUP_2017_lb.py']),
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
    #     'partition': cycle(['be-short']),
    #       'repeats': 5
    # },

    # 'LHC-weak-scale-mpich3': {
    #     'exe': cycle(['_LHC_BUP_2017.py']),
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
    #     'partition': cycle(['be-short']),
    #       'repeats': 5
    # },

    # 'LHC-weak-scale-openmpi3': {
    #     'exe': cycle(['_LHC_BUP_2017.py']),
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
    #     'partition': cycle(['be-long']),
    #       'repeats': 5
    # },

    # 'LHC-weak-scale-mvapich2': {
    #     'exe': cycle(['_LHC_BUP_2017.py']),
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
    #     'partition': cycle(['be-long']),
    #       'repeats': 5
    # },


}
