from cycler import cycle

case = 'EX01'

run_configs = [
    'mpich3-lb0-tp0-approx0',
    'mpich3-lb1-tp1-approx1',
    'mpich3-lb1-tp1-approx2',
]

base = {
    'exe': cycle(['EX_01_Acceleration.py']),
    'b': cycle([1]),
    't': cycle([5000]),
    'log': cycle([True]),
    'timing': cycle(['-time']),  # otherwise pass -time
    'partition': cycle(['be-short']),
    'o': cycle([10]),
    'repeats': 5
}


configs = {
    'small': {
        'mpich3-lb0-tp0-approx0': {
            'p': cycle([1000000]),
            's': cycle([400]),
            'approx': cycle([0]),
            'reduce': cycle([1]),
            'withtp': cycle([0]),
            'lb': cycle(['off']),
            'lba': cycle([500]),
            'w': [4, 8, 12, 16],
            'mpi': cycle(['mpich3']),
            'time': cycle([30]),
        },


        'mpich3-lb1-tp1-approx1': {
            'p': cycle([1000000]),
            's': cycle([400]),
            'approx': cycle([1]),
            'reduce': cycle([2]),
            'withtp': cycle([1]),
            'lb': cycle(['off']),
            'lba': cycle([500]),
            'w': [4, 8, 12, 16],
            'mpi': cycle(['mpich3']),
            'time': cycle([30]),
        },


        'mpich3-lb1-tp1-approx2': {
            'p': cycle([1000000]),
            's': cycle([400]),
            'approx': cycle([2]),
            'reduce': cycle([1]),
            'withtp': cycle([1]),
            'lb': cycle(['off']),
            'lba': cycle([500]),
            'w': [4, 8, 12, 16],
            'mpi': cycle(['mpich3']),
            'time': cycle([30]),
        },
    },
    'medium': {
        'mpich3-lb0-tp0-approx0': {
            'p': cycle([10000000]),
            's': cycle([4000]),
            'approx': cycle([0]),
            'reduce': cycle([1]),
            'withtp': cycle([0]),
            'lb': cycle(['off']),
            'lba': cycle([500]),
            'w': [4, 8, 12, 16],
            'mpi': cycle(['mpich3']),
            'time': cycle([30]),
        },


        'mpich3-lb1-tp1-approx1': {
            'p': cycle([10000000]),
            's': cycle([4000]),
            'approx': cycle([1]),
            'reduce': cycle([2]),
            'withtp': cycle([1]),
            'lb': cycle(['off']),
            'lba': cycle([500]),
            'w': [4, 8, 12, 16],
            'mpi': cycle(['mpich3']),
            'time': cycle([30]),
        },


        'mpich3-lb1-tp1-approx2': {
            'p': cycle([10000000]),
            's': cycle([4000]),
            'approx': cycle([2]),
            'reduce': cycle([1]),
            'withtp': cycle([1]),
            'lb': cycle(['off']),
            'lba': cycle([500]),
            'w': [4, 8, 12, 16],
            'mpi': cycle(['mpich3']),
            'time': cycle([30]),
        },
    },
    'large': {
        'mpich3-lb0-tp0-approx0': {
            'p': cycle([80000000]),
            's': cycle([8000]),
            'approx': cycle([0]),
            'reduce': cycle([1]),
            'withtp': cycle([0]),
            'lb': cycle(['off']),
            'lba': cycle([500]),
            'w': [4, 8, 12, 16],
            'mpi': cycle(['mpich3']),
            'time': cycle([30]),
        },


        'mpich3-lb1-tp1-approx1': {
            'p': cycle([80000000]),
            's': cycle([8000]),
            'approx': cycle([1]),
            'reduce': cycle([2]),
            'withtp': cycle([1]),
            'lb': cycle(['off']),
            'lba': cycle([500]),
            'w': [4, 8, 12, 16],
            'mpi': cycle(['mpich3']),
            'time': cycle([30]),
        },


        'mpich3-lb1-tp1-approx2': {
            'p': cycle([80000000]),
            's': cycle([8000]),
            'approx': cycle([2]),
            'reduce': cycle([1]),
            'withtp': cycle([1]),
            'lb': cycle(['off']),
            'lba': cycle([500]),
            'w': [4, 8, 12, 16],
            'mpi': cycle(['mpich3']),
            'time': cycle([30]),
        },
    }

}
