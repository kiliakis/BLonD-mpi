import argparse
import sys

parser = argparse.ArgumentParser(description='BLonD simulation mainfile.')

parser.add_argument('-w', '--workers', type=int, default=3,
                    help='Number of worker processes to spawn.'
                    '\nDefault: 3 (3 workers + 1 master)')

parser.add_argument('-p', '--particles', type=int, default=1000000,
                    help='Number of macro-particles.'
                    '\nDefault: 1 million')

parser.add_argument('-s', '--slices', type=int, default=500,
                    help='Number of slices.'
                    '\nDefault: 500')


parser.add_argument('-t', '--turns', type=int, default=2000,
                    help='Number of simulation turns.'
                    '\nDefault: 2000')

parser.add_argument('-o', '--omp', type=int, default=1,
                    help='Number of openmp threads to use.'
                    '\nDefault: 1')

parser.add_argument('-l', '--log', type=str, default=None,
                    nargs='?', const='logs',
                    help='Directory to store the log files.'
                    '\nDefault: Do not generate log files.')

parser.add_argument('-r', '--report', type=str, default=None,
                    nargs='?', const='reports',
                    help='Directory to store the report files.'
                    '\nDefault: Do not generate report files.')


parser.add_argument('-d', '--debug', action='store_true',
                    help='Run workers in debug mode.'
                    '\nDefault: No')


def parse():
    args = parser.parse_args()
    return vars(args)


if __name__ == '__main__':
    args = parser.parse_args()
    print('Parsed arguments: ', args)
