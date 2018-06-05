import argparse
import sys

parser = argparse.ArgumentParser(description='BLonD simulation mainfile.')

# parser.add_argument('-w', '--workers', type=int, default=3,
#                     help='Number of worker processes to spawn.'
#                     '\nDefault: 3 (3 workers + 1 master)')

parser.add_argument('-p', '--particles', type=int,
                    help='Number of macro-particles.')

parser.add_argument('-s', '--slices', type=int,
                    help='Number of slices.')


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

parser.add_argument('-r', '--report', type=str, default='reports',
                    nargs='?', const='reports',
                    help='Directory to store the timing reports.'
                    '\nDefault: Do not generate timing reports.')


parser.add_argument('-time', '--time', action='store_true',
                    help='Time the specified regions of interest.'
                    '\nDefault: No timing.')


parser.add_argument('-trace', '--trace', action='store_true',
                    help='Trace the specified regions of interest (MPE).'
                    '\nDefault: No tracing.')

parser.add_argument('-tracefile', '--tracefile', type=str, default='mpe-trace',
                    help='The file name to save the MPE trace (without the file type).'
                    '\nDefault: mpe-trace')


# parser.add_argument('-d', '--debug', action='store_true',
#                     help='Run workers in debug mode.'
#                     '\nDefault: No')


def parse():
    args = parser.parse_args()
    return vars(args)


if __name__ == '__main__':
    args = parser.parse_args()
    print('Parsed arguments: ', args)
