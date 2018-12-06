
# Copyright 2016 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''

@author: Danilo Quartullo
'''

# MAKE SURE YOU HAVE GCC 4.8.1 OR LATER VERSIONS ON YOUR SYSTEM LINKED TO YOUR
# SYSTEM PATH.IF YOU ARE ON CERN-LXPLUS YOU CAN TYPE FROM CONSOLE
# source /afs/cern.ch/sw/lcg/contrib/gcc/4.8.1/x86_64-slc6/setup.sh
# TO GET GCC 4.8.1 64 BIT. IN GENERAL IT IS ADVISED TO USE PYTHON 64 BIT PLUS
# GCC 64 BIT.

from __future__ import print_function
import os
import sys
import subprocess
import ctypes
import argparse

path = os.path.realpath(__file__)
basepath = os.sep.join(path.split(os.sep)[:-1])

parser = argparse.ArgumentParser(description='Run python compile.py to'
                                 ' compile the cpp routines needed from BLonD')

parser.add_argument('-p', '--parallel',
                    default=False, action='store_true',
                    help='Produce Multi-threaded code. Use the environment'
                    ' variable OMP_NUM_THREADS=xx to control the number of'
                    ' threads that will be used.'
                    ' Default: Serial code')

parser.add_argument('-b', '--boost', type=str, nargs='?', const='',
                    help='Use boost library to speedup synchrotron radiation'
                    ' routines. If the installation path of boost differs'
                    ' from the default, you have to pass it as an argument.'
                    ' Default: Boost will not be used')

parser.add_argument('-c', '--compiler', type=str, default='g++',
                    help='C++ compiler that will be used to compile the'
                    ' source files. Default: g++')

parser.add_argument('-l', '--libs', type=str, default='',
                    help='Any extra libraries needed to compile')

# If True you can launch with 'OMP_NUM_THREADS=xx python MAIN_FILE.py'
# where xx is the number of threads that you want to launch
parallel = False

# If True, the boost library would be used
boost = False
# Path to the boost library if not in your CPATH (recommended to use the
# latest version)
boost_path = None

# EXAMPLE FLAGS: -Ofast -std=c++11 -fopt-info-vec -march=native
#                -mfma4 -fopenmp -ftree-vectorizer-verbose=1
cflags = ['-Ofast', '-g', '-std=c++11',
          '-march=native', '-Wno-psabi', '-fopt-info-vec']
# cflags = ['-Ofast', '-std=c++11']
# libs = ['-L/afs/cern.ch/work/k/kiliakis/install/lib','-lfftw3', '-lm']
libs = []

cpp_files = [
    os.path.join(basepath, 'cpp_routines/kick.cpp'),
    os.path.join(basepath, 'cpp_routines/drift.cpp'),
    os.path.join(basepath, 'cpp_routines/fft.cpp'),
    os.path.join(basepath, 'cpp_routines/linear_interp_kick.cpp'),
    os.path.join(basepath, 'cpp_routines/histogram.cpp'),
    os.path.join(basepath, 'cpp_routines/music_track.cpp'),
    os.path.join(basepath, 'cpp_routines/blondmath.cpp'),
    os.path.join(basepath, 'cpp_routines/fast_resonator.cpp'),
    os.path.join(basepath, 'cpp_routines/beam_phase.cpp'),
    os.path.join(basepath, 'toolbox/tomoscope.cpp'),
    os.path.join(basepath, 'synchrotron_radiation/synchrotron_radiation.cpp'),
    os.path.join(basepath, 'beam/sparse_histogram.cpp')
]


if (__name__ == "__main__"):
    args = parser.parse_args()
    parallel = args.parallel
    if(args.boost is not None):
        boost = True
        if(args.boost):
            boost_path = os.path.abspath(args.boost)
        else:
            boost_path = ''
        cflags += ['-I', boost_path, '-DBOOST']

    compiler = args.compiler

    if (args.libs):
        libs = args.libs.split()

    print('Enable Multi-threaded code: ', parallel)
    print('Use of boost: ', boost)
    print('Boost installation path: ', boost_path)
    print('C++ Compiler: ', compiler)
    print('Extra libraries: ', libs)
    subprocess.call([compiler, '--version'])

    try:
        os.remove(os.path.join(basepath, 'cpp_routines/libblond.so'))
    except OSError as e:
        pass

    if (parallel is True):
        cflags += ['-fopenmp', '-DPARALLEL', '-D_GLIBCXX_PARALLEL']

    if ('posix' in os.name):
        cflags += ['-fPIC']
        libname = os.path.join(basepath, 'cpp_routines/libblond.so')
        command = [compiler] + cflags + ['-o', libname] + cpp_files + libs
        subprocess.call(command)

        print('\nIF THE COMPILATION IS CORRECT A FILE NAMED libblond.so SHOULD'
              ' APPEAR IN THE cpp_routines FOLDER. OTHERWISE YOU HAVE TO'
              ' CORRECT THE ERRORS AND COMPILE AGAIN.')

    elif ('win' in sys.platform):

        libname = os.path.join(basepath, 'cpp_routines/libblond.dll')

        command = [compiler] + cflags + ['-o', libname] + cpp_files + libs
        subprocess.call(command)

        print('\nIF THE COMPILATION IS CORRECT A FILE NAMED libblond.dll SHOULD'
              ' APPEAR IN THE cpp_routines FOLDER. OTHERWISE YOU HAVE TO'
              ' CORRECT THE ERRORS AND COMPILE AGAIN.')

    else:
        print(
            'YOU DO NOT HAVE A WINDOWS OR LINUX OPERATING SYSTEM. ABORTING...')
        sys.exit(-1)


#     if ('posix' in os.name):
#         cflags += ['-shared']
#         if('linux' in sys.platform):
#             cflags += ['-fPIC']
#         subprocess.call('rm -rf cpp_routines/*.so',
#                         shell=True, executable='/bin/bash')
#         subprocess.call('rm -rf synchrotron_radiation/*.so',
#                         shell=True, executable='/bin/bash')

#         # command = [compiler] + cflags + \
#         #     ['-o', 'cpp_routines/result.so'] + cpp_files
#         # subprocess.call(command)

#         # command = [compiler] + cflags + \
#         #     ['-o', 'synchrotron_radiation/sync_rad.so'] + cpp_files_SR
#         # subprocess.call(command)

#         command = [compiler] + cflags + \
#             ['-o', 'cpp_routines/libblondphysics.so'] + cpp_files_SR + cpp_files
#         subprocess.call(command)

#         command = [compiler] + cflags + \
#             ['-o', 'cpp_routines/libblondmath.so'] + \
#             ['cpp_routines/blondmath.cpp']
#         subprocess.call(command)

#         command = [compiler] + cflags + \
#             ['-o', 'cpp_routines/fft.so'] + cpp_files_fft + libs
#         subprocess.call(command)

#         print('\nIF THE COMPILATION IS CORRECT A FILE NAMED result.so SHOULD'
#               ' APPEAR IN THE cpp_routines FOLDER. OTHERWISE YOU HAVE TO'
#               ' CORRECT THE ERRORS AND COMPILE AGAIN.')
#         sys.exit()

#     elif ('win' in sys.platform):
#         os.system('gcc --version')
#         os.system('del /s/q ' + os.getcwd() + '\\cpp_routines\\*.dll')
#         os.system('del /s/q ' + os.getcwd() + '\\synchrotron_radiation\\*.dll')

#         cpp_files_join_list = os.getcwd()+'\\'+' '.join(cpp_files)
#         cpp_files_SR_join_list = os.getcwd()+'\\'+' '.join(cpp_files_SR)
#         cflags_join_list = ' ' + ' '.join(cflags)
#         cpp_files_bmath_join_list = ' '.join(
#             cpp_files)+' '+' '.join(cpp_files_SR)

#         command = compiler + cflags_join_list + ' -o ' + \
#             os.getcwd()+'\\cpp_routines\\result.dll -shared ' + \
#             cpp_files_join_list
#         os.system(command)

#         command = compiler + cflags_join_list + ' -o ' + \
#             os.getcwd()+'\\synchrotron_radiation\\sync_rad.dll -shared ' +\
#             cpp_files_SR_join_list
#         os.system(command)

#         command = compiler + cflags_join_list + ' -o ' + \
#             os.getcwd()+'\\cpp_routines\\libblondphysics.dll -shared ' + \
#             cpp_files_bmath_join_list
#         os.system(command)

#         command = compiler + cflags_join_list + ' -o ' + \
#             os.getcwd()+'\\cpp_routines\\libblondmath.dll -shared ' + \
#             os.getcwd() + '\\cpp_routines\\blondmath.cpp'
#         os.system(command)

#         print('\nIF THE COMPILATION IS CORRECT A FILE NAMED result.dll SHOULD'
#               ' APPEAR IN THE cpp_routines FOLDER. OTHERWISE YOU HAVE TO'
#               ' CORRECT THE ERRORS AND COMPILE AGAIN.')
#         sys.exit()

#     else:
#         print(
#             'YOU DO NOT HAVE A WINDOWS OR LINUX OPERATING SYSTEM. ABORTING...')
#         sys.exit()


# path = os.path.realpath(__file__)
# parent_path = os.sep.join(path.split(os.sep)[:-1])
# if ('posix' in os.name):
#     # libblond = ctypes.CDLL(parent_path+'/cpp_routines/result.so')
#     # libsrqe = ctypes.CDLL(parent_path+'/synchrotron_radiation/sync_rad.so')
#     libblondmath = ctypes.CDLL(parent_path+'/cpp_routines/libblondmath.so')
#     libblondphysics = ctypes.CDLL(
#         parent_path+'/cpp_routines/libblondphysics.so')
#     libfft = ctypes.CDLL(parent_path+'/cpp_routines/fft.so')

#     libblond = libblondphysics
#     libsrqe = libblondphysics
# elif ('win' in sys.platform):
#     libblond = ctypes.CDLL(parent_path+'\\cpp_routines\\result.dll')
#     libsrqe = ctypes.CDLL(parent_path+'\\synchrotron_radiation\\sync_rad.dll')
#     libblondmath = ctypes.CDLL(parent_path+'\\cpp_routines\\libblondmath.dll')
#     libblondphysics = ctypes.CDLL(
#         parent_path+'\\cpp_routines\\libblondphysics.dll')
# else:
#     print('YOU DO NOT HAVE A WINDOWS OR LINUX OPERATING SYSTEM. ABORTING...')
#     sys.exit()
