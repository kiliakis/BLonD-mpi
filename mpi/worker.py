from mpi4py import MPI
import time
import numpy as np
import sys
import os
from mpi import mpi_config as mpiconf
from utils.bphysics_wrap import __kick, __drift, __slice, __linear_interp_kick, __rf_volt_comp
import logging
# from pyprof import timing
from pyprof import mpiprof
# import argparse
from toolbox.input_parser import parse

worker = None


@mpiprof.trackit(key='comp:kick')
def kick():
    __kick(dt, dE, voltage, omegarf, phirf, n_rf, acc_kick)
    # comm.Barrier()


@mpiprof.trackit(key='comp:drift')
def drift():
    __drift(dt, dE, solver, t_rev, length_ratio, alpha_order,
            eta_0, eta_1, eta_2, beta, energy)
    # comm.Barrier()

# @mpiprof.trackit('comp:histo')
def histo():
    with mpiprof.tracked_region('comp:histo') as tr:
        # global profile
        profile = np.empty(n_slices, dtype='d')
        __slice(dt, profile, cut_left, cut_right)

    # with mpiprof.tracked_region('histo_extra') as tr:
    #     new_profile = np.empty(len(profile), dtype='d')
    #     worker.intercomm.Allreduce(profile, new_profile, op=MPI.SUM)
    #     profile = new_profile
    with mpiprof.tracked_region('comm:histo_extra') as tr:
        # new_profile = np.empty(len(profile), dtype='d')
        worker.intracomm.Allreduce(MPI.IN_PLACE, profile, op=MPI.SUM)
        # profile = new_profile

    # Or even better, allreduce it


@mpiprof.trackit(key='comp:LIKick')
def LIKick():
    __linear_interp_kick(dt, dE, total_voltage, bin_centers,
                         charge, acc_kick)


@mpiprof.trackit(key='comp:SR')
def SR():
    __sync_rad_full(dE, U0, tau_z, n_kicks, sigma_dE, energy)


# Perhaps this is not big enough to use mpi, an omp might be better
@mpiprof.trackit(key='comp:RFVCalc')
def RFVCalc():
    __rf_volt_comp(voltage, omegarf, phirf, bin_centers,
                   rf_voltage)


@mpiprof.trackit(key='comm:gather')
def gather():
    worker.multi_gather(globals())


@mpiprof.trackit(key='comm:bcast')
def bcast():
    globals().update(worker.multi_bcast())


@mpiprof.trackit(key='comm:scatter')
def scatter():
    globals().update(worker.multi_scatter())


@mpiprof.trackit(key='comm:barrier')
def barrier():
    worker.intercomm.Barrier()


@mpiprof.trackit(key='comm:quit')
def quit():
    sys.stdout.flush()
    sys.stderr.flush()
    worker.logger.debug('Going to disconnect()')

    worker.intercomm.Disconnect()
    sys.exit(0)


@mpiprof.trackit(key='comm:stop')
def stop():
    pass


task_dir = {
    0: kick,
    1: drift,
    2: histo,
    3: LIKick,
    4: RFVCalc,
    5: gather,
    6: bcast,
    7: scatter,
    8: barrier,
    9: quit,
    10: stop
}


# def parse():
#     parser = argparse.ArgumentParser(description='MPI Worker function.')

#     parser.add_argument('-l', '--log', type=str, default=None,
#                         nargs='?', const='logs',
#                         help='Directory to store the log files.'
#                         '\nDefault: Do not generate log files.')

#     parser.add_argument('-r', '--report', type=str, default=None,
#                         nargs='?', const='reports',
#                         help='Directory to store the report files.'
#                         '\nDefault: Do not generate report files.')
#     args = parser.parse_args()
#     return vars(args)


def main():
    global worker
    try:
        args = parse()
        # log = 'nolog' not in sys.argv
        # report = ''
        worker = mpiconf.Worker(log=args.get('log', None))
        os.environ['OMP_NUM_THREADS'] = os.environ.get('OMP_NUM_THREADS', '1')
        worker.logger.debug('OMP_NUM_THREADS=%s' %
                            os.environ['OMP_NUM_THREADS'])
        start_t = time.time()

        task = worker.recv_task()
        # This is the main loop
        while task != 10:
            try:
                task_dir[task]()
            except:
                raise ValueError('Invalid task: %d.' % task)
            task = worker.recv_task()

        end_t = time.time()

        worker.logger.debug('Wating on the final barrier.')
        worker.intercomm.Barrier()

        if args.get('report', None):
            mpiprof.finalize()
            # timing.report(total_time=1e3*(end_t-start_t),
            #               out_dir=args['report'],
            #               out_file='worker-%d.csv' % worker.rank)
    # Shutdown
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        worker.intercomm.Disconnect()
        exit(0)