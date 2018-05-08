from mpi4py import MPI
import time
import numpy as np
import sys
from mpi import mpi_config as mpiconf
from utils.bphysics_wrap import __kick, __drift, __slice, __linear_interp_kick, __rf_volt_comp
import logging
from pyprof import timing

worker = None


@timing.timeit(key='comp:kick')
def kick():
    __kick(dt, dE, voltage, omegarf, phirf, n_rf, acc_kick)
    # comm.Barrier()


@timing.timeit(key='comp:drift')
def drift():
    __drift(dt, dE, solver, t_rev, length_ratio, alpha_order,
            eta_0, eta_1, eta_2, beta, energy)
    # comm.Barrier()


def histo():
    with timing.timed_region('comp:histo') as tr:
        global profile
        profile = np.empty(n_slices, dtype='d')
        __slice(dt, profile, cut_left, cut_right)

    # with timing.timed_region('histo_extra') as tr:
    #     new_profile = np.empty(len(profile), dtype='d')
    #     worker.intercomm.Allreduce(profile, new_profile, op=MPI.SUM)
    #     profile = new_profile
    with timing.timed_region('comm:histo_extra') as tr:
        # new_profile = np.empty(len(profile), dtype='d')
        worker.intracomm.Allreduce(MPI.IN_PLACE, profile, op=MPI.SUM)
        # profile = new_profile

    # Or even better, allreduce it

@timing.timeit(key='comp:LIKick')
def LIKick():
    __linear_interp_kick(dt, dE, total_voltage, bin_centers,
                         charge, acc_kick)


@timing.timeit(key='comp:SR')
def SR():
    __sync_rad_full(dE, U0, tau_z, n_kicks, sigma_dE, energy)


# Perhaps this is not big enough to use mpi, an omp might be better
@timing.timeit(key='comp:RFVCalc')
def RFVCalc():
    __rf_volt_comp(voltage, omegarf, phirf, bin_centers,
                   rf_voltage)

@timing.timeit(key='comm:gather')
def gather():
    worker.multi_gather(globals())

@timing.timeit(key='comm:bcast')
def bcast():
    globals().update(worker.multi_bcast())

@timing.timeit(key='comm:scatter')
def scatter():
    globals().update(worker.multi_scatter())

@timing.timeit(key='comm:barrier')
def barrier():
    worker.intercomm.Barrier()


@timing.timeit(key='comm:quit')
def quit():
    sys.stdout.flush()
    sys.stderr.flush()
    worker.logger.debug('Going to disconnect()')

    worker.intercomm.Disconnect()
    sys.exit(0)


if __name__ == '__main__':

    try:
        log = 'nolog' not in sys.argv
        worker = mpiconf.Worker(log=log)

        # This is the main loop
        task = None
        task = worker.intercomm.bcast(task, root=0)
        logging.debug('Received a %s task.' % task)

        start_t = time.time()
        while task != 'stop':
            if task == 'kick':
                kick()
            elif task == 'drift':
                drift()
            elif task == 'histo':
                histo()
            elif task == 'LIKick':
                LIKick()
            elif task == 'RFVCalc':
                RFVCalc()
            elif task == 'gather':
                gather()
            elif task == 'bcast':
                bcast()
            elif task == 'scatter':
                scatter()
            elif task == 'barrier':
                barrier()
            elif task == 'quit':
                quit()
            else:
                raise ValueError('Invalid task: %s.' % task)
            worker.logger.debug('Completed the %s task.' % task)

            with timing.timed_region('comm:task_receive') as tr:
                task = worker.intercomm.bcast(task, root=0)

            worker.logger.debug('Received a %s task.' % task)
        end_t = time.time()

        worker.logger.debug('Wating on the barrier')
        worker.intercomm.Barrier()

        timing.report(total_time=1e3*(end_t-start_t),
                      out_file='report-worker-%d.csv' % worker.rank)
    # Shutdown
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        worker.intercomm.Disconnect()
