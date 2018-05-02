from mpi4py import MPI
import time
import numpy as np
import sys
from mpi import mpi_config as mpiconf
from utils.bphysics_wrap import __kick, __drift
import logging

worker = None


def kick():
    __kick(dt, dE, voltage, omegarf, phirf, n_rf, acc_kick)
    # comm.Barrier()


def drift():
    __drift(dt, dE, solver, t_rev, length_ratio, alpha_order,
            eta_0, eta_1, eta_2, beta, energy)
    # comm.Barrier()


def histo():
    __slice(dt, profile, cut_left, cut_right)
    new_profile = np.empty(len(profile), dtype='d')
    worker.intercomm.Allreduce(profile, new_profile, op=MPI.SUM, root=0)
    profile = new_profile
    # Or even better, allreduce it


def LIkick():
    __linear_interp_kick(dt, dE, total_voltage, bin_centers,
                         charge, acc_kick)


# Perhaps this is not big enough to use mpi, an omp might be better
def RFVCalc():
    __rf_volt_comp(voltages, omega_rf, phi_rf, bin_centers,
                   rf_voltage)


def gather():
    worker.multi_gather(globals())


def bcast():
    globals().update(worker.multi_bcast())


def scatter():
    globals().update(worker.multi_scatter())


def barrier():
    worker.intercomm.Barrier()


if __name__ == '__main__':

    try:

        worker = mpiconf.Worker()

        # This is the main loop
        task = None
        task = worker.intercomm.bcast(task, root=0)
        logging.debug('Received a %s task.' % task)
        while task != 'stop':
            if task == 'kick':
                kick()
            elif task == 'drift':
                drift()
            elif task == 'histo':
                histo()
            elif task == 'LIkick':
                LIkick()
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
            else:
                raise ValueError('Invalid task: %s.' % task)
            logging.debug('Completed the %s task.' % task)
            task = worker.intercomm.bcast(task, root=0)
            logging.debug('Received a %s task.' % task)
        logging.debug('Wating on the barrier')
        worker.intercomm.Barrier()

    # Shutdown
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        worker.intercomm.Disconnect()
