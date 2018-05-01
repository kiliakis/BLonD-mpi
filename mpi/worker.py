from mpi4py import MPI
import time
import numpy as np
import sys
from mpi import mpi_config as mpiconf
from utils.bphysics_wrap import __kick, __drift
import logging

comm = None
rank = -1


def kick():
    vars_dict = multi_bcast_worker(comm)
    globals().update(vars_dict)
    __kick(dt, dE, voltage, omega_rf, phi_rf, n_rf, acc_kick)


def drift():
    vars_dict = multi_bcast_worker(comm)
    globals().update(vars_dict)
    __drift(dt, dE, solver, t_rev, length_ratio, alpha_order,
            eta_0, eta_1, eta_2, beta, energy)


def histo():
    vars_dict = multi_bcast_worker(comm)
    globals().update(vars_dict)
    __slice(dt, profile, cut_left, cut_right)
    new_profile = np.empty(len(profile), dtype='d')
    comm.Allreduce(profile, new_profile, op=MPI.SUM, root=0)
    profile = new_profile
    # Or even better, allreduce it


def LIkick():
    vars_dict = multi_bcast_worker(comm)
    globals().update(vars_dict)
    __linear_interp_kick(dt, dE, total_voltage, bin_centers,
                         charge, acc_kick)


# Perhaps this is not big enough to use mpi, an omp might be better
def RFVCalc():
    vars_dict = multi_bcast_worker(comm)
    globals().update(vars_dict)
    __rf_volt_comp(voltages, omega_rf, phi_rf, bin_centers,
                   rf_voltage)


def gather():
    mpiconf.multi_gather_worker(comm, globals())


def bcast():
    globals().update(mpiconf.multi_bcast_worker(comm))


def scatter():
    globals().update(mpiconf.multi_scatter_worker(comm))


def barrier():
    comm.Barrier()


if __name__ == '__main__':

    try:
        # Connect to parent
        comm = MPI.Comm.Get_parent()
        rank = comm.Get_rank()
    except:
        raise ValueError('Could not connect to parent')

    try:
        logger = mpiconf.MPILog(rank)
        logging.debug('Hostname: %s' % MPI.Get_processor_name())

        # This is the main loop
        task = None
        task = comm.bcast(task, root=0)
        while task != 'stop':
            logging.debug('Received a %s task.' % task)
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
            task = comm.bcast(task, root=0)

    # Shutdown
    finally:
        comm.Disconnect()
