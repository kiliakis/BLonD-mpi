
from mpi4py import MPI
import time
import numpy as np
import sys
from mpi import mpi_config as mpiconf
from utils.bphysics_wrap import __kick, __drift
import logging

comm = None
rank = -1
# initial variables needed for kick:
# charge, n_rf + dt, dE
def kick():
    # first get needed parameters
    # these are: voltage, omega_rf, phi_rf, acc_kick
    vars_dict = multi_bcast_worker(comm)
    globals().update(vars_dict)

    # then compute
    __kick(dt, dE, voltage, omega_rf, phi_rf, n_rf, acc_kick)
    # finally send back any needed values
    # I don't need to send anything back

# initial variables needed for kick:
# solver, length_ratio, alpha_order + dt, dE
def drift():
    # logging.debug('I am running drift' % rank)
    # first get needed parameters
    # these are: t_rev, eta_0, eta_1, eta_2, beta, energy
    vars_dict = multi_bcast_worker(comm)
    globals().update(vars_dict)

    # then compute
    __drift(dt, dE, solver, t_rev, length_ratio, alpha_order,
            eta_0, eta_1, eta_2, beta, energy)
    # finally send back any needed values
    # I don't need to send anything back


if __name__ == '__main__':

    try:
        # Connect to parent
        comm = MPI.Comm.Get_parent()
        rank = comm.Get_rank()
    except:
        raise ValueError('Could not connect to parent')

    try:
        logger = mpiconf.Logger(rank)

        logging.debug('Hostname: %s' % MPI.Get_processor_name())

        # General initialization variables
        vars_dict = mpiconf.multi_bcast_worker(comm)
        comm.Barrier()
        globals().update(vars_dict)
        logging.debug('Initialization dictionary: ' % (rank), vars_dict)

        # Numpy arrays, coordinates etc
        vars_dict = mpiconf.multi_scatter_worker(comm)
        comm.Barrier()
        logging.debug('Received dict: ' % (rank), vars_dict)
        globals().update(vars_dict)

        # This is the main loop
        task = None
        task = comm.bcast(task, root=0)
        while task != 'stop':
            if task == 'kick':
                logging.debug('Received kick task')
                kick()
            elif task == 'drift':
                logging.debug('Computing drift task')
                drift()
            else:
                raise ValueError('Invalid task: %s' % task)
            task = comm.bcast(task, root=0)
        comm.Barrier()

    # Shutdown
    finally:
        comm.Disconnect()
