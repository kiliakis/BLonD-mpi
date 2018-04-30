
from mpi4py import MPI
import time
import numpy as np
import sys
from mpi import mpi_config as mpiconf
from utils.bphysics_wrap import __kick, __drift


def kick():
    # first get needed parameters
    # then compute
    __kick(...)
    # finally send back any needed values


def drift():
    # print('Worker[%d]: I am running drift' % rank)
    # first get needed parameters
    # then compute
    __drift(...)
    # finally send back any needed values


if __name__ == '__main__':

    try:
        # Connect to parent
        comm = MPI.Comm.Get_parent()
        rank = comm.Get_rank()
    except:
        raise ValueError('Could not connect to parent')

    try:
        print('Worker[%d]: Hostname: %s' % (rank, MPI.Get_processor_name()))

        # General initialization variables
        vars_dict = mpiconf.multi_bcast_worker(comm)
        comm.Barrier()
        locals().update(vars_dict)
        print('Worker[%d]: Initialization dictionary: ' % (rank), vars_dict)

        # Numpy arrays, coordinates etc
        vars_dict = mpiconf.multi_scatter_worker(comm)
        comm.Barrier()
        print('Worker[%d]: Received dict: ' % (rank), vars_dict)
        locals().update(vars_dict)

        # This is the main loop
        task = None
        task = comm.bcast(task, root=0)
        while task != 'stop':
            if task == 'kick':
                kick()
            elif task == 'drift':
                drift()
            else:
                raise ValueError('Invalid task: %s' % task)
            task = comm.bcast(task, root=0)
        comm.Barrier()

    # Shutdown
    finally:
        comm.Disconnect()
