import sys
from mpi4py import MPI
import numpy as np
import logging

mpi_type = {
    'd': MPI.DOUBLE,
    'i': MPI.INT
}

executable = 'python'
n_workers = 2
worker_script = sys.path[0] + '/worker.py'


def multi_scatter_master(comm, vars):

    var_list = [(k, v[1]) for k, v in vars.items()]
    comm.bcast(var_list, root=MPI.ROOT)

    for (name, dtype) in var_list:
        val = vars[name][0]
        # calculate counts and displs
        basesize = len(val) // n_workers
        plusone = len(val) - basesize * n_workers
        counts = np.array([basesize+1]*plusone + [basesize] *
                          (n_workers-plusone), dtype='i')
        displs = np.append([0], np.cumsum(counts[:-1]))

        recvbuf = np.array(0, dtype='i')
        comm.Scatter(counts, recvbuf, root=MPI.ROOT)

        recvbuf = np.empty(1, dtype=dtype)
        comm.Scatterv([val, counts, displs, mpi_type[dtype]],
                            recvbuf, root=MPI.ROOT)
    # comm.Barrier()


def multi_scatter_worker(comm):
    var_list = None
    var_list = comm.bcast(var_list, root=0)
    vals = {}
    sendbuf = None
    for name, dtype in var_list:
        count = np.array(0, dtype='i')
        comm.Scatter(sendbuf, count, root=0)
        # print('Worker[%d]: Received buffer: ' % (rank), count)
        recvbuf = np.empty(count, dtype=dtype)
        comm.Scatterv(sendbuf, recvbuf, root=0)
        vals[name] = recvbuf
        # print('Worker[%d]: Received buffer: ' % (rank), recvbuf)
    return vals
    # comm.Barrier()



def multi_bcast_master(comm, vars):
    comm.bcast(vars, root=MPI.ROOT)


def multi_bcast_worker(comm):
    vars = None
    vars = comm.bcast(vars, root=0)
    return vars

# def __scalar_multi_bcast_master(comm, vars):
#     # method 1
#     # comm.bcast(vars, root=MPI.ROOT)

#     # method 2
#     var_list = [(k, v[1]) for k, v in vars.items()]
#     comm.bcast(var_list, root=MPI.ROOT)

#     for (name, dtype) in var_list:
#         val = vars[name][0]
#         # calculate counts and displs
#         basesize = len(val) // n_workers
#         plusone = len(val) - basesize * n_workers
#         counts = np.array([basesize+1]*plusone + [basesize] *
#                           (n_workers-plusone), dtype='i')
#         displs = np.append([0], np.cumsum(counts[:-1]))

#         recvbuf = np.array(0, dtype='i')
#         comm.Scatter(counts, recvbuf, root=MPI.ROOT)

#         recvbuf = np.empty(1, dtype=dtype)
#         comm.Scatterv([val, counts, displs, mpi_type[dtype]],
#                             recvbuf, root=MPI.ROOT)
#     # comm.Barrier()



class Logger(object):
    """Class to log messages coming from other classes. Messages contain 
    {Time stamp} {Class name} {Log level} {Message}. Errors, warnings and info
    are logged into the console. To disable logging, call Logger().disable()
    
    Parameters
    ----------
    debug : bool
        Log DEBUG messages in 'debug.log'; default is False
    
    """
    
    def __init__(self, debug = False):

        # Root logger on DEBUG level
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # Console handler on INFO level        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        log_format = logging.Formatter(
            "%(asctime)s %(name)-25s %(levelname)-9s %(message)s")
        console_handler.setFormatter(log_format)
        root_logger.addHandler(console_handler)
        
        # File logger on DEBUG level
        if debug == True:       
            file_handler = logging.FileHandler('debug.log', mode = 'w')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(log_format)
            root_logger.addHandler(file_handler)
            
        logging.info("Start logging")
        if debug == True:
            logging.debug("Logger in debug mode")


    def disable(self):
        """Disables all logging."""
        
        logging.info("Disable logging")
        logging.disable(level=logging.NOTSET)
        