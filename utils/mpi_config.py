import sys
from mpi4py import MPI
import numpy as np
import logging
from pyprof import timing as mpiprof
# from pyprof import mpiprof as mpiprof
import os
from utils import worker

mpi_type = {
    'd': MPI.DOUBLE,
    'float64': MPI.DOUBLE,
    'numpy.float64': MPI.DOUBLE,
    '<f8': MPI.DOUBLE,
    'i': MPI.INT,
    'int32': MPI.INT,
    'numpy.int32': MPI.INT,
    '<i4': MPI.INT,
    'uint8': MPI.UINT8_T,
    'B': MPI.UINT8_T,
    '|u1': MPI.UINT8_T
}


task_id = {
    'kick': np.array(0, np.uint8),
    'drift': np.array(1, np.uint8),
    'histo': np.array(2, np.uint8),
    'LIKick': np.array(3, np.uint8),
    'RFVCalc': np.array(4, np.uint8),
    'gather': np.array(5, np.uint8),
    'bcast': np.array(6, np.uint8),
    'scatter': np.array(7, np.uint8),
    'barrier': np.array(8, np.uint8),
    'quit': np.array(9, np.uint8),
    'stop': np.array(10, np.uint8)
}

master = None


def init(track=False):
    rank = MPI.COMM_WORLD.rank
    if track==True:
        mpiprof.mode = 'timing'
        mpiprof.init()

    if rank != 0:
        worker.main()
        exit(0)


class Master:
    # @mpiprof.timeit(key='master:init')
    def __init__(self, log=None):
        global master

        rank = MPI.COMM_WORLD.rank
        self.intracomm = MPI.COMM_WORLD.Split(rank==0, rank)
        self.intercomm = self.intracomm.Create_intercomm(0, MPI.COMM_WORLD, 1)
        self.workers = self.intercomm.Get_remote_size()
        self.rank = self.intracomm.rank

        if self.intracomm.size != 1:
            'Only one process can be the master!\nRe-run with only 1 process.'

        self.hostname = MPI.Get_processor_name()
        if log:
            self.logger = MPILog(rank=-1, log_dir=log)
        else:
            self.logger = MPILog(rank=-1)
            self.logger.disable()
        logging.debug('Initialized.')
        master = self


    @mpiprof.timeit(key='master:multi_scatter')
    def multi_scatter(self, vars):
        self.intercomm.Bcast(task_id['scatter'], root=MPI.ROOT)

        var_list = [(k, v.dtype.char) for k, v in vars.items()]
        self.intercomm.bcast(var_list, root=MPI.ROOT)

        for name, dtype in var_list:
            val = vars[name]
            # calculate counts and displs
            basesize = len(val) // self.workers
            plusone = len(val) - basesize * self.workers
            counts = np.array([basesize+1]*plusone + [basesize] *
                              (self.workers-plusone), dtype='i')
            displs = np.append([0], np.cumsum(counts[:-1]))

            recvbuf = np.array(0, dtype='i')
            self.intercomm.Scatter(counts, recvbuf, root=MPI.ROOT)

            recvbuf = np.empty(1, dtype=dtype)
            self.intercomm.Scatterv([val, counts, displs, mpi_type[dtype]],
                                    recvbuf, root=MPI.ROOT)

    # args are the buffers to fill with the gathered values
    # e.g. (comm, beam.dt, beam.dE)
    @mpiprof.timeit(key='master:multi_gather')
    def multi_gather(self, gather_dict):
        self.intercomm.Bcast(task_id['gather'], root=MPI.ROOT)
        keys = list(gather_dict.keys())
        self.intercomm.bcast(keys, root=MPI.ROOT)
        sendbuf = None
        for k in gather_dict.keys():
            v = gather_dict[k]
            basesize = len(v) // self.workers
            plusone = len(v) - basesize * self.workers
            counts = np.array([basesize+1]*plusone + [basesize] *
                              (self.workers-plusone), dtype='i')
            displs = np.append([0], np.cumsum(counts[:-1]))

            self.intercomm.Gatherv(sendbuf, [v, counts, displs, mpi_type[v.dtype.char]],
                                   root=MPI.ROOT)

    @mpiprof.timeit(key='master:multi_bcast')
    def multi_bcast(self, vars):
        self.logger.debug('Broadcasting variables')
        self.intercomm.Bcast(task_id['bcast'], root=MPI.ROOT)
        self.intercomm.bcast(vars, root=MPI.ROOT)

    @mpiprof.timeit(key='master:bcast')
    def bcast(self, cmd):
        self.intercomm.Bcast(task_id[cmd], root=MPI.ROOT)

    @mpiprof.timeit(key='master:reduce')
    def reduce(self, x, y, op=MPI.SUM):
        self.intercomm.Reduce(x, y, op=op, root=MPI.ROOT)

    # @mpiprof.timeit(key='master:stop')
    def stop(self):
        self.logger.debug('Sending a stop signal')
        self.intercomm.Bcast(task_id['stop'], root=MPI.ROOT)
        # self.logger.debug('Waiting on the barrier')
        # self.intercomm.Barrier()

    @mpiprof.timeit(key='master:sync')
    def sync(self):
        self.intercomm.Bcast(task_id['barrier'], root=MPI.ROOT)
        self.intercomm.Barrier()

    # @mpiprof.timeit(key='master:disconnect')
    def disconnect(self):
        self.intercomm.Disconnect()

    # @mpiprof.timeit(key='master:quit')
    def quit(self):
        self.intercomm.Bcast(task_id['quit'], root=MPI.ROOT)



class MPILog(object):
    """Class to log messages coming from other classes. Messages contain 
    {Time stamp} {Class name} {Log level} {Message}. Errors, warnings and info
    are logged into the console. To disable logging, call Logger().disable()

    Parameters
    ----------
    debug : bool
        Log DEBUG messages in 'debug.log'; default is False

    """

    def __init__(self, rank=-1, log_dir='./'):

        # Root logger on DEBUG level
        self.disabled = False
        self.root_logger = logging.getLogger()
        self.root_logger.setLevel(logging.DEBUG)
        # if not os.path.exists(log_dir):
        #     os.mkdir(log_dir)
        if rank < 0:
            log_name = log_dir+'/master.log'
        else:
            log_name = log_dir+'/worker-%.3d.log' % rank
        # Console handler on INFO level
        # console_handler = logging.StreamHandler()
        # console_handler.setLevel(logging.INFO)
        log_format = logging.Formatter(
            "%(asctime)s %(name)-25s %(levelname)-9s %(message)s")
        # console_handler.setFormatter(log_format)
        # self.root_logger.addHandler(console_handler)

        self.file_handler = logging.FileHandler(log_name, mode='w')
        self.file_handler.setLevel(logging.DEBUG)
        self.file_handler.setFormatter(log_format)
        self.root_logger.addHandler(self.file_handler)
        logging.debug("Start logging")
        # if debug == True:
        #     logging.debug("Logger in debug mode")

    def disable(self):
        """Disables all logging."""

        logging.info("Disable logging")
        # logging.disable(level=logging.NOTSET)
        # self.root_logger.setLevel(logging.NOTSET)
        # self.file_handler.setLevel(logging.NOTSET)
        self.root_logger.disabled = True
        self.disabled = True

    def debug(self, string):
        if self.disabled == False:
            logging.debug(string)

    def info(self, string):
        if self.disabled == False:
            logging.info(string)
