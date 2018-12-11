import sys
import os
from mpi4py import MPI
import numpy as np
import logging
from functools import wraps
try:
    from pyprof import timing
    from pyprof import mpiprof
except ImportError:
    from ..utils import profile_mock as timing
    mpiprof = timing

from ..utils.input_parser import parse
from ..utils import bmath as bm

worker = None



def c_add_uint32(xmem, ymem, dt):
    x = np.frombuffer(xmem, dtype=np.uint32)
    y = np.frombuffer(ymem, dtype=np.uint32)
    bm.add(y, x, inplace=True)

def c_add_uint16(xmem, ymem, dt):
    x = np.frombuffer(xmem, dtype=np.uint16)
    y = np.frombuffer(ymem, dtype=np.uint16)
    bm.add(y, x, inplace=True)


def print_wrap(f):
    @wraps(f)
    def wrap(*args):
        if worker.rank == 0:
            # worker.logger.debug(*args)
            worker.logger.debug(' '.join([str(a) for a in args]))
            return f(*args)
        else:
            return worker.logger.debug(' '.join([str(a) for a in args]))
            # pass
    return wrap


class Worker:
    @timing.timeit(key='serial:init')
    @mpiprof.traceit(key='serial:init')
    def __init__(self):
        args = parse()
        self.indices = {}
        self.intracomm = MPI.COMM_WORLD
        self.rank = self.intracomm.rank

        # self.intercomm = MPI.COMM_WORLD.Split(self.rank == 0, self.rank)
        # self.intercomm = self.intercomm.Create_intercomm(0, MPI.COMM_WORLD, 1)

        self.workers = self.intracomm.size

        self.hostname = MPI.Get_processor_name()

        if args['log']:
            self.logger = MPILog(rank=self.rank, log_dir=args['logdir'])
        else:
            self.logger = MPILog(rank=self.rank)
            self.logger.disable()
        logging.debug('Initialized.')

        self.add_op_uint32 = MPI.Op.Create(c_add_uint32, commute=True)
        self.add_op_uint16 = MPI.Op.Create(c_add_uint16, commute=True)

        # master = self

        # Get the neighbors
        # self.intercomm.bcast(self.hostname, root=MPI.ROOT)
        # self.neighbors = np.empty(self.workers, dtype=int)
        # self.intercomm.Gather(self.neighbors, self.neighbors, root=MPI.ROOT)
        # self.weights = (1. + self.neighbors*add_load) / \
        #     np.sum(1. + self.neighbors*add_load)
        # print('Master, add_load: {}, weights {}'.format(add_load, self.weights))

    # Define the begin and size numbers in order to split a variable of length size
    @timing.timeit(key='serial:split')
    @mpiprof.traceit(key='serial:split')
    def split(self, size):

        counts = [size // self.workers + 1 if i < size % self.workers
                  else size // self.workers for i in range(self.workers)]
        displs = np.append([0], np.cumsum(counts[:-1])).astype(int)

        return displs[self.rank], counts[self.rank]


    # args are the buffers to fill with the gathered values
    # e.g. (comm, beam.dt, beam.dE)
    @timing.timeit(key='comm:gather')
    @mpiprof.traceit(key='comm:gather')
    def gather(self, var, size):
        if self.rank == 0:
            counts = [size // self.workers + 1 if i < size % self.workers
                      else size // self.workers for i in range(self.workers)]
            displs = np.append([0], np.cumsum(counts[:-1]))
            sendbuf = np.copy(var)
            recvbuf = np.resize(var, np.sum(counts))

            self.intracomm.Gatherv(sendbuf,
                                   [recvbuf, counts, displs, recvbuf.dtype.char], root=0)
            return recvbuf
        else:
            recvbuf = None
            self.intracomm.Gatherv(var, recvbuf, root=0)
            return var

    @timing.timeit(key='comm:allreduce')
    @mpiprof.traceit(key='comm:allreduce')
    def allreduce(self, sendbuf, recvbuf=None, dtype=np.uint32):
        if (recvbuf is None) or (sendbuf is recvbuf):
            if dtype == np.uint32:
                self.intracomm.Allreduce(
                    MPI.IN_PLACE, sendbuf, op=self.add_op_uint32)
            elif dtype == np.uint16:
                self.intracomm.Allreduce(
                    MPI.IN_PLACE, sendbuf, op=self.add_op_uint16)
        else:
            if dtype == np.uint32:
                self.intracomm.Allreduce(
                    sendbuf, recvbuf, op=self.add_op_uint32)
            elif dtype == np.uint16:
                self.intracomm.Allreduce(
                    sendbuf, recvbuf, op=self.add_op_uint16)

    @timing.timeit(key='serial:sync')
    @mpiprof.traceit(key='serial:sync')
    def sync(self):
        self.intracomm.Barrier()



class MPILog(object):
    """Class to log messages coming from other classes. Messages contain 
    {Time stamp} {Class name} {Log level} {Message}. Errors, warnings and info
    are logged into the console. To disable logging, call Logger().disable()

    Parameters
    ----------
    debug : bool
        Log DEBUG messages in 'debug.log'; default is False

    """

    def __init__(self, rank=-1, log_dir='./logs'):

        # Root logger on DEBUG level
        self.disabled = False
        self.root_logger = logging.getLogger()
        self.root_logger.setLevel(logging.DEBUG)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

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


# def init(trace=False, logfile='mpe-trace'):
#     rank = MPI.COMM_WORLD.rank
#     if trace == True:
#         mpiprof.mode = 'tracing'
#         mpiprof.init(logfile=logfile)

#     if rank != 0:
#         worker.main()
#         exit(0)

if worker is None:
    worker = Worker()


# mpi_type = {
#     'd': MPI.DOUBLE,
#     'float64': MPI.DOUBLE,
#     'numpy.float64': MPI.DOUBLE,
#     '<f8': MPI.DOUBLE,
#     'i': MPI.INT,
#     'int32': MPI.INT,
#     'numpy.int32': MPI.INT,
#     '<i4': MPI.INT,
#     'uint8': MPI.UINT8_T,
#     'B': MPI.UINT8_T,
#     '|u1': MPI.UINT8_T
# }


# task_id = {
#     'kick': np.uint8(0),
#     'drift': np.uint8(1),
#     'histo': np.uint8(2),
#     'LIKick': np.uint8(3),
#     'RFVCalc': np.uint8(4),
#     'gather': np.uint8(5),
#     'bcast': np.uint8(6),
#     'scatter': np.uint8(7),
#     'barrier': np.uint8(8),
#     'quit': np.uint8(9),
#     'switch_context': np.uint8(10),
#     'induced_voltage_sum': np.uint8(11),
#     # 'histo_and_induced_voltage': np.uint8(12),
#     'gather_single': np.uint8(13),
#     'beamFB': np.uint8(14),
#     'reduce_histo': np.uint8(15),
#     'scale_histo': np.uint8(16),
#     'LIKick_n_drift': np.uint8(17),
#     'impedance_reduction': np.uint8(18),
#     'induced_voltage_sum_packed': np.uint8(19),
#     'stop': np.uint8(255)
# }
