import sys
from mpi4py import MPI
import numpy as np
import logging


mpi_type = {
    'd': MPI.DOUBLE,
    'float64': MPI.DOUBLE,
    'numpy.float64': MPI.DOUBLE,
    '<f8': MPI.DOUBLE,
    'i': MPI.INT,
    'int32': MPI.INT,
    'numpy.int32': MPI.INT,
    '<i4': MPI.INT
}

_worker_script = sys.path[0] + '/worker.py'

_exec = 'python'
_args = []

_debug_exec = 'xterm'
_debug_args = ['-e', 'python', '-m', 'pdb']


master = None

class Master:
    def __init__(self, log=True):
        self.intercomm = None
        self.intracomm = MPI.COMM_WORLD

        if self.intracomm.size != 1:
            'Only one process can be the master!\nRe-run with only 1 process.'    

        self.workers = 0
        self.hostname = MPI.Get_processor_name()
        self.logger = MPILog(rank=-1)
        if log == False:
            self.logger.disable()
        logging.debug('Initialized.')
        global master
        master = self

    def spawn_workers(self, workers=1, worker_script=_worker_script,
                      debug=False, args=None):
        if args:
            self.args = args
        elif debug == False:
            self.args = _args
            self.exec = _exec
        else:
            self.args = _debug_args
            self.exec = _debug_exec

        self.intercomm = self.intracomm.Spawn(self.exec,
                                              args=self.args + [worker_script],
                                              maxprocs=workers)

        self.workers = self.intercomm.Get_remote_size()
        logging.debug('%d workers successfully initialized.' % self.workers)

    def multi_scatter(self, vars):
        self.intercomm.bcast('scatter', root=MPI.ROOT)

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
    def multi_gather(self, gather_dict):
        self.intercomm.bcast('gather', root=MPI.ROOT)
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

    def multi_bcast(self, vars):
        logging.debug('Broadcasting variables')
        self.intercomm.bcast('bcast', root=MPI.ROOT)
        self.intercomm.bcast(vars, root=MPI.ROOT)

    def stop(self):
        logging.debug('Sending a stop signal')
        self.intercomm.bcast('stop', root=MPI.ROOT)
        logging.debug('Waiting on the barrier')
        self.intercomm.Barrier()

    def sync(self):
        self.intercomm.bcast('barrier', root=MPI.ROOT)
        self.intercomm.Barrier()

    def disconnect(self):
        self.intercomm.Disconnect()


class Worker:
    def __init__(self, log=True):
        try:
            # Connect to parent
            self.intercomm = MPI.Comm.Get_parent()
            self.rank = self.intercomm.Get_rank()
            self.hostname = MPI.Get_processor_name()
        except:
            raise ValueError('Could not connect to parent')

        self.logger = MPILog(rank=self.rank)
        if log==False:
            self.logger.disable()
        sys.stdout = open('stdout-worker-%.3d.txt' % self.rank, 'w')
        sys.stderr = open('stderr-worker-%.3d.txt' % self.rank, 'w')
        self.intracomm = MPI.COMM_WORLD
        logging.debug('Hostname: %s' % self.hostname)

    def multi_scatter(self):
        var_list = None
        var_list = self.intercomm.bcast(var_list, root=0)
        vals = {}
        sendbuf = None
        for name, dtype in var_list:
            count = np.array(0, dtype='i')
            self.intercomm.Scatter(sendbuf, count, root=0)
            recvbuf = np.empty(count, dtype=dtype)
            self.intercomm.Scatterv(sendbuf, recvbuf, root=0)
            vals[name] = recvbuf
        return vals
        # self.intercomm.Barrier()

    def multi_gather(self, globs):
        vars = []
        vars = self.intercomm.bcast(vars, root=0)
        recvbuf = None
        for v in vars:
            self.intercomm.Gatherv(globs[v], recvbuf, root=0)

    def multi_bcast(self):
        vars = None
        vars = self.intercomm.bcast(vars, root=0)
        return vars


class MPILog(object):
    """Class to log messages coming from other classes. Messages contain 
    {Time stamp} {Class name} {Log level} {Message}. Errors, warnings and info
    are logged into the console. To disable logging, call Logger().disable()

    Parameters
    ----------
    debug : bool
        Log DEBUG messages in 'debug.log'; default is False

    """

    def __init__(self, rank=-1):

        # Root logger on DEBUG level
        self.root_logger = logging.getLogger()
        self.root_logger.setLevel(logging.DEBUG)
        if rank < 0:
            log_name = 'master.log'
        else:
            log_name = 'worker-%.3d.log' % rank
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