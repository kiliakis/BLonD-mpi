import sys
import os
from mpi4py import MPI
import numpy as np
import logging
from functools import wraps
import socket

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


add_op_uint32 = MPI.Op.Create(c_add_uint32, commute=True)


def c_add_uint16(xmem, ymem, dt):
    x = np.frombuffer(xmem, dtype=np.uint16)
    y = np.frombuffer(ymem, dtype=np.uint16)
    bm.add(y, x, inplace=True)


add_op_uint16 = MPI.Op.Create(c_add_uint16, commute=True)


def print_wrap(f):
    @wraps(f)
    def wrap(*args):
        msg = '[{}] '.format(worker.rank) + ' '.join([str(a) for a in args])
        if worker.isMaster:
            worker.logger.debug(msg)
            return f('[{}]'.format(worker.rank), *args)
        else:
            return worker.logger.debug(msg)
    return wrap


mpiprint = print_wrap(print)


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
        self.hostip = socket.gethostbyname(self.hostname)

        # Create communicator with process on the same host
        # TODO: very rare, but possibile hash collisions are not handled
        color = np.dot(np.array(self.hostip.split('.'), int)
                       [1:], [1, 256, 256**2])
        self.hostcomm = self.intracomm.Split(color, self.rank)
        self.hostrank = self.hostcomm.rank
        self.hostworkers = self.hostcomm.size

        self.log = args['log']
        self.trace = args['trace']

        if self.log:
            self.logger = MPILog(rank=self.rank, log_dir=args['logdir'])
        else:
            self.logger = MPILog(rank=self.rank)
            self.logger.disable()

        if self.trace:
            mpiprof.mode = 'tracing'
            mpiprof.init(logfile=args['tracefile'])

    def __del__(self):
        # if self.trace:
        mpiprof.finalize()

    @property
    def isMaster(self):
        return self.rank == 0

    @property
    def isHostFirst(self):
        return self.hostrank == 0

    @property
    def isHostLast(self):
        return self.hostrank == self.hostworkers-1

    # Define the begin and size numbers in order to split a variable of length size

    @timing.timeit(key='serial:split')
    @mpiprof.traceit(key='serial:split')
    def split(self, size):
        self.logger.debug('split')
        counts = [size // self.workers + 1 if i < size % self.workers
                  else size // self.workers for i in range(self.workers)]
        displs = np.append([0], np.cumsum(counts[:-1])).astype(int)

        return displs[self.rank], counts[self.rank]

    # args are the buffers to fill with the gathered values
    # e.g. (comm, beam.dt, beam.dE)

    @timing.timeit(key='comm:gather')
    @mpiprof.traceit(key='comm:gather')
    def gather(self, var, size):
        self.logger.debug('gather')
        if self.isMaster:
            counts = np.empty(self.workers, int)
            sendbuf = np.array(len(var), int)
            self.intracomm.Gather(sendbuf, counts, root=0)
            displs = np.append([0], np.cumsum(counts[:-1]))
            sendbuf = np.copy(var)
            recvbuf = np.resize(var, np.sum(counts))

            self.intracomm.Gatherv(sendbuf,
                                   [recvbuf, counts, displs, recvbuf.dtype.char], root=0)
            return recvbuf
        else:
            recvbuf = None
            sendbuf = np.array(len(var), int)
            self.intracomm.Gather(sendbuf, recvbuf, root=0)
            self.intracomm.Gatherv(var, recvbuf, root=0)
            return var

    @timing.timeit(key='comm:scatter')
    @mpiprof.traceit(key='comm:scatter')
    def scatter(self, var, size):
        self.logger.debug('scatter')
        if self.isMaster:
            counts = [size // self.workers + 1 if i < size % self.workers
                      else size // self.workers for i in range(self.workers)]
            displs = np.append([0], np.cumsum(counts[:-1]))
            # sendbuf = np.copy(var)
            recvbuf = np.empty(counts[worker.rank], dtype=var.dtype.char)
            self.intracomm.Scatterv([var, counts, displs, var.dtype.char],
                                    recvbuf, root=0)
        else:
            counts = [size // self.workers + 1 if i < size % self.workers
                      else size // self.workers for i in range(self.workers)]
            sendbuf = None
            recvbuf = np.empty(counts[worker.rank], dtype=var.dtype.char)
            self.intracomm.Scatterv(sendbuf, recvbuf, root=0)

        return recvbuf

    @timing.timeit(key='comm:allreduce')
    @mpiprof.traceit(key='comm:allreduce')
    def allreduce(self, sendbuf, recvbuf=None, dtype=np.uint32):
        self.logger.debug('allreduce')

        if dtype == np.uint32:
            op = add_op_uint32
        elif dtype == np.uint16:
            op = add_op_uint16
        else:
            print('Error: Not recognized dtype:{}'.format(dtype))
            exit(-1)

        if (recvbuf is None) or (sendbuf is recvbuf):
            self.intracomm.Allreduce(MPI.IN_PLACE, sendbuf, op=op)
        else:
            self.intracomm.Allreduce(sendbuf, recvbuf, op=op)

    @timing.timeit(key='serial:sync')
    @mpiprof.traceit(key='serial:sync')
    def sync(self):
        self.logger.debug('sync')
        self.intracomm.Barrier()

    @timing.timeit(key='serial:hostsync')
    @mpiprof.traceit(key='serial:hostsync')
    def hostsync(self):
        self.logger.debug('hostsync')
        self.hostcomm.Barrier()

    @timing.timeit(key='serial:finalize')
    @mpiprof.traceit(key='serial:finalize')
    def finalize(self):
        self.logger.debug('finalize')
        if not self.isMaster:
            sys.exit(0)

    @timing.timeit(key='comm:sendrecv')
    @mpiprof.traceit(key='comm:sendrecv')
    def sendrecv(self, sendbuf, recvbuf):
        self.logger.debug('sendrecv')
        if self.isHostFirst and not self.isHostLast:
            self.hostcomm.Sendrecv(sendbuf, dest=self.hostworkers-1, sendtag=0,
                                   recvbuf=recvbuf, source=self.hostworkers-1,
                                   recvtag=1)
        elif self.isHostLast and not self.isHostFirst:
            self.hostcomm.Sendrecv(recvbuf, dest=0, sendtag=1,
                                   recvbuf=sendbuf, source=0, recvtag=0)

    @timing.timeit(key='comm:redistribute')
    @mpiprof.traceit(key='comm:redistribute')
    def redistribute(self, beam, time):
        latency = time / beam.n_macroparticles
        self.logger.critical('[{}]: Time {} sec.'.format(self.rank, time))
        self.logger.critical('[{}]: Latency {} sec/particle.'.format(self.rank, latency))
        recvbuf = np.empty(2 * self.workers, dtype=float)
        self.intracomm.Allgather(
            np.array([latency, beam.n_macroparticles]), recvbuf)
        latencies = recvbuf[::2]
        Pi_old = recvbuf[1::2]
        P = np.sum(Pi_old)
        Pi = P / (latencies * np.sum(1./latencies))
        dPi = np.rint(Pi_old - Pi)
        # assert np.sum(dPi) == 0, 'Particles lost while rounding'
        transactions = calc_transactions(dPi, 0.01 * P)[self.rank]
        # transactions = transactions[self.rank]
        if dPi[self.rank] > 0 and len(transactions) > 0:
            req = []
            tot_to_send = np.sum(t[1] for t in transactions)
            i = beam.n_macroparticles - tot_to_send
            for t in transactions:
                # I need to send t[1] particles to t[0]
                # buf[:t[1]] de, then dt, then id
                # print('[{}]: Sending {} p to {}.'.format(self.rank, t[1], t[0]))
                buf = np.empty(3*t[1], float)
                buf[0:t[1]] = beam.dE[i:i+t[1]]
                buf[t[1]:2*t[1]] = beam.dt[i:i+t[1]]
                buf[2*t[1]:3*t[1]] = beam.id[i:i+t[1]]
                i += t[1]
                req.append(self.intracomm.Isend(buf, t[0]))
            # Then I need to resize local beam.dt and beam.dE, also
            # beam.n_macroparticles
            beam.dE = beam.dE[:beam.n_macroparticles-tot_to_send]
            beam.dt = beam.dt[:beam.n_macroparticles-tot_to_send]
            beam.id = beam.id[:beam.n_macroparticles-tot_to_send]
            beam.n_macroparticles -= tot_to_send
            req[0].Waitall(req)
        elif dPi[self.rank] < 0 and len(transactions) > 0:
            req = []
            recvbuf = []
            for t in transactions:
                # I need to receive t[1] particles from t[0]
                # The buffer contains: de, dt, id
                # print('[{}]: Receiving {} p from {}.'.format(self.rank, t[1], t[0]))

                buf = np.empty(3*t[1], float)
                recvbuf.append(buf)
                req.append(self.intracomm.Irecv(buf, t[0]))
            req[0].Waitall(req)
            # Then I need to resize local beam.dt and beam.dE, also
            # beam.n_macroparticles
            tot_to_recv = np.sum(t[1] for t in transactions)
            beam.dE = np.resize(beam.dE, beam.n_macroparticles + tot_to_recv)
            beam.dt = np.resize(beam.dt, beam.n_macroparticles + tot_to_recv)
            beam.id = np.resize(beam.id, beam.n_macroparticles + tot_to_recv)
            # beam.dE.resize(beam.n_macroparticles + tot_to_recv)
            # beam.dt.resize(beam.n_macroparticles + tot_to_recv)
            # beam.id.resize(beam.n_macroparticles + tot_to_recv)
            i = beam.n_macroparticles
            for buf, t in zip(recvbuf, transactions):
                beam.dE[i:i+t[1]] = buf[0:t[1]]
                beam.dt[i:i+t[1]] = buf[t[1]:2*t[1]]
                beam.id[i:i+t[1]] = buf[2*t[1]:3*t[1]]
                i += t[1]
            beam.n_macroparticles += tot_to_recv
        self.logger.critical('[{}]: Tracking {} particles.'.format(self.rank, beam.n_macroparticles))
        return

    def greet(self):
        self.logger.debug('greet')
        print('[{}]@{}: Hello World!'.format(self.rank, self.hostname))

    def print_version(self):
        self.logger.debug('version')
        # print('[{}] Library version: {}'.format(self.rank, MPI.Get_library_version()))
        # print('[{}] Version: {}'.format(self.rank,MPI.Get_version()))
        print('[{}] Library: {}'.format(self.rank, MPI.get_vendor()))

    def time(self):
        return MPI.Wtime()


def calc_transactions(temp, cutoff):
    trans = {}
    for i in range(len(temp)):
        trans[i] = []
    arr = [{'val': i[1], 'id':i[0]} for i in enumerate(temp)]

    # First pass is to prioritize transactions within the same node
    i = 0
    # e = len(arr)-1
    while i < len(arr)-1:
        if (arr[i]['val'] < 0) and (arr[i+1]['val'] > 0):
            s = i+1
            r = i
        elif (arr[i]['val'] > 0) and (arr[i+1]['val'] < 0):
            s = i
            r = i+1
        else:
            i+=2
            continue
        if (arr[s]['val'] > cutoff) and (abs(arr[r]['val']) > cutoff):
            diff = int(min(abs(arr[s]['val']), abs(arr[r]['val'])))
            trans[arr[s]['id']].append((arr[r]['id'], diff))
            trans[arr[r]['id']].append((arr[s]['id'], diff))
            arr[s]['val'] -= diff
            arr[r]['val'] += diff
        i+=2
    # Then the internode transactions
    arr = sorted(arr, key=lambda x: x['val'], reverse=True)
    s = 0
    e = len(arr)-1
    while s < e:
        if arr[s]['val'] <= cutoff:
            s += 1
            continue
        if abs(arr[e]['val']) <= cutoff:
            e -= 1
            continue
        diff = int(min(abs(arr[s]['val']), abs(arr[e]['val'])))
        trans[arr[s]['id']].append((arr[e]['id'], diff))
        trans[arr[e]['id']].append((arr[s]['id'], diff))
        arr[s]['val'] -= diff
        arr[e]['val'] += diff
    # all_sum = 0
    # for k, v in trans.items():
    #     all_sum += np.sign(temp[k]) * np.sum(i[1] for i in v)
    # assert all_sum == 0, 'Particles lost while rounding'

    return trans


class MPILog(object):
    """Class to log messages coming from other classes. Messages contain 
    {Time stamp} {Class name} {Log level} {Message}. Errors, warnings and info
    are logged into the console. To disable logging, call Logger().disable()

    Parameters
    ----------
    debug : bool
        Log DEBUG messages in 'debug.log'; default is False

    """

    def __init__(self, rank=0, log_dir='./logs'):

        # Root logger on DEBUG level
        self.disabled = False
        self.root_logger = logging.getLogger()
        self.root_logger.setLevel(logging.WARNING)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_name = log_dir+'/worker-%.3d.log' % rank
        # Console handler on INFO level
        # console_handler = logging.StreamHandler()
        # console_handler.setLevel(logging.INFO)
        log_format = logging.Formatter(
            "%(asctime)s %(name)-25s %(levelname)-9s %(message)s")
        # console_handler.setFormatter(log_format)
        # self.root_logger.addHandler(console_handler)

        self.file_handler = logging.FileHandler(log_name, mode='w')
        self.file_handler.setLevel(logging.WARNING)
        self.file_handler.setFormatter(log_format)
        self.root_logger.addHandler(self.file_handler)
        logging.info("Initialized")
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

    def critical(self, string):
        if self.disabled == False:
            logging.critical(string)



if worker is None:
    worker = Worker()
