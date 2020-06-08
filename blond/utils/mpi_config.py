import sys
import os
import numpy as np
import logging
from functools import wraps
from ..utils import bmath as bm
from mpi4py import MPI
import socket
import time

try:
    from pyprof import timing
    from pyprof import mpiprof
except ImportError:
    from ..utils import profile_mock as timing
    mpiprof = timing


worker = None


def mpiprint(*args, all=False):
    if worker.isMaster or all:
        print('[{}]'.format(worker.rank), *args)


def master_wrap(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if worker.isMaster:
            return f(*args, **kwargs)
        else:
            return None
    return wrap


def sequential_wrap(f, beam, split_args={}, gather_args={}):
    @wraps(f)
    def wrap(*args, **kw):
        beam.gather(**gather_args)
        if worker.isMaster:
            result = f(*args, **kw)
        else:
            result = None
        beam.split(**split_args)
        return result
    return wrap


class Worker:
    @timing.timeit(key='serial:init')
    @mpiprof.traceit(key='serial:init')
    def __init__(self):
        self.start_turn = 100
        self.start_interval = 500
        self.indices = {}
        self.interval = 500
        self.coefficients = {'particles': [0], 'times': [0.]}
        self.taskparallelism = False

        # Global inter-communicator
        self.intercomm = MPI.COMM_WORLD
        self.rank = self.intercomm.rank
        self.workers = self.intercomm.size

        # Setup TP intracomm
        self.hostname = MPI.Get_processor_name()
        self.hostip = socket.gethostbyname(self.hostname)

        # Create communicator with processes on the same host
        color = np.dot(np.array(self.hostip.split('.'), int)
                       [1:], [1, 256, 256**2])
        tempcomm = self.intercomm.Split(color, self.rank)
        temprank = tempcomm.rank
        # Break the hostcomm in neighboring pairs
        self.intracomm = tempcomm.Split(temprank//2, temprank)
        self.intraworkers = self.intracomm.size
        self.intrarank = self.intracomm.rank
        tempcomm.Free()
        self.log = False
        self.trace = False

    def initLog(self, log, logdir):
        self.log = log
        self.logger = MPILog(rank=self.rank, log_dir=logdir)
        if not self.log:
            self.logger.disable()

    def initTrace(self, trace, tracefile):
        self.trace = trace
        if self.trace:
            mpiprof.mode = 'tracing'
            mpiprof.init(logfile=tracefile)

    def __del__(self):
        # if self.trace:
        mpiprof.finalize()

    @property
    def isMaster(self):
        return self.rank == 0

    @property
    def isFirst(self):
        return (self.intrarank == 0) or (self.taskparallelism is False)

    @property
    def isLast(self):
        return (self.intrarank == self.intraworkers-1) or (self.taskparallelism is False)

    # Define the begin and size numbers in order to split a variable of length size

    # @timing.timeit(key='serial:split')
    # @mpiprof.traceit(key='serial:split')
    # def split(self, size):
    #     self.logger.debug('split')
    #     counts = [size // self.workers + 1 if i < size % self.workers
    #               else size // self.workers for i in range(self.workers)]
    #     displs = np.append([0], np.cumsum(counts[:-1])).astype(int)

    #     return displs[self.rank], counts[self.rank]

    # args are the buffers to fill with the gathered values
    # e.g. (comm, beam.dt, beam.dE)

    @timing.timeit(key='comm:gather')
    @mpiprof.traceit(key='comm:gather')
    def gather(self, var):
        if self.log:
            self.logger.debug('gather')

        # First I need to know the total size
        counts = np.zeros(self.workers, dtype=int)
        sendbuf = np.array([len(var)], dtype=int)
        self.intercomm.Gather(sendbuf, counts, root=0)
        total_size = np.sum(counts)

        if self.isMaster:
            # counts = [size // self.workers + 1 if i < size % self.workers
            #           else size // self.workers for i in range(self.workers)]
            displs = np.append([0], np.cumsum(counts[:-1]))
            sendbuf = np.copy(var)
            recvbuf = np.resize(var, total_size)

            self.intercomm.Gatherv(sendbuf,
                                   [recvbuf, counts, displs, recvbuf.dtype.char], root=0)
            return recvbuf
        else:
            recvbuf = None
            self.intercomm.Gatherv(var, recvbuf, root=0)
            return var

        # if self.isMaster:
        #     counts = np.empty(self.workers, int)
        #     sendbuf = np.array(len(var), int)
        #     self.intercomm.Gather(sendbuf, counts, root=0)
        #     displs = np.append([0], np.cumsum(counts[:-1]))
        #     sendbuf = np.copy(var)
        #     recvbuf = np.resize(var, np.sum(counts))

        #     self.intercomm.Gatherv(sendbuf,
        #                            [recvbuf, counts, displs, recvbuf.dtype.char], root=0)
        #     return recvbuf
        # else:
        #     recvbuf = None
        #     sendbuf = np.array(len(var), int)
        #     self.intercomm.Gather(sendbuf, recvbuf, root=0)
        #     self.intercomm.Gatherv(var, recvbuf, root=0)
        #     return var

    # All workers gather the variable var (from all workers)

    def allgather(self, var):
        if self.log:
            self.logger.debug('allgather')

        # One first gather to collect all the sizes
        counts = np.zeros(self.workers, dtype=int)
        sendbuf = np.array([len(var)], dtype=int)
        self.intercomm.Allgather(sendbuf, counts)

        total_size = np.sum(counts)
        # counts = [size // self.workers + 1 if i < size % self.workers
        #           else size // self.workers for i in range(self.workers)]
        displs = np.append([0], np.cumsum(counts[:-1]))
        sendbuf = np.copy(var)
        recvbuf = np.resize(var, total_size)

        self.intercomm.Allgatherv(sendbuf,
                                  [recvbuf, counts, displs, recvbuf.dtype.char])
        return recvbuf

    @timing.timeit(key='comm:scatter')
    @mpiprof.traceit(key='comm:scatter')
    def scatter(self, var):
        if self.log:
            self.logger.debug('scatter')
        # First broadcast the total_size from the master
        total_size = int(self.intercomm.bcast(len(var), root=0))

        # Then calculate the counts (size for each worker)
        counts = [total_size // self.workers + 1 if i < total_size % self.workers
                  else total_size // self.workers for i in range(self.workers)]

        if self.isMaster:
            displs = np.append([0], np.cumsum(counts[:-1]))
            recvbuf = np.empty(counts[worker.rank], dtype=var.dtype.char)
            self.intercomm.Scatterv([var, counts, displs, var.dtype.char],
                                    recvbuf, root=0)
        else:
            sendbuf = None
            recvbuf = np.empty(counts[worker.rank], dtype=var.dtype.char)
            self.intercomm.Scatterv(sendbuf, recvbuf, root=0)

        return recvbuf
        # if self.isMaster:
        #     counts = [size // self.workers + 1 if i < size % self.workers
        #               else size // self.workers for i in range(self.workers)]
        #     displs = np.append([0], np.cumsum(counts[:-1]))
        #     # sendbuf = np.copy(var)
        #     recvbuf = np.empty(counts[worker.rank], dtype=var.dtype.char)
        #     self.intercomm.Scatterv([var, counts, displs, var.dtype.char],
        #                             recvbuf, root=0)
        # else:
        #     counts = [size // self.workers + 1 if i < size % self.workers
        #               else size // self.workers for i in range(self.workers)]
        #     sendbuf = None
        #     recvbuf = np.empty(counts[worker.rank], dtype=var.dtype.char)
        #     self.intercomm.Scatterv(sendbuf, recvbuf, root=0)

        # return recvbuf

    # @timing.timeit(key='comm:allreduce')
    # @mpiprof.traceit(key='comm:allreduce')
    # def allreduce(self, sendbuf, recvbuf=None, dtype=np.uint32):
    #     self.logger.debug('allreduce')

    #     dtype = sendbuf.dtype.name
    #     if dtype == 'int16':
    #         op = add_op_int16
    #     elif dtype == 'int32':
    #         op = add_op_int32
    #     elif dtype == 'int64':
    #         op = add_op_int64
    #     elif dtype == 'uint16':
    #         op = add_op_uint16
    #     elif dtype == 'uint32':
    #         op = add_op_uint32
    #     elif dtype == 'uint64':
    #         op = add_op_uint64
    #     elif dtype == 'float32':
    #         op = add_op_float32
    #     elif dtype == 'float64':
    #         op = add_op_float64
    #     else:
    #         print('Error: Not recognized dtype:{}'.format(dtype))
    #         exit(-1)

    #     if (recvbuf is None) or (sendbuf is recvbuf):
    #         self.intercomm.Allreduce(MPI.IN_PLACE, sendbuf, op=op)
    #     else:
    #         self.intercomm.Allreduce(sendbuf, recvbuf, op=op)

    @timing.timeit(key='comm:reduce')
    @mpiprof.traceit(key='comm:reduce')
    def reduce(self, sendbuf, recvbuf=None, dtype=np.uint32, operator='custom_sum'):
        # supported ops:
        # sum, mean, std, max, min, prod, custom_sum
        if self.log:
            self.logger.debug('reduce')
        operator = operator.lower()
        if operator == 'custom_sum':
            dtype = sendbuf.dtype.name
            if dtype == 'int16':
                op = add_op_int16
            elif dtype == 'int32':
                op = add_op_int32
            elif dtype == 'int64':
                op = add_op_int64
            elif dtype == 'uint16':
                op = add_op_uint16
            elif dtype == 'uint32':
                op = add_op_uint32
            elif dtype == 'uint64':
                op = add_op_uint64
            elif dtype == 'float32':
                op = add_op_float32
            elif dtype == 'float64':
                op = add_op_float64
            else:
                print('Error: Not recognized dtype:{}'.format(dtype))
                exit(-1)
        elif operator == 'sum':
            op = MPI.SUM
        elif operator == 'max':
            op = MPI.MAX
        elif operator == 'min':
            op = MPI.MIN
        elif operator == 'prod':
            op = MPI.PROD
        elif operator in ['mean', 'avg']:
            op = MPI.SUM
        elif operator == 'std':
            recvbuf = self.gather(sendbuf)
            if worker.isMaster:
                assert len(recvbuf) == 3 * self.workers
                totals = np.sum((recvbuf[2::3] - 1) * recvbuf[1::3]**2 +
                                recvbuf[2::3] * (recvbuf[1::3] - bm.mean(recvbuf[0::3]))**2)
                return np.array([np.sqrt(totals / (np.sum(recvbuf[2::3]) - 1))])
            else:
                return np.array([sendbuf[1]])

        if worker.isMaster:
            if (recvbuf is None) or (sendbuf is recvbuf):
                self.intercomm.Reduce(MPI.IN_PLACE, sendbuf, op=op, root=0)
                recvbuf = sendbuf
            else:
                self.intercomm.Reduce(sendbuf, recvbuf, op=op, root=0)

            if operator in ['mean', 'avg']:
                return recvbuf / self.workers
            else:
                return recvbuf
        else:
            recvbuf = None
            self.intercomm.Reduce(sendbuf, recvbuf, op=op, root=0)
            return sendbuf

    @timing.timeit(key='comm:allreduce')
    @mpiprof.traceit(key='comm:allreduce')
    def allreduce(self, sendbuf, recvbuf=None, dtype=np.uint32, operator='custom_sum'):
        # supported ops:
        # sum, mean, std, max, min, prod, custom_sum
        if self.log:
            self.logger.debug('allreduce')
        operator = operator.lower()
        if operator == 'custom_sum':
            dtype = sendbuf.dtype.name
            if dtype == 'int16':
                op = add_op_int16
            elif dtype == 'int32':
                op = add_op_int32
            elif dtype == 'int64':
                op = add_op_int64
            elif dtype == 'uint16':
                op = add_op_uint16
            elif dtype == 'uint32':
                op = add_op_uint32
            elif dtype == 'uint64':
                op = add_op_uint64
            elif dtype == 'float32':
                op = add_op_float32
            elif dtype == 'float64':
                op = add_op_float64
            else:
                print('Error: Not recognized dtype:{}'.format(dtype))
                exit(-1)
        elif operator == 'sum':
            op = MPI.SUM
        elif operator == 'max':
            op = MPI.MAX
        elif operator == 'min':
            op = MPI.MIN
        elif operator == 'prod':
            op = MPI.PROD
        elif operator in ['mean', 'avg']:
            op = MPI.SUM
        elif operator == 'std':
            recvbuf = self.allgather(sendbuf)
            assert len(recvbuf) == 3 * self.workers
            totals = np.sum((recvbuf[2::3] - 1) * recvbuf[1::3]**2 +
                            recvbuf[2::3] * (recvbuf[1::3] - bm.mean(recvbuf[::3]))**2)
            return np.array([np.sqrt(totals / (np.sum(recvbuf[2::3]) - 1))])

        if (recvbuf is None) or (sendbuf is recvbuf):
            self.intercomm.Allreduce(MPI.IN_PLACE, sendbuf, op=op)
            recvbuf = sendbuf
        else:
            self.intercomm.Allreduce(sendbuf, recvbuf, op=op)

        if operator in ['mean', 'avg']:

            return recvbuf / self.workers
        else:
            return recvbuf

    @timing.timeit(key='serial:sync')
    @mpiprof.traceit(key='serial:sync')
    def sync(self):
        if self.log:
            self.logger.debug('sync')
        self.intercomm.Barrier()

    @timing.timeit(key='serial:intraSync')
    @mpiprof.traceit(key='serial:intraSync')
    def intraSync(self):
        if self.log:
            self.logger.debug('intraSync')
        self.intracomm.Barrier()

    @timing.timeit(key='serial:finalize')
    @mpiprof.traceit(key='serial:finalize')
    def finalize(self):
        if self.log:
            self.logger.debug('finalize')
        if not self.isMaster:
            sys.exit(0)

    @timing.timeit(key='comm:sendrecv')
    @mpiprof.traceit(key='comm:sendrecv')
    def sendrecv(self, sendbuf, recvbuf):
        if self.log:
            self.logger.debug('sendrecv')
        if self.isFirst and not self.isLast:
            self.intracomm.Sendrecv(sendbuf, dest=self.intraworkers-1, sendtag=0,
                                    recvbuf=recvbuf, source=self.intraworkers-1,
                                    recvtag=1)
        elif self.isLast and not self.isFirst:
            self.intracomm.Sendrecv(recvbuf, dest=0, sendtag=1,
                                    recvbuf=sendbuf, source=0, recvtag=0)

    @timing.timeit(key='comm:redistribute')
    @mpiprof.traceit(key='comm:redistribute')
    def redistribute(self, turn, beam, tcomp, tconst):
        self.coefficients['particles'].append(beam.n_macroparticles)
        self.coefficients['times'].append(tcomp)

        # We pass weights to the polyfit
        # The weight function I am using is:
        # e(-x/5), where x is the abs(distance) from the last
        # datapoint.
        ncoeffs = len(self.coefficients['times'])
        weights = np.exp(-(ncoeffs - 1 - np.arange(ncoeffs))/5)
        # weights = np.ones(len(self.coefficients['times']))
        # weights[-1] = np.sum(weights[:-1])
        # We model the runtime as latency * particles + c
        # where latency = p[1] and c = p[0]
        p = np.polynomial.polynomial.Polynomial.fit(
            self.coefficients['particles'],
            self.coefficients['times'],
            deg=1,
            w=weights).convert().coef
        latency = p[1]
        # assert latency != 0
        tconst += p[0]
        totalt = tcomp + tconst
        # latency = tcomp / beam.n_macroparticles
        recvbuf = np.empty(4 * self.workers, dtype=float)
        self.intercomm.Allgather(
            np.array([latency, tconst, totalt, beam.n_macroparticles]),
            recvbuf)

        latencies = recvbuf[::4]
        ctimes = recvbuf[1::4]
        totalt = recvbuf[2::4]
        Pi_old = recvbuf[3::4]

        avgt = np.mean(totalt)
        P = np.sum(Pi_old)

        # For the scheme to work I need that avgt > ctimes, if not
        # it means that a machine will be assigned negative number fo particles
        # I need to put a lower bound on the number of particles that
        # a machine can get, example 10% of the total/n_workers
        Pi = np.maximum((avgt - ctimes) / latencies, 0.1 * P/self.workers)

        # sum1 = np.sum(ctimes/latencies)
        # sum2 = np.sum(1./latencies)
        # Pi = (P + sum1 - ctimes * sum2)/(latencies * sum2)

        dPi = np.rint(Pi_old - Pi)

        for i in range(len(dPi)):
            if dPi[i] < 0 and -dPi[i] > Pi[i]:
                dPi[i] = -Pi[i]
            elif dPi[i] > Pi[i]:
                dPi[i] = Pi[i]

        # Need better definition of the cutoff
        # Maybe as a percentage of the number of particles
        # Let's say that each transaction has to be at least
        # 1% of total/n_workers
        transactions = calc_transactions(
            dPi, cutoff=0.01 * P / self.workers)[self.rank]
        if dPi[self.rank] > 0 and len(transactions) > 0:
            reqs = []
            tot_to_send = np.sum([t[1] for t in transactions])
            i = beam.n_macroparticles - tot_to_send
            for t in transactions:
                # I need to send t[1] particles to t[0]
                # buf[:t[1]] de, then dt, then id
                buf = np.empty(3*t[1], dtype=float)
                buf[0:t[1]] = beam.dE[i:i+t[1]]
                buf[t[1]:2*t[1]] = beam.dt[i:i+t[1]]
                buf[2*t[1]:3*t[1]] = beam.id[i:i+t[1]]
                i += t[1]
                # self.logger.critical(
                #     '[{}]: Sending {} parts to {}.'.format(self.rank, t[1], t[0]))
                reqs.append(self.intercomm.Isend(buf, t[0]))
            # Then I need to resize local beam.dt and beam.dE, also
            # beam.n_macroparticles
            beam.dE = beam.dE[:beam.n_macroparticles-tot_to_send]
            beam.dt = beam.dt[:beam.n_macroparticles-tot_to_send]
            beam.id = beam.id[:beam.n_macroparticles-tot_to_send]
            beam.n_macroparticles -= tot_to_send
            for req in reqs:
                req.Wait()
            # req[0].Waitall(req)
        elif dPi[self.rank] < 0 and len(transactions) > 0:
            reqs = []
            recvbuf = []
            for t in transactions:
                # I need to receive t[1] particles from t[0]
                # The buffer contains: de, dt, id
                buf = np.empty(3*t[1], float)
                recvbuf.append(buf)
                # self.logger.critical(
                #     '[{}]: Receiving {} parts from {}.'.format(self.rank, t[1], t[0]))
                reqs.append(self.intercomm.Irecv(buf, t[0]))
            for req in reqs:
                req.Wait()
            # req[0].Waitall(req)
            # Then I need to resize local beam.dt and beam.dE, also
            # beam.n_macroparticles
            tot_to_recv = np.sum([t[1] for t in transactions])
            beam.dE = np.resize(
                beam.dE, beam.n_macroparticles + tot_to_recv)
            beam.dt = np.resize(
                beam.dt, beam.n_macroparticles + tot_to_recv)
            beam.id = np.resize(
                beam.id, beam.n_macroparticles + tot_to_recv)
            i = beam.n_macroparticles
            for buf, t in zip(recvbuf, transactions):
                beam.dE[i:i+t[1]] = buf[0:t[1]]
                beam.dt[i:i+t[1]] = buf[t[1]:2*t[1]]
                beam.id[i:i+t[1]] = buf[2*t[1]:3*t[1]]
                i += t[1]
            beam.n_macroparticles += tot_to_recv

        if np.sum(np.abs(dPi))/2 < 1e-4 * P:
            self.interval = min(2*self.interval, 4000)
            return self.interval
        else:
            self.interval = self.start_interval
            return self.start_turn

    def report(self, turn, beam, tcomp, tcomm, tconst, tsync):
        latency = tcomp / beam.n_macroparticles
        if self.log:
            self.logger.critical('[{}]: Turn {}, Tconst {:g}, Tcomp {:g}, Tcomm {:g}, Tsync {:g}, Latency {:g}, Particles {:g}'.format(
                self.rank, turn, tconst, tcomp, tcomm, tsync, latency, beam.n_macroparticles))

    def greet(self):
        if self.log:
            self.logger.debug('greet')
        print('[{}]@{}: Hello World!'.format(self.rank, self.hostname))

    def print_version(self):
        if self.log:
            self.logger.debug('version')
        # print('[{}] Library version: {}'.format(self.rank, MPI.Get_library_version()))
        # print('[{}] Version: {}'.format(self.rank,MPI.Get_version()))
        print('[{}] Library: {}'.format(self.rank, MPI.get_vendor()))

    def timer_start(self, phase):
        if phase not in self.times:
            self.times[phase] = {'start': MPI.Wtime(), 'total': 0.}
        else:
            self.times[phase]['start'] = MPI.Wtime()

    def timer_stop(self, phase):
        self.times[phase]['total'] += MPI.Wtime() - self.times[phase]['start']

    def timer_reset(self, phase):
        self.times[phase] = {'start': MPI.Wtime(), 'total': 0.}

    def initDelay(self, delaystr):
        self.delay = {}
        self.delay['type'] = delaystr.split(',')[0]
        if self.delay['type'] == 'off':
            delayed_ids = np.array([], int)
        else:
            assert len(delaystr.split(',')
                       ) == 6, 'Artificial delay string missing arguments'
            init, incr, top, dcr, workers, delay = [
                int(a) for a in delaystr.split(',')]
            assert init > 0 and incr > 0 and top > 0 and dcr > 0, 'Wrong artificial delay values'
            self.delay['init'] = init
            self.delay['incr'] = incr + init
            self.delay['top'] = top + incr + init
            self.delay['dcr'] = dcr + top + incr + init

            # self.delay['init%'] = 0
            self.delay['incr%'] = delay / incr / 100
            self.delay['top%'] = delay / 100
            self.delay['dcr%'] = delay / dcr / 100
            self.delay['active%'] = 0.

            self.delay['tconst'] = 0.
            self.delay['tcomp'] = 0.
            # self.delay['tcomm'] = 0.
            # assert workers/100 * self.workers == int(workers/100)
            delayed_workers = int(np.ceil(int(workers)/100. * self.workers))

            assert delayed_workers > 0 and delayed_workers <= self.workers

            delayed_ids = np.array_split(
                np.arange(self.workers), delayed_workers)
            delayed_ids = [a[0] for a in delayed_ids]
            # delayed_ids = np.arange(self.workers)[::self.workers+1-delayed_workers]
            # delayed_ids = np.random.choice(
            # self.workers, delayed_workers, replace=False)

        if self.rank in delayed_ids:
            self.delay['delayed'] = True
        else:
            self.delay['delayed'] = False

        if self.isMaster:
            if self.log:
                self.logger.critical('[{}]: Delayed worker ids: {}'.format(
                    self.rank, ','.join(np.array(delayed_ids, str))))

    def trackDelay(self, turn):
        if self.delay['delayed']:
            modturn = turn % self.delay['dcr']

            if modturn < self.delay['init'] - 1:
                # Not need to do anything, just wait and collect stats
                pass
            elif modturn == self.delay['init'] - 1:
                # Last turn of the init interval, update the time values
                tconst = timing.get(['serial:'], exclude_lst=[
                    'serial:sync', 'serial:intraSync'])
                tcomp = timing.get(['comp:'])
                # tcomm = timing.get(['comm:'])

                self.delay['tconst'] = (
                    tconst - self.delay['tconst']) / self.delay['init'] / 1000
                self.delay['tcomp'] = (
                    tcomp - self.delay['tcomp']) / self.delay['init'] / 1000

                assert self.delay['tconst'] >= 0 and self.delay['tcomp'] >= 0
                # self.delay['tcomm'] = (
                #     tcomm - self.delay['tcomm']) / self.delay['init'] / 1000
            else:
                # Here I need to apply some delay
                # I update the active_percent and then apply the delay
                if modturn < self.delay['incr']:
                    self.delay['active%'] += self.delay['incr%']
                elif modturn < self.delay['top']:
                    pass
                elif modturn < self.delay['dcr']:
                    self.delay['active%'] -= self.delay['dcr%']
            # self.logger.critical('[{}]: tconst:{}, tcomp:{}, tcomm:{}'.format(
            #     self.rank, tconst, tcomp, tcomm))

                sleep = self.delay['active%']*self.delay['tconst']
                if sleep > 0.:
                    with timing.timed_region('serial:artificial') as tr:
                        time.sleep(sleep)

                # sleep = self.delay['active%']*self.delay['tcomm']
                # if sleep > 0.:
                #     with timing.timed_region('comm:artificial') as tr:
                #         time.sleep(sleep)

                sleep = self.delay['active%']*self.delay['tcomp']
                if sleep > 0.:
                    with timing.timed_region('comp:artificial') as tr:
                        time.sleep(sleep)

                if modturn == self.delay['dcr'] - 1:
                    # this should bring the added delay back to zero
                    assert np.isclose(self.delay['active%'], 0)
                    self.delay['tconst'] = timing.get(['serial:'], exclude_lst=[
                        'serial:sync', 'serial:intraSync'])
                    # self.delay['tcomm'] = timing.get(['comm:'])
                    self.delay['tcomp'] = timing.get(['comp:'])

    def initDLB(self, lb_type, lb_arg, n_iter):
        self.lb_turns = []
        self.lb_type = lb_type
        self.lb_arg = lb_arg
        if lb_type == 'times':
            if lb_arg != 0:
                intv = max(n_iter // (lb_arg+1), 1)
            else:
                intv = max(n_iter // (10 + 1), 1)
            self.lb_turns = np.arange(0, n_iter, intv)[1:]

        elif lb_type == 'interval':
            if lb_arg != 0:
                self.lb_turns = np.arange(0, n_iter, lb_arg)[1:]
            else:
                self.lb_turns = np.arange(0, n_iter, 1000)[1:]
        elif lb_type == 'dynamic':
            self.lb_turns = [self.start_turn]
        elif lb_type == 'reportonly':
            if lb_arg != 0:
                self.lb_turns = np.arange(0, n_iter, lb_arg)
            else:
                self.lb_turns = np.arange(0, n_iter, 100)
        self.dlb_times = {'tcomp': 0, 'tcomm': 0,
                          'tconst': 0, 'tsync': 0}
        return self.lb_turns

    def DLB(self, turn, beam):
        if turn not in self.lb_turns:
            return
        tcomp_new = timing.get(['comp:'])
        tcomm_new = timing.get(['comm:'])
        tconst_new = timing.get(['serial:'], exclude_lst=[
                                'serial:sync', 'serial:intraSync'])
        tsync_new = timing.get(['serial:sync', 'serial:intraSync'])
        if self.lb_type != 'reportonly':
            intv = self.redistribute(turn, beam,
                                     tcomp=tcomp_new-self.dlb_times['tcomp'],
                                     tconst=((tconst_new-self.dlb_times['tconst'])
                                             + (tcomm_new - self.dlb_times['tcomm'])))
        if self.lb_type == 'dynamic':
            self.lb_turns[0] += intv
        self.report(turn, beam, tcomp=tcomp_new-self.dlb_times['tcomp'],
                    tcomm=tcomm_new-self.dlb_times['tcomm'],
                    tconst=tconst_new-self.dlb_times['tconst'],
                    tsync=tsync_new-self.dlb_times['tsync'])
        self.dlb_times['tcomp'] = tcomp_new
        self.dlb_times['tcomm'] = tcomm_new
        self.dlb_times['tconst'] = tconst_new
        self.dlb_times['tsync'] = tsync_new


def calc_transactions(dpi, cutoff):
    trans = {}
    arr = []
    for i in enumerate(dpi):
        trans[i[0]] = []
        arr.append({'val': i[1], 'id':i[0]})

    # First pass is to prioritize transactions within the same node
    # basically transactions between worker i and i + 1, i: 0, 2, 4, ...
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
            i += 2
            continue
        diff = int(min(abs(arr[s]['val']), abs(arr[r]['val'])))
        if diff > cutoff:
            trans[arr[s]['id']].append((arr[r]['id'], diff))
            trans[arr[r]['id']].append((arr[s]['id'], diff))
            arr[s]['val'] -= diff
            arr[r]['val'] += diff
        i += 2
    # Then the internode transactions
    arr = sorted(arr, key=lambda x: x['val'], reverse=True)
    s = 0
    e = len(arr)-1
    while (s < e) and (arr[s]['val'] >= 0) and (arr[e]['val'] <= 0):
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
            os.makedirs(log_dir, exist_ok=True)

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


def c_add_float32(xmem, ymem, dt):
    x = np.frombuffer(xmem, dtype=np.float32)
    y = np.frombuffer(ymem, dtype=np.float32)
    bm.add(y, x, inplace=True)


add_op_float32 = MPI.Op.Create(c_add_float32, commute=True)


def c_add_float64(xmem, ymem, dt):
    x = np.frombuffer(xmem, dtype=np.float64)
    y = np.frombuffer(ymem, dtype=np.float64)
    bm.add(y, x, inplace=True)


add_op_float64 = MPI.Op.Create(c_add_float64, commute=True)


def c_add_uint16(xmem, ymem, dt):
    x = np.frombuffer(xmem, dtype=np.uint16)
    y = np.frombuffer(ymem, dtype=np.uint16)
    bm.add(y, x, inplace=True)


add_op_uint16 = MPI.Op.Create(c_add_uint16, commute=True)


def c_add_uint32(xmem, ymem, dt):
    x = np.frombuffer(xmem, dtype=np.uint32)
    y = np.frombuffer(ymem, dtype=np.uint32)
    bm.add(y, x, inplace=True)


add_op_uint32 = MPI.Op.Create(c_add_uint32, commute=True)


def c_add_uint64(xmem, ymem, dt):
    x = np.frombuffer(xmem, dtype=np.uint64)
    y = np.frombuffer(ymem, dtype=np.uint64)
    bm.add(y, x, inplace=True)


add_op_uint64 = MPI.Op.Create(c_add_uint64, commute=True)


def c_add_int16(xmem, ymem, dt):
    x = np.frombuffer(xmem, dtype=np.int16)
    y = np.frombuffer(ymem, dtype=np.int16)
    bm.add(y, x, inplace=True)


add_op_int16 = MPI.Op.Create(c_add_int16, commute=True)


def c_add_int32(xmem, ymem, dt):
    x = np.frombuffer(xmem, dtype=np.int32)
    y = np.frombuffer(ymem, dtype=np.int32)
    bm.add(y, x, inplace=True)


add_op_int32 = MPI.Op.Create(c_add_int32, commute=True)


def c_add_int64(xmem, ymem, dt):
    x = np.frombuffer(xmem, dtype=np.int64)
    y = np.frombuffer(ymem, dtype=np.int64)
    bm.add(y, x, inplace=True)


add_op_int64 = MPI.Op.Create(c_add_int64, commute=True)
