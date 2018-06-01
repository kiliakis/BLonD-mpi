from mpi4py import MPI
import time
import numpy as np
import sys
import os
import logging
from pyprof import timing as mpiprof
# from pyprof import mpiprof as mpiprof
from utils.bphysics_wrap import __kick, __drift, __slice, __linear_interp_kick, __rf_volt_comp
from utils import mpi_config as mpiconf
# from pyprof import timing
# import argparse
from toolbox.input_parser import parse

worker = None


class Worker:

    def __init__(self, log=None):
        rank = MPI.COMM_WORLD.rank

        self.intracomm = MPI.COMM_WORLD.Split(rank == 0, rank)
        self.intercomm = self.intracomm.Create_intercomm(0, MPI.COMM_WORLD, 0)
        self.rank = self.intracomm.rank
        self.hostname = MPI.Get_processor_name()
        if log:
            self.logger = mpiconf.MPILog(log_dir=log, rank=self.rank)
        else:
            self.logger = mpiconf.MPILog(rank=self.rank)
            self.logger.disable()

        self.logger.debug('Hostname: %s' % self.hostname)
        self.taskbuf = np.array(0, np.uint8)

    # @mpiprof.timeit(key='comm:multi_scatter')
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

    # @mpiprof.timeit(key='comm:multi_gather')
    def multi_gather(self, globs):
        vars = []
        vars = self.intercomm.bcast(vars, root=0)
        recvbuf = None
        for v in vars:
            self.intercomm.Gatherv(globs[v], recvbuf, root=0)

    # @mpiprof.timeit(key='comm:multi_bcast')
    def multi_bcast(self):
        vars = None
        vars = self.intercomm.bcast(vars, root=0)
        return vars

    @mpiprof.timeit(key='comm:recv_task')
    def recv_task(self):
        self.intercomm.Bcast(self.taskbuf, root=0)
        task = np.uint8(self.taskbuf)
        self.logger.debug('Received a %d task.' % task)
        return task


@mpiprof.timeit(key='comp:kick')
def kick():
    __kick(dt, dE, voltage, omegarf, phirf, n_rf, acc_kick)
    # comm.Barrier()


@mpiprof.timeit(key='comp:drift')
def drift():
    __drift(dt, dE, solver, t_rev, length_ratio, alpha_order,
            eta_0, eta_1, eta_2, beta, energy)
    # comm.Barrier()

# @mpiprof.timeit('comp:histo')


def histo():
    with mpiprof.timed_region('comp:histo') as tr:
        # global profile
        profile = np.empty(n_slices, dtype='d')
        __slice(dt, profile, cut_left, cut_right)

    # with mpiprof.timed_region('histo_extra') as tr:
    #     new_profile = np.empty(len(profile), dtype='d')
    #     worker.intercomm.Allreduce(profile, new_profile, op=MPI.SUM)
    #     profile = new_profile
    with mpiprof.timed_region('comm:histo_extra') as tr:
        # new_profile = np.empty(len(profile), dtype='d')
        # worker.intracomm.Allreduce(MPI.IN_PLACE, profile, op=MPI.SUM)
        recvbuf = None
        worker.intercomm.Reduce(profile, recvbuf, op=MPI.SUM, root=0)
        # profile = new_profile

    # Or even better, allreduce it


@mpiprof.timeit(key='comp:LIKick')
def LIKick():
    # print(dE, total_voltage, bin_centers, charge, acc_kick)
    __linear_interp_kick(dt, dE, total_voltage, bin_centers,
                         charge, acc_kick)
    # print(dE, total_voltage, bin_centers, charge, acc_kick)


@mpiprof.timeit(key='comp:SR')
def SR():
    __sync_rad_full(dE, U0, tau_z, n_kicks, sigma_dE, energy)


# Perhaps this is not big enough to use mpi, an omp might be better
@mpiprof.timeit(key='comp:RFVCalc')
def RFVCalc():
    __rf_volt_comp(voltage, omegarf, phirf, bin_centers,
                   rf_voltage)


@mpiprof.timeit(key='comm:gather')
def gather():
    worker.multi_gather(globals())


@mpiprof.timeit(key='comm:bcast')
def bcast():
    globals().update(worker.multi_bcast())


@mpiprof.timeit(key='comm:scatter')
def scatter():
    globals().update(worker.multi_scatter())


@mpiprof.timeit(key='comm:barrier')
def barrier():
    worker.intercomm.Barrier()


# @mpiprof.timeit(key='comm:quit')
def quit():
    sys.stdout.flush()
    sys.stderr.flush()
    worker.logger.debug('Going to disconnect()')

    worker.intercomm.Disconnect()
    sys.exit(0)


# @mpiprof.timeit(key='comm:stop')
def stop():
    pass
    # worker.logger.debug('Wating on the final barrier.')
    # worker.intercomm.Barrier()


task_dir = {
    0: kick,
    1: drift,
    2: histo,
    3: LIKick,
    4: RFVCalc,
    5: gather,
    6: bcast,
    7: scatter,
    8: barrier,
    9: quit,
    10: stop
}


def main():
    global worker
    try:
        args = parse()
        # log = 'nolog' not in sys.argv
        # report = ''
        worker = Worker(log=args.get('log', None))
        os.environ['OMP_NUM_THREADS'] = os.environ.get('OMP_NUM_THREADS', '1')
        worker.logger.debug('OMP_NUM_THREADS=%s' %
                            os.environ['OMP_NUM_THREADS'])
        start_t = time.time()

        task = worker.recv_task()
        # This is the main loop
        while task != 10:
            try:
                task_dir[task]()
            except:
                raise ValueError('Invalid task: %d.' % task)
            task = worker.recv_task()

        end_t = time.time()

        if args.get('report', None):
            mpiprof.finalize()
            mpiprof.report(total_time=1e3*(end_t-start_t),
                           out_dir=args['report'],
                           out_file='worker-%d.csv' % worker.rank)
    # Shutdown
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        worker.intercomm.Disconnect()
        exit(0)