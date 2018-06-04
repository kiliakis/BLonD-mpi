from mpi4py import MPI
import time
import numpy as np
import sys
import os
import logging
from pyprof import timing
# from pyprof import mpiprof
from utils.bphysics_wrap import __kick, __drift, __slice, __linear_interp_kick, __rf_volt_comp
from utils import mpi_config as mpiconf
# import argparse
from toolbox.input_parser import parse

worker = None


class Worker:

    def __init__(self, log=None):
        rank = MPI.COMM_WORLD.rank
        self.contexts = {0: {}}
        self.active = self.contexts[0]
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

    # @timing.timeit(key='comm:multi_scatter')
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

    # @timing.timeit(key='comm:multi_gather')
    def multi_gather(self):
        globals().update(self.active)
        _vars = []
        _vars = self.intercomm.bcast(_vars, root=0)
        recvbuf = None
        for v in _vars:
            self.intercomm.Gatherv(globals()[v], recvbuf, root=0)

    # @timing.timeit(key='comm:multi_bcast')
    def multi_bcast(self):
        _vars = None
        _vars = self.intercomm.bcast(_vars, root=0)
        return _vars

    @timing.timeit(key='comm:recv_task')
    def recv_task(self):
        self.intercomm.Bcast(self.taskbuf, root=0)
        task = np.uint8(self.taskbuf)
        self.logger.debug('Received a %d task.' % task)
        return task


@timing.timeit(key='comp:kick')
def kick():
    # globals().update(worker.active)
    __kick(dt, dE, voltage, omegarf, phirf, n_rf, acc_kick)
    # comm.Barrier()


@timing.timeit(key='comp:drift')
def drift():
    # globals().update(worker.active)
    __drift(dt, dE, solver, t_rev, length_ratio, alpha_order,
            eta_0, eta_1, eta_2, beta, energy)
    # comm.Barrier()


# @timing.timeit('comp:histo')
def histo():
    # globals().update(worker.active)
    with timing.timed_region('comp:histo') as tr:
        # global profile
        profile = np.empty(n_slices, dtype='d')
        __slice(dt, profile, cut_left, cut_right)

    # with timing.timed_region('histo_extra') as tr:
    #     new_profile = np.empty(len(profile), dtype='d')
    #     worker.intercomm.Allreduce(profile, new_profile, op=MPI.SUM)
    #     profile = new_profile
    with timing.timed_region('comm:histo_extra') as tr:
        # new_profile = np.empty(len(profile), dtype='d')
        # worker.intracomm.Allreduce(MPI.IN_PLACE, profile, op=MPI.SUM)
        recvbuf = None
        worker.intercomm.Reduce(profile, recvbuf, op=MPI.SUM, root=0)
        # profile = new_profile

    # Or even better, allreduce it


@timing.timeit(key='comp:LIKick')
def LIKick():
    # globals().update(worker.active)
    # print(dE, total_voltage, bin_centers, charge, acc_kick)
    __linear_interp_kick(dt, dE, total_voltage, bin_centers,
                         charge, acc_kick)
    # print(dE, total_voltage, bin_centers, charge, acc_kick)


@timing.timeit(key='comp:SR')
def SR():
    # globals().update(worker.active)
    __sync_rad_full(dE, U0, tau_z, n_kicks, sigma_dE, energy)


# Perhaps this is not big enough to use mpi, an omp might be better
@timing.timeit(key='comp:RFVCalc')
def RFVCalc():
    # globals().update(worker.active)
    __rf_volt_comp(voltage, omegarf, phirf, bin_centers,
                   rf_voltage)


@timing.timeit(key='comm:gather')
def gather():
    worker.multi_gather()


@timing.timeit(key='comm:bcast')
def bcast():
    # new_vars = worker.multi_bcast()
    # if '__id__' in new_vars:
    # _id = new_vars.get('__id__', 0)
    # if _id not in worker.contexts:
        # worker.contexts[_id] = {}
    # worker.contexts[_id].update(new_vars)
    worker.active.update(worker.multi_bcast())
    globals().update(worker.active)



@timing.timeit(key='comm:scatter')
def scatter():
    worker.active.update(worker.multi_scatter())
    globals().update(worker.active)


@timing.timeit(key='comm:barrier')
def barrier():
    worker.intercomm.Barrier()


# @timing.timeit(key='comm:quit')
def quit():
    sys.stdout.flush()
    sys.stderr.flush()
    worker.logger.debug('Going to disconnect()')
    worker.intercomm.Disconnect()
    exit(0)

@timing.timeit(key='comm:switch_context')
def switch_context():
    recvbuf = np.array(0, dtype='i')
    worker.intercomm.Bcast(recvbuf, root=0)
    context = np.int32(recvbuf)
    if context not in worker.contexts:
        worker.contexts[context] = {}
    worker.active = worker.contexts[context]
    globals().update(worker.active)


# @timing.timeit(key='comm:stop')
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
    10: switch_context,
    255: stop
}


def main():
    global worker
    try:
        args = parse()

        if 'omp' in args:
            os.environ['OMP_NUM_THREADS'] = str(args['omp'])

        worker = Worker(log=args.get('log', None))
        worker.logger.debug('OMP_NUM_THREADS=%s' %
                            os.environ['OMP_NUM_THREADS'])
        start_t = time.time()

        task = worker.recv_task()
        # This is the main loop
        while task != 255:
            try:
                task_dir[task]()
            except:
                raise ValueError('Invalid task: %d.' % task)
            task = worker.recv_task()

        end_t = time.time()
        
        # worker.logger.debug(worker.contexts)

        if args.get('report', None):
            timing.finalize()
            timing.report(total_time=1e3*(end_t-start_t),
                           out_dir=args['report'],
                           out_file='worker-%d.csv' % worker.rank)
    # Shutdown
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        worker.intercomm.Disconnect()
        exit(0)
