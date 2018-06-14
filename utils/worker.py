from mpi4py import MPI
import time
import numpy as np
import sys
import os
import logging
from scipy.constants import e
from pyprof import timing
from pyprof import mpiprof
from utils import bphysics_wrap as bph
from utils import mpi_config as mpiconf
from utils.input_parser import parse
from utils import bmath as bm


class Worker:

    def __init__(self, log=None):
        rank = MPI.COMM_WORLD.rank
        self.contexts = {0: {}}
        self.active = self.contexts[0]
        self.intracomm = MPI.COMM_WORLD.Split(rank == 0, rank)
        self.intercomm = self.intracomm.Create_intercomm(0, MPI.COMM_WORLD, 0)
        self.rank = self.intracomm.rank
        self.hostname = MPI.Get_processor_name()
        self.workers = self.intracomm.size
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
    # @mpiprof.traceit(key='multi_bcast')
    def multi_bcast(self):
        _vars = None
        _vars = self.intercomm.bcast(_vars, root=0)
        return _vars

    @timing.timeit(key='comm:recv_task')
    @mpiprof.traceit(key='comm:recv_task')
    def recv_task(self):
        self.intercomm.Bcast(self.taskbuf, root=0)
        task = np.uint8(self.taskbuf)
        self.logger.debug('Received a %d task.' % task)
        return task

    # @timing.timeit(key='comp:kick')
    # @mpiprof.traceit(key='kick')
    def kick(self):
        self.bcast()
        with timing.timed_region('comp:kick') as tr:
            with mpiprof.traced_region('comp:kick') as tr:
                bph._kick(dt, dE, voltage, omegarf, phirf, n_rf, acc_kick)

    # @timing.timeit(key='comp:drift')
    # @mpiprof.traceit(key='drift')
    def drift(self):
        self.bcast()
        with timing.timed_region('comp:drift') as tr:
            with mpiprof.traced_region('comp:drift') as tr:
                bph._drift(dt, dE, solver, t_rev, length_ratio, alpha_order,
                           eta_0, eta_1, eta_2, beta, energy)

    # @timing.timeit('comp:histo')
    # @mpiprof.traceit(key='histo')
    def histo(self):
        self.bcast()
        global profile

        with timing.timed_region('comp:histo') as tr:
            with mpiprof.traced_region('comp:histo') as tr:
                profile = np.empty(n_slices, dtype='d')
                bph._slice(dt, profile, cut_left, cut_right)

        with timing.timed_region('comm:histo_extra') as tr:
            with mpiprof.traced_region('comm:histo_reduce') as tr:
                # recvbuf = None
                # self.intercomm.Reduce(profile, recvbuf, op=MPI.SUM, root=0)
                # self.bcast()
                # new_profile = np.empty(len(profile), dtype='d')
                self.intracomm.Allreduce(MPI.IN_PLACE, profile, op=MPI.SUM)
                # recvbuf = None
                # basesize = len(profile) // self.workers
                # start = self.rank * basesize
                # end = (self.rank+1) * basesize
                # if end >= len(profile):
                #     end = len(profile)

                # self.intercomm.Gatherv(profile[start:end], recvbuf, root=0)

                # profile = new_profile
                # self.active.update({'profile': profile})
        # Or even better, allreduce it

    def induced_voltage_1turn(self):
        # for any per-turn updated variables
        global total_voltage
        self.bcast()

        with timing.timed_region('comp:indVolt1Turn') as tr:
            with mpiprof.traced_region('comp:indVolt1Turn') as tr:
                # Beam_spectrum_generation
                beam_spectrum = bm.rfft(profile, n_fft)

                induced_voltage = - (charge * e * beam_ratio *
                                     bm.irfft(total_impedance * beam_spectrum))
                induced_voltage = induced_voltage[:n_induced_voltage]

                total_voltage += induced_voltage[:n_slices]
                self.active.update({'total_voltage': total_voltage})

    def histo_and_induced_voltage(self):
        self.bcast()
        global profile
        global total_voltage

        with timing.timed_region('comp:histo') as tr:
            with mpiprof.traced_region('comp:histo') as tr:
                profile = np.empty(n_slices, dtype='d')
                bph._slice(dt, profile, cut_left, cut_right)

        with timing.timed_region('comm:histo_extra') as tr:
            with mpiprof.traced_region('comm:histo_reduce') as tr:
                # recvbuf = None
                # self.intercomm.Reduce(profile, recvbuf, op=MPI.SUM, root=0)
                # self.bcast()
                # new_profile = np.empty(len(profile), dtype='d')
                self.intracomm.Allreduce(MPI.IN_PLACE, profile, op=MPI.SUM)
                recvbuf = None
                basesize = len(profile) // self.workers
                start = self.rank * basesize
                end = min((self.rank+1) * basesize, len(profile))
                self.intercomm.Gatherv(profile[start:end], recvbuf, root=0)

        with timing.timed_region('comp:indVolt1Turn') as tr:
            with mpiprof.traced_region('comp:indVolt1Turn') as tr:
                # Beam_spectrum_generation
                beam_spectrum = bm.rfft(profile, n_fft)

                induced_voltage = - (charge * e * beam_ratio *
                                     bm.irfft(total_impedance * beam_spectrum))
                induced_voltage = induced_voltage[:n_induced_voltage]

                total_voltage += induced_voltage[:n_slices]
                self.active.update({'total_voltage': total_voltage})

                # profile = new_profile
                # self.active.update({'profile': profile})
        # Or even better, allreduce it

        # for any per-turn updated variables
        # self.bcast()

    # @timing.timeit(key='comp:LIKick')
    # @mpiprof.traceit(key='LIKick')
    def LIKick(self):
        self.bcast()
        with timing.timed_region('comp:LIKick') as tr:
            with mpiprof.traced_region('comp:LIKick') as tr:
                bph._linear_interp_kick(dt, dE, total_voltage, bin_centers,
                                        charge, acc_kick)

    # @timing.timeit(key='comp:SR')
    def SR(self):
        self.bcast()
        with timing.timed_region('comp:SR') as tr:
            with mpiprof.traced_region('comp:SR') as tr:
                bph._sync_rad_full(dE, U0, tau_z, n_kicks, sigma_dE, energy)

    # Perhaps this is not big enough to use mpi, an omp might be better
    # @timing.timeit(key='comp:RFVCalc')
    def RFVCalc(self):
        self.bcast()
        global total_voltage
        with timing.timed_region('comp:RFVCalc') as tr:
            with mpiprof.traced_region('comp:RFVCalc') as tr:
                rf_voltage = np.empty(len(bin_centers), dtype='d')
                bph._rf_volt_comp(voltages, omega_rf, phi_rf, bin_centers,
                                  rf_voltage)

                total_voltage += rf_voltage

    @timing.timeit(key='comm:gather')
    @mpiprof.traceit(key='comm:gather')
    def gather(self):
        self.multi_gather()

    @timing.timeit(key='comm:gather_single')
    @mpiprof.traceit(key='comm:gather_single')
    def gather_single(self):
        globals().update(self.active)
        _vars = []
        _vars = self.intercomm.bcast(_vars, root=0)
        recvbuf = None
        for v in _vars:
            basesize = len(globals()[v]) // self.workers
            start = self.rank * basesize
            end = min((self.rank+1) * basesize, len(globals()[v]))
            self.intercomm.Gatherv(globals()[v][start:end], recvbuf, root=0)

    @timing.timeit(key='comm:bcast')
    @mpiprof.traceit(key='comm:bcast')
    def bcast(self):
        self.active.update(self.multi_bcast())
        globals().update(self.active)

    @timing.timeit(key='comm:scatter')
    @mpiprof.traceit(key='comm:scatter')
    def scatter(self):
        self.active.update(self.multi_scatter())
        globals().update(self.active)

    @timing.timeit(key='comm:barrier')
    def barrier(self):
        self.intercomm.Barrier()

    # @timing.timeit(key='comm:quit')
    def quit(self):
        sys.stdout.flush()
        sys.stderr.flush()
        self.logger.debug('Going to disconnect()')
        self.intercomm.Disconnect()
        exit(0)

    @timing.timeit(key='comm:switch_context')
    @mpiprof.traceit(key='comm:switch_context')
    def switch_context(self):
        recvbuf = np.array(0, dtype='i')
        self.intercomm.Bcast(recvbuf, root=0)
        context = np.int32(recvbuf)
        if context not in self.contexts:
            self.contexts[context] = {}
        self.active = self.contexts[context]
        globals().update(self.active)

    # @timing.timeit(key='comm:stop')
    def stop(self):
        pass
        # worker.logger.debug('Wating on the final barrier.')
        # worker.intercomm.Barrier()


def main():
    # global worker
    try:
        args = parse()

        if 'omp' in args:
            os.environ['OMP_NUM_THREADS'] = str(args['omp'])

        if args.get('time', False) == True:
            timing.mode = 'timing'

        if args.get('trace', False) == True:
            mpiprof.mode = 'tracing'

        worker = Worker(log=args.get('log', None))
        task_dir = {
            0: worker.kick,
            1: worker.drift,
            2: worker.histo,
            3: worker.LIKick,
            4: worker.RFVCalc,
            5: worker.gather,
            6: worker.bcast,
            7: worker.scatter,
            8: worker.barrier,
            9: worker.quit,
            10: worker.switch_context,
            11: worker.induced_voltage_1turn,
            12: worker.histo_and_induced_voltage,
            13: worker.gather_single,
            255: worker.stop
        }

        # worker.logger.debug('OMP_NUM_THREADS=%s' %
        #                     os.environ['OMP_NUM_THREADS'])

        # Doing the first task receive manually to exclude the initialization
        # time

        worker.intercomm.Bcast(worker.taskbuf, root=0)
        task = np.uint8(worker.taskbuf)
        # task = worker.recv_task()

        start_t = time.time()

        # This is the main loop
        while task != 255:
            try:
                # getattr(worker, task_dir[task])()
                task_dir[task]()
            except Exception as e:
                print(e)
                raise AttributeError('Invalid task: %d.' % task)
            task = worker.recv_task()

        end_t = time.time()

        # worker.logger.debug(worker.contexts)
        # if args.get('trace', False) == True:
        mpiprof.finalize()

        # if args.get('time', False) == True:
        timing.report(total_time=1e3*(end_t-start_t),
                      out_dir=args['report'],
                      out_file='worker-%d.csv' % worker.rank)
    # Shutdown
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        worker.intercomm.Disconnect()
        exit(0)
