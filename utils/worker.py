from mpi4py import MPI
import time
import numpy as np
import sys
import os
import logging
from collections import deque

from scipy.constants import e
from pyprof import timing
from pyprof import mpiprof
from utils import bphysics_wrap as bph
from utils import mpi_config as mpiconf
from utils.input_parser import parse
from utils import bmath as bm


def c_add(xmem, ymem, dt):
    x = np.frombuffer(xmem, dtype=np.int32)
    y = np.frombuffer(ymem, dtype=np.int32)
    bm.add(y, x, inplace=True)
    # y[:] = bm.add(x, y)


add_op = MPI.Op.Create(c_add, commute=True)


def c_add_uint16(xmem, ymem, dt):
    x = np.frombuffer(xmem, dtype=np.uint16)
    y = np.frombuffer(ymem, dtype=np.uint16)
    # y[:] = bm.add(x, y)
    bm.add(y, x, inplace=True)


add_op_uint16 = MPI.Op.Create(c_add_uint16, commute=True)


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
        self.task_queue = deque()
        self.turn = 0

        self.master_hostname = None
        self.master_hostname = self.intercomm.bcast(
            self.master_hostname, root=0)

        # Send the neighbors array to master
        temp = np.array([self.hostname == self.master_hostname], int)
        self.intercomm.Gather(temp, temp, root=0)
        # if self.hostname == self.master_hostname:
        #     os.environ['OMP_NUM_THREADS'] = '20'

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
            # print('Worker {}: len {}: {}'.format(self.rank, name, len(recvbuf)))

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
        if(len(self.task_queue) == 0):
            # receive new tasks
            tasks = None
            tasks = self.intercomm.bcast(tasks, root=0)
            self.task_queue.extend(tasks)
            # self.logger.debug('Received a %d task.' % task)

        task = self.task_queue.popleft()
        self.logger.debug('Returning a %d task.' % task)
        return task

    @timing.timeit(key='overhead:update')
    @mpiprof.traceit(key='overhead:update')
    def update(self):
        self.active.update(globals())

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

    @timing.timeit(key='overhead:switch_context')
    @mpiprof.traceit(key='overhead:switch_context')
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

    # @timing.timeit(key='comp:kick')
    # @mpiprof.traceit(key='kick')
    def kick(self):
        # self.bcast()
        turn = self.turn
        with timing.timed_region('comp:kick'):
            with mpiprof.traced_region('comp:kick'):
                voltage = np.ascontiguousarray(charge * rfp_voltage[:, turn])
                omegarf = np.ascontiguousarray(rfp_omega_rf[:, turn])
                phirf = np.ascontiguousarray(rfp_phi_rf[:, turn])
                bph._kick(dt, dE, voltage, omegarf, phirf,
                          n_rf, tracker_acc_kick[turn])
        self.update()

    # @timing.timeit(key='comp:drift')
    # @mpiprof.traceit(key='drift')
    def drift(self):
        # self.bcast()
        turn = self.turn + 1
        with timing.timed_region('comp:drift'):
            with mpiprof.traced_region('comp:drift'):
                bph._drift(dt, dE, solver, tracker_t_rev[turn],
                           length_ratio, alpha_order,
                           tracker_eta_0[turn], tracker_eta_1[turn],
                           tracker_eta_2[turn], rfp_beta[turn], rfp_energy[turn])
        self.turn += 1
        self.update()

    # @timing.timeit('comp:histo')
    # @mpiprof.traceit(key='histo')
    def histo(self):
        # self.bcast()
        global profile

        with timing.timed_region('comp:histo'):
            with mpiprof.traced_region('comp:histo'):
                profile = np.empty(n_slices, dtype='d')
                bph._slice(dt, profile, cut_left, cut_right)
        # Or even better, allreduce it
        self.update()

    # @timing.timeit(key='comm:histo_reduce')
    # @mpiprof.traceit(key='comm:histo_reduce')
    # def reduce_histo(self):

    #     global profile
    #     with timing.timed_region('comm:conversions'):
    #         with mpiprof.traced_region('comm:conversions'):
    #             profile = profile.astype(np.int32, order='C')

    #     with timing.timed_region('comm:histo_reduce'):
    #         with mpiprof.traced_region('comm:histo_reduce'):
    #             self.intracomm.Allreduce(MPI.IN_PLACE, profile, op=add_op)
    #             # self.intracomm.Allreduce(MPI.IN_PLACE, profile, op=MPI.SUM)

    #     with timing.timed_region('comm:conversions'):
    #         with mpiprof.traced_region('comm:conversions'):
    #             profile = profile.astype(np.float64, order='C')

    #     self.update()

    def reduce_histo(self):

        global profile
        with timing.timed_region('comm:conversions'):
            with mpiprof.traced_region('comm:conversions'):
                profile = profile.astype(np.uint16, order='C')

        with timing.timed_region('comm:histo_reduce'):
            with mpiprof.traced_region('comm:histo_reduce'):
                self.intracomm.Allreduce(
                    MPI.IN_PLACE, profile, op=add_op_uint16)
                # self.intracomm.Allreduce(MPI.IN_PLACE, profile, op=MPI.SUM)

        with timing.timed_region('comm:conversions'):
            with mpiprof.traced_region('comm:conversions'):
                profile = profile.astype(np.float64, order='C')

        self.update()

    @timing.timeit(key='comm:histo_scale')
    @mpiprof.traceit(key='comm:histo_scale')
    def scale_histo(self):
        global profile
        profile *= self.workers
        # self.intracomm.Allreduce(MPI.IN_PLACE, profile, op=MPI.SUM)

    def induced_voltage_sum_packed(self):
        # for any per-turn updated variables
        global induced_voltage
        # self.bcast()
        beam_spectrum = None
        induced_voltage = []
        min_idx = n_slices

        with timing.timed_region('serial:indVoltSum'):
            with mpiprof.traced_region('serial:indVoltSum'):
                for imped in impedList.values():
                    # Beam_spectrum_generation
                    if beam_spectrum is None:
                        #  with timing.timed_region('serial:indVoltRfft'):
                        beam_spectrum = bm.rfft(profile, imped['n_fft'])

                    #  with timing.timed_region('serial:indVoltMul1'):
                    induced_voltage.append(
                        bm.mul(imped['total_impedance'], beam_spectrum))
                    min_idx = min(imped['n_induced_voltage'], min_idx)

                #  with timing.timed_region('serial:indVoltIrfft'):
                induced_voltage = bm.irfft_packed(induced_voltage)[:, :min_idx]
                #  with timing.timed_region('serial:indVoltMul2'):
                induced_voltage = -charge * e * beam_ratio * induced_voltage
                # for i in range(len(induced_voltage)):
                #     induced_voltage[i] = bm.mul(induced_voltage[i], -charge * e * beam_ratio)
                #  with timing.timed_region('serial:indVoltAcc'):
                induced_voltage = np.sum(induced_voltage, axis=0)

        self.update()

    def induced_voltage_sum(self):
        # for any per-turn updated variables
        global induced_voltage, n_slices, profile, beam_ration, charge, \
            impedList, induced_voltage

        temp_induced_voltage = 0
        beam_spectrum = None
        min_idx = n_slices

        for imped in impedList.values():
            # Beam_spectrum_generation
            if 'type' not in imped:
                induced_voltage, beam_spectrum = self.inducedVoltage1Turn(
                    imped, beam_spectrum)
            elif imped['type'] == 'inductive':
                induced_voltage = self.inducedVoltageInductive(imped)
            elif imped['type'] == 'mtw':
                induced_voltage, beam_spectrum = self.inducedVoltageMTW(
                    imped, beam_spectrum)
            else:
                self.logger.debug(
                    'Unrecognized impedance type: {}'.format(imped['type']))
                raise RuntimeError('Unrecognized impedance type.')

            with timing.timed_region('serial:indVoltSum'):
                with mpiprof.traced_region('serial:indVoltSum'):
                    min_idx = min(imped['n_induced_voltage'], min_idx)
                    temp_induced_voltage += induced_voltage[:min_idx]

        induced_voltage = temp_induced_voltage
        self.update()

    def inducedVoltageMTW(self, imped, beam_spectrum=None):

        global tracker_t_rev
        turn = self.turn

        if imped['mtw_mode'] == 'time':
            with timing.timed_region('serial:indVoltMTW'):
                with mpiprof.traced_region('serial:indVoltMTW'):
                    imped['mtw_memory'] = bm.interp_const_space(
                        imped['time_mtw'] + tracker_t_rev[turn],
                        imped['time_mtw'],
                        imped['mtw_memory'],
                        left=0, right=0)

            induced_voltage, beam_spectrum = self.inducedVoltage1Turn(
                imped, beam_spectrum)
            with timing.timed_region('serial:indVoltMTW'):
                with mpiprof.traced_region('serial:indVoltMTW'):
                    induced_voltage[(imped['n_induced_voltage'] -
                                     imped['front_wake_buffer']):] = 0
                    imped['mtw_memory'][:imped['n_induced_voltage']] += \
                        induced_voltage
                    induced_voltage = imped['mtw_memory'][:imped['n_induced_voltage']]
            return induced_voltage, beam_spectrum

        elif imped['mtw_mode'] == 'freq':
            self.logger.debug('The freq mtw mode is not yet implemented.')
            raise RuntimeError('The freq mtw mode is not yet implemented.')
        else:
            self.logger.debug(
                'Unrecognized mtw mode: {}'.format(imped['mtw_mode']))
            raise RuntimeError('Unrecognized mtw mode.')

    def inducedVoltageInductive(self, imped):
        global charge, beam_ratio, Z_over_n, tracker_t_rev, bin_size, deriv_mode
        turn = self.turn
        with timing.timed_region('serial:indVoltInductive'):
            with mpiprof.traced_region('serial:indVoltInductive'):
                induced_voltage = - (charge * e / (2 * np.pi) *
                                     beam_ratio * imped['Z_over_n'][turn] *
                                     tracker_t_rev[turn] / bin_size *
                                     self.beam_profile_derivative(imped['deriv_mode'])[1])

        # self.induced_voltage = induced_voltage[:self.n_induced_voltage]
        return induced_voltage[:imped['n_induced_voltage']]

    def inducedVoltage1Turn(self, imped, beam_spectrum=None):
        # for any per-turn updated variables
        global induced_voltage, n_slices, profile, beam_ration, charge, \
            impedList, induced_voltage
        # self.bcast()

        with timing.timed_region('serial:indVolt1Turn'):
            with mpiprof.traced_region('serial:indVolt1Turn'):
                if beam_spectrum is None:
                    beam_spectrum = bm.rfft(profile, imped['n_fft'])

                induced_voltage = - (charge * e * beam_ratio *
                                     bm.irfft(imped['total_impedance'] * beam_spectrum))
        return induced_voltage[:imped['n_induced_voltage']], beam_spectrum

    # @timing.timeit(key='comp:LIKick')
    # @mpiprof.traceit(key='LIKick')
    def LIKick(self):
        # self.bcast()
        turn = self.turn
        with timing.timed_region('comp:LIKick'):
            with mpiprof.traced_region('comp:LIKick'):
                bph._linear_interp_kick(dt, dE, total_voltage, bin_centers,
                                        charge, tracker_acc_kick[turn])
        self.update()

    def LIKick_n_drift(self):
        # self.bcast()
        # global acc_kick
        turn = self.turn
        with timing.timed_region('comp:LIKick_n_drift'):
            with mpiprof.traced_region('comp:LIKick_n_drift'):
                bph._LIKick_n_drift(dt, dE, total_voltage, bin_centers,
                                    charge, tracker_acc_kick[turn], solver,
                                    tracker_t_rev[turn], length_ratio,
                                    alpha_order, tracker_eta_0[turn],
                                    tracker_eta_1[turn], tracker_eta_2[turn],
                                    rfp_beta[turn], rfp_energy[turn])
        # update the turn
        self.turn += 1
        self.update()

    # @timing.timeit(key='comp:SR')
    def SR(self):
        self.bcast()
        with timing.timed_region('comp:SR'):
            with mpiprof.traced_region('comp:SR'):
                bph._sync_rad_full(dE, U0, tau_z, n_kicks, sigma_dE, energy)
        self.update()

    # Perhaps this is not big enough to use mpi, an omp might be better
    # @timing.timeit(key='comp:RFVCalc')
    def RFVCalc(self):
        # self.bcast()
        global total_voltage, induced_voltage
        total_voltage = 0.
        turn = self.turn
        with timing.timed_region('serial:RFVCalc'):
            with mpiprof.traced_region('serial:RFVCalc'):
                voltages = np.ascontiguousarray(rfp_voltage[:, turn])
                omega_rf = np.ascontiguousarray(rfp_omega_rf[:, turn])
                phi_rf = np.ascontiguousarray(rfp_phi_rf[:, turn])
                rf_voltage = np.zeros(len(bin_centers), dtype='d')

                bph._rf_volt_comp(voltages, omega_rf, phi_rf, bin_centers,
                                  rf_voltage)

                total_voltage = rf_voltage + induced_voltage
        self.update()

    # @timing.timeit(key='comp:RFVCalc')
    def impedance_reduction(self):
        # self.bcast()
        turn = self.turn
        with timing.timed_region('serial:imped_red'):
            with mpiprof.traced_region('serial:imped_red'):
                for impRed in impedanceReduction:
                    impedList[impRed['impedance']]['total_impedance'][impRed['affected_indices']] = impRed['initial_impedance'][impRed['affected_indices']] \
                        * impRed['filter_func']**(impRed['reduction_factor'][turn]
                                                  * impRed['FB_strength'])

        self.update()

    def beamFB(self):
        # self.bcast()
        global lhc_y, rfp_dphi_rf, rfp_omega_rf, rfp_phi_rf, machine, time_offset
        turn = self.turn
        with timing.timed_region('serial:beamFB'):
            with mpiprof.traced_region('serial:beamFB'):
                if machine == 'LHC':
                    coeff = bph._beam_phase(
                        bin_centers, profile, alpha,
                        rfp_omega_rf[0, turn],
                        rfp_phi_rf[0, turn],
                        bin_size)

                    phi_beam = np.arctan(coeff) + np.pi

                    dphi = phi_beam - rfp_phi_s[turn]

                    if len(globals().get('lhc_noise_dphi', [])) > 0:
                        dphi += lhc_noise_dphi[turn]

                    domega_rf = - gain*dphi - gain2 * \
                        (lhc_y + lhc_a[turn] *
                         (rfp_dphi_rf[0] + reference))

                    lhc_y = (1 - lhc_t[turn]) * lhc_y + \
                        (1 - lhc_a[turn]) * lhc_t[turn] * \
                        (rfp_dphi_rf[0] + reference)

                    # Update the RF frequency of all systems for the next turn
                    turn = turn + 1
                    rfp_omega_rf[:, turn] += domega_rf * \
                        rfp_harmonic[:, turn] / \
                        rfp_harmonic[0, turn]

                    # Update the RF phase of all systems for the next turn
                    # Accumulated phase offset due to PL in each RF system
                    rfp_dphi_rf += 2.*np.pi*rfp_harmonic[:, turn] * \
                        (rfp_omega_rf[:, turn] -
                         rfp_omega_rf_d[:, turn]) / \
                        rfp_omega_rf_d[:, turn]

                    # Total phase offset
                    rfp_phi_rf[:, turn] += rfp_dphi_rf
                elif machine == 'SPS_F':
                    if alpha != 0.0:
                        indexes = np.logical_and(
                            (time_offset - np.pi /
                             rfp_omega_rf[0, turn]) <= bin_centers,
                            bin_centers <= (-1/alpha + time_offset -
                                            2 * np.pi / rfp_omega_rf[0, turn]))
                    else:
                        indexes = np.ones(n_slices, dtype=bool)

                    scoeff = np.trapz(np.sin(rfp_omega_rf[0, turn]*bin_centers[indexes]
                                             + rfp_phi_rf[0, turn])
                                      * profile[indexes], dx=bin_size)

                    ccoeff = np.trapz(np.cos(rfp_omega_rf[0, turn]*bin_centers[indexes]
                                             + rfp_phi_rf[0, turn])
                                      * profile[indexes], dx=bin_size)

                    # Project beam phase to (pi/2,3pi/2) range
                    phi_beam = np.arctan(scoeff/ccoeff) + np.pi

                    dphi = phi_beam - rfp_phi_s[turn]

                    domega_dphi = - gain * dphi
                    domega_df = -gain2*(rfp_omega_rf[0, turn]
                                        - rfp_omega_rf_d[0, turn])

                    domega_rf = domega_dphi + domega_df

                    # Update the RF frequency of all systems for the next turn
                    turn = turn + 1
                    rfp_omega_rf[:, turn] += domega_rf * \
                        rfp_harmonic[:, turn] / \
                        rfp_harmonic[0, turn]

                    # Update the RF phase of all systems for the next turn
                    # Accumulated phase offset due to PL in each RF system
                    rfp_dphi_rf += 2.*np.pi*rfp_harmonic[:, turn] * \
                        (rfp_omega_rf[:, turn] -
                         rfp_omega_rf_d[:, turn]) / \
                        rfp_omega_rf_d[:, turn]

                    # Total phase offset
                    rfp_phi_rf[:, turn] += rfp_dphi_rf

                else:
                    self.logger.debug('Unrecognized machine type: %s' % machine)

        self.update()

    def beam_profile_derivative(self, mode='gradient'):
        """
        The input is one of the three available methods for differentiating
        a function. The two outputs are the bin centres and the discrete
        derivative of the Beam profile respectively.*
        """
        from scipy import ndimage
        global bin_centers, profile
        dist_centers = bin_centers[1] - bin_centers[0]

        if mode == 'filter1d':
            derivative = ndimage.gaussian_filter1d(profile, sigma=1, order=1,
                                                   mode='wrap') / dist_centers
        elif mode == 'gradient':
            derivative = np.gradient(profile, dist_centers)
        elif mode == 'diff':
            derivative = np.diff(profile) / dist_centers
            diffCenters = bin_centers[0:-1] + dist_centers/2
            derivative = np.interp(bin_centers, diffCenters, derivative)
        else:
            self.logger.debug(
                'Option for derivative is not recognized: {}'.format(mode))
            raise RuntimeError('Option for derivative is not recognized.')

        return bin_centers, derivative


def main():
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
            11: worker.induced_voltage_sum,
            # 12: worker.histo_and_induced_voltage,
            13: worker.gather_single,
            14: worker.beamFB,
            15: worker.reduce_histo,
            16: worker.scale_histo,
            17: worker.LIKick_n_drift,
            18: worker.impedance_reduction,
            19: worker.induced_voltage_sum_packed,
            255: worker.stop
        }

        # worker.logger.debug('OMP_NUM_THREADS=%s' %
        #                     os.environ['OMP_NUM_THREADS'])

        task = worker.recv_task()

        # This is the main loop
        start_t = time.time()
        while task != 255:
            try:
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
