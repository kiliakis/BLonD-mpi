'''
BLonD physics wrapper functions

@author Konstantinos Iliakis
@date 20.10.2017
'''

import ctypes as ct
import numpy as np
from setup_cpp import libblondphysics as __lib

from pyprof import timing
# from pyprof import mpiprof


def __getPointer(x):
    return x.ctypes.data_as(ct.c_void_p)


def __getLen(x):
    return ct.c_int(len(x))


def beam_phase(beamFB, omegarf, phirf):
    return _beam_phase(beamFB.profile.bin_centers,
                       beamFB.profile.n_macroparticles,
                       beamFB.alpha, omegarf, phirf)
    
def _beam_phase(bin_centers, profile, alpha, omegarf, phirf):
    __lib.beam_phase.restype = ct.c_double
    coeff = __lib.beam_phase(__getPointer(bin_centers),
                             __getPointer(profile),
                             ct.c_double(alpha),
                             ct.c_double(omegarf),
                             ct.c_double(phirf),
                             __getLen(profile))
    return coeff


def rf_volt_comp(voltages, omega_rf, phi_rf, ring):

    rf_voltage = np.zeros(len(ring.profile.bin_centers))
    _rf_volt_comp(voltages, omega_rf, phi_rf,
                  ring.profile.bin_centers, rf_voltage)

    return rf_voltage


def _rf_volt_comp(voltages, omega_rf, phi_rf,
                  bin_centers, rf_voltage):

    __lib.rf_volt_comp(__getPointer(voltages),
                       __getPointer(omega_rf),
                       __getPointer(phi_rf),
                       __getPointer(bin_centers),
                       __getLen(voltages),
                       __getLen(rf_voltage),
                       __getPointer(rf_voltage))


def kick(ring, dt, dE, turn):
    voltage = np.ascontiguousarray(ring.charge * ring.voltage[:, turn])
    omegarf = np.ascontiguousarray(ring.omega_rf[:, turn])
    phirf = np.ascontiguousarray(ring.phi_rf[:, turn])

    _kick(dt, dE, voltage, omegarf, phirf,
          ring.n_rf, ring.acceleration_kick[turn])


def kick_mpi(ring, turn):

    import utils.mpi_config as mpiconf
    from mpi4py import MPI
    master = mpiconf.master
    master.bcast('kick')

    with timing.timed_region('master:kick') as tr:
        voltage_kick = np.ascontiguousarray(ring.charge*ring.voltage[:, turn])
        omegarf_kick = np.ascontiguousarray(ring.omega_rf[:, turn])
        phirf_kick = np.ascontiguousarray(ring.phi_rf[:, turn])

    vars_dict = {
        'voltage': voltage_kick,
        'omegarf': omegarf_kick,
        'phirf': phirf_kick,
        'acc_kick': ring.acceleration_kick[turn]
    }

    master.multi_bcast(vars_dict, msg=False)
    # workercomm.Barrier()


def _kick(dt, dE, voltage, omega_rf,
          phi_rf, n_rf, acc_kick):
    __lib.kick(__getPointer(dt),
               __getPointer(dE),
               ct.c_int(n_rf),
               __getPointer(voltage),
               __getPointer(omega_rf),
               __getPointer(phi_rf),
               __getLen(dt),
               ct.c_double(acc_kick))


def drift(ring, dt, dE, turn):
    _drift(dt, dE, ring.solver, ring.t_rev[turn],
           ring.length_ratio, ring.alpha_order,
           ring.eta_0[turn], ring.eta_1[turn],
           ring.eta_2[turn], ring.rf_params.beta[turn],
           ring.rf_params.energy[turn])


def drift_mpi(ring, turn):
    import utils.mpi_config as mpiconf
    from mpi4py import MPI
    # with mpiprof.timed_region('master:drift') as tr:

    master = mpiconf.master
    master.bcast('drift')
    vars_dict = {
        't_rev': ring.t_rev[turn],
        'eta_0': ring.eta_0[turn],
        'eta_1': ring.eta_1[turn],
        'eta_2': ring.eta_2[turn],
        'beta': ring.rf_params.beta[turn],
        'energy': ring.rf_params.energy[turn]
    }
    master.multi_bcast(vars_dict, msg=False)


def _drift(dt, dE, solver,
           t_rev, length_ratio, alpha_order,
           eta_0, eta_1, eta_2,
           beta, energy):

    __lib.drift(__getPointer(dt),
                __getPointer(dE),
                ct.c_char_p(solver),
                ct.c_double(t_rev),
                ct.c_double(length_ratio),
                ct.c_double(alpha_order),
                ct.c_double(eta_0),
                ct.c_double(eta_1),
                ct.c_double(eta_2),
                ct.c_double(beta),
                ct.c_double(energy),
                __getLen(dt))


def LIKick(ring, dt, dE, turn):
    _linear_interp_kick(dt, dE, ring.total_voltage,
                        ring.profile.bin_centers, ring.beam.Particle.charge,
                        ring.acceleration_kick[turn])


def LIKick_mpi(ring, turn):
    import utils.mpi_config as mpiconf
    from mpi4py import MPI
    # with mpiprof.timed_region('master:LIKick') as tr:

    master = mpiconf.master
    master.bcast('LIKick')

    vars_dict = {
        # 'total_voltage': ring.total_voltage,
        # 'bin_centers': ring.profile.bin_centers,
        # 'charge': ring.beam.Particle.charge,
        'acc_kick': ring.acceleration_kick[turn]
    }

    master.multi_bcast(vars_dict, msg=False)


def _linear_interp_kick(dt, dE, total_voltage, bin_centers,
                        charge, acc_kick):
    __lib.linear_interp_kick(__getPointer(dt),
                             __getPointer(dE),
                             __getPointer(total_voltage),
                             __getPointer(bin_centers),
                             ct.c_double(charge),
                             __getLen(bin_centers),
                             __getLen(dt),
                             ct.c_double(acc_kick))


def linear_interp_time_translation(ring, dt, dE, turn):
    pass


def slice(profile):
    _slice(__getPointer(profile.Beam.dt),
           __getPointer(profile.n_macroparticles),
           ct.c_double(profile.cut_left),
           ct.c_double(profile.cut_right))


def slice_mpi(profile):
    import utils.mpi_config as mpiconf
    from mpi4py import MPI
    # with mpiprof.timed_region('master:histo') as tr:

    master = mpiconf.master
    master.bcast('histo')

    vars_dict = {
        'cut_left': profile.cut_left,
        'cut_right': profile.cut_right
    }

    master.multi_bcast(vars_dict, msg=False)
    # zero = np.zeros(profile.n_slices, dtype='d')
    # master.reduce(zero, profile.n_macroparticles)
    # master.multi_bcast({'profile': profile.n_macroparticles}, msg=False)
    master.gather_single({'profile': profile.n_macroparticles})
    # zero = np.zeros(profile.n_slices, dtype='d')
    # profile.n_macroparticles = np.zeros(profile.n_slices, dtype='d')
    # master.intracomm.Allreduce(
    # MPI.IN_PLACE, profile.n_macroparticles, op=MPI.SUM)
    # master.intercomm.Reduce(zero, profile.n_macroparticles, op=MPI.SUM, root=MPI.)


def _slice(dt, profile, cut_left, cut_right):
    __lib.histogram(__getPointer(dt),
                    __getPointer(profile),
                    ct.c_double(cut_left),
                    ct.c_double(cut_right),
                    __getLen(profile),
                    __getLen(dt))


def slice_smooth(profile):
    __lib.smooth_histogram(__getPointer(profile.Beam.dt),
                           __getPointer(profile.n_macroparticles),
                           ct.c_double(profile.cut_left),
                           ct.c_double(profile.cut_right),
                           ct.c_int(profile.n_slices),
                           ct.c_int(profile.Beam.n_macroparticles))


def music_track(music):
    __lib.music_track(__getPointer(music.beam.dt),
                      __getPointer(music.beam.dE),
                      __getPointer(music.induced_voltage),
                      __getPointer(music.array_parameters),
                      __getLen(music.beam.dt),
                      ct.c_double(music.alpha),
                      ct.c_double(music.omega_bar),
                      ct.c_double(music.const),
                      ct.c_double(music.coeff1),
                      ct.c_double(music.coeff2),
                      ct.c_double(music.coeff3),
                      ct.c_double(music.coeff4))


def music_track_multiturn(music):
    __lib.music_track_multiturn(__getPointer(music.beam.dt),
                                __getPointer(music.beam.dE),
                                __getPointer(music.induced_voltage),
                                __getPointer(music.array_parameters),
                                __getLen(music.beam.dt),
                                ct.c_double(music.alpha),
                                ct.c_double(music.omega_bar),
                                ct.c_double(music.const),
                                ct.c_double(music.coeff1),
                                ct.c_double(music.coeff2),
                                ct.c_double(music.coeff3),
                                ct.c_double(music.coeff4))


def SR(SyncRad, turn):
    _synch_rad(SyncRad.beam.dE, SyncRad.U0,
               SyncRad.tau_z, SyncRad.n_kicks)


def _synch_rad(dE, U0, tau_z, n_kicks):
    __lib.synchrotron_radiation(
        __getPointer(dE), ct.c_double(U0 / n_kicks),
        __getLen(dE), ct.c_double(tau_z * n_kicks), ct.c_int(n_kicks))


def SR_full(SyncRad, turn):
    __sync_rad_full(SyncRad.beam.dE, SyncRad.U0,
                    SyncRad.tau_z, SyncRad.n_kicks,
                    SyncRad.sigma_dE, SyncRad.general_params.energy[0, turn],
                    SyncRad.random_array)


def SR_full_mpi(SyncRad, turn):
    _sync_rad_full(SyncRad.beam.dE, SyncRad.U0,
                   SyncRad.tau_z, SyncRad.n_kicks,
                   SyncRad.sigma_dE, SyncRad.general_params.energy[0, turn],
                   SyncRad.random_array)


def _sync_rad_full(dE, U0, tau_z, n_kicks,
                   sigma_dE, energy, random_array=None):
    if random_array == None:
        random_array = np.empty(len(dE), dtype='d')

    __lib.synchrotron_radiation_full(
        __getPointer(dE), ct.c_double(U0 / n_kicks),
        __getLen(dE), ct.c_double(sigma_dE),
        ct.c_double(tau_z * n_kicks), ct.c_double(energy),
        __getPointer(random_array), ct.c_int(n_kicks))
