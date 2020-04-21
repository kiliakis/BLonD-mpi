'''
BLonD physics wrapper functions

@author Konstantinos Iliakis
@date 20.10.2017
'''

import ctypes as ct
import numpy as np
# from setup_cpp import libblondphysics as __lib
from .. import libblond as __lib


def __getPointer(x):
    return x.ctypes.data_as(ct.c_void_p)


def __getLen(x):
    return ct.c_int(len(x))


def beam_phase(beamFB, omegarf, phirf):
    return _beam_phase(beamFB.profile.bin_centers,
                       beamFB.profile.n_macroparticles,
                       beamFB.alpha, omegarf, phirf,
                       beamFB.profile.bin_size)


def _beam_phase(bin_centers, profile, alpha, omegarf, phirf, bin_size):
    __lib.beam_phase.restype = ct.c_double
    coeff = __lib.beam_phase(__getPointer(bin_centers),
                             __getPointer(profile),
                             ct.c_double(alpha),
                             ct.c_double(omegarf),
                             ct.c_double(phirf),
                             ct.c_double(bin_size),
                             __getLen(profile))
    return coeff


def rf_volt_comp(voltages, omega_rf, phi_rf, bin_centers):

    rf_voltage = np.zeros(len(bin_centers))

    __lib.rf_volt_comp(__getPointer(voltages),
                       __getPointer(omega_rf),
                       __getPointer(phi_rf),
                       __getPointer(bin_centers),
                       __getLen(voltages),
                       __getLen(rf_voltage),
                       __getPointer(rf_voltage))
    return rf_voltage


def kick(dt, dE, voltage, omega_rf, phi_rf, charge, n_rf, acceleration_kick):
    voltage_kick = np.ascontiguousarray(charge*voltage)
    omegarf_kick = np.ascontiguousarray(omega_rf)
    phirf_kick = np.ascontiguousarray(phi_rf)

    __lib.kick(__getPointer(dt),
               __getPointer(dE),
               ct.c_int(n_rf),
               __getPointer(voltage_kick),
               __getPointer(omegarf_kick),
               __getPointer(phirf_kick),
               __getLen(dt),
               ct.c_double(acceleration_kick))


def drift(dt, dE, solver, t_rev, length_ratio, alpha_order, eta_0,
          eta_1, eta_2, alpha_0, alpha_1, alpha_2, beta, energy):

    __lib.drift(__getPointer(dt),
                __getPointer(dE),
                ct.c_char_p(solver),
                ct.c_double(t_rev),
                ct.c_double(length_ratio),
                ct.c_double(alpha_order),
                ct.c_double(eta_0),
                ct.c_double(eta_1),
                ct.c_double(eta_2),
                ct.c_double(alpha_0),
                ct.c_double(alpha_1),
                ct.c_double(alpha_2),
                ct.c_double(beta),
                ct.c_double(energy),
                __getLen(dt))


def linear_interp_kick(dt, dE, voltage,
                       bin_centers, charge,
                       acceleration_kick):
    __lib.linear_interp_kick(__getPointer(dt),
                             __getPointer(dE),
                             __getPointer(voltage),
                             __getPointer(bin_centers),
                             ct.c_double(charge),
                             __getLen(bin_centers),
                             __getLen(dt),
                             ct.c_double(acceleration_kick))


def linear_interp_kick_n_drift(dt, dE, total_voltage, bin_centers, charge, acc_kick,
                               solver, t_rev, length_ratio, alpha_order, eta_0, eta_1,
                               eta_2, beta, energy):
    __lib.linear_interp_kick_n_drift(__getPointer(dt),
                                     __getPointer(dE),
                                     __getPointer(total_voltage),
                                     __getPointer(bin_centers),
                                     __getLen(bin_centers),
                                     __getLen(dt),
                                     ct.c_double(acc_kick),
                                     ct.c_char_p(solver),
                                     ct.c_double(t_rev),
                                     ct.c_double(length_ratio),
                                     ct.c_double(alpha_order),
                                     ct.c_double(eta_0),
                                     ct.c_double(eta_1),
                                     ct.c_double(eta_2),
                                     ct.c_double(beta),
                                     ct.c_double(energy),
                                     ct.c_double(charge))


def linear_interp_time_translation(ring, dt, dE, turn):
    pass


def slice(profile):
    __lib.histogram(__getPointer(profile.Beam.dt),
                    __getPointer(profile.n_macroparticles),
                    ct.c_double(profile.cut_left),
                    ct.c_double(profile.cut_right),
                    ct.c_int(profile.n_slices),
                    ct.c_int(profile.Beam.n_macroparticles))


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


def synchrotron_radiation(SyncRad, turn):
    __lib.synchrotron_radiation(
        __getPointer(SyncRad.beam.dE),
        ct.c_double(SyncRad.U0 / SyncRad.n_kicks),
        ct.c_int(SyncRad.beam.n_macroparticles),
        ct.c_double(SyncRad.tau_z * SyncRad.n_kicks),
        ct.c_int(SyncRad.n_kicks))


def synchrotron_radiation_full(SyncRad, turn):
    __lib.synchrotron_radiation_full(
        __getPointer(SyncRad.beam.dE),
        ct.c_double(SyncRad.U0 / SyncRad.n_kicks),
        ct.c_int(SyncRad.beam.n_macroparticles),
        ct.c_double(SyncRad.sigma_dE),
        ct.c_double(SyncRad.tau_z * SyncRad.n_kicks),
        ct.c_double(SyncRad.general_params.energy[0, turn]),
        __getPointer(SyncRad.random_array),
        ct.c_int(SyncRad.n_kicks))
