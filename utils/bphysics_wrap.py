'''
BLonD physics wrapper functions

@author Konstantinos Iliakis
@date 20.10.2017
'''

import ctypes as ct
import numpy as np
from setup_cpp import libblondphysics as __lib


def __getPointer(x):
    return x.ctypes.data_as(ct.c_void_p)


def __getLen(x):
    return ct.c_int(len(x))


def rf_volt_comp(voltages, omega_rf, phi_rf, ring):

    rf_voltage = np.zeros(len(ring.profile.bin_centers))
    __rf_volt_comp(voltages, omega_rf, phi_rf,
                   ring.profile.bin_centers, rf_voltage)

    return rf_voltage


def __rf_volt_comp(voltages, omega_rf, phi_rf,
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

    __kick(dt, dE, voltage, omegarf, phirf,
           ring.n_rf, ring.acceleration_kick[turn])


def __kick(dt, dE, voltage, omega_rf,
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
    __drift(dt, dE, ring.solver, ring.t_rev[turn],
            ring.length_ratio, ring.alpha_order,
            ring.eta_0[turn], ring.eta_1[turn],
            ring.eta_2[turn], ring.rf_params.beta[turn],
            ring.rf_params.energy[turn])


def __drift(dt, dE, solver,
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


def linear_interp_kick(ring, dt, dE, turn):
    __linear_interp_kick(dt, dE, ring.total_voltage,
                         ring.profile.bin_centers, ring.beam.Particle.charge,
                         ring.acceleration_kick[turn])
    # __lib.linear_interp_kick(__getPointer(dt),
    #                          __getPointer(dE),
    #                          __getPointer(ring.total_voltage),
    #                          __getPointer(ring.profile.bin_centers),
    #                          ct.c_double(ring.beam.Particle.charge),
    #                          ct.c_int(ring.profile.n_slices),
    #                          ct.c_int(ring.beam.n_macroparticles),
    #                          ct.c_double(ring.acceleration_kick[turn]))


def __linear_interp_kick(dt, dE, total_voltage, bin_centers,
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
    __slice(__getPointer(profile.Beam.dt),
            __getPointer(profile.n_macroparticles),
            ct.c_double(profile.cut_left),
            ct.c_double(profile.cut_right))


def __slice(dt, profile, cut_left, cut_right):
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
