# Copyright 2016 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/


'''
Microwave instability in FCC-ee the Z machine with parameters table from 
October 2017.

:Authors: **Ivan Karpov**
'''

from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from input_parameters.ring import Ring
from beam.beam import Beam, Electron
from beam.distributions import bigaussian
from input_parameters.rf_parameters import RFStation
from beam.profile import Profile
from trackers.tracker import RingAndRFTracker, FullRingAndRF
from synchrotron_radiation.synchrotron_radiation import SynchrotronRadiation
from impedances.impedance import InducedVoltageFreq
from impedances.impedance import TotalInducedVoltage
from impedances.impedance_sources import ResistiveWall

from scipy.constants import c, e, m_e
from beam.profile import CutOptions
import yaml
import datetime
from setup_cpp import libblond
import ctypes



# SIMULATION PARAMETERS -------------------------------------------------------

# Import beam parameters from the yaml file
# Read beam parameters
fcc_machine = r'Z.yaml'
with open(fcc_machine, 'r') as inputfile:
    fcc_params = yaml.load(inputfile)


particle_type = Electron()
n_particles = int(float(fcc_params['Np']))
print(r"Bunch intensity %1.2e" % n_particles)
n_macroparticles = int(1e6)
sync_momentum = float(fcc_params['Eb'])  # [eV]


distribution_type = 'gaussian'
emittance = 0.0013
distribution_variable = 'Action'

# Machine and RF parameters
h = int(fcc_params['h'])  # harmonic number
frf = float(fcc_params['frf'])  # RF frequency
C = h*c/frf  # machine circumference [m]
print("C", C)


# Tracking details
n_turns = int(100)

# Derived parameters
E_0 = m_e * c**2 / e    # [eV]
tot_beam_energy = np.sqrt(sync_momentum**2 + E_0**2)  # [eV]
momentum_compaction = float(fcc_params['eta'])
rho = 10.7e3
print(momentum_compaction)

# Cavities parameters
n_rf_systems = 1
harmonic_numbers = [h]
voltage_program = [float(fcc_params['Vrf'])]
print("voltage_program", voltage_program)
phi_offset = [np.pi]  # [0] #

bucket_length = C / c / harmonic_numbers[0]
print(bucket_length)


# WITH QUANTUM EXCITATION

# DEFINE RING------------------------------------------------------------------

n_sections = 2
n_kicks = 1
general_params = Ring(np.ones(n_sections) * C/n_sections,
                      np.tile(momentum_compaction, (1, n_sections)).T,
                      np.tile(sync_momentum, (n_sections, n_turns+1)),
                      particle_type, n_turns=n_turns, n_sections=n_sections)

RF_sct_par = []
for i in np.arange(n_sections)+1:
    RF_sct_par.append(RFStation(general_params,
                                harmonic_numbers, [
                                    v/n_sections for v in voltage_program],
                                phi_offset, section_index=i, n_rf=n_rf_systems))

print("phi_s ", RF_sct_par[0].phi_s[0])

# DEFINE BEAM------------------------------------------------------------------

beam = Beam(general_params, n_macroparticles, n_particles)


# BEAM GENERATION--------------------------------------------------------------

bigaussian(general_params, RF_sct_par[0], beam, sigma_dt=15e-12, seed=1234,
           reinsertion='on')

beam.dt += 1.25e-9


# Define Synchrotron radiation objects with quantum excitation
SR = []
for i in range(n_sections):
    SR.append(SynchrotronRadiation(general_params, RF_sct_par[
              i], beam, rho, n_kicks=n_kicks, python=False))

SR[0].print_SR_params()


print(beam.n_macroparticles)
# DEFINE SLICES----------------------------------------------------------------
factor = 2
number_slices = factor*100  # *100
cut_length = factor*125e-12
shift = SR[0].beam_phase_to_compensate_SR*RF_sct_par[0].t_rf[0,0] / (2.0*np.pi)

cut_options = CutOptions(cut_left=bucket_length/2.-cut_length-shift,
                         cut_right=bucket_length/2.+cut_length-shift, n_slices=number_slices)
profile = Profile(beam, CutOptions=cut_options)

# DEFINE TRACKER---------------------------------------------------------------
longitudinal_tracker = []
for i in range(n_sections):
    longitudinal_tracker.append(RingAndRFTracker(
        RF_sct_par[i], beam, Profile=profile))

full_tracker = FullRingAndRF(longitudinal_tracker)


# R_S = 5e4
# frequency_R = 10e6
# Q = 1e2
# Resistive wall impedance parameters
pipe_radius = 35e-3  # [m]
conductivity = 3.77e7  # [S]
pipe_length = C

resistivewall = ResistiveWall(
    pipe_radius, pipe_length/n_sections, conductivity=conductivity)

# INDUCED VOLTAGE FROM IMPEDANCE-----------------------------------------------

imp_list = [resistivewall]

ind_volt_freq = InducedVoltageFreq(
    beam, Profile=profile, impedance_source_list=imp_list)

total_ind_volt_freq = TotalInducedVoltage(
    beam, Profiles=profile, induced_voltage_list=[ind_volt_freq])


print('Bunch position %.4e s' % np.mean(beam.dt))
print('bin_size = {0:1.3f} [fs]'.format(profile.bin_size*1e15))
print('n_slices = {0:d}'.format(profile.n_slices))
print('bunch length (4 sigma) = {0:1.3e} [ps]'.format(4e12 * np.std(beam.dt)))
print('energy spread = {0:1.3f} [MeV]'.format(1e-6 * np.std(beam.dE)))


# ACCELERATION MAP-------------------------------------------------------------
map_ = []
for i in range(n_sections):
    map_ += [longitudinal_tracker[i]] + [SR[i]] + \
        [profile] + [total_ind_volt_freq]

# TRACKING + PLOTS-------------------------------------------------------------


std_dt = np.zeros(n_turns)
std_dE = np.zeros(n_turns)
avg_dt = np.zeros(n_turns)
# plt.figure()

# plt.plot(beam.dt,beam.dE)
print(np.mean(beam.dt))
print("Start tracking", datetime.datetime.now())

for i in range(n_turns):
    # for m in map_:
    #     m.track()
    for s in range(n_sections):
        longitudinal_tracker[s].track()

        SR[s].track()
        profile.track()
        total_ind_volt_freq.track()

    # avg_dt[i] = c_mean(beam.dt.ctypes.data_as(ctypes.c_void_p),
    #                    ctypes.c_int(beam.n_macroparticles))
    avg_dt[i] = np.mean(beam.dt)
    std_dt[i] = np.std(beam.dt)
    # std_dt[i] = c_std_w_mean(beam.dt.ctypes.data_as(ctypes.c_void_p),
    #                          ctypes.c_int(beam.n_macroparticles),
    #                          ctypes.c_double(avg_dt[i]))
    std_dE[i] = np.std(beam.dE)
    # std_dE[i] = c_std_wo_mean(beam.dt.ctypes.data_as(ctypes.c_void_p),
    #                           ctypes.c_int(beam.n_macroparticles))

print("End tracking", datetime.datetime.now())
#
# plt.scatter(beam.dt[::10],beam.dE[::10])
# plt.xlim(-1e-8,1e-8)
# plt.show()


# plt.figure(figsize=[6,4.5])
# plt.plot(avg_dt*1e9,lw=2)

# plt.xlabel('Turns')
# plt.ylabel('Bunch position [ns]')

# plt.savefig('pos.png',bbox_inches='tight')
# plt.close()


# plt.figure(figsize=[6,4.5])
# plt.plot(1e-6*std_dE, lw=2)
# plt.plot(np.arange(len(std_dE)), [1e-6*SR[0].sigma_dE*sync_momentum] *
#         len(std_dE), 'r--', lw=2)
# print('Equilibrium energy spread = {0:1.3f} [MeV]'.format(1e-6 *
#         std_dE[-10:].mean()))
# plt.xlabel('Turns')
# plt.ylabel('Energy spread [MeV]')
# plt.savefig('std_dE_QE.png',bbox_inches='tight')
# plt.close()


# plt.figure(figsize=[6,4.5])
# plt.plot(1e12*4.0*std_dt, lw=2)
# print('Equilibrium bunch length = {0:1.3f} [ps]'.format(4e12 *
#         std_dt[-10:].mean()))
# plt.xlabel('Turns')
# plt.ylabel('Bunch length [ps]')
# plt.savefig('bl_QE.png',bbox_inches='tight')
# plt.close()


# print("\n\n")

print("Done!")
