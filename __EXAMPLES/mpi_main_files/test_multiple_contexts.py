
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
SPS simulation with intensity effects in time and frequency domains using
a table of resonators. The input beam has been cloned to show that the two
methods are equivalent (compare the two figure folders). Note that to create an
exact clone of the beam, the option seed=0 in the generation has been used.
This script shows also an example of how to use the class SliceMonitor (check
the corresponding h5 files).

:Authors: **Danilo Quartullo**
'''

from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
from input_parameters.ring import Ring
from input_parameters.rf_parameters import RFStation
from trackers.tracker import RingAndRFTracker
from beam.distributions import bigaussian
from monitors.monitors import BunchMonitor
from beam.profile import Profile, CutOptions, FitOptions
from impedances.impedance import InducedVoltageTime, InducedVoltageFreq
from impedances.impedance import InducedVoltageResonator, TotalInducedVoltage
from impedances.induced_voltage_analytical import analytical_gaussian_resonator
from beam.beam import Beam, Proton
from plots.plot import Plot
from plots.plot_impedance import plot_induced_voltage_vs_bin_centers
from impedances.impedance_sources import Resonators
import os

import time
import datetime
from toolbox.input_parser import parse
from utils import mpi_config as mpiconf
from pyprof import timing as mpiprof
# from pyprof import mpiprof as mpiprof

args = parse()

mpiconf.init(track=False)


print(args)
# try:
#     os.mkdir('../output_files')
# except:
#     pass
# try:
#     os.mkdir('../output_files/EX_05_fig')
# except:
#     pass

# SIMULATION PARAMETERS -------------------------------------------------------

# Beam parameters
n_particles = 1e10
n_macroparticles = 5*1e6
tau_0 = 2e-9  # [s]

# Machine and RF parameters
gamma_transition = 1/np.sqrt(0.00192)   # [1]
C = 6911.56  # [m]

# Tracking details
n_turns = 2
dt_plt = 1

# Derived parameters
sync_momentum = 25.92e9  # [eV / c]
momentum_compaction = 1 / gamma_transition**2  # [1]

# Cavities parameters
n_rf_systems = 1
harmonic_number = 4620
voltage_program = 0.9e6  # [V]
phi_offset = 0.0
number_slices = 2**8


workers = 3
debug = False
log = None
report = None

if 'turns' in args:
    n_turns = args['turns']
if 'particles' in args:
    n_macroparticles = args['particles']
if 'slices' in args:
    number_slices = args['slices']

if 'omp' in args:
    os.environ['OMP_NUM_THREADS'] = str(args['omp'])
if 'workers' in args:
    workers = args['workers']
if 'log' in args:
    log = args['log']
if 'report' in args:
    report = args['report']
if 'debug' in args:
    debug = args['debug']

master = mpiconf.Master(log=log)

# class A:
#     def __init__(self, a1=0, a2=0):
#         self.a1 = a1
#         self.a2 = a2

# a = A(4, 2)
# b = A(1, 3)

# init_dict = {
#     'n_rf': 10,
#     'n_slices': number_slices,
#     'pi': 3.14,
#     'a1': a.a1,
#     'a2': a.a2
# }
# master.multi_bcast(init_dict)


# master.switch_context(1)
# init_dict = {
#     'a1': b.a1,
#     'a2': b.a2
# }

# master.multi_bcast(init_dict)


# master.stop()
# master.disconnect()

# Scatter coordinates etc
# vars_dict = {
#     'dt': my_beam.dt,
#     'dE': my_beam.dE
# }

# master.logger.debug('Scattered initial coordinates')
# master.multi_scatter(vars_dict)

# exit(0)

# DEFINE RING------------------------------------------------------------------
print("Setting up the simulation...")

general_params = Ring(C, momentum_compaction,
                      sync_momentum, Proton(), n_turns)
general_params_freq = Ring(C, momentum_compaction,
                           sync_momentum, Proton(), n_turns)
# general_params_res = Ring(C, momentum_compaction,
#                           sync_momentum, Proton(), n_turns)


RF_sct_par = RFStation(general_params, [harmonic_number],
                       [voltage_program], [phi_offset], n_rf_systems)
RF_sct_par_freq = RFStation(general_params_freq,
                            [harmonic_number], [voltage_program],
                            [phi_offset], n_rf_systems)
# RF_sct_par_res = RFStation(general_params_res,
#                            [harmonic_number], [voltage_program],
#                            [phi_offset], n_rf_systems)

my_beam = Beam(general_params, n_macroparticles, n_particles)
my_beam_freq = Beam(general_params_freq, n_macroparticles, n_particles)
# my_beam_res = Beam(general_params_res, n_macroparticles, n_particles)

ring_RF_section = RingAndRFTracker(RF_sct_par, my_beam)
ring_RF_section_freq = RingAndRFTracker(RF_sct_par_freq, my_beam_freq)
# ring_RF_section_res = RingAndRFTracker(RF_sct_par_res, my_beam_res)

# DEFINE BEAM------------------------------------------------------------------

bigaussian(general_params, RF_sct_par, my_beam, tau_0/4,
           seed=1)
bigaussian(general_params_freq, RF_sct_par_freq, my_beam_freq,
           tau_0/4, seed=1)
# bigaussian(general_params_res, RF_sct_par_res, my_beam_res,
#            tau_0/4, seed=1)

print('dE mean: ', np.mean(my_beam.dE))
print('dE freq mean: ', np.mean(my_beam_freq.dE))

cut_options = CutOptions(cut_left=0, cut_right=2*np.pi, n_slices=number_slices,
                         RFSectionParameters=RF_sct_par, cuts_unit='rad')
slice_beam = Profile(my_beam, cut_options, FitOptions(fit_option='gaussian'))
cut_options_freq = CutOptions(cut_left=0, cut_right=2*np.pi, n_slices=number_slices,
                              RFSectionParameters=RF_sct_par_freq, cuts_unit='rad')
slice_beam_freq = Profile(my_beam_freq, cut_options_freq,
                          FitOptions(fit_option='gaussian'))
# cut_options_res = CutOptions(cut_left=0, cut_right=2*np.pi, n_slices=number_slices,
#                              RFSectionParameters=ring_RF_section_res, cuts_unit='rad')
# slice_beam_res = Profile(my_beam_res, cut_options_res,
#                          FitOptions(fit_option='gaussian'))


# MONITOR----------------------------------------------------------------------

# bunchmonitor = BunchMonitor(general_params, ring_RF_section, my_beam,
#                             '../output_files/EX_05_output_data',
#                             Profile=slice_beam, buffer_time=1)

# bunchmonitor_freq = BunchMonitor(general_params_freq, ring_RF_section_freq,
#                                  my_beam_freq, '../output_files/EX_05_output_data_freq',
#                                  Profile=slice_beam_freq, buffer_time=1)
# bunchmonitor_res = BunchMonitor(general_params_res, ring_RF_section_res,
#                                 my_beam_res, '../output_files/EX_05_output_data_res',
#                                 Profile=slice_beam_res, buffer_time=1)


# LOAD IMPEDANCE TABLE--------------------------------------------------------

# print(os.getcwd())
table = np.loadtxt(
    '__EXAMPLES/input_files/EX_05_new_HQ_table.dat', comments='!')

R_shunt = table[:, 2] * 10**6
f_res = table[:, 0] * 10**9
Q_factor = table[:, 1]
resonator = Resonators(R_shunt, f_res, Q_factor)

ind_volt_time = InducedVoltageTime(my_beam, slice_beam, [resonator])
ind_volt_freq = InducedVoltageFreq(
    my_beam_freq, slice_beam_freq, [resonator], 1e5)
# ind_volt_res = InducedVoltageResonator(my_beam_res, slice_beam_res, resonator)

tot_vol = TotalInducedVoltage(my_beam, slice_beam, [ind_volt_time])
tot_vol_freq = TotalInducedVoltage(my_beam_freq, slice_beam_freq,
                                   [ind_volt_freq])
# tot_vol_res = TotalInducedVoltage(my_beam_res, slice_beam_res,
#                                   [ind_volt_res])

# Analytic result-----------------------------------------------------------
# VindGauss = np.zeros(len(slice_beam.bin_centers))
# for r in range(len(Q_factor)):
#     # Notice that the time-argument of inducedVoltageGauss is shifted by
#     # mean(my_slices.bin_centers), because the analytical equation assumes the
#     # Gauss to be centered at t=0, but the line density is centered at
#     # mean(my_slices.bin_centers)
#     tmp = analytical_gaussian_resonator(tau_0/4,
#                                         Q_factor[r], R_shunt[r], 2 *
#                                         np.pi*f_res[r],
#                                         slice_beam.bin_centers -
#                                         np.mean(slice_beam.bin_centers),
#                                         my_beam.intensity)
#     VindGauss += tmp.real

# # PLOTS

# format_options = {'dirname': '../output_files/EX_05_fig/1', 'linestyle': '.'}
# plots = Plot(general_params, RF_sct_par, my_beam, dt_plt, n_turns, 0,
#              0.0014*harmonic_number, -1.5e8, 1.5e8, xunit='rad',
#              separatrix_plot=True, Profile=slice_beam,
#              h5file='../output_files/EX_05_output_data',
#              histograms_plot=True, sampling=50, format_options=format_options)

# format_options = {'dirname': '../output_files/EX_05_fig/2', 'linestyle': '.'}
# plots_freq = Plot(general_params_freq, RF_sct_par_freq, my_beam_freq, dt_plt,
#                   n_turns, 0, 0.0014*harmonic_number, -1.5e8, 1.5e8,
#                   xunit='rad', separatrix_plot=True, Profile=slice_beam_freq,
#                   h5file='../output_files/EX_05_output_data_freq',
#                   histograms_plot=True, sampling=50,
#                   format_options=format_options)
# format_options = {'dirname': '../output_files/EX_05_fig/3', 'linestyle': '.'}
# plots_res = Plot(general_params_res, RF_sct_par_res, my_beam_res, dt_plt,
#                  n_turns, 0, 0.0014*harmonic_number, -1.5e8, 1.5e8,
#                  xunit='rad', separatrix_plot=True, Profile=slice_beam_res,
#                  h5file='../output_files/EX_05_output_data_res',
#                  histograms_plot=True, sampling=50,
#                  format_options=format_options)


# ACCELERATION MAP-------------------------------------------------------------

map_ = [tot_vol] + [ring_RF_section] + \
    [slice_beam]  # + [bunchmonitor] + [plots]
map_freq = [tot_vol_freq] + [ring_RF_section_freq] + [slice_beam_freq]
# map_freq = [tot_vol_freq] + [ring_RF_section_freq] + [slice_beam_freq] \
#     + [bunchmonitor_freq] + [plots_freq]
# map_res = [tot_vol_res] + [ring_RF_section_res] + [slice_beam_res] \
#     + [bunchmonitor_res] + [plots_res]


# map_ = [ring_RF_section, slice_beam]
# TRACKING + PLOTS-------------------------------------------------------------
print('Map set')


start_t = time.time()
print(datetime.datetime.now().time())

init_dict = {
    'n_rf': ring_RF_section.n_rf,
    'solver': ring_RF_section.solver,
    'length_ratio': ring_RF_section.length_ratio,
    'alpha_order': ring_RF_section.alpha_order,
    'n_slices': slice_beam.n_slices,
    'bin_centers': slice_beam.bin_centers,
    'charge': my_beam.Particle.charge
}
master.multi_bcast(init_dict)
master.logger.debug('Broadcasted initial variables')


# Scatter coordinates etc
vars_dict = {
    'dt': my_beam.dt,
    'dE': my_beam.dE
}
master.multi_scatter(vars_dict)
master.logger.debug('Scattered initial coordinates')


slice_beam.track()


master.switch_context(1)

init_dict = {
    'bin_centers': slice_beam_freq.bin_centers
}
master.logger.debug('Broadcasted initial variables')
master.multi_bcast(init_dict)

vars_dict = {
    'dt': my_beam_freq.dt,
    'dE': my_beam_freq.dE
}
master.multi_scatter(vars_dict)
master.logger.debug('Scattered initial coordinates')
slice_beam_freq.track()
# slice_beam_res.track()

# print('dE std: ', np.std(my_beam.dE))

# print('dt mean: ', np.mean(my_beam.dt))
# print('dt std: ', np.std(my_beam.dt))

for i in np.arange(1, n_turns+1):


    if i % 200 == 0:
        print(i)

    master.switch_context(0)
    for m in map_:
        m.track()

    master.switch_context(1)
    for m in map_freq:
        m.track()

    # for m in map_res:
    #     m.track()

    # Plots
    # if (i % dt_plt) == 0:
    #     plot_induced_voltage_vs_bin_centers(i, general_params, tot_vol,
    #                                         style='.', dirname='../output_files/EX_05_fig/1')
    #     plot_induced_voltage_vs_bin_centers(i, general_params_freq,
    #                                         tot_vol_freq, style='.', dirname='../output_files/EX_05_fig/2')
    #     plot_induced_voltage_vs_bin_centers(i, general_params_res,
    #                                         tot_vol_res, style='.', dirname='../output_files/EX_05_fig/3')


master.switch_context(0)
vars_dict = {
    'dt': my_beam.dt,
    'dE': my_beam.dE
}
master.multi_gather(vars_dict)

master.switch_context(1)
vars_dict = {
    'dt': my_beam_freq.dt,
    'dE': my_beam_freq.dE
}
master.multi_gather(vars_dict)


master.stop()
master.disconnect()
end_t = time.time()
print(datetime.datetime.now().time())


print('dE mean: ', np.mean(my_beam.dE))
print('dE freq mean: ', np.mean(my_beam_freq.dE))

# print('dE mean: ', np.mean(my_beam.dE))
# print('dE std: ', np.std(my_beam.dE))

# print('dt mean: ', np.mean(my_beam.dt))
# print('dt std: ', np.std(my_beam.dt))


if report:
    mpiprof.finalize()
    mpiprof.report(total_time=1e3*(end_t-start_t),
                   out_dir=report,
                   out_file='master.csv')

# Plotting induced voltages---------------------------------------------------
# plt.clf()
# plt.ylabel("induced voltage [arb. unit]")
# plt.xlabel("time [ns]")
# plt.plot(1e9*slice_beam.bin_centers, tot_vol.induced_voltage, label='Time')
# plt.plot(1e9*slice_beam_freq.bin_centers, tot_vol_freq.induced_voltage,
#          label='Freq')
# plt.plot(1e9*slice_beam_res.bin_centers, tot_vol_res.induced_voltage,
#          label='Resonator')
# plt.plot(1e9*slice_beam.bin_centers, VindGauss, label='Analytic')
# plt.legend()
# dirname = '../output_files/EX_05_fig'
# fign = dirname + '/comparison_induced_voltage.png'
# plt.savefig(fign)

print("Done!")
