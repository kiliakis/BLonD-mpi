
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
Example input for simulation of acceleration
No intensity effects

:Authors: **Helga Timko**
'''
#  General Imports
from __future__ import division, print_function
from builtins import range
import numpy as np

#  BLonD Imports
from input_parameters.ring import Ring
from input_parameters.rf_parameters import RFStation
from trackers.tracker import RingAndRFTracker
from beam.beam import Beam, Proton
from beam.distributions import bigaussian
from beam.profile import CutOptions, FitOptions, Profile
from monitors.monitors import BunchMonitor
from plots.plot import Plot
import os
import sys
import matplotlib.pyplot as plt
import time

#from toolbox.logger import Logger

import logging

from mpi import mpi_config as mpiconf
from pyprof import timing

# Simulation parameters -------------------------------------------------------
# Bunch parameters
N_b = 1e9           # Intensity
N_p = 1e8         # Macro-particles
tau_0 = 0.4e-9          # Initial bunch length, 4 sigma [s]

# Machine and RF parameters
C = 26658.883        # Machine circumference [m]
p_i = 450e9         # Synchronous momentum [eV/c]
p_f = 460.005e9      # Synchronous momentum, final
h = 35640            # Harmonic number
V = 6e6                # RF voltage [V]
dphi = 0             # Phase modulation/offset
gamma_t = 55.759505  # Transition gamma
alpha = 1./gamma_t/gamma_t        # First order mom. comp. factor

# Tracking details
N_t = 2000           # Number of turns to track
dt_plt = 200         # Time steps between plots

# try:
#     os.mkdir('../output_files')
# except:
#     pass
# try:
#     os.mkdir('../output_files/EX_01_fig')
# except:
#     pass


# Simulation setup ------------------------------------------------------------
print("Setting up the simulation...")


# # Define general parameters
ring = Ring(C, alpha, np.linspace(p_i, p_f, N_t+1), Proton(), N_t)

# # Define beam and distribution
beam = Beam(ring, N_p, N_b)


# # Define RF station parameters and corresponding tracker
rf = RFStation(ring, [h], [V], [dphi])

bigaussian(ring, rf, beam, tau_0/4, reinsertion=True, seed=1)


# # Need slices for the Gaussian fit
profile = Profile(beam, CutOptions(n_slices=100),
                  FitOptions(fit_option='gaussian'))

long_tracker = RingAndRFTracker(rf, beam)

# # Define what to save in file
# # bunchmonitor = BunchMonitor(ring, rf, beam,
# #                             '../output_files/EX_01_output_data', Profile=profile)
# #
# # format_options = {'dirname': '../output_files/EX_01_fig'}
# # plots = Plot(ring, rf, beam, dt_plt, N_t, 0, 0.0001763*h,
# #              -400e6, 400e6, xunit='rad', separatrix_plot=True,
# #              Profile=profile, h5file='../output_files/EX_01_output_data',
# #              format_options=format_options)
# #
# Accelerator map
map_ = [long_tracker, profile]
print("Map set")
# sys.stdout.flush()


# Workers initialization
# logging.debug('master: Spawning the workers')

# workercomm = MPI.COMM_WORLD.Spawn(mpiconf.executable,
#                                   args=mpiconf.args,
#                                   maxprocs=mpiconf.n_workers)

# mpiconf.n_workers = workercomm.Get_remote_size()
# mpiconf.workercomm = workercomm
# logging.debug('master: %d workers successfully initialized' % mpiconf.n_workers)

import datetime



# start_t = time.time()
print(datetime.datetime.now().time())

master = mpiconf.Master(log=False)
master.spawn_workers(workers=7, debug=False, log=False)

# Send initial data to the workers
init_dict = {
    'n_rf': long_tracker.n_rf,
    'solver': long_tracker.solver,
    'length_ratio': long_tracker.length_ratio,
    'alpha_order': long_tracker.alpha_order,
    'n_slices': profile.n_slices
}
master.logger.debug('Broadcasted initial variables')
master.multi_bcast(init_dict)


# Scatter coordinates etc
vars_dict = {
    'dt': beam.dt,
    'dE': beam.dE
    # 'id': (id, 'i')
}

master.logger.debug('Scattered initial coordinates')
master.multi_scatter(vars_dict)
# workercomm.Barrier()


N_t = 2000
print('dE mean: ', np.mean(beam.dE))
print('dE std: ', np.std(beam.dE))

# Tracking --------------------------------------------------------------------
for i in range(1, N_t+1):
    # print('Turn: ', i)

    #     # Plot has to be done before tracking (at least for cases with separatrix)
    if (i % dt_plt) == 0:
        print("Outputting at time step %d..." % i)
        print("   Beam momentum %.6e eV" % beam.momentum)
        print("   Beam gamma %3.3f" % beam.gamma)
        print("   Beam beta %3.3f" % beam.beta)
        print("   Beam energy %.6e eV" % beam.energy)
        print("   Four-times r.m.s. bunch length %.4e s" % (4.*beam.sigma_dt))
        print("   Gaussian bunch length %.4e s" % profile.bunchLength)
        print("")

#     # Track
    for m in map_:
        m.track()

    # master.sync()
#     # Define losses according to separatrix and/or longitudinal position
#     # beam.losses_separatrix(ring, rf)
#     # beam.losses_longitudinal_cut(0., 2.5e-9)

master.multi_gather(vars_dict)
master.stop()
master.disconnect()

print(datetime.datetime.now().time())

# end_t = time.time()
timing.report(#total_time=1e3*(end_t-start_t),
              out_file='report-master.csv')

print('dE mean: ', np.mean(beam.dE))
print('dE std: ', np.std(beam.dE))
# plt.figure()
# plt.plot(profile.n_macroparticles)
# plt.show()

# print(beam.dE)
# print(beam.dt)

print("Done!")
