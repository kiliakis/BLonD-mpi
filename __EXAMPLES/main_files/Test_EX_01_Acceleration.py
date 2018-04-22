
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
from mpi4py import MPI 
import sys

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

# Simulation parameters -------------------------------------------------------
# Bunch parameters
N_b = 1e9           # Intensity
N_p = 8         # Macro-particles
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

if rank == 0:
    try:
        os.mkdir('../output_files')
    except:
        pass
    try:
        os.mkdir('../output_files/EX_01_fig')
    except:
        pass



# Simulation setup ------------------------------------------------------------
    print("Setting up the simulation...")
    print("")


# Define general parameters
    ring = Ring(C, alpha, np.linspace(p_i, p_f, N_t+1), Proton(), N_t)

# Define beam and distribution
    beam = Beam(ring, N_p, N_b)


# Define RF station parameters and corresponding tracker
    rf = RFStation(ring, [h], [V], [dphi])


    bigaussian(ring, rf, beam, tau_0/4, reinsertion=True, seed=1)


# Need slices for the Gaussian fit
    profile = Profile(beam, CutOptions(n_slices=100),
                      FitOptions(fit_option='gaussian'))

    long_tracker = RingAndRFTracker(rf, beam)

# Define what to save in file
    # bunchmonitor = BunchMonitor(ring, rf, beam,
    #                             '../output_files/EX_01_output_data', Profile=profile)
    # 
    # format_options = {'dirname': '../output_files/EX_01_fig'}
    # plots = Plot(ring, rf, beam, dt_plt, N_t, 0, 0.0001763*h,
    #              -400e6, 400e6, xunit='rad', separatrix_plot=True,
    #              Profile=profile, h5file='../output_files/EX_01_output_data',
    #              format_options=format_options)
    # 
# Accelerator map
    map_ = [long_tracker]
    print("Map set")
    print("")
    sys.stdout.flush()
    # print("dE: ", beam.dE)

# scatter the data

my_N_p = (N_p + size - 1)//comm.size

# if comm.rank==comm.size-1:
#     my_N_p = N_p - (comm.size-1)*my_N_p

my_dE = np.empty(my_N_p, dtype=np.float64)
my_dt = np.empty(my_N_p, dtype=np.float64)


# for r in range(size):
#     if rank == r:
#         print("[%d] Size: %d" % (rank, my_N_p))
#     comm.Barrier()

dE = None
dt = None
if rank == 0:
    dE = beam.dE
    dt = beam.dt
    dE = np.append(dE, np.zeros(int(comm.size * my_N_p - N_p)))
    dt = np.append(dt, np.zeros(int(comm.size * my_N_p - N_p)))


comm.Scatter(dE, my_dE, root=0)

# for r in range(comm.size):
#     if comm.rank == r:
#         print("[%d] :" % comm.rank, my_dE)
#     comm.Barrier()


sys.exit(0)

# print(long_tracker.rf_voltage[0])

# Tracking --------------------------------------------------------------------
for i in range(1, N_t+1):
    # Plot has to be done before tracking (at least for cases with separatrix)
    if (i % dt_plt) == 0 and (comm.rank == 0):
        print("Outputting at time step %d..." % i)
        print("   Beam momentum %.6e eV" % beam.momentum)
        print("   Beam gamma %3.3f" % beam.gamma)
        print("   Beam beta %3.3f" % beam.beta)
        print("   Beam energy %.6e eV" % beam.energy)
        print("   Four-times r.m.s. bunch length %.4e s" % (4.*beam.sigma_dt))
        print("   Gaussian bunch length %.4e s" % profile.bunchLength)
        print("")

    # Track
    for m in map_:
        m.track()

    # Define losses according to separatrix and/or longitudinal position
    # beam.losses_separatrix(ring, rf)
    # beam.losses_longitudinal_cut(0., 2.5e-9)

print("Done!")
