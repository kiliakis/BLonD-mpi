
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
import os
import sys
import matplotlib.pyplot as plt
import time
import datetime
try:
    from pyprof import timing
    from pyprof import mpiprof
except ImportError:
    from blond.utils import profile_mock as timing
    mpiprof = timing
#  BLonD Imports
from blond.input_parameters.ring import Ring
from blond.input_parameters.rf_parameters import RFStation
from blond.trackers.tracker import RingAndRFTracker
from blond.beam.beam import Beam, Proton
from blond.beam.distributions import bigaussian
from blond.beam.profile import CutOptions, FitOptions, Profile
from blond.monitors.monitors import BunchMonitor
from blond.plots.plot import Plot
from blond.utils.input_parser import parse
from blond.utils import mpi_config as mpiconf

this_directory = os.path.dirname(os.path.realpath(__file__)) + '/'

args = parse()

mpiconf.init(trace=args['trace'], logfile=args['tracefile'])
print(args)
# Simulation parameters -------------------------------------------------------
# Bunch parameters
N_b = 1e9           # Intensity
N_p = 1e6         # Macro-particles
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
N_slices = 500
n_bunches = 1
workers = 3
debug = False
log = None
report = None
seed = 0
N_t_reduce = 1
N_t_monitor = 0

if args.get('turns', None) is not None:
    N_t = args['turns']

if args.get('particles', None) is not None:
    N_p = args['particles']

if args.get('slices', None) is not None:
    N_slices = args['slices']

if args.get('time', False) is True:
    timing.mode = 'timing'

if args.get('omp', None) is not None:
    os.environ['OMP_NUM_THREADS'] = str(args['omp'])

if args.get('bunches', None) is not None:
    n_bunches = args['bunches']

if args.get('reduce', None) is not None:
    N_t_reduce = args['reduce']

if args.get('monitor', None) is not None:
    N_t_monitor = args['monitor']

if args.get('seed', None) is not None:
    seed = args['seed']

if 'log' in args:
    log = args['log']


print({'N_t': N_t, 'n_macroparticles': N_p,
       'N_slices': N_slices,
       'timing.mode': timing.mode,
       'n_bunches': n_bunches,
       'N_t_reduce': N_t_reduce,
       'N_t_monitor': N_t_monitor, 'seed': seed, 'log': log})
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
profile = Profile(beam, CutOptions(n_slices=N_slices),
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


master = mpiconf.Master(log=log)
start_t = time.time()
print(datetime.datetime.now().time())

# master.spawn_workers(workers=workers, debug=debug, log=log, report=report)

try: 
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
    }

    master.logger.debug('Scattered initial coordinates')
    master.multi_scatter(vars_dict)
    # workercomm.Barrier()


    # N_t = 600
    # print('dE mean: ', np.mean(beam.dE))
    # print('dE std: ', np.std(beam.dE))

    # N_t=1
    # Tracking --------------------------------------------------------------------

    task_list = []
    for turn in range(N_t):
        task_list += ['kick', 'drift']

        if (turn % N_t_reduce == 0):
            task_list += ['histo', 'reduce_histo']

        if (N_t_monitor > 0) and (turn % N_t_monitor == 0):
            task_list += ['gather_single']

    master.bcast(task_list)

    for i in range(1, N_t+1):
        # print('Turn: ', i)

        # Plot has to be done before tracking (at least for cases with separatrix)
        # if (i % dt_plt) == 0:
        #     print("Outputting at time step %d..." % i)
        #     print("   Beam momentum %.6e eV" % beam.momentum)
        #     print("   Beam gamma %3.3f" % beam.gamma)
        #     print("   Beam beta %3.3f" % beam.beta)
        #     print("   Beam energy %.6e eV" % beam.energy)
        #     print("   Four-times r.m.s. bunch length %.4e s" % (4.*beam.sigma_dt))
        #     print("   Gaussian bunch length %.4e s" % profile.bunchLength)
        #     print("")

        # Track
        for m in map_:
            m.track()

        # Define losses according to separatrix and/or longitudinal position
        # beam.losses_separatrix(ring, rf)
        # beam.losses_longitudinal_cut(0., 2.5e-9)

    master.multi_gather(vars_dict)
    master.stop()
    master.disconnect()

except Exception as e:
    print(e)
    master.quit()
    master.disconnect()


end_t = time.time()
print(datetime.datetime.now().time())
# if report:
mpiprof.finalize()
timing.report(total_time=1e3*(end_t-start_t),
              out_dir=args['report'],
              out_file='master.csv')

print('dE mean: ', np.mean(beam.dE))
print('dE std: ', np.std(beam.dE))
# plt.figure()
# plt.plot(profile.n_macroparticles)
# plt.show()

# print(beam.dE)
# print(beam.dt)

print("Done!")
