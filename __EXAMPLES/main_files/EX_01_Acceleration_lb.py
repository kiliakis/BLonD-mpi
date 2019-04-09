
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
# from __future__ import division, print_function
# from builtins import range
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
from blond.monitors.monitors import SlicesMonitor
from blond.input_parameters.ring import Ring
from blond.input_parameters.rf_parameters import RFStation
from blond.trackers.tracker import RingAndRFTracker
from blond.beam.beam import Beam, Proton
from blond.beam.distributions import bigaussian
from blond.beam.profile import CutOptions, FitOptions, Profile
from blond.monitors.monitors import BunchMonitor
from blond.plots.plot import Plot
from blond.utils.input_parser import parse

from blond.utils.mpi_config import worker, mpiprint

this_directory = os.path.dirname(os.path.realpath(__file__)) + '/'


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
dt_plt = 1000         # Time steps between plots
N_slices = 100
n_bunches = 1
workers = 3
debug = False
log = None
report = None
seed = 1
N_t_reduce = 1
N_t_monitor = 0
approx = 0


worker.greet()
if worker.isMaster:
    worker.print_version()

# mpiprint('Done!')
# worker.finalize()
# exit()

args = parse()
if args.get('turns', None) is not None:
    N_t = args['turns']

if args.get('particles', None) is not None:
    N_p = args['particles']

if args.get('slices', None) is not None:
    N_slices = args['slices']

if args.get('time', False) is True:
    timing.mode = 'timing'

if args.get('trace', False) == True:
    mpiprof.mode = 'tracing'

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

if args.get('log', None) is not None:
    log = args['log']

if args.get('approx', None) is not None:
    approx = int(args['approx'])


mpiprint({'N_t': N_t, 'n_macroparticles': N_p,
          'N_slices': N_slices,
          'timing.mode': timing.mode,
          'n_bunches': n_bunches,
          'N_t_reduce': N_t_reduce,
          'N_t_monitor': N_t_monitor, 'seed': seed, 'log': log,  'approx': approx})


# Simulation setup ------------------------------------------------------------
mpiprint("Setting up the simulation...")


# Define general parameters
ring = Ring(C, alpha, np.linspace(p_i, p_f, N_t+1), Proton(), N_t)

# Define beam and distribution
beam = Beam(ring, N_p, N_b)


# Define RF station parameters and corresponding tracker
rf = RFStation(ring, [h], [V], [dphi])

bigaussian(ring, rf, beam, tau_0/4, reinsertion=True, seed=seed)


# Need slices for the Gaussian fit
# TODO add the gaussian fit
profile = Profile(beam, CutOptions(n_slices=N_slices))
# FitOptions(fit_option='gaussian'))

long_tracker = RingAndRFTracker(rf, beam)

beam.split_random()


if N_t_monitor > 0 and worker.isMaster:
    if args.get('monitorfile', None):
        filename = args['monitorfile']
    else:
        filename = 'profiles/{}-t{}-p{}-b{}-sl{}-r{}-m{}-se{}-w{}'.format('EX_01_Acceleration',
                                                                          N_t, N_p,
                                                                          n_bunches, N_slices,
                                                                          N_t_reduce,
                                                                          N_t_monitor,
                                                                          seed,
                                                                          worker.workers)
    slicesMonitor = SlicesMonitor(filename=filename,
                                  n_turns=np.ceil(1.0 * N_t / N_t_monitor),
                                  profile=profile,
                                  rf=rf,
                                  Nbunches=n_bunches)


# Accelerator map
# map_ = [long_tracker, profile]
mpiprint("Map set")


# mpiprint(datetime.datetime.now().time())


lbturns = []
if args['loadbalance'] == 'times':
    if args['loadbalancearg'] != 0:
        intv = N_t // (args['loadbalancearg']+1)
    else:
        intv = N_t // (100 + 1)
    lbturns = np.arange(0, N_t, intv)[1:]

elif args['loadbalance'] == 'interval':
    if args['loadbalancearg'] != 0:
        lbturns = np.arange(0, N_t, args['loadbalancearg'])
    else:
        lbturns = np.arange(0, N_t, 100)

elif args['loadbalance'] == 'dynamic':
    lbturns = [worker.interval]

    # print('Warning: Dynamic load balance policy not supported.')

worker.sync()
timing.reset()
start_t = time.time()
tcomp_old = tcomm_old = tconst_old = 0
for turn in range(N_t):

    # Plot has to be done before tracking (at least for cases with separatrix)
    # if (turn % dt_plt) == 0:
    #     mpiprint("Outputting at time step %d..." % turn)
    #     mpiprint("   Beam momentum %.6e eV" % beam.momentum)
    #     mpiprint("   Beam gamma %3.3f" % beam.gamma)
    #     mpiprint("   Beam beta %3.3f" % beam.beta)
    #     mpiprint("   Beam energy %.6e eV" % beam.energy)
    #     mpiprint("   Four-times r.m.s. bunch length %.4e s" % (4.*beam.sigma_dt))
    #     mpiprint("   Gaussian bunch length %.4e s" % profile.bunchLength)
    #     mpiprint("")

    # Track
    long_tracker.track()

    # Update profile
    if (approx == 0):
        profile.track()
        profile.reduce_histo()
    elif (approx == 1) and (turn % N_t_reduce == 0):
        profile.track()
        profile.reduce_histo()
    elif (approx == 2):
        profile.track()
        profile.scale_histo()

    if (N_t_monitor > 0) and (turn % N_t_monitor == 0):
        beam.losses_separatrix(ring, rf)
        beam.statistics()
        beam.gather_statistics()
        if worker.isMaster:
            profile.fwhm()
            slicesMonitor.track(turn)

    if turn in lbturns:
        tcomp_new = timing.get(['comp:'])
        tcomm_new = timing.get(['comm:'])
        tconst_new = timing.get(['serial:'])
        intv = worker.redistribute(turn, beam, tcomp=tcomp_new-tcomp_old,
                                   tcomm=tcomm_new-tcomm_old,
                                   tconst=tconst_new-tconst_old)
        if args['loadbalance'] == 'dynamic':
            lbturns[0] += intv
        worker.report(turn, beam, tcomp=tcomp_new-tcomp_old,
                      tcomm=tcomm_new-tcomm_old,
                      tconst=tconst_new-tconst_old)
        tcomp_old = tcomp_new
        tcomm_old = tcomm_new
        tconst_old = tconst_new


beam.gather()
end_t = time.time()

timing.report(total_time=1e3*(end_t-start_t),
              out_dir=args['timedir'],
              out_file='worker-{}.csv'.format(os.getpid()))
worker.finalize()
if N_t_monitor > 0:
    slicesMonitor.close()

mpiprint('dE mean: ', np.mean(beam.dE))
mpiprint('dE std: ', np.std(beam.dE))
mpiprint('profile mean: ', np.mean(profile.n_macroparticles))
mpiprint('profile std: ', np.std(profile.n_macroparticles))

mpiprint('Done!')
