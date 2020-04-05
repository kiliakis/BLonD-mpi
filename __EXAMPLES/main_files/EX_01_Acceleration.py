
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
import time
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
n_particles = 1e6         # Macro-particles
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
n_turns = 2000           # Number of turns to track
dt_plt = 1000         # Time steps between plots
n_slices = 100
n_bunches = 1
n_turns_reduce = 1
n_iterations = n_turns

worker.greet()
if worker.isMaster:
    worker.print_version()

# mpiprint('Done!')
# worker.finalize()
# exit()

args = parse()

n_iterations = args.get('turns', n_iterations)
n_particles = args.get('particles', n_particles)
n_bunches = args.get('bunches', n_bunches)
n_turns_reduce = args.get('reduce', n_turns_reduce)
if args.get('time', False) is True:
    timing.mode = 'timing'
os.environ['OMP_NUM_THREADS'] = str(args.get('omp', '1'))
seed = args.get('seed')
approx = args.get('approx')
withtp = int(args.get('withtp'))

worker.initLog(args['log'], args['logdir'])
worker.initTrace(args['trace'], args['tracefile'])
worker.taskparallelism(withtp)

mpiprint(args)


# Simulation setup ------------------------------------------------------------
mpiprint("Setting up the simulation...")


# Define general parameters
ring = Ring(C, alpha, np.linspace(p_i, p_f, n_turns+1), Proton(), n_turns)

# Define beam and distribution
beam = Beam(ring, n_particles, N_b)


# Define RF station parameters and corresponding tracker
rf = RFStation(ring, [h], [V], [dphi])


bigaussian(ring, rf, beam, tau_0/4, reinsertion=True, seed=seed)


# Need slices for the Gaussian fit
# TODO add the gaussian fit
profile = Profile(beam, CutOptions(n_slices=n_slices))
# FitOptions(fit_option='gaussian'))

long_tracker = RingAndRFTracker(rf, beam)

beam.split_random()


# Accelerator map
# map_ = [long_tracker, profile]
mpiprint("Map set")

lbturns = []
if args['loadbalance'] == 'times':
    if args['loadbalancearg'] != 0:
        intv = n_iterations // (args['loadbalancearg']+1)
    else:
        intv = n_iterations // (10 + 1)
    lbturns = np.arange(worker.start_turn, n_iterations, intv)[1:]

elif args['loadbalance'] == 'interval':
    if args['loadbalancearg'] != 0:
        lbturns = np.arange(worker.start_turn, n_iterations, args['loadbalancearg'])
    else:
        lbturns = np.arange(worker.start_turn, n_iterations, 1000)

elif args['loadbalance'] == 'dynamic':
    lbturns = [worker.start_turn]
    # print('Warning: Dynamic load balance policy not supported.')

worker.sync()
timing.reset()
start_t = time.time()
# mpiprint(datetime.datetime.now().time())
tcomp_old = tcomm_old = tconst_old = tsync_old = 0


for turn in range(n_iterations):

    # Plot has to be done before tracking (at least for cases with separatrix)
    if (turn % dt_plt) == 0:
        mpiprint("Outputting at time step %d..." % turn)
        mpiprint("   Beam momentum %.6e eV" % beam.momentum)
        mpiprint("   Beam gamma %3.3f" % beam.gamma)
        mpiprint("   Beam beta %3.3f" % beam.beta)
        mpiprint("   Beam energy %.6e eV" % beam.energy)
        mpiprint("   Four-times r.m.s. bunch length %.4e s" % (4.*beam.sigma_dt))
        mpiprint("   Gaussian bunch length %.4e s" % profile.bunchLength)
        mpiprint("")

    # Track
    long_tracker.track()

    # Update profile
    if (approx == 0):
        profile.track()
        profile.reduce_histo()
    elif (approx == 1) and (turn % n_turns_reduce == 0):
        profile.track()
        profile.reduce_histo()
    elif (approx == 2):
        profile.track()
        profile.scale_histo()
        

    if (turn in lbturns):
        tcomp_new = timing.get(['comp:'])
        tcomm_new = timing.get(['comm:'])
        tconst_new = timing.get(['serial:'], ['serial:sync'])
        tsync_new = 0
        # intv = worker.redistribute(turn, beam, tcomp=tcomp_new-tcomp_old,
        #                            tconst=(tconst_new-tconst_old) + (tcomm_new - tcomm_old))
        # if args['loadbalance'] == 'dynamic':
        #     lbturns[0] += intv
        worker.report(turn, beam, tcomp=tcomp_new-tcomp_old,
                      tcomm=tcomm_new-tcomm_old,
                      tconst=tconst_new-tconst_old,
                      tsync=tsync_new-tsync_old)
        tcomp_old = tcomp_new
        tcomm_old = tcomm_new
        tconst_old = tconst_new
        tsync_old = tsync_new


beam.gather()
end_t = time.time()

timing.report(total_time=1e3*(end_t-start_t),
              out_dir=args['timedir'],
              out_file='worker-{}.csv'.format(os.getpid()))
worker.finalize()

mpiprint('dE mean: ', np.mean(beam.dE))
mpiprint('dE std: ', np.std(beam.dE))
mpiprint('profile mean: ', np.mean(profile.n_macroparticles))
mpiprint('profile std: ', np.std(profile.n_macroparticles))

mpiprint('Done!')
