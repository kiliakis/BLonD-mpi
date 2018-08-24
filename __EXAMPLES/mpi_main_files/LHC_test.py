# Simulation of LHC acceleration ramp to 6.5 TeV
# Noise injection through PL; both PL & SL closed
# Pre-distort noise spectrum to counteract PL action
# WITH intensity effects
#
# Run first:
# Preprocess_ramp.py
# Preprocess_LHC_noise.py
#
# H. Timko


import time
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import sys


REAL_RAMP = False    # track full ramp
MONITORING = False   # turn off plots and monitors


from input_parameters.ring import Ring
from input_parameters.rf_parameters import RFStation
from trackers.tracker import RingAndRFTracker, FullRingAndRF
from llrf.beam_feedback import BeamFeedback
from llrf.rf_noise import FlatSpectrum, LHCNoiseFB
from beam.beam import Beam, Proton
from beam.distributions import bigaussian  # matched_from_distribution_function
from beam.profile import Profile, CutOptions
from impedances.impedance_sources import InputTable
from impedances.impedance import InducedVoltageFreq, TotalInducedVoltage
from toolbox.next_regular import next_regular
if MONITORING:
    from monitors.monitors import BunchMonitor
    from plots.plot import Plot
    from plots.plot_beams import plot_long_phase_space
    from plots.plot_slices import plot_beam_profile

from monitors.monitors import SlicesMonitor

import datetime
from utils.input_parser import parse
from utils import mpi_config as mpiconf
from pyprof import timing
from pyprof import mpiprof
import os

args = parse()
mpiconf.init(trace=args['trace'], logfile=args['tracefile'])
print(args)


# Simulation parameters --------------------------------------------------------
# Bunch parameters
N_b = 1.2e9          # Intensity
N_p = 250000         # Macro-particles
NB = 48              # Number of bunches

# Machine and RF parameters
C = 26658.883        # Machine circumference [m]
h = 35640            # Harmonic number
dphi = 0.            # Phase modulation/offset
gamma_t = 55.759505  # Transition gamma
alpha = 1./gamma_t/gamma_t        # First order mom. comp. factor

# Tracking details
dt_plt = 10000      # Time steps between plots
dt_mon = 1           # Time steps between monitoring
dt_save = 1000000    # Time steps between saving coordinates
if REAL_RAMP:
    N_t = 14000000       # Number of turns to track; full ramp: 8700001
else:
    N_t = 500000
bl_target = 1.25e-9  # 4 sigma r.m.s. target bunch length in [ns]


N_t_reduce = 1
N_t_monitor = 0
seed = 0

if args.get('turns', None):
    N_t = args['turns']
if args.get('particles', None):
    N_p = args['particles']

if args.get('bunches', None):
    NB = args['bunches']

if args.get('reduce', None):
    N_t_reduce = args['reduce']

if args.get('monitor', None):
    N_t_monitor = args['monitor']

if args.get('omp', None):
    os.environ['OMP_NUM_THREADS'] = str(args['omp'])
if 'log' in args:
    log = args['log']

if args.get('time', False) == True:
    timing.mode = 'timing'

if args.get('seed', None):
    seed = args['seed']


# Simulation setup -------------------------------------------------------------
print("Setting up the simulation...")
print("")
wrkDir = r'/afs/cern.ch/work/k/kiliakis/public/helga/'

# Import pre-processed momentum and voltage for the acceleration ramp
ps = 450.e9*np.ones(N_t+1)
print("Flat top momentum %.4e eV" % ps[-1])
V = 6.e6*np.ones(N_t+1)
print("Flat top voltage %.4e V" % V[-1])
print("Momentum and voltage loaded...")

# Define general parameters
ring = Ring(C, alpha, ps[0:N_t+1], Proton(), n_turns=N_t)
print("General parameters set...")

# Define RF parameters (noise to be added for CC case)
rf = RFStation(ring, [h], [V[0:N_t+1]], [0.])
print("RF parameters set...")

# FULL BEAM
bunch = Beam(ring, N_p, N_b)
beam = Beam(ring, N_p*NB, N_b)
bigaussian(ring, rf, bunch, 0.3e-9, reinsertion=True, seed=seed)
for i in np.arange(NB):
    beam.dt[i*N_p:(i+1)*N_p] = bunch.dt[0:N_p] + i*25.e-9
    beam.dE[i*N_p:(i+1)*N_p] = bunch.dE[0:N_p]


# Profile required for PL
cutRange = (NB-1)*25.e-9+3.5e-9
nSlices = np.int(cutRange/0.025e-9 + 1)
nSlices = next_regular(nSlices)
profile = Profile(beam, CutOptions(n_slices=nSlices, cut_left=-0.5e-9,
                                   cut_right=(cutRange-0.5e-9)))
print("Beam generated, profile set...")
print("Using %d slices" % nSlices)


tracker = RingAndRFTracker(rf, beam, interpolation=False)

map_ = [profile] + [tracker]
print("Map set")


print('dE mean: ', np.mean(beam.dE))
print('dE std: ', np.std(beam.dE))

if N_t_monitor > 0:
    filename = 'profiles/LHC-v0-t{}-p{}-b{}-sl{}-r{}-m{}-se{}'.format(
        N_t, N_p, NB, nSlices, N_t_reduce, N_t_monitor, seed)
    slicesMonitor = SlicesMonitor(filename=filename,
                                  n_turns=np.ceil(1.0 * N_t / N_t_monitor),
                                  profile=profile)

master = mpiconf.Master(log=log)
start_t = time.time()
try:

    init_dict = {
        'tracker_t_rev': tracker.t_rev,
        'tracker_eta_0': tracker.eta_0,
        'tracker_eta_1': tracker.eta_1,
        'tracker_eta_2': tracker.eta_2,
        'rfp_beta': rf.beta,
        'rfp_energy': rf.energy,
        'n_rf': tracker.n_rf,
        'solver': tracker.solver,
        'length_ratio': tracker.length_ratio,
        'alpha_order': tracker.alpha_order,
        'n_slices': profile.n_slices,
        'bin_size': profile.bin_size,
        'bin_centers': profile.bin_centers,
        'cut_left': profile.cut_left,
        'cut_right': profile.cut_right,
        'charge': beam.Particle.charge,
        'beam_ratio': beam.ratio,
        'total_voltage': 0.,
        'induced_voltage': 0.,
        'rfp_omega_rf': rf.omega_rf,
        'rfp_omega_rf_d': rf.omega_rf_d,
        'rfp_phi_rf': rf.phi_rf,
        'rfp_dphi_rf': rf.dphi_rf,
        'rfp_harmonic': rf.harmonic,
        'rfp_voltage': rf.voltage,
        'rfp_phi_s': rf.phi_s,
        'tracker_acc_kick': tracker.acceleration_kick
    }

    master.multi_bcast(init_dict)

    vars_dict = {
        'dt': beam.dt,
        'dE': beam.dE
    }
    master.multi_scatter(vars_dict)
    # master.bcast(['histo', 'gather_single'])
    master.bcast(['histo', 'reduce_histo'])
    profile.track()
    print("Ready for tracking!")
    print("")

    # Tracking --------------------------------------------------------------------
    for i in range(N_t):
        # for i in range(turns):
        t0 = time.clock()

        # task_list = ['bcast']
        task_list = []

        if (i % N_t_reduce == 0):
            task_list += ['histo', 'reduce_histo']

        if (N_t_monitor > 0) and (i % N_t_monitor == 0):
            task_list += ['gather_single']

        task_list += ['kick', 'drift']
        master.bcast(task_list)

        profile.track()

        if (N_t_monitor > 0) and (i % N_t_monitor == 0):
            master.gather_single(
                {'profile': profile.n_macroparticles}, msg=False)
            slicesMonitor.track(i)

        tracker.track()

    master.multi_gather(vars_dict)
    master.stop()
    master.disconnect()
except Exception as e:
    print(e)
    master.quit()
    master.disconnect()

end_t = time.time()
print('Total time: ', end_t - start_t)
# if report:
mpiprof.finalize()
timing.report(total_time=1e3*(end_t-start_t),
              out_dir=args['report'],
              out_file='master.csv')

print('dE mean: ', np.mean(beam.dE))
print('dE std: ', np.std(beam.dE))

# np.savetxt('out/coords_' "%d" % rf.counter[0] + '.dat',
# np.c_[beam.dt, beam.dE], fmt='%.10e')
if MONITORING:
    plots.track()

print("Done!")
print("")
