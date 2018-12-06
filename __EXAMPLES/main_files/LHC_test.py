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

import datetime
from pyprof import timing
from pyprof import mpiprof
import os

REAL_RAMP = True    # track full ramp
MONITORING = False   # turn off plots and monitors


from blond.input_parameters.ring import Ring
from blond.input_parameters.rf_parameters import RFStation
from blond.trackers.tracker import RingAndRFTracker, FullRingAndRF
from blond.llrf.beam_feedback import BeamFeedback
from blond.llrf.rf_noise import FlatSpectrum, LHCNoiseFB
from blond.beam.beam import Beam, Proton
from blond.beam.distributions import bigaussian  # matched_from_distribution_function
from blond.beam.profile import Profile, CutOptions
from blond.impedances.impedance_sources import InputTable
from blond.impedances.impedance import InducedVoltageFreq, TotalInducedVoltage
from blond.toolbox.next_regular import next_regular
# if MONITORING:
from blond.monitors.monitors import BunchMonitor
from blond.plots.plot import Plot
from blond.plots.plot_beams import plot_long_phase_space
from blond.plots.plot_slices import plot_beam_profile

from blond.monitors.monitors import SlicesMonitor

from blond.utils.input_parser import parse
from blond.utils import mpi_config as mpiconf

this_directory = os.path.dirname(os.path.realpath(__file__)) + '/'


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

if args.get('turns', None) is not None:
    N_t = args['turns']
if args.get('particles', None) is not None:
    N_p = args['particles']

if args.get('bunches', None) is not None:
    NB = args['bunches']

if args.get('reduce', None) is not None:
    N_t_reduce = args['reduce']

if args.get('monitor', None) is not None:
    N_t_monitor = args['monitor']

if args.get('omp', None) is not None:
    os.environ['OMP_NUM_THREADS'] = str(args['omp'])
if 'log' in args:
    log = args['log']

if args.get('time', False) is True:
    timing.mode = 'timing'

if args.get('seed', None) is not None:
    seed = args['seed']


# Simulation setup -------------------------------------------------------------
print("Setting up the simulation...")
print("")
wrkDir = r'/afs/cern.ch/work/k/kiliakis/public/helga/'

# Import pre-processed momentum and voltage for the acceleration ramp
if REAL_RAMP:
    ps = np.loadtxt(wrkDir+r'input/LHC_momentum_programme_6.5TeV.dat',
                    unpack=True)
    ps = np.ascontiguousarray(ps)
    ps = np.concatenate((ps, np.ones(436627)*6.5e12))
else:
    ps = 450.e9*np.ones(N_t+1)


# ps = 450.e9*np.ones(N_t+1)
print("Flat top momentum %.4e eV" % ps[-1])
if REAL_RAMP:
    V = np.concatenate((np.linspace(6.e6, 12.e6, 13563374),
                        np.ones(436627)*12.e6))
else:
    V = 6.e6*np.ones(N_t+1)

# V = 6.e6*np.ones(N_t+1)
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


# Define machine impedance from http://impedance.web.cern.ch/impedance/
ZTot = np.loadtxt(wrkDir + r'input/Zlong_Allthemachine_450GeV_B1_LHC_inj_450GeV_B1.dat',
                  skiprows=1)
ZTable = InputTable(ZTot[:, 0], ZTot[:, 1], ZTot[:, 2])
indVoltage = InducedVoltageFreq(
    beam, profile, [ZTable], frequency_resolution=4.e5)
totVoltage = TotalInducedVoltage(beam, profile, [indVoltage])

# tracker = RingAndRFTracker(rf, beam, BeamFeedback=None, Profile=profile,
#                            interpolation=True, TotalInducedVoltage=totVoltage)

tracker = RingAndRFTracker(rf, beam, BeamFeedback=None, Profile=profile,
                           interpolation=False, TotalInducedVoltage=totVoltage)


print("PL, SL, and tracker set...")
# Fill beam distribution
fullring = FullRingAndRF([tracker])


map_ = [profile] + [tracker]
print("Map set")


print('dE mean: ', np.mean(beam.dE))
print('dE std: ', np.std(beam.dE))

# plot_long_phase_space(ring, rf, beam, 0, 2.5e-9, -500e6, 500e6,
#                       separatrix_plot=True)
# plot_beam_profile(profile, 0)

if N_t_monitor > 0:
    if args.get('monitorfile', None):
        filename = args['monitorfile']
    else:
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
        'impedList': {
            'indVoltage': {'total_impedance': indVoltage.total_impedance,
                           'n_fft': indVoltage.n_fft,
                           'n_induced_voltage': indVoltage.n_induced_voltage}
        },
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

    task_list = []
    for i in range(N_t):
        if (i % N_t_reduce == 0):
            # task_list += ['induced_voltage_sum', 'histo', 'reduce_histo']
            task_list += ['histo', 'reduce_histo']

        if (N_t_monitor > 0) and (i % N_t_monitor == 0):
            task_list += ['gather_single']
        task_list += ['kick', 'drift']

        # task_list += ['RFVCalc', 'LIKick_n_drift']

    master.bcast(task_list)

    # Tracking --------------------------------------------------------------------
    for i in range(N_t):
        # if (i % N_t_reduce == 0):
        #     totVoltage.induced_voltage_sum()

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

# plot_long_phase_space(ring, rf, beam, 24 * 25e-9, 2.5e-9 + 24 * 25e-9, -500e6, 500e6,
#                       separatrix_plot=True)
# plot_beam_profile(profile, 0)

# np.savetxt('out/coords_' "%d" % rf.counter[0] + '.dat',
# np.c_[beam.dt, beam.dE], fmt='%.10e')
if MONITORING:
    plots.track()

print("Done!")
print("")
