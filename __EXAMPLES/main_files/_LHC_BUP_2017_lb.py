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
import os
import datetime
import sys
import time
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
try:
    from pyprof import timing
    from pyprof import mpiprof
except ImportError:
    from blond.utils import profile_mock as timing
    mpiprof = timing

# H. Timko
from blond.utils.input_parser import parse
from blond.utils.mpi_config import worker, mpiprint
from blond.input_parameters.ring import Ring
from blond.input_parameters.rf_parameters import RFStation
from blond.trackers.tracker import RingAndRFTracker, FullRingAndRF
from blond.llrf.beam_feedback import BeamFeedback
from blond.llrf.rf_noise import FlatSpectrum, LHCNoiseFB
from blond.beam.beam import Beam, Proton
from blond.beam.distributions import bigaussian
from blond.beam.profile import Profile, CutOptions
from blond.impedances.impedance_sources import InputTable
from blond.impedances.impedance import InducedVoltageFreq, TotalInducedVoltage
from blond.toolbox.next_regular import next_regular
from blond.monitors.monitors import SlicesMonitor

REAL_RAMP = True    # track full ramp
MONITORING = False   # turn off plots and monitors

if MONITORING:
    from blond.monitors.monitors import BunchMonitor
    from blond.plots.plot import Plot
    from blond.plots.plot_beams import plot_long_phase_space
    from blond.plots.plot_slices import plot_beam_profile


# matched_from_distribution_function

this_directory = os.path.dirname(os.path.realpath(__file__)) + '/'

worker.greet()
if worker.isMaster:
    worker.print_version()

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

log = None
report = None
N_t_reduce = 1
N_t_monitor = 0
seed = 0
approx = 0

args = parse()

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

if args.get('log', None) is not None:
    log = args['log']

if args.get('time', False) is True:
    timing.mode = 'timing'

if args.get('seed', None) is not None:
    seed = args['seed']

if args.get('approx', None) is not None:
    approx = int(args['approx'])


mpiprint({'N_t': N_t, 'N_p': N_p,
          'timing.mode': timing.mode, 'n_bunches': NB,
          'N_t_reduce': N_t_reduce,
          'N_t_monitor': N_t_monitor,
          'seed': seed, 'log': log,  'approx': approx})


# Simulation setup -------------------------------------------------------------
mpiprint("Setting up the simulation...")
mpiprint("")
wrkDir = r'/afs/cern.ch/work/k/kiliakis/public/helga/'

# Import pre-processed momentum and voltage for the acceleration ramp
if REAL_RAMP:
    ps = np.load(wrkDir+r'input/LHC_momentum_programme_6.5TeV.npz')['arr_0']
    # ps = np.loadtxt(wrkDir+r'input/LHC_momentum_programme_6.5TeV.dat',
    # unpack=True)
    ps = np.ascontiguousarray(ps)
    ps = np.concatenate((ps, np.ones(436627)*6.5e12))
else:
    ps = 450.e9*np.ones(N_t+1)
mpiprint("Flat top momentum %.4e eV" % ps[-1])
if REAL_RAMP:
    V = np.concatenate((np.linspace(6.e6, 12.e6, 13563374),
                        np.ones(436627)*12.e6))
else:
    V = 6.e6*np.ones(N_t+1)
mpiprint("Flat top voltage %.4e V" % V[-1])
mpiprint("Momentum and voltage loaded...")

# Define general parameters
ring = Ring(C, alpha, ps[0:N_t+1], Proton(), n_turns=N_t)
mpiprint("General parameters set...")

# Define RF parameters (noise to be added for CC case)
rf = RFStation(ring, [h], [V[0:N_t+1]], [0.])
mpiprint("RF parameters set...")

# Generate RF phase noise
LHCnoise = FlatSpectrum(ring, rf, fmin_s0=0.8571, fmax_s0=1.001,
                        initial_amplitude=1.e-5,
                        predistortion='weightfunction')
# LHCnoise.dphi = np.genfromtxt(
#     wrkDir+r'input/LHCNoise_fmin0.8571_fmax1.001_ampl1e-5_weightfct_6.5TeV.dat',
#     # (?# wrkDir+r'input/LHCNoise_fmin0.8571_fmax1.001_ampl1e-5_weightfct.dat',)
#     unpack=True,
#     max_rows=N_t+1)
LHCnoise.dphi = np.load(
    wrkDir+r'input/LHCNoise_fmin0.8571_fmax1.001_ampl1e-5_weightfct_6.5TeV.npz')['arr_0']
LHCnoise.dphi = np.ascontiguousarray(LHCnoise.dphi[0:N_t+1])
mpiprint("RF phase noise loaded...")

# FULL BEAM
bunch = Beam(ring, N_p, N_b)
beam = Beam(ring, N_p*NB, N_b)
bigaussian(ring, rf, bunch, 0.3e-9, reinsertion=True, seed=seed)
bunch_spacing_buckets = 10

for i in np.arange(NB):
    beam.dt[i*N_p:(i+1)*N_p] = bunch.dt[0:N_p] + i*rf.t_rf[0, 0]*10
    beam.dE[i*N_p:(i+1)*N_p] = bunch.dE[0:N_p]


# Profile required for PL
cutRange = (NB-1)*25.e-9+3.5e-9
nSlices = np.int(cutRange/0.025e-9 + 1)
nSlices = next_regular(nSlices)
profile = Profile(beam, CutOptions(n_slices=nSlices, cut_left=-0.5e-9,
                                   cut_right=(cutRange-0.5e-9)))
mpiprint("Beam generated, profile set...")
mpiprint("Using %d slices" % nSlices)

# Define emittance BUP feedback
noiseFB = LHCNoiseFB(rf, profile, bl_target)
mpiprint("Phase noise feedback set...")

# Define phase loop and frequency loop gain
PL_gain = 1./(5.*ring.t_rev[0])
SL_gain = PL_gain/10.

# Noise injected in the PL delayed by one turn and opposite sign
config = {'machine': 'LHC', 'PL_gain': PL_gain, 'SL_gain': SL_gain}
PL = BeamFeedback(ring, rf, profile, config, PhaseNoise=LHCnoise,
                  LHCNoiseFB=noiseFB)
mpiprint("   PL gain is %.4e 1/s for initial turn T0 = %.4e s" % (PL.gain,
                                                                  ring.t_rev[0]))
mpiprint("   SL gain is %.4e turns" % PL.gain2)
mpiprint("   Omega_s0 = %.4e s at flat bottom, %.4e s at flat top"
         % (rf.omega_s0[0], rf.omega_s0[N_t]))
mpiprint("   SL a_i = %.4f a_f = %.4f" % (PL.lhc_a[0], PL.lhc_a[N_t]))
mpiprint("   SL t_i = %.4f t_f = %.4f" % (PL.lhc_t[0], PL.lhc_t[N_t]))

# Injecting noise in the cavity, PL on

# Define machine impedance from http://impedance.web.cern.ch/impedance/
ZTot = np.loadtxt(wrkDir + r'input/Zlong_Allthemachine_450GeV_B1_LHC_inj_450GeV_B1.dat',
                  skiprows=1)
ZTable = InputTable(ZTot[:, 0], ZTot[:, 1], ZTot[:, 2])
indVoltage = InducedVoltageFreq(
    beam, profile, [ZTable], frequency_resolution=4.e5)
totVoltage = TotalInducedVoltage(beam, profile, [indVoltage])

# TODO add the noiseFB
tracker = RingAndRFTracker(rf, beam, BeamFeedback=PL, Profile=profile,
                           interpolation=True, TotalInducedVoltage=totVoltage)
# interpolation=True, TotalInducedVoltage=None)
mpiprint("PL, SL, and tracker set...")
# Fill beam distribution
fullring = FullRingAndRF([tracker])
# Juan's fit to LHC profiles: binomial w/ exponent 1.5
# matched_from_distribution_function(beam, fullring,
#    main_harmonic_option = 'lowest_freq',
#    distribution_exponent = 1.5, distribution_type='binomial',
#    bunch_length = 1.1e-9, bunch_length_fit = 'fwhm',
#    distribution_variable = 'Action')

# Initial losses, slicing, statistics
beam.losses_separatrix(ring, rf)

mpiprint('dE mean: ', np.mean(beam.dE))
mpiprint('dE std: ', np.std(beam.dE))
mpiprint('dt mean, 1st bunch: ', np.mean(beam.dt[:N_p]))
mpiprint('shift ', rf.phi_rf[0, 0]/rf.omega_rf[0, 0])
# plots = Plot(ring, rf, beam, dt_plt, dt_save, 0, 2.5e-9, -1500e6, 1500e6,
#              separatrix_plot=True, Profile=profile, h5file='output_data',
#              output_frequency=dt_mon, PhaseLoop=PL, LHCNoiseFB=None)
# if worker.isMaster:
#     plot_long_phase_space(ring, rf, beam, 0, 2.5e-9, -1500e6, 1500e6,
#                           separatrix_plot=True)

beam.split_random()

mpiprint("Statistics set...")

# Define what to save in file
# if MONITORING:
#     monitor = BunchMonitor(ring, rf, beam, 'output_data', buffer_time=dt_save,
#                            Profile=profile, PhaseLoop=PL, LHCNoiseFB=noiseFB)
#     monitor.track()

#     # Set up plotting
#     plots = Plot(ring, rf, beam, dt_plt, dt_save, 0, 2.5e-9, -1500e6, 1500e6,
#                  separatrix_plot=True, Profile=profile, h5file='output_data',
#                  output_frequency=dt_mon, PhaseLoop=PL, LHCNoiseFB=noiseFB)

#     # Plot initial distribution
#     plot_long_phase_space(ring, rf, beam, 0, 2.5e-9, -500e6, 500e6,
#                           separatrix_plot=True)
#     plot_beam_profile(profile, 0)

#     mpiprint("Initial mean bunch position %.4e s" % (beam.mean_dt))
#     mpiprint("Initial four-times r.m.s. bunch length %.4e s" % (4.*beam.sigma_dt))

#     # Accelerator map
#     map_ = [totVoltage] + [profile] + [tracker] + \
#         [monitor] + [plots] + [noiseFB]
# else:
#     map_ = [totVoltage] + [profile] + [tracker] + [noiseFB]


if N_t_monitor > 0 and worker.isMaster:
    if args.get('monitorfile', None):
        filename = args['monitorfile']
    else:
        filename = 'profiles/LHC-v0-t{}-p{}-b{}-sl{}-r{}-m{}-se{}-w{}'.format(
            N_t, N_p, NB, nSlices, N_t_reduce, N_t_monitor, seed, worker.workers)
    slicesMonitor = SlicesMonitor(filename=filename,
                                  n_turns=np.ceil(1.0 * N_t / N_t_monitor),
                                  profile=profile,
                                  rf=rf,
                                  Nbunches=NB)

mpiprint("Map set")


timing.reset()
start_t = time.time()

lbturns = []
if args['loadbalance'] == 'times':
    if args['loadbalancearg'] != 0:
        intv = N_t // (args['loadbalancearg']+1)
    else:
        intv = N_t // (10 +1)
    lbturns = np.arange(0, N_t, intv)[1:]

elif args['loadbalance'] == 'interval':
    if args['loadbalancearg'] != 0:
        lbturns = np.arange(0, N_t, args['loadbalancearg'])
    else:
        lbturns = np.arange(0, N_t, 1000)

elif args['loadbalance'] == 'dynamic':
    print('Warning: Dynamic load balance policy not supported.')
ts = worker.time()

for turn in range(N_t):
    # Plots and outputting
    # if MONITORING and (i % dt_plt) == 0:
    # if (i % dt_plt) == 0:
    #     mpiprint("Outputting at time step %d, tracking time %.4e s..." % (i, t0))
    #     mpiprint("RF tracker counter is %d" % rf.counter[0])
    #     mpiprint("   Beam momentum %0.6e eV" % beam.momentum)
    #     mpiprint("   Beam energy %.6e eV" % beam.energy)
    #     mpiprint("   Design RF revolution frequency %.10e Hz" %
    #           rf.omega_rf_d[0, i])
    #     mpiprint("   RF revolution frequency %.10e Hz" % rf.omega_rf[0, i])
    #     mpiprint("   RF phase %.4f rad" % rf.phi_rf[0, i])
    #     mpiprint("   Beam phase %.4f rad" % PL.phi_beam)
    #     mpiprint("   Phase noise %.4f rad" % (noiseFB.x*LHCnoise.dphi[i]))
    #     mpiprint("   PL phase error %.4f rad" % PL.RFnoise.dphi[i])
    #     mpiprint("   Synchronous phase %.4f rad" % rf.phi_s[i])
    #     mpiprint("   PL phase correction %.4f rad" % PL.dphi)
    #     mpiprint("   SL recursion variable %.4e" % PL.lhc_y)
    #     mpiprint("   Mean bunch position %.4e s" % (beam.mean_dt))
    #     mpiprint("   Four-times r.m.s. bunch length %.4e s" %
    #           (4.*beam.sigma_dt))
    #     mpiprint("   FWHM bunch length %.4e s" % noiseFB.bl_meas)
    #     mpiprint("")
    #     sys.stdout.flush()
    # Remove lost particles to obtain a correct r.m.s. value
    # if (i % 1000) == 0:  # reduce computational costs
    #     master.multi_gather({'dt': beam.dt, 'dE': beam.dE})
    #     beam.losses_separatrix(ring, rf)

    # After the first 2/3 of the ramp, regulate down the bunch length
    if turn == 9042249:
        noiseFB.bl_targ = 1.1e-9

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
        beam.statistics()
        beam.gather_statistics()
        if worker.isMaster:
            profile.fwhm_multibunch(NB, bunch_spacing_buckets, rf.t_rf[0, turn],
                                    bucket_tolerance=0.,
                                    shiftX=-rf.phi_rf[0, turn]/rf.omega_rf[0, turn])
            slicesMonitor.track(turn)

    if worker.isHostFirst:
        if (approx == 0) or (approx == 2):
            totVoltage.induced_voltage_sum()
        elif (approx == 1) and (turn % N_t_reduce == 0):
            totVoltage.induced_voltage_sum()
    if worker.isHostLast:
        tracker.pre_track()

    worker.sendrecv(totVoltage.induced_voltage, tracker.rf_voltage)

    tracker.track_only()

    if turn in lbturns:
        worker.redistribute(beam, worker.time() - ts)
        ts = worker.time()
        
    # worker.hostsync()
    # worker.sync()
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(profile.bin_centers[-200:], profile.n_macroparticles[-200:])
    # plt.savefig('plot' + str(turn) + '.png')

    # plt.show()

beam.gather()
end_t = time.time()

timing.report(total_time=1e3*(end_t-start_t),
              out_dir=args['timedir'],
              out_file='worker-{}.csv'.format(os.getpid()))

# mpiprof.finalize()
worker.finalize()


# plot_long_phase_space(ring, rf, beam, 0, 2.5e-9, -1500e6, 1500e6,
#                       separatrix_plot=True)

if N_t_monitor > 0:
    slicesMonitor.close()


mpiprint('dE mean: ', np.mean(beam.dE))
mpiprint('dE std: ', np.std(beam.dE))
mpiprint('dt mean, 1st bunch: ', np.mean(beam.dt[:N_p]))
mpiprint('shift ', rf.phi_rf[0, turn]/rf.omega_rf[0, turn])

mpiprint('profile mean: ', np.mean(profile.n_macroparticles))
mpiprint('profile std: ', np.std(profile.n_macroparticles))

mpiprint('Done!')
