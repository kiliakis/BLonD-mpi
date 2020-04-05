# Simulation of LHC acceleration ramp to 6.5 TeV
# Noise injection through PL; both PL & SL closed
# Pre-distort noise spectrum to counteract PL action
# WITH intensity effects
#
# authors: H. Timko, K. Iliakis
#
import os
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

from blond.monitors.monitors import SlicesMonitor
from blond.toolbox.next_regular import next_regular
from blond.impedances.impedance import InducedVoltageFreq, TotalInducedVoltage
from blond.impedances.impedance_sources import InputTable
from blond.beam.profile import Profile, CutOptions
from blond.beam.distributions import bigaussian
from blond.beam.beam import Beam, Proton
from blond.llrf.rf_noise import FlatSpectrum, LHCNoiseFB
from blond.llrf.beam_feedback import BeamFeedback
from blond.trackers.tracker import RingAndRFTracker, FullRingAndRF
from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring
from blond.utils.mpi_config import worker, mpiprint
from blond.utils.input_parser import parse


REAL_RAMP = True    # track full ramp
MONITORING = False   # turn off plots and monitors

if MONITORING:
    from blond.monitors.monitors import BunchMonitor
    from blond.plots.plot import Plot
    from blond.plots.plot_beams import plot_long_phase_space
    from blond.plots.plot_slices import plot_beam_profile


this_directory = os.path.dirname(os.path.realpath(__file__)) + '/'
inputDir = os.path.join(this_directory, '../input_files/LHC/')

worker.greet()
if worker.isMaster:
    worker.print_version()

# Simulation parameters --------------------------------------------------------
# Bunch parameters
N_b = 1.2e9          # Intensity
n_particles = 250000         # Macro-particles
n_bunches = 48              # Number of bunches
freq_res = 2.09e5
# freq_res = 4.e5
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
    n_turns = 14000000       # Number of turns to track; full ramp: 8700001
else:
    n_turns = 500000
bl_target = 1.25e-9  # 4 sigma r.m.s. target bunch length in [ns]

n_turns_reduce = 1
n_iterations = n_turns
seed = 0
args = parse()

n_iterations = n_iterations if args['turns'] == None else args['turns']
n_particles = n_particles if args['particles'] == None else args['particles']
n_bunches = n_bunches if args['bunches'] == None else args['bunches']
n_turns_reduce = n_turns_reduce if args['reduce'] == None else args['reduce']
seed = seed if args['seed'] == None else args['seed']
approx = args['approx']
timing.mode = args['time']
os.environ['OMP_NUM_THREADS'] = str(args['omp'])
withtp = args['withtp']


mpiprint(args)
# mpiprint({'iterations': n_iterations, 'particles_per_bunch': n_particles,
#           'timing.mode': timing.mode, 'n_bunches': n_bunches,
#           'n_turns_reduce': n_turns_reduce,
#           'seed': seed, 'log': args['log'], 'trace': args['trace'], 'approx': approx,
#           'withtp': withtp})


# Simulation setup -------------------------------------------------------------
mpiprint("Setting up the simulation...")
mpiprint("")

# Import pre-processed momentum and voltage for the acceleration ramp
if REAL_RAMP:
    ps = np.load(os.path.join(inputDir,'LHC_momentum_programme_6.5TeV.npz'))['arr_0']
    # ps = np.loadtxt(wrkDir+r'input/LHC_momentum_programme_6.5TeV.dat',
    # unpack=True)
    ps = np.ascontiguousarray(ps)
    ps = np.concatenate((ps, np.ones(436627)*6.5e12))
else:
    ps = 450.e9*np.ones(n_turns+1)
mpiprint("Flat top momentum %.4e eV" % ps[-1])
if REAL_RAMP:
    V = np.concatenate((np.linspace(6.e6, 12.e6, 13563374),
                        np.ones(436627)*12.e6))
else:
    V = 6.e6*np.ones(n_turns+1)
mpiprint("Flat top voltage %.4e V" % V[-1])
mpiprint("Momentum and voltage loaded...")

# Define general parameters
ring = Ring(C, alpha, ps[0:n_turns+1], Proton(), n_turns=n_turns)
mpiprint("General parameters set...")

# Define RF parameters (noise to be added for CC case)
rf = RFStation(ring, [h], [V[0:n_turns+1]], [0.])
mpiprint("RF parameters set...")

# Generate RF phase noise
LHCnoise = FlatSpectrum(ring, rf, fmin_s0=0.8571, fmax_s0=1.001,
                        initial_amplitude=1.e-5,
                        predistortion='weightfunction')
LHCnoise.dphi = np.load(
    os.path.join(inputDir, 'LHCNoise_fmin0.8571_fmax1.001_ampl1e-5_weightfct_6.5TeV.npz'))['arr_0']
LHCnoise.dphi = np.ascontiguousarray(LHCnoise.dphi[0:n_turns+1])
mpiprint("RF phase noise loaded...")

# FULL BEAM
bunch = Beam(ring, n_particles, N_b)
beam = Beam(ring, n_particles*n_bunches, N_b)
bigaussian(ring, rf, bunch, 0.3e-9, reinsertion=True, seed=seed)
bunch_spacing_buckets = 10

for i in np.arange(n_bunches):
    beam.dt[i*n_particles:(i+1)*n_particles] = bunch.dt[0:n_particles] + i*rf.t_rf[0, 0]*10
    beam.dE[i*n_particles:(i+1)*n_particles] = bunch.dE[0:n_particles]


# Profile required for PL
cutRange = (n_bunches-1)*25.e-9+3.5e-9
n_slices = np.int(cutRange/0.025e-9 + 1)
n_slices = next_regular(n_slices)
profile = Profile(beam, CutOptions(n_slices=n_slices, cut_left=-0.5e-9,
                                   cut_right=(cutRange-0.5e-9)))
mpiprint("Beam generated, profile set...")
mpiprint("Using %d slices" % n_slices)

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
         % (rf.omega_s0[0], rf.omega_s0[n_turns]))
mpiprint("   SL a_i = %.4f a_f = %.4f" % (PL.lhc_a[0], PL.lhc_a[n_turns]))
mpiprint("   SL t_i = %.4f t_f = %.4f" % (PL.lhc_t[0], PL.lhc_t[n_turns]))

# Injecting noise in the cavity, PL on

# Define machine impedance from http://impedance.web.cern.ch/impedance/
ZTot = np.loadtxt(os.path.join(inputDir, 'Zlong_Allthemachine_450GeV_B1_LHC_inj_450GeV_B1.dat'),
                  skiprows=1)
ZTable = InputTable(ZTot[:, 0], ZTot[:, 1], ZTot[:, 2])
indVoltage = InducedVoltageFreq(
    beam, profile, [ZTable], frequency_resolution=freq_res)
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
mpiprint('dt mean, 1st bunch: ', np.mean(beam.dt[:n_particles]))
mpiprint('shift ', rf.phi_rf[0, 0]/rf.omega_rf[0, 0])

beam.split_random()

mpiprint("Statistics set...")


# if N_t_monitor > 0 and worker.isMaster:
#     if args.get('monitorfile', None):
#         filename = args['monitorfile']
#     else:
#         filename = 'profiles/LHC-v0-t{}-p{}-b{}-sl{}-r{}-m{}-se{}-w{}'.format(
#             n_turns, n_particles, n_bunches, n_slices, n_turns_reduce, N_t_monitor, seed, worker.workers)
#     slicesMonitor = SlicesMonitor(filename=filename,
#                                   n_turns=np.ceil(1.0 * n_turns / N_t_monitor),
#                                   profile=profile,
#                                   rf=rf,
#                                   Nbunches=n_bunches)

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
elif args['loadbalance'] == 'reportonly':
    if args['loadbalancearg'] != 0:
        lbturns = np.arange(worker.start_turn, n_iterations, args['loadbalancearg'])
    else:
        lbturns = np.arange(worker.start_turn, n_iterations, 100)


worker.sync()
timing.reset()
start_t = time.time()
tcomp_old = tcomm_old = tconst_old = tsync_old = 0

for turn in range(n_iterations):
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
        worker.sync()
        profile.reduce_histo()
    elif (approx == 1) and (turn % n_turns_reduce == 0):
        profile.track()
        worker.sync()
        profile.reduce_histo()
    elif (approx == 2):
        profile.track()
        profile.scale_histo()

    # if (approx == 0) or (approx == 2):
    #     totVoltage.induced_voltage_sum()
    # elif (approx == 1) and (turn % n_turns_reduce == 0):
    #     totVoltage.induced_voltage_sum()

    # tracker.track()

    if worker.isFirst:
        if (approx == 0) or (approx == 2):
            totVoltage.induced_voltage_sum()
        elif (approx == 1) and (turn % n_turns_reduce == 0):
            totVoltage.induced_voltage_sum()
    if worker.isLast:
        tracker.pre_track()

    worker.intraSync()
    worker.sendrecv(totVoltage.induced_voltage, tracker.rf_voltage)

    tracker.track_only()

    if (turn in lbturns):
        tcomp_new = timing.get(['comp:'])
        tcomm_new = timing.get(['comm:'])
        tconst_new = timing.get(['serial:'], ['serial:sync'])
        tsync_new = timing.get(['serial:sync'])
        if args['loadbalance'] != 'reportonly':
            intv = worker.redistribute(turn, beam, tcomp=tcomp_new-tcomp_old,
                                       # tconst=(tconst_new-tconst_old))
                                       tconst=(tconst_new-tconst_old) + (tcomm_new - tcomm_old))
        if args['loadbalance'] == 'dynamic':
            lbturns[0] += intv
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
# mpiprint('dt mean, 1st bunch: ', np.mean(beam.dt[:n_particles]))
# mpiprint('shift ', rf.phi_rf[0, turn]/rf.omega_rf[0, turn])

mpiprint('profile mean: ', np.mean(profile.n_macroparticles))
mpiprint('profile std: ', np.std(profile.n_macroparticles))

mpiprint('Done!')
