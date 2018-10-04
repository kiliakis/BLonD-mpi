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

addload = 0.0

if args.get('turns', None) is not None:
    N_t = args['turns']
if args.get('particles', None) is not None:
    N_p = args['particles']

if args.get('bunches', None) is not None:
    NB = args['bunches']

if args.get('reduce', None) is not None:
    N_t_reduce = args['reduce']

if args.get('addload', None) is not None:
    addload = args['addload']

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



print({'N_t':N_t, 'N_p':N_p, 
        'timing.mode':timing.mode, 'NB':NB, 
        'addload': addload,
        'N_t_reduce':N_t_reduce,
        'N_t_monitor':N_t_monitor, 'seed':seed, 'log':log})

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
print("Flat top momentum %.4e eV" % ps[-1])
if REAL_RAMP:
    V = np.concatenate((np.linspace(6.e6, 12.e6, 13563374),
                        np.ones(436627)*12.e6))
else:
    V = 6.e6*np.ones(N_t+1)
print("Flat top voltage %.4e V" % V[-1])
print("Momentum and voltage loaded...")

# Define general parameters
ring = Ring(C, alpha, ps[0:N_t+1], Proton(), n_turns=N_t)
print("General parameters set...")

# Define RF parameters (noise to be added for CC case)
rf = RFStation(ring, [h], [V[0:N_t+1]], [0.])
print("RF parameters set...")

# Generate RF phase noise
LHCnoise = FlatSpectrum(ring, rf, fmin_s0=0.8571, fmax_s0=1.001,
                        initial_amplitude=1.e-5,
                        predistortion='weightfunction')
LHCnoise.dphi = np.genfromtxt(
    # wrkDir+r'input/LHCNoise_fmin0.8571_fmax1.001_ampl1e-5_weightfct_6.5TeV.dat',
    wrkDir+r'input/LHCNoise_fmin0.8571_fmax1.001_ampl1e-5_weightfct.dat',
    unpack=True,
    max_rows=N_t+1)
LHCnoise.dphi = np.ascontiguousarray(LHCnoise.dphi[0:N_t+1])
print("RF phase noise loaded...")

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

# Define emittance BUP feedback
noiseFB = LHCNoiseFB(rf, profile, bl_target)
print("Phase noise feedback set...")

# Define phase loop and frequency loop gain
PL_gain = 1./(5.*ring.t_rev[0])
SL_gain = PL_gain/10.

# Noise injected in the PL delayed by one turn and opposite sign
config = {'machine': 'LHC', 'PL_gain': PL_gain, 'SL_gain': SL_gain}
PL = BeamFeedback(ring, rf, profile, config, PhaseNoise=LHCnoise)
# LHCNoiseFB=noiseFB)
print("   PL gain is %.4e 1/s for initial turn T0 = %.4e s" % (PL.gain,
                                                               ring.t_rev[0]))
print("   SL gain is %.4e turns" % PL.gain2)
print("   Omega_s0 = %.4e s at flat bottom, %.4e s at flat top"
      % (rf.omega_s0[0], rf.omega_s0[N_t]))
print("   SL a_i = %.4f a_f = %.4f" % (PL.lhc_a[0], PL.lhc_a[N_t]))
print("   SL t_i = %.4f t_f = %.4f" % (PL.lhc_t[0], PL.lhc_t[N_t]))

# Injecting noise in the cavity, PL on

# Define machine impedance from http://impedance.web.cern.ch/impedance/
ZTot = np.loadtxt(wrkDir + r'input/Zlong_Allthemachine_450GeV_B1_LHC_inj_450GeV_B1.dat',
                  skiprows=1)
ZTable = InputTable(ZTot[:, 0], ZTot[:, 1], ZTot[:, 2])
indVoltage = InducedVoltageFreq(
    beam, profile, [ZTable], frequency_resolution=4.e5)
totVoltage = TotalInducedVoltage(beam, profile, [indVoltage])

tracker = RingAndRFTracker(rf, beam, BeamFeedback=PL, Profile=profile,
                           interpolation=True, TotalInducedVoltage=totVoltage)
print("PL, SL, and tracker set...")
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
print("Statistics set...")

# Define what to save in file
if MONITORING:
    monitor = BunchMonitor(ring, rf, beam, 'output_data', buffer_time=dt_save,
                           Profile=profile, PhaseLoop=PL, LHCNoiseFB=noiseFB)
    monitor.track()

    # Set up plotting
    plots = Plot(ring, rf, beam, dt_plt, dt_save, 0, 2.5e-9, -1500e6, 1500e6,
                 separatrix_plot=True, Profile=profile, h5file='output_data',
                 output_frequency=dt_mon, PhaseLoop=PL, LHCNoiseFB=noiseFB)

    # Plot initial distribution
    plot_long_phase_space(ring, rf, beam, 0, 2.5e-9, -500e6, 500e6,
                          separatrix_plot=True)
    plot_beam_profile(profile, 0)

    print("Initial mean bunch position %.4e s" % (beam.mean_dt))
    print("Initial four-times r.m.s. bunch length %.4e s" % (4.*beam.sigma_dt))

    # Accelerator map
    map_ = [totVoltage] + [profile] + [tracker] + \
        [monitor] + [plots] + [noiseFB]
else:
    map_ = [totVoltage] + [profile] + [tracker] + [noiseFB]
print("Map set")


print('dE mean: ', np.mean(beam.dE))
print('dE std: ', np.std(beam.dE))

if N_t_monitor > 0:
    if args.get('monitorfile', None):
        filename = args['monitorfile']
    else:
        filename = 'profiles/LHC-v0-t{}-p{}-b{}-sl{}-r{}-m{}-se{}'.format(
            N_t, N_p, NB, nSlices, N_t_reduce, N_t_monitor, seed)
    slicesMonitor = SlicesMonitor(filename=filename,
                                  n_turns=np.ceil(1.0 * N_t / N_t_monitor),
                                  profile=profile)

master = mpiconf.Master(log=log, add_load=addload)
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
        'tracker_acc_kick': tracker.acceleration_kick,
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
        'lhc_noise_dphi': LHCnoise.dphi,
        'gain': PL.gain,
        'gain2': PL.gain2,
        'lhc_a': PL.lhc_a,
        'lhc_t': PL.lhc_t,
        'lhc_y': PL.lhc_y,
        'alpha': PL.alpha,
        'reference': PL.reference,
        'machine': PL.machine
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

    # Tracking --------------------------------------------------------------------
    for i in range(N_t):

        if (i % N_t_reduce == 0):
            task_list += ['histo', 'reduce_histo']

        if (N_t_monitor > 0) and (i % N_t_monitor == 0):
            task_list += ['gather_single']

        if (i % N_t_reduce == 0):
            task_list += ['induced_voltage_sum']

        task_list += ['beamFB', 'RFVCalc', 'LIKick_n_drift']

    master.bcast(task_list)

    # Tracking --------------------------------------------------------------------
    for i in range(N_t):
        # t0 = time.clock()

        # Remove lost particles to obtain a correct r.m.s. value
        # if (i % 1000) == 0:  # reduce computational costs
        #     master.multi_gather({'dt': beam.dt, 'dE': beam.dE})
        #     beam.losses_separatrix(ring, rf)

        # After the first 2/3 of the ramp, regulate down the bunch length
        if i == 9042249:
            noiseFB.bl_targ = 1.1e-9

        # totVoltage.induced_voltage_sum_and_histo()
        if (i % N_t_reduce == 0):
            totVoltage.induced_voltage_sum()

        if (N_t_monitor > 0) and (i % N_t_monitor == 0):
            master.gather_single(
                {'profile': profile.n_macroparticles}, msg=False)
            slicesMonitor.track(i)

        tracker.track()

        profile.track()

        # noiseFB.track()

        # Track
        # for m in map_:
        #     m.track()

        # Plots and outputting
        # if MONITORING and (i % dt_plt) == 0:
        # if (i % dt_plt) == 0:
        #     print("Outputting at time step %d, tracking time %.4e s..." % (i, t0))
        #     print("RF tracker counter is %d" % rf.counter[0])
        #     print("   Beam momentum %0.6e eV" % beam.momentum)
        #     print("   Beam energy %.6e eV" % beam.energy)
        #     print("   Design RF revolution frequency %.10e Hz" %
        #           rf.omega_rf_d[0, i])
        #     print("   RF revolution frequency %.10e Hz" % rf.omega_rf[0, i])
        #     print("   RF phase %.4f rad" % rf.phi_rf[0, i])
        #     print("   Beam phase %.4f rad" % PL.phi_beam)
        #     print("   Phase noise %.4f rad" % (noiseFB.x*LHCnoise.dphi[i]))
        #     print("   PL phase error %.4f rad" % PL.RFnoise.dphi[i])
        #     print("   Synchronous phase %.4f rad" % rf.phi_s[i])
        #     print("   PL phase correction %.4f rad" % PL.dphi)
        #     print("   SL recursion variable %.4e" % PL.lhc_y)
        #     print("   Mean bunch position %.4e s" % (beam.mean_dt))
        #     print("   Four-times r.m.s. bunch length %.4e s" %
        #           (4.*beam.sigma_dt))
        #     print("   FWHM bunch length %.4e s" % noiseFB.bl_meas)
        #     print("")
        #     sys.stdout.flush()

        # # Save phase space data
        # if MONITORING and (i % dt_save) == 0:
        #     np.savetxt('out/coords_' "%d" % rf.counter[0] + '.dat',
        #                np.c_[beam.dt, beam.dE, beam.id], fmt='%.10e')

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
