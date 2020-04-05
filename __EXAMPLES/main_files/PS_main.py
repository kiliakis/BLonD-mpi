'''
PS longitudinal instability simulation along the ramp
'''
# General imports
import numpy as np
import time
import os
import sys
from scipy.constants import c
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
try:
    from pyprof import timing
    from pyprof import mpiprof
except ImportError:
    from blond.utils import profile_mock as timing
    mpiprof = timing
# import yaml


# BLonD imports
#from blond.beams.distributions import matched_from_line_density
from blond.utils.input_parser import parse
from blond.utils.mpi_config import worker, mpiprint
from blond.beam.beam import Proton, Beam
from blond.input_parameters.ring import Ring, RingOptions
from blond.input_parameters.rf_parameters import RFStation
from blond.beam.profile import Profile, CutOptions
from blond.beam.distributions_multibunch import match_beam_from_distribution
from blond.trackers.tracker import RingAndRFTracker, FullRingAndRF
from blond.impedances.impedance import InducedVoltageTime, InducedVoltageFreq, TotalInducedVoltage, InductiveImpedance
from blond.impedances.impedance_sources import Resonators
from blond.monitors.monitors import SlicesMonitor
# Other imports
from colormap import colormap
# LoCa imports
import LoCa.Base.Machine as mach
import LoCa.Base.RFProgram as rfp
import LoCa.Base.Bare_RF as brf
# Impedance scenario import
from PS_impedance.impedance_scenario import scenario
cmap = colormap.cmap_white_blue_red


this_directory = os.path.dirname(os.path.realpath(__file__)) + '/'
inputDir = os.path.join(this_directory, '../input_files/PS/')

worker.greet()
if worker.isMaster:
    worker.print_version()



# Simulation parameters -------------------------------------------------------

# Output parameters
bunch_by_bunch_output_step = 10
datamatrix_output_step = 1000
plot_step = 10000

# intensity_per_bunch = float(loadedParams['intensity_per_bunch'])
# bunch_length = float(loadedParams['bunch_length'])

intensity_per_bunch = 259999999999.99997
bunch_length = 2.9e-08


output_folder = this_directory + '/../output_files/bl_%.1f_int_%.2f/' % (
    bunch_length*1e9, intensity_per_bunch/1e11)

try:
    os.mkdir(output_folder)
except:
    pass

# Simulation inputs

loaded_program = np.load(os.path.join(inputDir, 'LHC1.npz'))
momentumTime = loaded_program['momentumTime'] / 1e3  # s
momentum = loaded_program['momentum'] * 1e9  # eV/c

rfPerHamonicTime = loaded_program['rfPerHamonicTime'] / 1e3  # s
rfPerHamonicDict = loaded_program['rfPerHamonicDict']
rfProgH21 = rfPerHamonicDict.item()['21'] * 1e3
rfProgH21[rfPerHamonicTime >= 2.710] = rfProgH21[rfPerHamonicTime >= 2.710][0]

c_time_injection = 0.170
c_time_extraction = 2.850

c_time_start = 2.055  # s
c_time_end = c_time_extraction

# General parameters PS
particle_type = Proton()
circumference = 2*np.pi*100                     # Machine circumference [m]
gamma_transition = 6.1                          # Transition gamma
momentum_compaction = 1./gamma_transition**2    # Momentum compaction array

# RF parameters PS
n_rf_systems = 1              # Number of rf systems second section
harmonic_number = 21           # Harmonic number section
phi_offset = 0                 # Phase offset

voltage_ratio = 0.15
harmonic_ratio = 4


# Beam parameters
n_bunches = 21
n_particles = 1e6
#exponent = 1.0

# Profile parameters PS
n_slices = 2**7

# Impedance parameters PS
n_turns_memory = 100
n_turns_reduce = 1
n_turns = 378708
n_iterations = n_turns

args = parse()


n_iterations = args.get('turns', n_iterations)
n_particles = args.get('particles', n_particles)
n_slices = args.get('slices', n_slices)
n_bunches = args.get('bunches', n_bunches)
n_turns_reduce = args.get('reduce', n_turns_reduce)
n_turns_memory = args.get('mtw', n_turns_memory)
timing.mode = args.get('time')
os.environ['OMP_NUM_THREADS'] = str(args.get('omp', '1'))
seed = args.get('seed')
approx = args.get('approx')
withtp = int(args.get('withtp'))

worker.initLog(args['log'], args['logdir'])
worker.initTrace(args['trace'], args['tracefile'])
worker.taskparallelism(withtp)

mpiprint(args)

# mpiprint({'n_iterations': n_iterations, 'particles_per_bunch': n_particles,
#        'n_slices': n_slices,
#        'n_turns_memory': n_turns_memory,
#        'timing.mode': timing.mode, 'n_bunches': n_bunches,
#        'n_turns_reduce': n_turns_reduce,
#        'seed': seed, 'log': log,
#        'approx': approx, 'withtp': withtp})


n_macroparticles = n_bunches * n_particles
intensity = (4*n_bunches*intensity_per_bunch)
intensity_per_bunch = intensity/n_bunches
bunch_spacing_buckets = 1
bunch_length = bunch_length

n_slices_total = n_slices * harmonic_number
cut_left = 0.
cut_right = harmonic_number*2*np.pi

filter_front_wake = 0.5
model = 'plateau_h21'
emittance_x_norm = 1.5e-6  # for space charge
emittance_y_norm = 1.5e-6  # for space charge


# Simulation setup ------------------------------------------------------------
# General parameters

momentum = momentum[(momentumTime >= c_time_injection) *
                    (momentumTime <= c_time_extraction)]
momentumTime = momentumTime[(
    momentumTime >= c_time_injection)*(momentumTime <= c_time_extraction)]

ring_options = RingOptions(t_start=c_time_start, interpolation='derivative')
ring = Ring(circumference, momentum_compaction, (momentumTime, momentum),
            particle_type, 1, RingOptions=ring_options)

n_turns = ring.n_turns

# RF parameters
rf_params = RFStation(ring, harmonic_number, (rfPerHamonicTime, rfProgH21), phi_offset,
                      1)

if n_rf_systems > 1:
    rf_params = RFStation(ring,
                          [harmonic_number, harmonic_ratio*harmonic_number],
                          [rf_params.voltage[0, :],
                           voltage_ratio*rf_params.voltage[0, :]],
                          [phi_offset*np.ones(len(rf_params.phi_s)),
                           np.pi-harmonic_ratio*rf_params.phi_s],
                          n_rf_systems)


# Evaluation of ramp parameters through LoCa
machine_LoCa = mach.Machine_Parameters(
    (ring.cycle_time, ring.momentum[0, :]),
    ring.Particle.mass,
    ring.Particle.charge,
    ring.alpha_0[0, :],
    ring.ring_length)

rf_prog_LoCa = rfp.RFProgram()
rf_prog_LoCa.add_system(harmonic=rf_params.harmonic[0, 0],
                        voltage=(ring.cycle_time, rf_params.voltage[0, :]),
                        phase=(ring.cycle_time, rf_params.phi_rf[0, :]))

rf_LoCa = brf.Bare_RF(machine_LoCa,
                      rf_prog_LoCa,
                      harmonic_divide=rf_params.harmonic[0, 0],
                      emittance=1.4)


turns_SC = []
momentum_SC = []
momentum_spread_SC = []

for bucket in rf_LoCa.buckets:
    turns_SC.append(bucket[0])
    momentum_SC.append(rf_LoCa.buckets[bucket].momentum)
    momentum_spread_SC.append(rf_LoCa.buckets[bucket].bunch_dp_over_p * 2 / 4.)


# Beam
beam = Beam(ring, n_macroparticles, intensity)

# Profile
cut_options = CutOptions(cut_left, cut_right, n_slices_total, cuts_unit='rad',
                         RFSectionParameters=rf_params)
profile = Profile(beam, cut_options)


# 10 MHz cavities are treated separately
impedance10MHzCavities = scenario(MODEL=model,
                                  method_10MHz='/rf_cavities/10MHz/All/Resonators/multi_resonators_h21.txt')
impedance10MHzCavities.importCavities10MHz(
    impedance10MHzCavities.freq_10MHz,
    method=impedance10MHzCavities.method_10MHz,
    RshFactor=impedance10MHzCavities.RshFactor_10MHz,
    QFactor=impedance10MHzCavities.QFactor_10MHz)

# The rest of the impedance model
impedanceRestOfMachine = scenario(MODEL=model,
                                  method_10MHz='/rf_cavities/10MHz/All/Resonators/multi_resonators_h21.txt')

impedanceRestOfMachine.importCavities20MHz(impedanceRestOfMachine.freq_20MHz,
                                           impedanceRestOfMachine.filename_20MHz,
                                           RshFactor=impedanceRestOfMachine.RshFactor_20MHz,
                                           QFactor=impedanceRestOfMachine.QFactor_20MHz)

impedanceRestOfMachine.importCavities40MHz(
    impedanceRestOfMachine.filename_40MHz)

impedanceRestOfMachine.importCavities40MHz_HOMs(impedanceRestOfMachine.filename_40MHz_HOMs,
                                                impedanceRestOfMachine.RshFactor_40MHz_HOMs,
                                                impedanceRestOfMachine.QFactor_40MHz_HOMs)

impedanceRestOfMachine.importCavities80MHz(
    impedanceRestOfMachine.filename_80MHz)

impedanceRestOfMachine.importCavities80MHz_HOMs(impedanceRestOfMachine.filename_80MHz_HOMs,
                                                impedanceRestOfMachine.RshFactor_80MHz_HOMs,
                                                impedanceRestOfMachine.QFactor_80MHz_HOMs)

impedanceRestOfMachine.importCavities200MHz(impedanceRestOfMachine.filename_200MHz,
                                            impedanceRestOfMachine.RshFactor_200MHz,
                                            impedanceRestOfMachine.QFactor_200MHz)

impedanceRestOfMachine.importKickers(impedanceRestOfMachine.filename_kickers)

impedanceRestOfMachine.importDump(impedanceRestOfMachine.filename_dump, impedanceRestOfMachine.ZFactor_dump,
                                  impedanceRestOfMachine.RshFactor_dump)

impedanceRestOfMachine.importValve(impedanceRestOfMachine.filename_valves, impedanceRestOfMachine.ZFactor_valves,
                                   impedanceRestOfMachine.RshFactor_valves)

impedanceRestOfMachine.importMUSectionUp(impedanceRestOfMachine.filename_mu_sections_up,
                                         impedanceRestOfMachine.ZFactor_mu_sections_up,
                                         impedanceRestOfMachine.RshFactor_mu_sections_up)

impedanceRestOfMachine.importMUSectionDown(impedanceRestOfMachine.filename_mu_sections_down,
                                           impedanceRestOfMachine.ZFactor_mu_sections_down,
                                           impedanceRestOfMachine.RshFactor_mu_sections_down)

impedanceRestOfMachine.importResistiveWall(np.linspace(0, 1e9, 1000))

# Space charge program
space_charge_z_over_n = impedanceRestOfMachine.importSpaceCharge(
    emittance_x_norm, emittance_y_norm, particle_type.mass, momentum_SC,
    momentum_spread_SC)


space_charge_z_over_n = np.interp(
    ring.cycle_time, ring.cycle_time[turns_SC], space_charge_z_over_n)

imp10MHzToBLonD = impedance10MHzCavities.export2BLonD()
impRestToBLonD = impedanceRestOfMachine.export2BLonD()


# Program for the 10 MHz caivties
time_gap_close = 24e-3

close_group_3 = {'enable': True, 'time': 2456e-3, 'n_cavities': 3}
close_group_4 = {'enable': True, 'time': 2566e-3, 'n_cavities': 3}
close_group_2 = {'enable': True, 'time': 2686e-3, 'n_cavities': 3}
close_group_1 = {'enable': False, 'time': 2769e-3, 'n_cavities': 1}

real_c_time = ring.cycle_time + c_time_start


def generate_gap_prog(close_group):

    gap_prog_group = np.ones(n_turns+1)

    if close_group['enable']:

        start = close_group['time']
        stop = close_group['time'] + time_gap_close

        slope = -1/time_gap_close
        origin = 1-slope*start

        prog_time = real_c_time[(real_c_time > start)*(real_c_time < stop)]
        gap_prog_group[(real_c_time > start)*(real_c_time < stop)
                       ] = slope*prog_time + origin

        gap_prog_group[real_c_time >= stop] = 0

    gap_prog_group *= close_group['n_cavities']

    return gap_prog_group


gap_prog_group_3 = generate_gap_prog(close_group_3)
gap_prog_group_4 = generate_gap_prog(close_group_4)
gap_prog_group_2 = generate_gap_prog(close_group_2)
gap_prog_group_1 = generate_gap_prog(close_group_1)

R_S_10MHz_save = np.array(imp10MHzToBLonD.wakeList[0].R_S)
R_S_program_10MHz = (gap_prog_group_3+gap_prog_group_4 +
                     gap_prog_group_2+gap_prog_group_1)/10.


# Building up BLonD objects
ResonatorsList10MHz = imp10MHzToBLonD.wakeList
ImpedanceTableList10MHz = imp10MHzToBLonD.impedanceList

ResonatorsListRest = impRestToBLonD.wakeList
ImpedanceTableListRest = impRestToBLonD.impedanceList


frequency_step = 1/(ring.t_rev[0]*n_turns_memory)  # [Hz]
front_wake_length = filter_front_wake * ring.t_rev[0]*n_turns_memory


PS_intensity_freq_Rest = InducedVoltageFreq(beam,
                                            profile,
                                            ResonatorsList10MHz+ImpedanceTableList10MHz +
                                            ResonatorsListRest+ImpedanceTableListRest,
                                            frequency_step,
                                            RFParams=rf_params,
                                            multi_turn_wake=True,
                                            front_wake_length=front_wake_length)

PS_inductive = InductiveImpedance(
    beam, profile, space_charge_z_over_n, rf_params, deriv_mode='gradient')

PS_intensity_plot = InducedVoltageFreq(beam,
                                       profile,
                                       ResonatorsList10MHz+ImpedanceTableList10MHz +
                                       ResonatorsListRest+ImpedanceTableListRest,
                                       frequency_step,
                                       RFParams=rf_params,
                                       multi_turn_wake=True,
                                       front_wake_length=front_wake_length)

# PS_longitudinal_intensity = TotalInducedVoltage(
#     beam, profile, [PS_intensity_freq_10MHz, PS_intensity_freq_Rest, PS_inductive])

PS_longitudinal_intensity = TotalInducedVoltage(
    beam, profile, [PS_intensity_freq_Rest, PS_inductive])


# RF tracker
tracker = RingAndRFTracker(
    rf_params, beam, interpolation=True, Profile=profile,
    TotalInducedVoltage=PS_longitudinal_intensity)
full_tracker = FullRingAndRF([tracker])

# Beam generation
distribution_options = {'type': 'parabolic_amplitude',
                        'density_variable': 'Hamiltonian',
                        'bunch_length': bunch_length}

match_beam_from_distribution(beam, full_tracker, ring,
                             distribution_options, n_bunches,
                             bunch_spacing_buckets,
                             main_harmonic_option='lowest_freq',
                             TotalInducedVoltage=PS_longitudinal_intensity,
                             n_iterations=10,
                             n_points_potential=int(1e3),
                             dt_margin_percent=0.1, seed=seed)

mpiprint('dE mean:', np.mean(beam.dE))
mpiprint('dE std:', np.std(beam.dE))

beam.split_random()

# Tracking -------------------------------------------------------------------


mpiprint("Ready for tracking!\n")
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


# for i in range(n_iterations):
for turn in range(n_iterations):

    # if (i > 0) and (i % datamatrix_output_step) == 0:
    #     t0 = time.time()

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

    # Change impedance of 10 MHz only if it changes
    # if (i > 0) and (R_S_program_10MHz[i] != R_S_program_10MHz[i-1]):
    #     PS_intensity_freq_10MHz.impedance_source_list[0].R_S[:] = \
    #         R_S_10MHz_save * R_S_program_10MHz[i]
    #     PS_intensity_freq_10MHz.sum_impedances(PS_intensity_freq_10MHz.freq)


    if worker.isFirst:
        if (approx == 0) or (approx == 2):
            PS_longitudinal_intensity.induced_voltage_sum()
        elif (approx == 1) and (turn % n_turns_reduce == 0):
            PS_longitudinal_intensity.induced_voltage_sum()
    if worker.isLast:
        tracker.pre_track()

    worker.intraSync()
    worker.sendrecv(PS_longitudinal_intensity.induced_voltage, tracker.rf_voltage)

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
mpiprint('profile mean: ', np.mean(profile.n_macroparticles))
mpiprint('profile std: ', np.std(profile.n_macroparticles))

mpiprint('Done!')

# Analysis