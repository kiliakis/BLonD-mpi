'''
PS longitudinal instability simulation along the ramp
'''

# General imports
import numpy as np
import time
import os
import sys
# import yaml
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# BLonD imports
from beam.beam import Proton, Beam
from input_parameters.ring import Ring, RingOptions
from input_parameters.rf_parameters import RFStation
from beam.profile import Profile, CutOptions
#from beams.distributions import matched_from_line_density
from beam.distributions_multibunch import match_beam_from_distribution
from trackers.tracker import RingAndRFTracker, FullRingAndRF
from impedances.impedance_sources import Resonators
from impedances.impedance import InducedVoltageTime, InducedVoltageFreq, \
    TotalInducedVoltage, InductiveImpedance

# LoCa imports
import LoCa.Base.Machine as mach
import LoCa.Base.RFProgram as rfp
import LoCa.Base.Bare_RF as brf

# Impedance scenario import
from impedances.PS_impedance.impedance_scenario import scenario

# Other imports
from scipy.constants import c
from colormap import colormap

from monitors.monitors import SlicesMonitor
from utils.input_parser import parse
from utils import mpi_config as mpiconf
from pyprof import timing
# from pyprof import mpiprof

cmap = colormap.cmap_white_blue_red
currdir = os.path.dirname(os.path.realpath(__file__))

args = parse()
mpiconf.init(trace=args['trace'], logfile=args['tracefile'])

print(args)

# Simulation parameters -------------------------------------------------------

# Output parameters
bunch_by_bunch_output_step = 10
datamatrix_output_step = 1000
plot_step = 10000

# intensity_per_bunch = float(loadedParams['intensity_per_bunch'])
# bunch_length = float(loadedParams['bunch_length'])

intensity_per_bunch = 259999999999.99997
bunch_length = 2.9e-08


output_folder = currdir + '/../output_files/bl_%.1f_int_%.2f/' % (
    bunch_length*1e9, intensity_per_bunch/1e11)

# try:
#     os.mkdir(output_folder)
# except:
#     pass

# Simulation inputs

loaded_program = np.load(currdir + '/../input_files/LHC1.npz')
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
n_macroparticles_per_bunch = 1e6
#exponent = 1.0

# Profile parameters PS
n_slices_per_bunch = 2**7

# Impedance parameters PS
n_turns_memory = 100


N_t_reduce = 1
N_t_monitor = 0
seed = 0
N_t = 378708


if args.get('turns', None) is not None:
    N_t = args['turns']
if args.get('particles', None) is not None:
    n_macroparticles_per_bunch = args['particles']

if args.get('bunches', None) is not None:
    n_bunches = args['bunches']

if args.get('slices', None) is not None:
    n_slices_per_bunch = args['slices']

if args.get('mtw', None) is not None:
    n_turns_memory = args['mtw']


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


n_macroparticles = n_bunches * n_macroparticles_per_bunch
intensity = (4*n_bunches*intensity_per_bunch)
intensity_per_bunch = intensity/n_bunches
bunch_spacing_buckets = 1
bunch_length = bunch_length

n_slices = n_slices_per_bunch * harmonic_number
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

# plt.figure('programs')
# plt.clf()
# plt.plot(ring.cycle_time*1e3, ring.momentum[0, :]/1e9, 'k')
# plt.xlabel('C-time [ms]')
# plt.ylabel('Momentum [GeV/c]')
# plt.twinx()
# for indexRF in range(rf_params.n_rf):
#     plt.plot(ring.cycle_time*1e3,
#              rf_params.voltage[indexRF, :]/1e3, label='h%d' % (rf_params.harmonic[indexRF, 0]))
# plt.ylabel('RF voltage [kV]')
# plt.legend(loc='best')
# plt.savefig(output_folder+'/programs')

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
cut_options = CutOptions(cut_left, cut_right, n_slices, cuts_unit='rad',
                         RFSectionParameters=rf_params)
profile = Profile(beam, cut_options)

# Loading impedance scenario
# Import impedance sources one by one

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

# plt.figure('Space charge')
# plt.clf()
# plt.plot(ring.cycle_time*1e3, ring.momentum[0, :]/1e9, 'k')
# plt.xlabel('C-time [ms]')
# plt.ylabel('Momentum [GeV/c]')
# plt.twinx()
# plt.plot(ring.cycle_time[turns_SC]*1e3, space_charge_z_over_n)
# plt.ylabel('Space charge $\\mathrm{Im}\\mathcal{Z}/n$ [GeV/c]')
# plt.savefig(output_folder+'/space_charge')

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

# plt.figure('10 MHz prog')
# plt.clf()
# plt.plot(ring.cycle_time*1e3, R_S_program_10MHz*10)
# plt.xlabel('C-time [ms]')
# plt.ylabel('10 MHz gaps')
# plt.twinx()
# plt.plot(ring.cycle_time*1e3, rf_params.voltage[0, :]/1e3, 'k', label='Vprog')
# plt.plot(ring.cycle_time*1e3, R_S_program_10MHz*10*20e3/1e3, 'r', label='Vmax')
# plt.ylabel('RF voltage [kV]')
# plt.legend(loc='best')
# plt.savefig(output_folder+'/10MHz_prog')

# Building up BLonD objects
ResonatorsList10MHz = imp10MHzToBLonD.wakeList
ImpedanceTableList10MHz = imp10MHzToBLonD.impedanceList

ResonatorsListRest = impRestToBLonD.wakeList
ImpedanceTableListRest = impRestToBLonD.impedanceList


frequency_step = 1/(ring.t_rev[0]*n_turns_memory)  # [Hz]
front_wake_length = filter_front_wake * ring.t_rev[0]*n_turns_memory

# PS_intensity_freq_10MHz = InducedVoltageFreq(beam,
#                                              profile,
#                                              ResonatorsList10MHz+ImpedanceTableList10MHz,
#                                              frequency_step,
#                                              RFParams=rf_params,
#                                              multi_turn_wake=True,
#                                              front_wake_length=front_wake_length)

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

# Tracking -------------------------------------------------------------------
# profile.track()
# PS_longitudinal_intensity.induced_voltage_sum()

# plt.figure('Generated beam')
# plt.clf()
# plt.plot(profile.bin_centers, profile.n_macroparticles)


# plt.figure('impedance')
# plt.clf()
# plt.plot(profile.beam_spectrum_freq/1e6, np.abs(profile.beam_spectrum)/np.max(np.abs(profile.beam_spectrum))
#          * np.max(np.abs(PS_intensity_plot.total_impedance*profile.bin_size))/1e3, label='Beam')

# plt.plot(PS_intensity_plot.freq/1e6, np.abs(PS_intensity_plot.total_impedance*profile.bin_size
#                                             + 1j*PS_inductive.Z_over_n[0]*(PS_intensity_plot.freq/ring.f_rev[0]))/1e3, label='Full impedance + SC')

# plt.xlabel('Frequency [MHz]')
# plt.ylabel('Abs. Impedance [$\\mathrm{k\\Omega}$]')
# plt.legend(loc='best', ncol=3)
# plt.savefig(output_folder+'/impedance')


# fwhmBunchPosition = np.zeros(
#     (n_bunches, int(n_turns/bunch_by_bunch_output_step)+1))
# fwhmBunchLength = np.zeros(
#     (n_bunches, int(n_turns/bunch_by_bunch_output_step)+1))

# profile.fwhm_multibunch(n_bunches, bunch_spacing_buckets,
#                         rf_params.t_rf[0, 0], bucket_tolerance=0.)

# fwhmBunchPosition[:, 0] = profile.bunchPosition - \
#     np.arange(n_bunches)*rf_params.t_rf[0, 0]
# fwhmBunchLength[:, 0] = profile.bunchLength

# rmsBunchPosition = np.zeros(
#     (n_bunches, int(n_turns/bunch_by_bunch_output_step)+1))
# rmsBunchLength = np.zeros(
#     (n_bunches, int(n_turns/bunch_by_bunch_output_step)+1))

# profile.rms_multibunch(n_bunches, bunch_spacing_buckets,
#                        rf_params.t_rf[0, 0], bucket_tolerance=0.)

# rmsBunchPosition[:, 0] = profile.bunchPosition - \
#     np.arange(n_bunches)*rf_params.t_rf[0, 0]
# rmsBunchLength[:, 0] = profile.bunchLength

# dataMatrix = np.zeros((int(n_turns/datamatrix_output_step)+1, profile.n_slices))
# dataMatrix[0, :] = np.array(profile.n_macroparticles)

# perf_turn = []
# perf_time = []


print('dE mean:', np.mean(beam.dE))
print('dE std:', np.std(beam.dE))

if N_t_monitor > 0:
    filename = 'profiles/sps-t{}-p{}-b{}-sl{}-r{}-m{}-se{}'.format(
        N_t, n_macroparticles_per_bunch, n_bunches, n_slices,
        N_t_reduce, N_t_monitor, seed)
    slicesMonitor = SlicesMonitor(filename=filename,
                                  n_turns=np.ceil(1.0 * N_t / N_t_monitor),
                                  profile=profile)

master = mpiconf.Master(log=log)

timing.reset()
start_t = time.time()


try:

    init_dict = {
        'tracker_t_rev': tracker.t_rev,
        'tracker_eta_0': tracker.eta_0,
        'tracker_eta_1': tracker.eta_1,
        'tracker_eta_2': tracker.eta_2,
        'n_rf': tracker.n_rf,
        'solver': tracker.solver,
        'tracker_acc_kick': tracker.acceleration_kick,
        'length_ratio': tracker.length_ratio,
        'alpha_order': tracker.alpha_order,
        'rfp_beta': rf_params.beta,
        'rfp_energy': rf_params.energy,
        'rfp_omega_rf': rf_params.omega_rf,
        'rfp_omega_rf_d': rf_params.omega_rf_d,
        'rfp_phi_rf': rf_params.phi_rf,
        'rfp_dphi_rf': rf_params.dphi_rf,
        'rfp_harmonic': rf_params.harmonic,
        'rfp_voltage': rf_params.voltage,
        'rfp_phi_s': rf_params.phi_s,
        'n_slices': profile.n_slices,
        'bin_size': profile.bin_size,
        'bin_centers': profile.bin_centers,
        'cut_left': profile.cut_left,
        'cut_right': profile.cut_right,
        'charge': beam.Particle.charge,
        'beam_ratio': beam.ratio,
        'impedList': {
            # 'PS_intensity_freq_10MHz': {
            #     'type': 'mtw',
            #     'mtw_mode': 'time',
            #     'total_impedance': PS_intensity_freq_10MHz.total_impedance,
            #     'n_fft': PS_intensity_freq_10MHz.n_fft,
            #     'n_induced_voltage': PS_intensity_freq_10MHz.n_induced_voltage,
            #     'front_wake_buffer': PS_intensity_freq_10MHz.front_wake_buffer,
            #     'time_mtw': PS_intensity_freq_10MHz.time_mtw,
            #     'mtw_memory': PS_intensity_freq_10MHz.mtw_memory},
            'PS_intensity_freq_Rest': {
                'type': 'mtw',
                'mtw_mode': 'time',
                'total_impedance': PS_intensity_freq_Rest.total_impedance,
                'n_fft': PS_intensity_freq_Rest.n_fft,
                'n_induced_voltage': PS_intensity_freq_Rest.n_induced_voltage,
                'front_wake_buffer': PS_intensity_freq_Rest.front_wake_buffer,
                'time_mtw': PS_intensity_freq_Rest.time_mtw,
                'mtw_memory': PS_intensity_freq_Rest.mtw_memory},
            'PS_inductive': {
                'type': 'inductive',
                'deriv_mode': PS_inductive.deriv_mode,
                'Z_over_n': PS_inductive.Z_over_n,
                'n_induced_voltage': PS_inductive.n_induced_voltage}
        },
        'total_voltage': 0.,
        'induced_voltage': 0.,
    }

    master.multi_bcast(init_dict)

    vars_dict = {
        'dt': beam.dt,
        'dE': beam.dE
    }
    master.multi_scatter(vars_dict)

    print("Ready for tracking!\n")

    task_list = []
    for i in range(N_t):

        if (i % N_t_reduce == 0):
            task_list += ['histo', 'reduce_histo']

        if (N_t_monitor > 0) and (i % N_t_monitor == 0):
            task_list += ['gather_single']

        if (i % N_t_reduce == 0):
            task_list += ['induced_voltage_sum']

        task_list += ['RFVCalc', 'LIKick_n_drift']

    master.bcast(task_list)

    # for i in range(n_turns):
    for i in range(N_t):

        # if (i > 0) and (i % datamatrix_output_step) == 0:
        #     t0 = time.time()

        if (i % N_t_reduce == 0):
            profile.track()

        if (N_t_monitor > 0) and (i % N_t_monitor == 0):
            master.gather_single(
                {'profile': profile.n_macroparticles}, msg=False)
            slicesMonitor.track(i)

        # Change impedance of 10 MHz only if it changes
        # if (i > 0) and (R_S_program_10MHz[i] != R_S_program_10MHz[i-1]):
        #     PS_intensity_freq_10MHz.impedance_source_list[0].R_S[:] = \
        #         R_S_10MHz_save * R_S_program_10MHz[i]
        #     PS_intensity_freq_10MHz.sum_impedances(PS_intensity_freq_10MHz.freq)

        if (i % N_t_reduce == 0):
            PS_longitudinal_intensity.induced_voltage_sum()

        # Track
        tracker.track()

        # Analysis

        # if (i > 0) and (i % bunch_by_bunch_output_step) == 0:

        #     profile.fwhm_multibunch(n_bunches, bunch_spacing_buckets,
        #                             rf_params.t_rf[0, i], bucket_tolerance=0.)

        #     fwhmBunchPosition[:, int(i/bunch_by_bunch_output_step)
        #                       ] = profile.bunchPosition - np.arange(n_bunches)*rf_params.t_rf[0, i]
        #     fwhmBunchLength[:, int(i/bunch_by_bunch_output_step)
        #                     ] = profile.bunchLength

        #     profile.rms_multibunch(n_bunches, bunch_spacing_buckets,
        #                            rf_params.t_rf[0, i], bucket_tolerance=0.)

        #     rmsBunchPosition[:, int(i/bunch_by_bunch_output_step)
        #                      ] = profile.bunchPosition - np.arange(n_bunches)*rf_params.t_rf[0, i]
        #     rmsBunchLength[:, int(i/bunch_by_bunch_output_step)
        #                    ] = profile.bunchLength

        # if (i > 0) and (i % datamatrix_output_step) == 0:
        #     t1 = time.time()
        #     print('Turn %d' % (i))
        #     # print('One turn time %.2e' % (t1-t0))
        #     print('Bunch length :',
        #           (rmsBunchLength[:, int(i/bunch_by_bunch_output_step)]))
        #     print('Bunch position :',
        #           (rmsBunchPosition[:, int(i/bunch_by_bunch_output_step)]))

        #     dataMatrix[int(i/datamatrix_output_step),
        #                :] = np.array(profile.n_macroparticles)
        #     dataMatrix_extent = [profile.bin_centers[0]*1e6,
        #                          profile.bin_centers[-1]*1e6,
        #                          ring.cycle_time[0]*1e3,
        #                          ring.cycle_time[::datamatrix_output_step][int(i/datamatrix_output_step)]*1e3]

        # if (i % plot_step == 0) and (i > 0):

        #     plt.figure('Profile')
        #     plt.clf()
        #     plt.imshow(dataMatrix[:int(i/datamatrix_output_step), :], origin='bottom', aspect='auto',
        #                extent=dataMatrix_extent, cmap=cmap)
        #     plt.xlabel('Time $\\tau$ [$\\mathrm{\\mu s}$]')
        #     plt.ylabel('Time [ms]')
        #     plt.savefig(output_folder+'/dataMatrix')
        #     for indexBunch in range(n_bunches):
        #         plt.xlim((indexBunch*rf_params.t_rf[0, -1]*1e6,
        #                   (indexBunch+1)*rf_params.t_rf[0, -1]*1e6))
        #         plt.savefig(output_folder+'/dataMatrix_%02d' % (indexBunch+1))

        #     plt.figure('Bunch length')
        #     plt.clf()
        #     for indexBunch in range(n_bunches):
        #         plt.plot(ring.cycle_time[::bunch_by_bunch_output_step][:int(
        #             i/bunch_by_bunch_output_step)]*1e3, rmsBunchLength[indexBunch, :int(i/bunch_by_bunch_output_step)]*1e9)
        #     plt.xlabel('Time [ms]')
        #     plt.ylabel('Bunch length $4\\sigma$ [ns]')
        #     plt.savefig(output_folder+'/bunch_length')

        #     plt.figure('Bunch position')
        #     plt.clf()
        #     for indexBunch in range(n_bunches):
        #         plt.plot(ring.cycle_time[::bunch_by_bunch_output_step][:int(
        #             i/bunch_by_bunch_output_step)]*1e3, rmsBunchPosition[indexBunch, :int(i/bunch_by_bunch_output_step)]*1e9)
        #     plt.xlabel('Time [ms]')
        #     plt.ylabel('Bunch position $\\bar{\\tau}$ [ns]')
        #     plt.savefig(output_folder+'/bunch_position')

        #     perf_turn.append(i)
        #     # perf_time.append(t1-t0)
        #     np.savetxt(output_folder+'/performance.txt',
        #                np.hstack((np.array(perf_turn, ndmin=2).T,
        #                           np.array(perf_time, ndmin=2).T)))

        # np.savez_compressed(output_folder+'/saved_data.npz',
        #                     fwhmBunchPosition=fwhmBunchPosition,
        #                     fwhmBunchLength=fwhmBunchLength,
        #                     rmsBunchPosition=rmsBunchPosition,
        #                     rmsBunchLength=rmsBunchLength,
        #                     cycle_time=ring.cycle_time[::bunch_by_bunch_output_step])

    master.multi_gather(vars_dict)
    master.stop()
    master.disconnect()
except Exception as e:
    print(e)
    master.quit()
    master.disconnect()


end_t = time.time()
print('Total time: ', (end_t-start_t))
print('dE mean: ', np.mean(beam.dE))
print('dE std: ', np.std(beam.dE))

timing.report(total_time=1e3*(end_t-start_t),
              out_dir=args['report'],
              out_file='report.csv')
print('Done!')


# np.savez_compressed(output_folder+'/saved_data.npz',
#                     fwhmBunchPosition=fwhmBunchPosition,
#                     fwhmBunchLength=fwhmBunchLength,
#                     rmsBunchPosition=rmsBunchPosition,
#                     rmsBunchLength=rmsBunchLength,
#                     cycle_time=ring.cycle_time[::bunch_by_bunch_output_step])


# dataMatrix_extent = [profile.bin_centers[0]*1e6,
#                      profile.bin_centers[-1]*1e6,
#                      ring.cycle_time[0]*1e3,
#                      ring.cycle_time[::datamatrix_output_step][-1]*1e3]


# plt.figure('Profile')
# plt.clf()
# plt.imshow(dataMatrix, origin='bottom', aspect='auto',
#            extent=dataMatrix_extent, cmap=cmap)
# plt.xlabel('Time $\\tau$ [$\\mathrm{\\mu s}$]')
# plt.ylabel('Time [ms]')
# plt.savefig(output_folder+'/dataMatrix')
# for indexBunch in range(n_bunches):
#     plt.xlim((indexBunch*rf_params.t_rf[0, -1]*1e6,
#               (indexBunch+1)*rf_params.t_rf[0, -1]*1e6))
#     plt.savefig(output_folder+'/dataMatrix_%02d' % (indexBunch+1))

# plt.figure('Bunch length')
# plt.clf()
# for indexBunch in range(n_bunches):
#     plt.plot(ring.cycle_time[::bunch_by_bunch_output_step]
#              * 1e3, rmsBunchLength[indexBunch, :]*1e9)
# plt.xlabel('Time [ms]')
# plt.ylabel('Bunch length $4\\sigma$ [ns]')
# plt.savefig(output_folder+'/bunch_length')

# plt.figure('Bunch position')
# plt.clf()
# for indexBunch in range(n_bunches):
#     plt.plot(ring.cycle_time[::bunch_by_bunch_output_step]
#              * 1e3, rmsBunchPosition[indexBunch, :]*1e9)
# plt.xlabel('Time [ms]')
# plt.ylabel('Bunch position $\\bar{\\tau}$ [ns]')
# plt.savefig(output_folder+'/bunch_position')
