# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 16:11:42 2018

@author: schwarz
"""

import numpy as np
import os
import h5py
from pyprof import timing
from pyprof import mpiprof

import time
from scipy.constants import c

# BLonD imports
from beam.distributions import matched_from_distribution_function
from input_parameters.ring import Ring
from input_parameters.rf_parameters import RFStation
from beam.beam import Beam, Proton
from beam.profile import Profile, CutOptions
from impedances.impedance import InducedVoltageFreq, TotalInducedVoltage
from impedances.impedance_sources import TravelingWaveCavity
from trackers.tracker import RingAndRFTracker, FullRingAndRF
from llrf.beam_feedback import BeamFeedback
from impedances.impedance_scenario import scenario, impedance2blond
from impedances.impedance_reduction import ImpedanceReduction
from utils.input_parser import parse
from monitors.monitors import SlicesMonitor
from utils import mpi_config as mpiconf


currdir = os.path.dirname(os.path.realpath(__file__))

args = parse()
mpiconf.init(trace=args['trace'], logfile=args['tracefile'])

print(args)


# --- Simulation parameters -------------------------------------

# where and at what turns to save, only relevant if SAVE_DATA is True
save_folder = currdir + '/../output_files/lossInSimulation_scanSimParam/'
save_turn_fine = 5
save_turn_coarse = 1

# bunch parameters
BUNCHLENGTH_MODULATION = False
if BUNCHLENGTH_MODULATION is False:
    PS_case = 'rms13.0ns_full15ns'
else:
    PS_case = 'blMod'

optics = 'Q22'
intensity_pb = 1.7e11
V1 = 2.0e6
n_bunches = 72
bunch_shift = 0  # how many degrees to displace the bunch [deg]

INTENSITY_MODULATION = False
# number of turns to simulate in the SPS
n_turns = 43348+1  # 1s

# impedance & LLRF parameters
SPS_IMPEDANCE = True
impedance_model_str = 'present'  # present or future
cavities = impedance_model_str

SPS_PHASELOOP = True
PLrange = 5*12
PL_2ndLoop = 'F_Loop'

FB_strength = 'present'

# simulation parameters
seed = 1980
n_macroparticles_pb = int(4e6)  # 4M macroparticles per bunch
n_bins_rf = 256  # number of slices per RF-bucket
nFrev = 2  # multiples of f_rev for frequency resolution

N_t_reduce = 1
N_t_monitor = 0

if args.get('turns', None):
    N_t = args['turns']

if args.get('particles', None):
    n_macroparticles_pb = args['particles']

if args.get('time', False) == True:
    timing.mode = 'timing'

if args.get('omp', None):
    os.environ['OMP_NUM_THREADS'] = str(args['omp'])

if args.get('bunches', None):
    n_bunches = args['bunches']

if args.get('reduce', None):
    N_t_reduce = args['reduce']

if args.get('monitor', None):
    N_t_monitor = args['monitor']

if args.get('seed', None):
    seed = args['seed']

if 'log' in args:
    log = args['log']


# initialize simulation

np.random.seed(seed)

SAVE_DATA = False

FB_model = 'ImpRed'
reduction_model = 'filter'  # reduce via a 'filter' or R_S of the 200 MHz'cavity
filter_type = 'fb_reduction'  # apply 'fb_reduction' or 'none' filter

save_folder += optics+'/'+FB_model+'/'+PS_case+'/'

if n_turns == 43348+1:
    save_folder += '1s_sim/'
else:
    save_folder += str(n_turns)+'turns/'

case = 'int'+str(intensity_pb/1e11)+'_'+'V' + \
    str(V1/1e6)+'_'+str(n_bunches)+'Nbun'

if bunch_shift != 0:
    case += '_'+str(bunch_shift)+'deg'

if SPS_IMPEDANCE is True:
    case += '_'+impedance_model_str+'Imp'

if SPS_PHASELOOP is True:
    case += '_PL2'+str(int(PLrange/5))+'b'+PL_2ndLoop[0]

case += '_'+FB_strength+'FBstr'

case += '_seed'+str(seed) + '_'+str(n_macroparticles_pb/1e6)+'Mmppb'
case += '_'+str(n_bins_rf)+'binRF_' + str(nFrev)+'fRes'
print('simulating case: '+case)
print('saving in: '+save_folder)
save_file_name = case + '_data'

if INTENSITY_MODULATION:
    save_folder += 'intMod/'
if BUNCHLENGTH_MODULATION:
    save_folder += 'blMod/'

# SPS --- Ring Parameters -------------------------------------------

bunch_spacing = 5   # how many SPS RF buckets between bunches in the SPS
# 5*t_rf_SPS spacing = 5*5ns = 25ns

intensity = n_bunches * intensity_pb     # total intensity SPS

# Ring parameters SPS
circumference = 6911.5038  # Machine circumference [m]
sync_momentum = 25.92e9  # SPS momentum at injection [eV/c]

if optics == 'Q20':
    gamma_transition = 17.95142852  # Q20 Transition gamma
elif optics == 'Q22':
    gamma_transition = 20.071  # Q22 Transition gamma
else:
    raise RuntimeError('No gamma_transition specified')
momentum_compaction = 1./gamma_transition**2  # Momentum compaction array

ring = Ring(circumference, momentum_compaction, sync_momentum, Proton(),
            n_turns=n_turns)
tRev = ring.t_rev[0]

# RF parameters SPS
n_rf_systems = 1    # Number of rf systems

if n_rf_systems == 2:
    V2_ratio = 1.00e-01     # voltage 800 MHz  [V]
    harmonic_numbers = [4620, 18480]             # Harmonic numbers
    voltage = [V1, V1*V2_ratio]
    phi_offsets = [0, np.pi]
elif n_rf_systems == 1:
    harmonic_numbers = 4620  # Harmonic numbers
    voltage = V1
    phi_offsets = 0

rf_station = RFStation(ring, harmonic_numbers, voltage, phi_offsets,
                       n_rf=n_rf_systems)
t_rf = rf_station.t_rf[0, 0]

# calculate fs in case of two RF systems
if n_rf_systems == 2:
    h = rf_station.harmonic[0, 0]
    omega0 = ring.omega_rev[0]
    phiS = rf_station.phi_s[0]
    eta = rf_station.eta_0[0]
    V0 = rf_station.voltage[0, 0]
    beta0 = ring.beta[0, 0]
    E0 = ring.energy[0, 0]
    omegaS0 = np.sqrt(-h*omega0**2*eta*np.cos(phiS)*V0/(2*np.pi*beta0**2*E0))

    nh = rf_station.harmonic[1, 0] / rf_station.harmonic[0, 0]
    V2 = rf_station.voltage[1, 0]
    phi2 = rf_station.phi_offset[1, 0]
    omegaS = omegaS0 * \
        np.sqrt(1 + nh*V2*np.cos(nh*phiS+phi2) / (V0*np.cos(phiS)))
    fs = omegaS/(2*np.pi)
elif n_rf_systems == 1:
    fs = rf_station.omega_s0[0]/(2*np.pi)


# --- PS beam --------------------------------------------------------
n_macroparticles = n_bunches * n_macroparticles_pb
beam = Beam(ring, n_macroparticles, intensity)

# PS_n_bunches = 1

n_shift = 500  # how many rf-buckets to shift beam

# PS_folder = currdir+'/../input_files/'

# if BUNCHLENGTH_MODULATION is False:
#     print('Loading PS beam')
#     with h5py.File(PS_folder+'bunch_rotation_'+PS_case
#                    + '_bunch_rotation/after_rotation_PS_beam.hd5', "r") as h5file:
#         PS_dt = h5file['PS_dt'].value
#         # place PS beam in SPS RF-bucket 0
#         PS_dt += 0.5*t_rf - np.mean(PS_dt)
#         PS_dE = h5file['PS_dE'].value
#         PS_n_macroparticles = h5file['PS_n_macroparticles'].value

#     # SPS --- Beam Setup -------------------------------------------
#     PS_bunchCopies = int(n_bunches / PS_n_bunches)

#     # shift beam by n_shift rf buckets
#     PS_dt += (n_shift + bunch_shift/180/2) * t_rf

#     if INTENSITY_MODULATION:
#         intensityModulation = np.array(np.linspace(0.9, 1.1, num=4).tolist()*18)
#     else:
#         intensityModulation = np.ones(PS_bunchCopies)

#     print('Creating SPS bunch from PS bunch')
#     # create 72 bunches from PS bunch
#     beginIndex = 0
#     endIndex = 0

#     for copy in range(PS_bunchCopies):
#         # randomly select macroparticles from PS bunch according to
#         # intensity modulation
#         numSelectedMPs = int(np.round(n_bunches*n_macroparticles_pb/PS_bunchCopies
#                                       * intensityModulation[copy]))
#         indices = np.zeros(len(PS_dt), dtype=bool)
#         randices = np.random.choice(len(indices), numSelectedMPs, replace=False)
#         indices[randices] = True

#         endIndex = beginIndex + numSelectedMPs

#         # now place PS bunch at correct position
#         beam.dt[beginIndex:endIndex] \
#             = PS_dt[indices] + copy * t_rf * PS_n_bunches*bunch_spacing

#         beam.dE[beginIndex:endIndex] = PS_dE[indices]

#         beginIndex = endIndex
# else:  # use bunch length modulation
#     print('creating SPS beam')
#     PS_cases = ['rms13.5ns_full20ns', 'rms13.4ns_full20ns', 'rms13.0ns_full20ns',
#                 'rms12.6ns_full20ns']

#     beginIndex = 0
#     for case, PS_case in enumerate(PS_cases):
#         with h5py.File(PS_folder+'bunch_rotation_'+PS_case
#                        + '_bunch_rotation/after_rotation_PS_beam.hd5', "r") as h5file:
#             PS_dt = h5file['PS_dt'].value
#             # place PS beam in SPS RF-bucket 0
#             PS_dt += 0.5*t_rf - np.mean(PS_dt)
#             PS_dE = h5file['PS_dE'].value
#             PS_n_macroparticles = h5file['PS_n_macroparticles'].value

#         PS_bunchCopies = int(n_bunches / len(PS_cases))

#         # shift beam by n_shift rf buckets
#         PS_dt += (n_shift + case*bunch_spacing + bunch_shift/180/2) * t_rf

#         for copy in range(PS_bunchCopies):
#             # randomly select macroparticles from PS bunch
#             numSelectedMPs = n_macroparticles_pb
#             indices = np.zeros(len(PS_dt), dtype=bool)
#             randices = np.random.choice(len(indices), numSelectedMPs,
#                                         replace=False)
#             indices[randices] = True

#             endIndex = beginIndex + numSelectedMPs

#             # now place PS bunch at correct position
#             beam.dt[beginIndex:endIndex] = PS_dt[indices] \
#                 + copy * t_rf * n_bunches/PS_bunchCopies * bunch_spacing

#             beam.dE[beginIndex:endIndex] = PS_dE[indices]
#             beginIndex = endIndex


# SPS --- Profile -------------------------------------------
print('Setting up profile')

profile_margin = 20 * t_rf

t_batch_begin = n_shift * t_rf
t_batch_end = t_rf * (bunch_spacing * (n_bunches-1) + 1+n_shift)

cut_left = t_batch_begin - profile_margin
cut_right = t_batch_end + profile_margin

# number of rf-buckets of the beam
# + rf-buckets before the beam + rf-buckets after the beam
n_slices = n_bins_rf * (bunch_spacing * (n_bunches-1) + 1
                        + int(np.round((t_batch_begin - cut_left)/t_rf))
                        + int(np.round((cut_right - t_batch_end)/t_rf)))

profile = Profile(beam, CutOptions=CutOptions(cut_left=cut_left,
                                              cut_right=cut_right, n_slices=n_slices))



print('Profile set!')


# SPS --- Impedance and induced voltage ------------------------------
print('Setting up impedance')

frequency_step = nFrev*ring.f_rev[0]

if SPS_IMPEDANCE == True:

    if impedance_model_str == 'present':

        number_vvsa = 28

        number_vvsb = 36

        shield_vvsa = False

        shield_vvsb = False

        HOM_630_factor = 1

        UPP_factor = 25/25

        new_MKE = True

        BPH_shield = False

        BPH_factor = 1

    elif impedance_model_str == 'future':

        number_vvsa = 28

        number_vvsb = 36

        shield_vvsa = False

        shield_vvsb = False

        HOM_630_factor = 1/3

        UPP_factor = 10/25

        new_MKE = True

        BPH_shield = False

        BPH_factor = 0

    

    # The main 200MHz impedance is effectively 0.0

    impedance_scenario = scenario(MODEL=impedance_model_str,

                                  Flange_VVSA_R_factor=number_vvsa/31,

                                  Flange_VVSB_R_factor=number_vvsb/33,

                                  HOM_630_R_factor=HOM_630_factor,

                                  HOM_630_Q_factor=HOM_630_factor,

                                  UPP_R_factor=UPP_factor,

                                  Flange_BPHQF_R_factor=BPH_factor,

                                  FB_attenuation=-1000)

    impedance_scenario.importImpedanceSPS(VVSA_shielded=shield_vvsa,

                                          VVSB_shielded=shield_vvsb,

                                          kickerMario=new_MKE, noMKP=False,

                                          BPH_shield=BPH_shield)

    

    # Convert to formats known to BLonD

    impedance_model = impedance2blond(impedance_scenario.table_impedance)

    

    # Induced voltage calculated by the 'frequency' method

    SPS_freq = InducedVoltageFreq(beam, profile,

                         impedance_model.impedanceListToPlot, frequency_step)


    # # The main 200MHz impedance is effectively 0.0
    # impedance_scenario = scenario(MODEL=impedance_model_str,
    #                               FB_attenuation=-1000)

    # impedance_scenario.importImpedanceSPS()

    # # Convert to formats known to BLonD
    # impedance_model = impedance2blond(impedance_scenario.table_impedance)

    # # Induced voltage calculated by the 'frequency' method
    # SPS_freq = InducedVoltageFreq(beam, profile,
    #                               impedance_model.impedanceListToPlot, frequency_step)

#    induced_voltage = TotalInducedVoltage(beam, profile, [SPS_freq])

    print('SPS impedance model set!')

R2 = 27.1e3  # series impedance [kOhm/m^2]
vg = 0.0946*c  # group velocity [m/s]
fr = 200.222e6  # resonant frequency [Hz]

if cavities == 'present':

    L_long = 54*0.374  # interaction length [m]
    R_shunt_long = L_long**2*R2/8  # shunt impedance [Ohm]
    damping_time_long = 2*np.pi*L_long/vg*(1+0.0946)
    n_cav_long = 2  # factor 2 because of two cavities are used for tracking

    L_short = 43*0.374  # interaction length [m]
    R_shunt_short = L_short**2*R2/8  # shunt impedance [Ohm]
    damping_time_short = 2*np.pi*L_short/vg*(1+0.0946)
    n_cav_short = 2  # factor 2 because of two cavities are used for tracking
elif cavities == 'future':

    L_long = 43*0.374  # interaction length [m]
    R_shunt_long = L_long**2*R2/8  # shunt impedance [Ohm]
    damping_time_long = 2*np.pi*L_long/vg*(1+0.0946)
    n_cav_long = 2  # factor 2 because of two cavities are used for tracking

    L_short = 32*0.374  # interaction length [m]
    R_shunt_short = L_short**2*R2/8  # shunt impedance [Ohm]
    damping_time_short = 2*np.pi*L_short/vg*(1+0.0946)
    n_cav_short = 4  # factor 4 because of four cavities are used for tracking

longCavity = TravelingWaveCavity(n_cav_long*R_shunt_long, fr,
                                 damping_time_long)
longCavityFreq = InducedVoltageFreq(beam, profile, [longCavity],
                                    frequency_step)
longCavityIntensity = TotalInducedVoltage(beam, profile, [longCavityFreq])

shortCavity = TravelingWaveCavity(n_cav_short*R_shunt_short, fr,
                                  damping_time_short)
shortCavityFreq = InducedVoltageFreq(beam, profile, [shortCavity],
                                     frequency_step)
shortCavityIntensity = TotalInducedVoltage(beam, profile, [shortCavityFreq])

# FB parameters
if FB_strength == 'present':
    FBstrengthLong = 1.05
    FBstrengthShort = 0.73
elif FB_strength == 'future':
    #-26dB
    FBstrengthLong = 1.8
    FBstrengthShort = FBstrengthLong

filter_center_frequency = 200.222e6  # center frequency of filter [Hz]
filter_bandwidth = 2e6  # filter bandwidth [Hz]

# bandwidth should be 3MHz, but then 'reduction' is greater 1...
longCavityImpedanceReduction = ImpedanceReduction(ring, rf_station, longCavityFreq,
                                                  filter_type, filter_center_frequency,
                                                  filter_bandwidth,
                                                  2*tRev, L_long, start_time=2*tRev,
                                                  FB_strength=FBstrengthLong)

shortCavityImpedanceReduction = ImpedanceReduction(ring, rf_station, shortCavityFreq,
                                                   filter_type, filter_center_frequency,
                                                   filter_bandwidth,
                                                   2*tRev, L_short, start_time=2*tRev,
                                                   FB_strength=FBstrengthShort)

if SPS_IMPEDANCE:
    inducedVoltage = TotalInducedVoltage(beam, profile,
                                         [longCavityFreq, shortCavityFreq, SPS_freq])
else:
    inducedVoltage = TotalInducedVoltage(beam, profile,
                                         [longCavityFreq, shortCavityFreq])


# SPS --- Phase Loop Setup -------------------------------------

if SPS_PHASELOOP is True:

    print('Setting up phase-loop')
    PLgain = 5e3  # [1/s]
    try:
        PLalpha = -1/PLrange / t_rf
        PLoffset = n_shift * t_rf
    except ZeroDivisionError:
        PLalpha = 0.0
        PLoffset = None
    PLdict = {'time_offset': PLoffset, 'PL_gain': PLgain,
              'window_coefficient': PLalpha}
    PL_save_turns = 50
    if PL_2ndLoop == 'R_Loop':
        gain2nd = 5e9
        PLdict['machine'] = 'SPS_RL'
        PLdict['RL_gain'] = gain2nd
    elif PL_2ndLoop == 'F_Loop':
        gain2nd = 0.9e-1
        PLdict['machine'] = 'SPS_F'
        PLdict['FL_gain'] = gain2nd
    phaseLoop = BeamFeedback(ring, rf_station, profile, PLdict)
    beamPosPrev = t_batch_begin + 0.5*t_rf


# SPS --- Tracker Setup ----------------------------------------

print('Setting up tracker')
tracker = RingAndRFTracker(rf_station, beam, Profile=profile,
                           TotalInducedVoltage=inducedVoltage,
                           interpolation=True)
fulltracker = FullRingAndRF([tracker])




print('Creating SPS bunch from PS bunch')
# create 72 bunches from PS bunch

beginIndex = 0
endIndex = 0
PS_beam = Beam(ring, n_macroparticles_pb, 0.0)
PS_n_bunches = 1
for copy in range(n_bunches):
    # create binomial distribution;
    # use different seed for different bunches to avoid cloned bunches
    matched_from_distribution_function(PS_beam, fulltracker, seed=seed+copy,
                               distribution_type='binomial',
                               distribution_exponent=0.7,
                               emittance=0.35)

    endIndex = beginIndex + n_macroparticles_pb

    # now place PS bunch at correct position
    beam.dt[beginIndex:endIndex] \
        = PS_beam.dt + copy * t_rf * PS_n_bunches * bunch_spacing

    beam.dE[beginIndex:endIndex] = PS_beam.dE
    beginIndex = endIndex

# do profile on inital beam
profile.track()




# SPS --- Tracking -------------------------------------
# to save computation time, compute the reduction only for times < 8*FBtime
FBtime = max(longCavityImpedanceReduction.FB_time,
             shortCavityImpedanceReduction.FB_time)/tRev


print('dE mean: ', np.mean(beam.dE))
print('dE std: ', np.std(beam.dE))

if N_t_monitor > 0:
    filename = 'profiles/sps-t{}-p{}-b{}-sl{}-r{}-m{}-se{}'.format(
        N_t, n_macroparticles_pb, n_bunches, n_slices,
        N_t_reduce, N_t_monitor, seed)
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
        'rfp_beta': rf_station.beta,
        'rfp_energy': rf_station.energy,
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
        'impedList': {
            'longCavityFreq': {'total_impedance': longCavityFreq.total_impedance,
                               'n_fft': longCavityFreq.n_fft,
                               'n_induced_voltage': longCavityFreq.n_induced_voltage},
            'shortCavityFreq': {'total_impedance': shortCavityFreq.total_impedance,
                                'n_fft': shortCavityFreq.n_fft,
                                'n_induced_voltage': shortCavityFreq.n_induced_voltage},
            'SPS_freq': {'total_impedance': SPS_freq.total_impedance,
                         'n_fft': SPS_freq.n_fft,
                         'n_induced_voltage': SPS_freq.n_induced_voltage}
        },
        'total_voltage': 0.,
        'induced_voltage': 0.,
        'impedanceReduction': [
            {
                'impedance': 'longCavityFreq',
                'initial_impedance': longCavityImpedanceReduction.initial_impedance,
                'affected_indices': longCavityImpedanceReduction.affected_indices,
                'filter_func': longCavityImpedanceReduction.filter_func,
                'reduction_factor': longCavityImpedanceReduction.reduction_factor,
                'FB_strength': longCavityImpedanceReduction.FB_strength
            },
            {
                'impedance': 'shortCavityFreq',
                'initial_impedance': shortCavityImpedanceReduction.initial_impedance,
                'affected_indices': shortCavityImpedanceReduction.affected_indices,
                'filter_func': shortCavityImpedanceReduction.filter_func,
                'reduction_factor': shortCavityImpedanceReduction.reduction_factor,
                'FB_strength': shortCavityImpedanceReduction.FB_strength
            }

        ],
        'rfp_omega_rf': rf_station.omega_rf,
        'rfp_omega_rf_d': rf_station.omega_rf_d,
        'rfp_phi_rf': rf_station.phi_rf,
        'rfp_dphi_rf': rf_station.dphi_rf,
        'rfp_harmonic': rf_station.harmonic,
        'rfp_voltage': rf_station.voltage,
        'rfp_phi_s': rf_station.phi_s,
        'tracker_acc_kick': tracker.acceleration_kick,
        'gain': phaseLoop.gain,
        'gain2': phaseLoop.gain2,
        'alpha': phaseLoop.alpha,
        'reference': phaseLoop.reference,
        'machine': phaseLoop.machine,
        'time_offset': phaseLoop.time_offset
    }

    master.multi_bcast(init_dict)

    vars_dict = {
        'dt': beam.dt,
        'dE': beam.dE
    }
    master.multi_scatter(vars_dict)

    print("Ready for tracking!\n")

    # task_list generation

    task_list = []
    for turn in range(N_t):
        if (turn % N_t_reduce == 0):
            task_list += ['histo', 'reduce_histo']

        if (N_t_monitor > 0) and (turn % N_t_monitor == 0):
            task_list += ['gather_single']

        task_list += ['beamFB']
        if (turn < 8*int(FBtime)):
            task_list += ['impedance_reduction']

        if (turn % N_t_reduce == 0):
            task_list += ['induced_voltage_sum_packed']

        task_list += ['RFVCalc', 'LIKick_n_drift']

    master.bcast(task_list)

    # for turn in range(ring.n_turns):
    for turn in range(N_t):

        if ring.n_turns <= 450 and turn % 10 == 0:
            print('turn: '+str(turn))
        elif turn % 1000 == 0:
            print('turn: '+str(turn))

        # Update profile
        if (turn % N_t_reduce == 0):
            profile.track()

        if (N_t_monitor > 0) and (turn % N_t_monitor == 0):
            master.gather_single(
                {'profile': profile.n_macroparticles}, msg=False)
            slicesMonitor.track(turn)

        if SPS_PHASELOOP is True:
            phaseLoop.track()

        # reduce impedance, poor man's feedback
        if (turn < 8*int(FBtime)):
            longCavityImpedanceReduction.track()
            # shortCavityImpedanceReduction.track()

        # applying this voltage is done by tracker if interpolation=True
        if (turn % N_t_reduce == 0):
            inducedVoltage.induced_voltage_sum()

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

mpiprof.finalize()
# if report:
timing.report(total_time=1e3*(end_t-start_t),
              out_dir=args['report'],
              out_file='master.csv')

print('dE mean: ', np.mean(beam.dE))
print('dE std: ', np.std(beam.dE))

# --- Saving results ----------------------------------------------------

print('Done!')
