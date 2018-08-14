# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 16:11:42 2018

@author: schwarz
"""

import numpy as np
import os
import h5py

# BLonD imports
from blond.input_parameters.ring import Ring
from blond.input_parameters.rf_parameters import RFStation
from blond.beam.beam import Beam, Proton
from blond.beam.profile import Profile, CutOptions
from blond.impedances.impedance import InducedVoltageFreq, TotalInducedVoltage
from blond.impedances.impedance_sources import TravelingWaveCavity
from blond.trackers.tracker import RingAndRFTracker, FullRingAndRF
from blond.llrf.beam_feedback import BeamFeedback

from blond.impedance_scenario import scenario, impedance2blond

#from mySeparatrix import mySeparatrix
from blond.impedance_reduction import ImpedanceReduction

from scipy.constants import c

def saveH5(fileh5, datas, key):
    with h5py.File(fileh5, 'w') as h5file:
        for it,item in enumerate(datas):
            if type(item).__name__ == 'ndarray':
                h5file.create_dataset(key[it],data=item,
                        compression="gzip", compression_opts=9, shuffle=True)
            elif type(item).__name__ == 'NoneType':
                h5file.create_dataset(key[it],data='None')
            else:
                h5file.create_dataset(key[it],data=item)
        h5file.close()


### --- Simulation parameters -------------------------------------

# where and at what turns to save, only relevant if SAVE_DATA is True
save_folder = '../simulation_results/lossInSimulation_scanSimParam/'
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
bunch_shift = 0 # how many degrees to displace the bunch [deg]

INTENSITY_MODULATION = False
# number of turns to simulate in the SPS
n_turns = 43348+1 # 1s

# impedance & LLRF parameters
SPS_IMPEDANCE = True
impedance_model_str = 'present' #present or future
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

### initialize simulation

np.random.seed(seed)

SAVE_DATA = False

FB_model = 'ImpRed'
reduction_model = 'filter' # reduce via a 'filter' or R_S of the 200 MHz'cavity
filter_type = 'fb_reduction' # apply 'fb_reduction' or 'none' filter

save_folder += optics+'/'+FB_model+'/'+PS_case+'/'

if n_turns == 43348+1:
    save_folder += '1s_sim/'
else:
    save_folder += str(n_turns)+'turns/'

case = 'int'+str(intensity_pb/1e11)+'_'+'V'+str(V1/1e6)+'_'+str(n_bunches)+'Nbun'

if bunch_shift != 0:
    case += '_'+str(bunch_shift)+'deg'

if SPS_IMPEDANCE is True:
    case += '_'+impedance_model_str+'Imp'

if SPS_PHASELOOP is True:
    case += '_PL2'+str(int(PLrange/5))+'b'+PL_2ndLoop[0]

case += '_'+FB_strength+'FBstr'

case += '_seed'+str(seed) +'_'+str(n_macroparticles_pb/1e6)+'Mmppb'
case += '_'+str(n_bins_rf)+'binRF_' + str(nFrev)+'fRes'
print('simulating case: '+case)
print('saving in: '+save_folder)
save_file_name = case + '_data'

if INTENSITY_MODULATION:
    save_folder += 'intMod/'
if BUNCHLENGTH_MODULATION:
    save_folder += 'blMod/'

### SPS --- Ring Parameters -------------------------------------------

bunch_spacing = 5   # how many SPS RF buckets between bunches in the SPS
                        # 5*t_rf_SPS spacing = 5*5ns = 25ns

intensity = n_bunches * intensity_pb     # total intensity SPS

# Ring parameters SPS
circumference = 6911.5038 # Machine circumference [m]
sync_momentum = 25.92e9 # SPS momentum at injection [eV/c]

if optics == 'Q20':
    gamma_transition = 17.95142852  # Q20 Transition gamma
elif optics == 'Q22':
    gamma_transition = 20.071 # Q22 Transition gamma
else:
    raise RuntimeError('No gamma_transition specified')
momentum_compaction = 1./gamma_transition**2  # Momentum compaction array

ring = Ring(circumference, momentum_compaction, sync_momentum, Proton(),
            n_turns=n_turns)
tRev = ring.t_rev[0]

# RF parameters SPS 
n_rf_systems = 1    # Number of rf systems 

if n_rf_systems==2:
    V2_ratio = 1.00e-01     # voltage 800 MHz  [V]
    harmonic_numbers = [4620, 18480]             # Harmonic numbers
    voltage = [V1, V1*V2_ratio]
    phi_offsets =  [0, np.pi]
elif n_rf_systems==1:
    harmonic_numbers = 4620 # Harmonic numbers
    voltage = V1
    phi_offsets = 0

rf_station = RFStation(ring, harmonic_numbers, voltage, phi_offsets,
                       n_rf=n_rf_systems)
t_rf = rf_station.t_rf[0,0]

#calculate fs in case of two RF systems
if n_rf_systems == 2:
    h=rf_station.harmonic[0,0]
    omega0 = ring.omega_rev[0]
    phiS=rf_station.phi_s[0]
    eta = rf_station.eta_0[0]
    V0=rf_station.voltage[0,0]
    beta0 = ring.beta[0,0]
    E0 = ring.energy[0,0]
    omegaS0 = np.sqrt(-h*omega0**2*eta*np.cos(phiS)*V0/(2*np.pi*beta0**2*E0))
    
    nh = rf_station.harmonic[1,0] / rf_station.harmonic[0,0]
    V2 = rf_station.voltage[1,0]
    phi2 = rf_station.phi_offset[1,0]
    omegaS = omegaS0 * np.sqrt(1 + nh*V2*np.cos(nh*phiS+phi2) / (V0*np.cos(phiS)))
    fs = omegaS/(2*np.pi)
elif n_rf_systems == 1:
    fs = rf_station.omega_s0[0]/(2*np.pi)
    

### --- PS beam --------------------------------------------------------
n_macroparticles = n_bunches * n_macroparticles_pb
beam = Beam(ring, n_macroparticles, intensity)

PS_n_bunches = 1

n_shift = 500 # how many rf-buckets to shift beam

PS_folder = 'C:/Users/schwarz/Work/PS_SPS_transfer/'

if BUNCHLENGTH_MODULATION is False:
    print('Loading PS beam')
    with h5py.File(PS_folder+'bunch_rotation_'+PS_case\
              +'_bunch_rotation/after_rotation_PS_beam.hd5', "r") as h5file:
        PS_dt = h5file['PS_dt'].value
        # place PS beam in SPS RF-bucket 0
        PS_dt += 0.5*t_rf - np.mean(PS_dt)
        PS_dE = h5file['PS_dE'].value
        PS_n_macroparticles = h5file['PS_n_macroparticles'].value
    
    ### SPS --- Beam Setup -------------------------------------------
    PS_bunchCopies = int(n_bunches / PS_n_bunches)
    
    #shift beam by n_shift rf buckets
    PS_dt += (n_shift + bunch_shift/180/2) * t_rf
    
    
    if INTENSITY_MODULATION:
        intensityModulation = np.array(np.linspace(0.9,1.1,num=4).tolist()*18)
    else:
        intensityModulation = np.ones(PS_bunchCopies)        
    
    print('Creating SPS bunch from PS bunch')
    # create 72 bunches from PS bunch
    beginIndex = 0
    endIndex = 0
    
    for copy in range(PS_bunchCopies):
        # randomly select macroparticles from PS bunch according to 
        # intensity modulation
        numSelectedMPs = int(np.round(n_bunches*n_macroparticles_pb/PS_bunchCopies
                             * intensityModulation[copy]))
        indices = np.zeros(len(PS_dt), dtype=bool)
        randices = np.random.choice(len(indices), numSelectedMPs, replace=False)
        indices[randices] = True
        
        endIndex = beginIndex + numSelectedMPs
        
        # now place PS bunch at correct position
        beam.dt[beginIndex:endIndex] \
            = PS_dt[indices] + copy * t_rf * PS_n_bunches*bunch_spacing
        
        beam.dE[beginIndex:endIndex] = PS_dE[indices]
        
        beginIndex = endIndex
else: # use bunch length modulation
    print('creating SPS beam')
    PS_cases = ['rms13.5ns_full20ns', 'rms13.4ns_full20ns', 'rms13.0ns_full20ns',
            'rms12.6ns_full20ns']

    beginIndex = 0
    for case, PS_case in enumerate(PS_cases):
        with h5py.File(PS_folder+'bunch_rotation_'+PS_case\
              +'_bunch_rotation/after_rotation_PS_beam.hd5', "r") as h5file:
            PS_dt = h5file['PS_dt'].value
            # place PS beam in SPS RF-bucket 0
            PS_dt += 0.5*t_rf - np.mean(PS_dt)
            PS_dE = h5file['PS_dE'].value
            PS_n_macroparticles = h5file['PS_n_macroparticles'].value
        
        PS_bunchCopies = int(n_bunches / len(PS_cases))
    
        #shift beam by n_shift rf buckets
        PS_dt += (n_shift + case*bunch_spacing + bunch_shift/180/2) * t_rf
        
        for copy in range(PS_bunchCopies):
            # randomly select macroparticles from PS bunch
            numSelectedMPs = n_macroparticles_pb
            indices = np.zeros(len(PS_dt), dtype=bool)
            randices = np.random.choice(len(indices), numSelectedMPs,
                                        replace=False)
            indices[randices] = True
            
            endIndex = beginIndex + numSelectedMPs
            
            # now place PS bunch at correct position
            beam.dt[beginIndex:endIndex] = PS_dt[indices] \
                + copy * t_rf * n_bunches/PS_bunchCopies * bunch_spacing
            
            beam.dE[beginIndex:endIndex] = PS_dE[indices]
            beginIndex = endIndex
    

### SPS --- Profile -------------------------------------------
print('Setting up profile')

profile_margin = 20 * t_rf

t_batch_begin = n_shift * t_rf
t_batch_end = t_rf * (bunch_spacing * (n_bunches-1) + 1+n_shift)

cut_left = t_batch_begin - profile_margin
cut_right = t_batch_end + profile_margin

# number of rf-buckets of the beam 
# + rf-buckets before the beam + rf-buckets after the beam
n_slices    = n_bins_rf * (bunch_spacing * (n_bunches-1) + 1 \
            + int(np.round((t_batch_begin - cut_left)/t_rf)) \
            + int(np.round((cut_right - t_batch_end)/t_rf)))

profile = Profile(beam, CutOptions = CutOptions(cut_left=cut_left,
                                    cut_right=cut_right, n_slices=n_slices))

# do profile on inital beam
profile.track()

# profile with 50ps resolution for saving; same resolution as measurement
nSlicesCoarse = 100 * (bunch_spacing * (n_bunches-1) + 1 \
            + int(np.round((t_batch_begin - cut_left)/t_rf)) \
            + int(np.round((cut_right - t_batch_end)/t_rf)))
profileCoarse = Profile(beam, CutOptions = CutOptions(cut_left=cut_left,
                                cut_right=cut_right, n_slices=nSlicesCoarse))
profileCoarse.track()

print('Profile set!')


### SPS --- Impedance and induced voltage ------------------------------
print('Setting up impedance')

frequency_step = nFrev*ring.f_rev[0]

if SPS_IMPEDANCE == True:
    
    # The main 200MHz impedance is effectively 0.0
    impedance_scenario = scenario(MODEL=impedance_model_str,
                                  FB_attenuation=-1000)
        
    impedance_scenario.importImpedanceSPS()
    
    # Convert to formats known to BLonD
    impedance_model = impedance2blond(impedance_scenario.table_impedance)
    
    # Induced voltage calculated by the 'frequency' method
    SPS_freq = InducedVoltageFreq(beam, profile,
                         impedance_model.impedanceListToPlot, frequency_step)
    
#    induced_voltage = TotalInducedVoltage(beam, profile, [SPS_freq])
        
    print('SPS impedance model set!')

R2 = 27.1e3 # series impedance [kOhm/m^2]
vg = 0.0946*c # group velocity [m/s]
fr = 200.222e6 # resonant frequency [Hz]

if cavities == 'present':
    
    L_long = 54*0.374 # interaction length [m]
    R_shunt_long = L_long**2*R2/8 # shunt impedance [Ohm]
    damping_time_long = 2*np.pi*L_long/vg*(1+0.0946)
    n_cav_long = 2 #factor 2 because of two cavities are used for tracking
    
    L_short = 43*0.374 # interaction length [m]
    R_shunt_short = L_short**2*R2/8 # shunt impedance [Ohm]
    damping_time_short = 2*np.pi*L_short/vg*(1+0.0946)
    n_cav_short = 2 #factor 2 because of two cavities are used for tracking
elif cavities == 'future':
    
    L_long = 43*0.374 # interaction length [m]
    R_shunt_long = L_long**2*R2/8 # shunt impedance [Ohm]
    damping_time_long = 2*np.pi*L_long/vg*(1+0.0946)
    n_cav_long = 2 #factor 2 because of two cavities are used for tracking
    
    L_short = 32*0.374 # interaction length [m]
    R_shunt_short = L_short**2*R2/8 # shunt impedance [Ohm]
    damping_time_short = 2*np.pi*L_short/vg*(1+0.0946)
    n_cav_short = 4 #factor 4 because of four cavities are used for tracking

longCavity = TravelingWaveCavity(n_cav_long*R_shunt_long, fr,
                                 damping_time_long)
longCavityFreq = InducedVoltageFreq(beam, profile, [longCavity],
                                      frequency_step) 
longCavityIntensity = TotalInducedVoltage(beam, profile, [longCavityFreq])

shortCavity = TravelingWaveCavity(n_cav_short*R_shunt_short, fr,
                                  damping_time_short)
shortCavityFreq = InducedVoltageFreq(beam, profile, [shortCavity],
                                      frequency_step) 
shortCavityIntensity = TotalInducedVoltage(beam, profile,[shortCavityFreq])

# FB parameters
if FB_strength == 'present':
    FBstrengthLong = 1.05
    FBstrengthShort = 0.73
elif FB_strength == 'future':
    #-26dB
    FBstrengthLong = 1.8
    FBstrengthShort = FBstrengthLong

filter_center_frequency = 200.222e6 # center frequency of filter [Hz]
filter_bandwidth = 2e6 # filter bandwidth [Hz]

#bandwidth should be 3MHz, but then 'reduction' is greater 1...
longCavityImpedanceReduction = ImpedanceReduction(ring,rf_station,longCavityFreq,
                    filter_type,filter_center_frequency,filter_bandwidth,
                    2*tRev, L_long, start_time=2*tRev, FB_strength=FBstrengthLong)

shortCavityImpedanceReduction = ImpedanceReduction(ring,rf_station,shortCavityFreq,
                    filter_type,filter_center_frequency,filter_bandwidth,
                    2*tRev, L_short, start_time=2*tRev, FB_strength=FBstrengthShort)

if SPS_IMPEDANCE:
    inducedVoltage = TotalInducedVoltage(beam, profile,
                                [longCavityFreq, shortCavityFreq, SPS_freq])
else:
    inducedVoltage = TotalInducedVoltage(beam, profile, 
                                         [longCavityFreq, shortCavityFreq])


### SPS --- Phase Loop Setup -------------------------------------

if SPS_PHASELOOP is True:

    print('Setting up phase-loop')
    PLgain = 5e3 # [1/s]
    try:
        PLalpha = -1/PLrange / t_rf
        PLoffset = n_shift * t_rf
    except ZeroDivisionError:
        PLalpha = 0.0
        PLoffset = None
    PLdict = {'time_offset':PLoffset, 'PL_gain':PLgain,
              'window_coefficient':PLalpha}
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


### SPS --- Tracker Setup ----------------------------------------

print('Setting up tracker')
section_tracker = RingAndRFTracker(rf_station, beam, Profile=profile,
                                   TotalInducedVoltage=inducedVoltage,
                                   interpolation=True)
tracker = FullRingAndRF([section_tracker])


### ------ Variables to save -----------------------------------------------------

#save every turn for first 'save_turn_fine' turns
#save every 'save_turn_coarse'-th turn after 'save_turn_fine' turns
#save_turn_fine = 170
#save_turn_coarse = 100

if SAVE_DATA == True:
    n_saves = save_turn_fine +1 + (n_turns - save_turn_fine)//save_turn_coarse
    save_counter = 0
    
    # at which turns to save
    save_turn = np.zeros(n_saves, dtype=int)
    save_turn[save_counter] = 0
    
#    # setup to save cavity voltage
#    induced_voltage_total = np.zeros(shape=(n_saves,
#                                            len(inducedVoltage.time_array)))
#
#    induced_voltage_long = np.zeros(shape=induced_voltage_total.shape)
#    induced_voltage_short = np.zeros(shape=induced_voltage_long.shape)
#    rf_voltage = np.zeros(shape=(n_saves, profile.n_slices))
#    # compute initial voltage (without beam loading)
#    section_tracker.rf_voltage_calculation()
#    rf_voltage[save_counter,:] = section_tracker.rf_voltage
#    save_induced_voltage = [inducedVoltage.time_array,
#                            induced_voltage_long, induced_voltage_short,
#                            induced_voltage_total, rf_voltage]
#    save_induced_voltage_names = ['induced_voltage_time','induced_voltage_long',
#                                 'induced_voltage_short','induced_voltage_total',
#                                 'rf_voltage']
#
#    induced_voltage_long[save_counter,:] = 0.5*longCavityIntensity.induced_voltage
#    induced_voltage_short[save_counter,:] = 0.5*shortCavityIntensity.induced_voltage
#
#    induced_voltage_total[save_counter,:] = inducedVoltage.induced_voltage
    save_induced_voltage = [0]
    save_induced_voltage_names = ['empty']
    
    #general parameters to save
    save_ring = [n_turns, circumference, momentum_compaction, sync_momentum]
    save_ring_names = ['n_turns','circumference', 'momentum_compaction', 
                                'sync_momentum']
    save_rf = [n_rf_systems, harmonic_numbers, voltage, phi_offsets, fs]
    save_rf_names = ['n_rf', 'harmonic', 'voltage', 'phi_offset', 'fs']
    save_beam_param = [n_macroparticles, intensity]
    save_beam_param_names = ['n_macroparticles', 'intensity']
    misc_param = [n_bunches, n_bins_rf, bunch_spacing, 
                  save_turn_fine, save_turn_coarse, save_turn,
                  intensity_pb, PS_case]
    misc_param_names = ['n_bunches', 'n_slices_per_bunch', 'bunch_spacing', 
                        'save_turn_fine', 'save_turn_coarse', 'save_turn',
                        'intensity_pb', 'PS_case']
    
    #beam profile to save
    avg_dE = np.zeros(shape=(n_saves), dtype=np.float32)
    avg_dE[save_counter] = np.mean(beam.dE)
    coarse_nMacroPart = np.zeros(shape=(n_saves, profileCoarse.n_slices),
                                 dtype=np.uint32)
    coarse_binCenters = np.copy(profileCoarse.bin_centers)
    coarse_binSize = profileCoarse.bin_size
    coarse_nMacroPart[save_counter] \
        = (profileCoarse.n_macroparticles).astype(dtype=np.uint32)
    
    save_profile_param = [coarse_nMacroPart, coarse_binSize, coarse_binCenters,
                          n_shift, profile_margin, avg_dE]
    save_profile_names = ['nMacroPart', 'binSize', 'binCenters', 'n_shift',
                          'profile_margin', 'avg_dE']
    
    #feedback parameters
    save_fb_param = [impedance_model_str, reduction_model, filter_type,
                 longCavityImpedanceReduction.FB_time,
                 longCavityImpedanceReduction.FB_strength,
                 longCavityImpedanceReduction.start_time,
                 shortCavityImpedanceReduction.FB_time,
                 shortCavityImpedanceReduction.FB_strength,
                 shortCavityImpedanceReduction.start_time]
    save_fb_param_names = ['impedance_model', 'reduction_model', 'filter_type',
                'longCavFBtimeConst','longCavFBstrength','longCavFBstart',
                'shortCavFBtimeConst','shortCavFBstrength','shortCavFBstart']
    
    if SPS_PHASELOOP is True:
        PLdelta = np.zeros(n_turns//PL_save_turns)
        PL_save_counter = 0
        save_PL_param = [PLalpha, PLoffset, PLgain, gain2nd, PLdelta,
                         PL_save_turns, PL_2ndLoop]
        save_PL_param_names = ['PL_alpha', 'PL_offset', 'PL_gain', 'gain2nd',
                               'delta', 'PL_save_turns', '2nd_loop']
    # phase space density
    margin = 0.1
    histBins = 150
    
    bunches_to_plot = [0]
    if n_bunches >= 12:
        bunches_to_plot.append(11)
    if n_bunches >= 24:
        bunches_to_plot.append(23)
    if n_bunches >= 36:
        bunches_to_plot.append(35)
    if n_bunches >= 48:
        bunches_to_plot.append(47)
    if n_bunches >= 72:
        bunches_to_plot.append(71)
    
    hists = np.zeros(shape=(len(bunches_to_plot), n_saves, histBins, histBins),
                     dtype=np.uint16)
    xedges = np.zeros(shape=(len(bunches_to_plot), n_saves, histBins+1),
                      dtype=np.float32)
    yedges = np.zeros(shape=xedges.shape, dtype=np.float32)
    
    l_bounds = np.array(list((t_batch_begin + t_rf*(bunch*bunch_spacing - margin)
                    for bunch in bunches_to_plot)))
    r_bounds = l_bounds + t_rf*(1 + 2*margin)

    for it, bunch in enumerate(bunches_to_plot):
        (l_bound, r_bound) = (l_bounds[it], r_bounds[it])
        l_indeces = beam.dt >= l_bound
        r_indeces = beam.dt <= r_bound
        hist, xedge, yedge = np.histogram2d(beam.dt[l_indeces*r_indeces]*1e9,
                                            beam.dE[l_indeces*r_indeces]/1e6,
            bins=histBins, range=[[l_bound*1e9,r_bound*1e9], [-150,150]])
        hists[it, save_counter] = (hist.T).astype(np.uint16)
        xedges[it, save_counter] = xedge.astype(np.float32)
        yedges[it, save_counter] = yedge.astype(np.float32)
    
    save_hists = [bunches_to_plot, margin, histBins, hists, xedges, yedges]
    save_hists_names = ['bunchesPlotted', 'histMargin', 'histBins',
                        'histogramms', 'xedges', 'yedges']

### SPS --- Tracking -------------------------------------
#to save computation time, compute the reduction only for times < 8*FBtime
FBtime = max(longCavityImpedanceReduction.FB_time,
             shortCavityImpedanceReduction.FB_time)/tRev
             
for turn in range(ring.n_turns):
    
    if ring.n_turns <= 450:
        if turn%10 == 0:
            print('turn: '+str(turn))
    else:
        if turn%1000 == 0:
            print('turn: '+str(turn))

    # Update profile
    profile.track()
    profileCoarse.track()
    
    if SPS_PHASELOOP is True:
        phaseLoop.track()
    
    #reduce impedance, poor man's feedback
    if (turn < 8*int(FBtime)):
        longCavityImpedanceReduction.track() 
        shortCavityImpedanceReduction.track()
    #end if time<8*FB_time
    
    # applying this voltage is done by tracker if interpolation=True
    inducedVoltage.induced_voltage_sum()

    tracker.track()
                
    if SAVE_DATA == True:
        # update data to save and update plots
        if (turn<save_turn_fine 
            or (turn!=save_turn_fine \
                and (turn-save_turn_fine)%save_turn_coarse==0)):
            save_counter += 1
            if save_counter >= 450 or save_counter%10 == 0:
                print('saving at ', turn, save_counter)
            
#            induced_voltage_long[save_counter,:] = 0.5 * inducedVoltage \
#                .induced_voltage_list[0].induced_voltage[:profile.n_slices]
#            induced_voltage_short[save_counter,:] = 0.5 * inducedVoltage \
#                .induced_voltage_list[1].induced_voltage[:profile.n_slices]
#        
#            induced_voltage_total[save_counter,:] \
#                = inducedVoltage.induced_voltage
#
#            rf_voltage[save_counter,:] = section_tracker.total_voltage
            
            # saved after present turn has completed
            save_turn[save_counter] = turn + 1
            coarse_nMacroPart[save_counter] \
                = (profileCoarse.n_macroparticles).astype(dtype=np.uint32)
            
            if PL_2ndLoop == 'R_Loop':
                avg_dE[save_counter] = phaseLoop.average_dE
            elif PL_2ndLoop == 'F_Loop':
                avg_dE[save_counter] = np.mean(beam.dE)
                
            for it, bunch in enumerate(bunches_to_plot):
                (l_bound, r_bound) = (l_bounds[it], r_bounds[it])
                l_indeces = beam.dt >= l_bound
                r_indeces = beam.dt <= r_bound
                hist, xedge, yedge = np.histogram2d(beam.dt[l_indeces*r_indeces]*1e9,
                                                    beam.dE[l_indeces*r_indeces]/1e6,
                                                    bins=histBins,
                                range=[[l_bound*1e9,r_bound*1e9], [-150,150]])
                hists[it, save_counter] = (hist.T).astype(np.uint16)
                xedges[it, save_counter] = xedge.astype(np.float32)
                yedges[it, save_counter] = yedge.astype(np.float32)
            
    # shift the profile bin_centers by the amount that the beam has drifted
    if SPS_PHASELOOP == True:
        if turn%PL_save_turns == 0 and turn>0:
            # present beam position
            beamPosFromPhase = (phaseLoop.phi_beam - rf_station.phi_rf[0,turn])\
                / rf_station.omega_rf[0,turn] + t_batch_begin
            # how much to shift the bin_centers
            delta = beamPosPrev - beamPosFromPhase
            beamPosPrev = beamPosFromPhase
            
            if SAVE_DATA == True:
                PLdelta[PL_save_counter] = delta
                PL_save_counter += 1
    
            profile.bin_centers -= delta
            profile.cut_left -= delta
            profile.cut_right -= delta
            profile.edges -= delta
            
            profileCoarse.bin_centers -= delta
            profileCoarse.cut_left -= delta
            profileCoarse.cut_right -= delta
            profileCoarse.edges -= delta
            
            # shift time_offset of phase loop as well, so that it starts at correct
            # bin_center corresponding to time_offset
            if phaseLoop.alpha != 0:
                phaseLoop.time_offset -= delta
            
            # update plot ranges
            for it, bunch in enumerate(bunches_to_plot):
                l_bounds[it] = profile.cut_left + profile_margin \
                    + t_rf*(bunch*bunch_spacing - margin)
                r_bounds[it] = l_bounds[it] + t_rf*(1 + 2*margin)
            
#end track
print('tracking done')


### --- Saving results ----------------------------------------------------

if SAVE_DATA is True:
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    print('saving data in: '+save_folder)
    
    save_params = save_ring + save_rf + save_beam_param + misc_param \
        + save_induced_voltage + save_profile_param + save_hists
    save_params_names = save_ring_names + save_rf_names + save_beam_param_names\
        + misc_param_names + save_induced_voltage_names + save_profile_names\
        + save_hists_names
    
    if SPS_PHASELOOP is True:
        save_params += save_PL_param
        save_params_names += save_PL_param_names
    
    saveH5(save_folder+save_file_name+ '.hd5', save_params, save_params_names)
    
print('Done')
