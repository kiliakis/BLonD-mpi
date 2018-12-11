# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 14:09:16 2018

@author: schwarz
"""

import numpy as np
from copy import deepcopy
from scipy.constants import c
try:
    from pyprof import timing
    from pyprof import mpiprof
except ImportError:
    from ..blond.utils import profile_mock as timing
    mpiprof = timing


### SPS --- Impedance reduction definition
class ImpedanceReduction():
    def __init__(self, Ring, RFStation, 
                 InducedVoltageFreq, filter_type, center_frequency, bandwidth,
                 time_constant, cavity_length, start_time = -1, FB_strength = 1):
        self.time = Ring.cycle_time
        self.counter = RFStation.counter
        self.frequencies = InducedVoltageFreq.freq
        self.impedance = InducedVoltageFreq.total_impedance
        self.initial_impedance = deepcopy(InducedVoltageFreq.total_impedance)
        self.freq_center = center_frequency
        self.delta_f = bandwidth
        self.FB_time = time_constant
        self.FB_strength = FB_strength
        if start_time == -1:
            self.start_time = Ring.t_rev[0]
        else:
            self.start_time = start_time
        
        if filter_type == 'fb_reduction':
#            self.L = 43*0.374   # cavity length [m]
            self.L = cavity_length # cavity length [m]
            self.vg = 0.0946*c  # group velocity [m/s]
            self.v = -Ring.beta[0,0]*c # particle speed [m/s]
            self.R2 = 27.1e3    # series impedance [Ohm/m^2]
            self.Z0 = 50        # characteristic impedance [Ohm]
            self.N = 924        # comb filter parameter, samplings per turn [1]
            self.a = 15/16      # comb filter parameter, bandwidth [1]
            # filter clock [Hz]
            self.f_clock = self.N * Ring.f_rev[0]
            self.b = 1/2        # high-pass filter parameter [1]
            # high-pass filter parameter [1] M=25 for 4-sections
            self.M = int(np.round(2*(self.L*(1+self.vg/c)/(2*self.vg))
                                    * self.f_clock))
            self.ghp = 3        # high-pass filter gain [1]
            self.glp = self.ghp/30  # low-pass filter gain [1]
            self.k = 1/120      # open-loop gain
            
            if np.round(self.L/0.374) == 43.0:
                self.lpf_par1 = 2.6
                self.lpf_par2 = 0.7
            elif np.round(self.L/0.374) == 54.0:
                self.lpf_par1 = 2.0
                self.lpf_par2 = 0.66
            elif np.round(self.L/0.374) == 32.0:
                self.lpf_par1 = 3.0
                self.lpf_par2 = 0.66
            else:
                raise RuntimeError('Cavity not recognized')
            
        # set minimum and maximum of affected frequencies to be 5*delta_f/2
        # and test that you don't under-/overshoot
        if (self.freq_center - 5*self.delta_f/2) < 0: 
            self.min_freq = 0.0
        else:
            self.min_freq = self.freq_center - 5*self.delta_f/2 # [Hz]
        if (self.freq_center + 5*self.delta_f/2) > max(self.frequencies):
            self.max_freq = max(self.frequencies)
        else:
            self.max_freq = self.freq_center + 5*self.delta_f/2 # [Hz]
        self.cond1 = self.frequencies <= self.max_freq
        self.cond2 = self.frequencies >= self.min_freq
        self.affected_indices = np.where(self.cond1*self.cond2)
        
        if filter_type == 'none':
            self.affected_indices = np.where(False)
            self.filter_func =  self.none_filter(
                    self.frequencies[self.affected_indices],
                    self.freq_center, self.delta_f)
        elif filter_type == 'fb_reduction':
            self.filter_func = self.fb_filter(
                    self.frequencies[self.affected_indices])
        else:
            raise RuntimeError('Filter type '+str(filter_type)+
                    ' not recognized! Must be: fb_reduction or none')
        
        if filter_type == 'none':
            self.reduction_factor = np.ones(len(self.time))
        elif filter_type =='fb_reduction':
            self.reduction_factor = \
                1-np.exp(-(self.time-start_time)/self.FB_time)
            self.reduction_factor[np.argwhere(self.time<self.start_time)] = 0
    #end __init__
    
    def none_filter(self,x,x0,delta_x):
        return np.ones(len(x))
    
    # contains filter and impedance reduction
    def fb_filter(self,f):
        z = np.exp(np.pi*2j*(f-self.freq_center)/self.f_clock)
        self.z = z
        
        # caution: np.sinc(x) = sin(pi*x)/(pi*x)
        Z_rf = self.L*(self.Z0*self.R2/2)**0.5 \
            * np.sinc(self.L * (f-self.freq_center) \
                      * (1-self.vg/self.v) / self.vg)
        self.Z_rf = Z_rf
        
#        H_comb = (1-self.a)/(1-self.a*z**-self.N) * z**-self.N
        H_comb = 1
        self.H_comb = H_comb
        
        H_lp = self.glp/self.M * (1-z**-self.M)/(1-z**(-1)) * z**((self.M-1)/2)
        self.H_lp = H_lp
        H_hp = -self.ghp * (1-self.b)/4 * (1-z**-self.M)/(1-self.b*z**-self.M)\
            * (1-z**-1) * z**((self.M+1)/2)
        self.H_hp = H_hp
        # fudge implementation of low-pass filter
        # achieves -3dB at 2MHz single-sided bandwidth
        lpf = self.lpf_par1*np.sinc((self.lpf_par2/1e6)*(f-self.freq_center)/np.pi)
        self.lpf = lpf
        
        return 1/(1 + 0.45*self.k * H_comb * (H_lp + lpf*H_hp) * Z_rf)
    #end fb_reduction
    
    @timing.timeit(key='serial:impedanceReduction')
    @mpiprof.traceit(key='serial:impedanceReduction')    
    def track(self):
        self.impedance[self.affected_indices] = \
            self.initial_impedance[self.affected_indices] \
            * self.filter_func**(self.reduction_factor[self.counter] \
                                 * self.FB_strength)
    #end track
#end definition ImpedanceReduction