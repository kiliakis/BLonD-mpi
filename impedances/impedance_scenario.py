# coding: utf-8
# Copyright 2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
**Module used to import the SPS impedance scenario and apply wanted modifications**

:Authors: **Joel Repond**
"""

from __future__ import division
import numpy as np

from impedances.impedance_sources import TravelingWaveCavity, Resonators, InputTable
from impedances.impedance import InducedVoltageTime, InducedVoltageFreq

from scipy.constants import c
import os,inspect


class handleImpedance(object):
    """Fundamental class defining how to interact with the impedance tables

    This class contains the main methods to import the data from the impedance
    folder and to act on it (damping, frequency shifting)
    
    It contains a dictionary called self.table_impedance which contains:
        -- the .keys() of the dictionary are based on the file name and arborescence
        -- the ['type'] of impedance ('resonator', 'twc', 'inputtable')
        -- the data depending in the type of impedance
    
    The three types of impedance used are as followed:
        -- 'resonator': resonating impedance defined by fr, Rsh and Q.
        -- 'inputtable': raw data table (from CST for example), three column
            containing fr, the frequency, ReZ the real part of Z and ImZ the
            imaginary part of the impedance.
        -- 'twc': theoretical travelling wave cavity impedance, containing fr,
            Rsh and alpha the time constant [1]_.

    Attributes
    ----------
    table_impedance : dict
        contains the impedance data.
    impedanceFolder : str
        root of the impedance folder.
    
    References
    ----------
    .. [1]  "The SPS acceleration system travelling wave drif-tube structure for
            the CERN SPS", G. Dôme, CERN-SPS-ARF-77-11, 1977

    Examples
    --------
    >>> from blabla import blibli
    >>> 
    """

# Init method
# ------------

    def __init__(self, **kwargs):
        self.table_impedance = dict()

        if 'folder' in kwargs:
            self.impedanceFolder = kwargs.get('folder')
        else:
            fold = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
            self.impedanceFolder = fold + '/impedance'        


# Importation method
# -------------------

    def check_Q(self,tableimport):
        """
        This method checks (for a resonator) if Q < 0.5 and rescals everything
        accordingly
        
        Parameters
        ----------
        tableimport : np.array
            imported 3D array containing [fr, Rsh, Q]
        
        Returns
        --------
        tableimport : np.array
            rescaled array if necessary
        """
        if len(tableimport[:,2][tableimport[:,2]<=0.5]) > 0:
            tableimport[:,1][tableimport[:,2]<1.] /= tableimport[:,2][tableimport[:,2]<1.]
            tableimport[:,2][tableimport[:,2]<1.] = 1.
            return tableimport
        else:
            return tableimport    

    def importResonatorFromFile(self,filestr,unitFreq=1e9, unitRsh=1e3,
                                  RshFactor=1.,QFactor=1.):

        """
        This method import the data for resonators from a file and store it in
        table_impedance
        
        Parameters
        ----------
        filestr : str
            path of the corresponding file, relative to impedanceFolder, this
            str is used a key in the table_impedance dictionary to easily find
            your impedance if necessary.
        unitFreq : float
            the frequency must be in Hz
        unitRsh : float
            the shunt impedance must be in Ohm
        RshFactor : float
            used to modify the shunt impedance of the entire file loaded
        QFactor : float
            used to modify the quality factor Q of the entire file loaded

        """

        if not isinstance(filestr,str):
            raise SystemError('The first argument of import_resonator_fromfile must be a string containing the name of the file')
            
        importedFile =  np.atleast_3d(np.loadtxt(self.impedanceFolder + '/' + filestr, comments=['!','%']))
        importedFile[:,0] *= unitFreq
        importedFile[:,1] *= unitRsh * RshFactor
        importedFile[:,2] *= QFactor
        self.check_Q(importedFile)

        self.table_impedance[filestr] = dict()
        
        self.table_impedance[filestr]['fr'] = importedFile[:,0]
        self.table_impedance[filestr]['Rsh'] = importedFile[:,1]
        self.table_impedance[filestr]['Q'] = importedFile[:,2]
        self.table_impedance[filestr]['type'] = 'resonator'
        self.table_impedance[filestr]['quantity'] = RshFactor


    def importInputTableFromFile(self, filestr, unitFreq=1., ZFactor=1.):

        """
        This method import the data from an inputtable in a file and store it in
        table_impedance
        
        Parameters
        ----------
        filestr : str
            path of the corresponding file, relative to impedanceFolder, this
            str is used a key in the table_impedance dictionary to easily find
            your impedance if necessary.
        unitFreq : float
            the frequency must be in Hz
        ZFactor : float
            used to modify the real and imaginary part of the impedance of the 
            entire file loaded
        """

        if not isinstance(filestr,str):
            raise SystemError('The first argument of import_inputtable must be a string containing the name of the file')

        importedFile =  np.loadtxt(self.impedanceFolder + '/' + filestr, comments=['!','%'])
        importedFile[:,0] *= unitFreq
        importedFile[:,1] *= ZFactor
        importedFile[:,2] *= ZFactor
        
        self.table_impedance[filestr] = dict()
        
        self.table_impedance[filestr]['fr'] = importedFile[:,0]
        self.table_impedance[filestr]['ReZ'] = importedFile[:,1]
        self.table_impedance[filestr]['ImZ'] = importedFile[:,2]
        self.table_impedance[filestr]['type'] = 'inputtable'

    def importInputTableFromList(self, thelist, thestr, unitFreq=1., ZFactor=1.):

        """
        This method import an inputtable from a list ([f,ReZ,ImZ]) and store it in
        table_impedance
        
        Parameters
        ----------
        thelist : list
            The list containing the impedance table in the form [freq, ReZ, ImZ]
        thestr : str
            this str is used a key in the table_impedance dictionary to easily find
            your impedance if necessary.
        unitFreq : float
            the frequency must be in Hz
        ZFactor : float
            used to modify the real and imaginary part of the impedance of the 
            entire file loaded
        """

#        if not isinstance(filestr,str):
#            raise SystemError('The first argument of import_inputtable must be a string containing the name of the file')

#        importedFile =  np.loadtxt(self.impedanceFolder + '/' + filestr, comments=['!','%'])
        frequency = thelist[0]*unitFreq
        RealZ = thelist[1]*ZFactor
        ImagZ = thelist[2]*ZFactor
        
        self.table_impedance[thestr] = dict()
        
        self.table_impedance[thestr]['fr'] = frequency
        self.table_impedance[thestr]['ReZ'] = RealZ
        self.table_impedance[thestr]['ImZ'] = ImagZ
        self.table_impedance[thestr]['type'] = 'inputtable'

    def importTWCFromFile(self, filestr,unitFreq=1e9, unitRsh=1e3, unitAlpha=1e-6,
                          RshFactor=1.):

        """
        This method import the data for a travelling wave cavity from a file
        and store it in table_impedance
        
        Parameters
        ----------
        filestr : str
            path of the corresponding file, relative to impedanceFolder, this
            str is used a key in the table_impedance dictionary to easily find
            your impedance if necessary.
        unitFreq : float
            the frequency must be in Hz
        unitRsh : float
            the shunt impedance must be in Ohm
        unitAlpha : float
            alpha must be in sec
        RshFactor : float
            used to modify the shunt impedance of the entire file loaded
        """

        if not isinstance(filestr,str):
            raise SystemError('The first argument of import_inputtable must be a string containing the name of the file')

        importedFile =  np.atleast_3d(np.loadtxt(self.impedanceFolder + '/' + filestr, comments=['!','%']))
        importedFile[:,0] *= unitFreq
        importedFile[:,1] *= unitRsh * RshFactor
        importedFile[:,2] *= unitAlpha
        
        self.table_impedance[filestr] = dict()
        
        self.table_impedance[filestr]['fr'] = importedFile[:,0]
        self.table_impedance[filestr]['Rsh'] = importedFile[:,1]
        self.table_impedance[filestr]['alpha'] = importedFile[:,2]
        self.table_impedance[filestr]['type'] = 'twc'


# Method to modify the loaded data
# ---------------------------------

    def move_fr_resonatorOrTWC(self,key,freq,delta_freq):

        """
        Shift the frequency freq by delta_freq in the data corresponding to key
        
        if freq is a list [f1,f2], all the component between f1 and f2 are shifted
        by delta_freq
        
        Parameters
        ----------
        key : str
            corresponding impedance data
        freq : float or list
            frequency or boundary between which the frequencies will be shifted.
            The frequency must be in Hz.
        delta_freq : float
            frequency shift.
        """

        frequencyToMove_array = self.table_impedance[key]['fr']
        
        if isinstance(freq,float):
            frequencyToMove_array[frequencyToMove_array == freq] += delta_freq
            
        elif isinstance(freq,list):
            cond1 = frequencyToMove_array >= freq[0]
            cond2 = frequencyToMove_array <= freq[1]
            
            frequencyToMove_array[cond1*cond2] += delta_freq

    def damp_R_resonatorOrTWC(self,key,freq,R_factor):

        """
        Multiply the shunt impedance in the data corresponding to key by R_factor
        
        if freq is a list [f1,f2], all the component between f1 and f2 are modify
        
        Parameters
        ----------
        key : str
            corresponding impedance data
        freq : float or list
            frequency or boundary of freq between which Rsh will be multiplied.
            The frequency must be in Hz.
        R_factor : float
            factor by which the shunt impedance is multiplied.
        """

        frequencyToDamp_array = self.table_impedance[key]['fr']
        Rshunt_to_damp_array = self.table_impedance[key]['Rsh']
        
        if isinstance(freq,float):
            Rshunt_to_damp_array[frequencyToDamp_array == freq] *= R_factor
            
        elif isinstance(freq,list):
            cond1 = frequencyToDamp_array >= freq[0]
            cond2 = frequencyToDamp_array <= freq[1]
            
            Rshunt_to_damp_array[cond1*cond2] *= R_factor

    def damp_Q_resonatorOrTWC(self,key,freq,Q_factor):

        """
        Multiply the Q factor in the data corresponding to key by Q_factor
        
        if freq is a list [f1,f2], all the component between f1 and f2 are modify
        
        Parameters
        ----------
        key : str
            corresponding impedance data
        freq : float or list
            frequency or boundary of freq between which Q will be multiplied.
            The frequency must be in Hz.
        Q_factor : float
            factor by which the Q factor is multiplied.
        """

        frequencyToDamp_array = self.table_impedance[key]['fr']
        Q_to_damp_array = self.table_impedance[key]['Q']
        
        if isinstance(freq,float):
            Q_to_damp_array[frequencyToDamp_array == freq] *= Q_factor
            
        elif isinstance(freq,list):
            cond1 = frequencyToDamp_array >= freq[0]
            cond2 = frequencyToDamp_array <= freq[1]
            
            Q_to_damp_array[cond1*cond2] *= Q_factor

    def print_resonator_table(self,key, save=False):
        s = ''
        if self.table_impedance[key]['type'] == 'resonator':      
            fr = self.table_impedance[key]['fr']
            Rsh = self.table_impedance[key]['Rsh']
            Q = self.table_impedance[key]['Q']
            
            # first line
            s += '!Resonator name: '+key+', quantity = '+str(self.table_impedance[key]['quantity'])+'\n\n'
#            s += '!fr [GHz] Rsh [kOhm] Q R/Q [Ohm] Q/(pi*fr) [ns] #b coupled\n'
            s += '!{:^10s} {:^10s} {:^10s} {:^10s} {:^10s} {:^10s}\n'.format('fr [GHz]', 'Rsh [kOhm]', 'Q', 'R/Q [Ohm]', 'Q/(pi*fr) [ns]', '#b coupled')
            
            for it in range(len(fr)):
                s += '%10.3f %10.3f %10.3f %10.3f %10.3f %10.3f\n' % (fr[it]/1e9, Rsh[it]/1e3, Q[it], Rsh[it]/Q[it], Q[it]/np.pi/fr[it]*1e9, Q[it]/np.pi/fr[it]/25e-9)
#                s += '{^10.3f} {^10.3f} {^10.3f} {^10.3f} {^10.3f} {^10.3f}\n'.format((str(fr[it]/1e9), str(Rsh[it]/1e3), str(Q[it]), str(Rsh[it]/Q[it]), str(Q[it]/np.pi/fr[it]*1e9), str(Q[it]/np.pi/fr[it]/25e-9)))
        else:
            s += 'name: '+key+' ... not a resonator'
            print('the function print_resonator_table can be use for resonator type impedance only')
        if save:
            with open('./'+str(key.split('/')[-1])+'.txt', 'w') as file:
                file.write(s)
        return s
class scenario(handleImpedance):

    """Class containing the SPS impedance scenario

    This class contains the specific scenario used in beam dynamics simulations
    in the SPS
    
    This class inherits of the handleImpedance class which contains the methods
    used to load the data. This class which files are used, depending on the
    scenario ('present' or 'future') and apply various modifications (number of
    element in the ring, damping wanted etc...)

    Attributes
    ----------
    MODEL : str
        scenario, can be 'present' or 'future'. In case you want a user defined
        sceneario, you can overwrite all the various options (see __init__).
        You can also use the method in handleImpedance to modify the impedance
        before converting it in BLonD variable with the impedance2blond class

    Examples
    --------
    >>> from blabla import blibli
    >>> 
    """

    def __init__(self, MODEL='present', **kwargs):

        handleImpedance.__init__(self, **kwargs)
        self.MODEL = MODEL

        # Set the variable for future and present
        if self.MODEL == 'future':
            self.FB_attenuation = -26.0

            self.HOM_630_R_factor = 1/1.
            self.HOM_630_Q_factor = 1/1.
            self.HOM_915_R_factor = 1/1.
            self.HOM_915_Q_factor = 1/1.
            self.HOM_1130_R_factor = 1./1.
            self.HOM_1130_Q_factor = 1./1.
            self.HOM_1500_R_factor = 1./1.
            self.HOM_1500_Q_factor = 1./1.

            self.Gr2flangesShield = 'Shield'

            # MQF ENAMEL
            self.Flange_QFMBA_R_factor = 1.
            self.Flange_MBAMBA_R_factor = 1.
            # MQF NON ENAMEL
            self.Flange_QFQFWB_R_factor = 1.
            # BQX ENAMEL
            self.Flange_BPVQD_R_factor = 1.
            self.Flange_BPHQF_R_factor = 1.

            self.ThomasVVSA = 'True'
            self.Flange_VVSA_R_factor = 1.#  +12 -> *29/17
            self.Flange_VVSA_Q_factor = 1.

            self.ThomasVVSB = 'True'
            self.Flange_VVSB_R_factor = 1.#  +12 -> *29/17
            self.Flange_VVSB_Q_factor = 1.

            # UPP NON ENAMEL
            self.UPP_R_factor = 1.
            #UPP_R_factor = 1.
            # MKP
            self.mkpAttenuation_R = 1/1.
            self.nMKEL = 4
            self.nMKES = 2

        elif self.MODEL == 'present':
            self.FB_attenuation = -20.0
            
            self.HOM_630_R_factor = 1/1.
            self.HOM_630_Q_factor = 1/1.
            self.HOM_915_R_factor = 1/1.
            self.HOM_915_Q_factor = 1/1.
            self.HOM_1130_R_factor = 1./1.
            self.HOM_1130_Q_factor = 1./1.
            self.HOM_1500_R_factor = 1./1.
            self.HOM_1500_Q_factor = 1./1.

            self.Gr2flangesShield = 'noShield'

            # MQF ENAMEL
            self.Flange_QFMBA_R_factor = 1.
            self.Flange_MBAMBA_R_factor = 1.
            # MQF NON ENAMEL
            self.Flange_QFQFWB_R_factor = 1.
            # BQX ENAMEL
            self.Flange_BPVQD_R_factor = 1.
            self.Flange_BPHQF_R_factor = 1.

            self.ThomasVVSA = True
            self.Flange_VVSA_R_factor = 1.#  +12 -> *29/17
            self.Flange_VVSA_Q_factor = 1.

            self.ThomasVVSB = True
            self.Flange_VVSB_R_factor = 1.#  +12 -> *29/17
            self.Flange_VVSB_Q_factor = 1.

            # UPP NON ENAMEL
            self.UPP_R_factor = 1.
            # MKP
            self.mkpAttenuation_R = 1/1.
            self.nMKEL = 4
            self.nMKES = 2

        # If other options are specified, overwrite
        if 'FB_attenuation' in kwargs:
            self.FB_attenuation = kwargs.get('FB_attenuation')
            
        if 'HOM_630_R_factor' in kwargs:
            self.HOM_630_R_factor = kwargs.get('HOM_630_R_factor')
        if 'HOM_630_Q_factor' in kwargs:
            self.HOM_630_Q_factor = kwargs.get('HOM_630_Q_factor')
        if 'HOM_915_R_factor' in kwargs:
            self.HOM_915_R_factor = kwargs.get('HOM_915_R_factor')
        if 'HOM_915_Q_factor' in kwargs:
            self.HOM_915_Q_factor = kwargs.get('HOM_915_Q_factor')
        if 'HOM_1130_R_factor' in kwargs:
            self.HOM_1130_R_factor = kwargs.get('HOM_1130_R_factor')
        if 'HOM_1130_Q_factor' in kwargs:
            self.HOM_1130_Q_factor = kwargs.get('HOM_1130_Q_factor')
        if 'HOM_1500_R_factor' in kwargs:
            self.HOM_1500_R_factor = kwargs.get('HOM_1500_R_factor')
        if 'HOM_1500_Q_factor' in kwargs:
            self.HOM_1500_Q_factor = kwargs.get('HOM_1500_Q_factor')
            
        if 'Flange_QFMBA_R_factor' in kwargs:
            self.Flange_QFMBA_R_factor = kwargs.get('Flange_QFMBA_R_factor')
        if 'Flange_QFQFWB_R_factor' in kwargs:
            self.Flange_QFQFWB_R_factor = kwargs.get('Flange_QFQFWB_R_factor')
        if 'Flange_MBAMBA_R_factor' in kwargs:
            self.Flange_MBAMBA_R_factor = kwargs.get('Flange_MBAMBA_R_factor')
        if 'Flange_BPVQD_R_factor' in kwargs:
            self.Flange_BPVQD_R_factor = kwargs.get('Flange_BPVQD_R_factor')
        if 'Flange_BPHQF_R_factor' in kwargs:
            self.Flange_BPHQF_R_factor = kwargs.get('Flange_BPHQF_R_factor')

        if 'Flange_VVSA_R_factor' in kwargs:
            self.Flange_VVSA_R_factor = kwargs.get('Flange_VVSA_R_factor')
        if 'Flange_VVSA_Q_factor' in kwargs:
            self.Flange_VVSA_Q_factor = kwargs.get('Flange_VVSA_Q_factor')
        if 'Flange_VVSA_FREQ_target' in kwargs:
            self.Flange_VVSA_FREQ_target = kwargs.get('Flange_VVSA_FREQ_target')
            
        if 'Flange_VVSB_R_factor' in kwargs:
            self.Flange_VVSB_R_factor = kwargs.get('Flange_VVSB_R_factor')
        if 'Flange_VVSB_Q_factor' in kwargs:
            self.Flange_VVSB_Q_factor = kwargs.get('Flange_VVSB_Q_factor')
        if 'Flange_VVSB_FREQ_target' in kwargs:
            self.Flange_VVSB_FREQ_target = kwargs.get('Flange_VVSB_FREQ_target')
            
        if 'Gr2flangesShield' in kwargs:
            self.Gr2flangesShield = kwargs.get('Gr2flangesShield')
        if 'nMKEL' in kwargs:
            self.nMKEL = kwargs.get('nMKEL')
        if 'nMKES' in kwargs:
            self.nMKES = kwargs.get('nMKES')
        if 'mkpAttenuation_R' in kwargs:
            self.mkpAttenuation_R = kwargs.get('mkpAttenuation_R')
        if 'UPP_R_factor' in kwargs:
            self.UPP_R_factor = kwargs.get('UPP_R_factor')
        if 'ThomasVVSA' in kwargs:
            self.ThomasVVSA = kwargs.get('ThomasVVSA')
        if 'ThomasVVSB' in kwargs:
            self.ThomasVVSB = kwargs.get('ThomasVVSB')


#------------------------------------------------------------------------------
#
# IMPORT FUNCTIONS OF ALL ELEMENTS SEPARATED
#
#------------------------------------------------------------------------------


# -----------------
# CAVITIES
# --------
    def import_200MHz_5_HOM(self, quantity, fitWakefield=False):
        fileHOM = 'cavities/200MHz/5sections/TWC200_5sections_HOM.dat'
        self.importResonatorFromFile(fileHOM, RshFactor=quantity)
        # Apply damping on HOM
        # ---------------------        
        # -- 630 MHz HOM
        if fitWakefield:
            self.damp_R_resonatorOrTWC(fileHOM,[0.6291e9,0.6299e9],0)
            self.damp_Q_resonatorOrTWC(fileHOM,[0.6291e9,0.6299e9],0)
            self.importResonatorFromFile('cavities/200MHz/5sections/fit_5sec_628_orig.txt', unitFreq=1e6, RshFactor=quantity*2*self.HOM_630_R_factor, QFactor=self.HOM_630_Q_factor)
        else:
            self.damp_R_resonatorOrTWC(fileHOM,[0.6291e9,0.6299e9],self.HOM_630_R_factor)
            self.damp_Q_resonatorOrTWC(fileHOM,[0.6291e9,0.6299e9],self.HOM_630_Q_factor)
        # -- 915 MHz HOM
        self.damp_R_resonatorOrTWC(fileHOM,[0.9142e9,0.9148e9],self.HOM_915_R_factor)
        self.damp_Q_resonatorOrTWC(fileHOM,[0.9142e9,0.9148e9],self.HOM_915_Q_factor)        
        # -- 1130 MHz HOM
        self.damp_R_resonatorOrTWC(fileHOM,[1.1317e9,1.1332e9],self.HOM_1130_R_factor)
        self.damp_Q_resonatorOrTWC(fileHOM,[1.1317e9,1.1332e9],self.HOM_1130_Q_factor)        
        # -- 1500 MHz HOM
        self.damp_R_resonatorOrTWC(fileHOM,[1.5074e9,1.5078e9],self.HOM_1500_R_factor)
        self.damp_Q_resonatorOrTWC(fileHOM,[1.5074e9,1.5078e9],self.HOM_1500_Q_factor)

    def import_200MHz_5_main(self, quantity):
        self.importResonatorFromFile('cavities/200MHz/5sections/TWC200_5sections_MAIN.dat',RshFactor=10**(self.FB_attenuation/20.)*quantity)

    def import_200MHz_5_main_dome(self, quantity):
        self.importTWCFromFile('cavities/200MHz/5sections/TWC200_5sections_dome_MAIN.dat',RshFactor=quantity * 10**(self.FB_attenuation/20.))
        
    def import_200MHz_4_HOM(self, quantity, fitWakefield=False):
        fileHOM = 'cavities/200MHz/4sections/TWC200_4sections_HOM.dat'
        self.importResonatorFromFile(fileHOM, RshFactor=quantity)
        # Apply damping on HOM
        # ---------------------        
        # -- 630 MHz HOM
        if fitWakefield:
            self.damp_R_resonatorOrTWC(fileHOM,[0.6289e9,0.6299e9],0)
            self.damp_Q_resonatorOrTWC(fileHOM,[0.6289e9,0.6299e9],0)
            self.importResonatorFromFile('cavities/200MHz/4sections/fit_4sec_628_orig.txt', unitFreq=1e6, RshFactor=quantity*self.HOM_630_R_factor, QFactor=self.HOM_630_Q_factor)
        else:
            self.damp_R_resonatorOrTWC(fileHOM,[0.6289e9,0.6299e9],self.HOM_630_R_factor)
            self.damp_Q_resonatorOrTWC(fileHOM,[0.6289e9,0.6299e9],self.HOM_630_Q_factor)
        # -- 915 MHz HOM
        self.damp_R_resonatorOrTWC(fileHOM,[0.914e9,0.9148e9],self.HOM_915_R_factor)
        self.damp_Q_resonatorOrTWC(fileHOM,[0.914e9,0.9148e9],self.HOM_915_Q_factor)        
        # -- 1130 MHz HOM
        self.damp_R_resonatorOrTWC(fileHOM,[1.1306e9,1.1335e9],self.HOM_1130_R_factor)
        self.damp_Q_resonatorOrTWC(fileHOM,[1.1306e9,1.1335e9],self.HOM_1130_Q_factor)        
        # -- 1500 MHz HOM
        self.damp_R_resonatorOrTWC(fileHOM,[1.5073e9,1.5076e9],self.HOM_1500_R_factor)
        self.damp_Q_resonatorOrTWC(fileHOM,[1.5073e9,1.5076e9],self.HOM_1500_Q_factor)
    
    def import_200MHz_4_HOM_postLS2(self, quantity):
        self.importResonatorFromFile('cavities/200MHz/4sections/4Section_HOM_PostLS2.txt', RshFactor=quantity, unitFreq=1e6)

    def import_200MHz_4_main(self, quantity):
        self.importResonatorFromFile('cavities/200MHz/4sections/TWC200_4sections_MAIN.dat',RshFactor=10**(self.FB_attenuation/20.)*quantity)

    def import_200MHz_4_main_dome(self, quantity):
        self.importTWCFromFile('cavities/200MHz/4sections/TWC200_4sections_dome_MAIN.dat',RshFactor=quantity * 10**(self.FB_attenuation/20.))

    def import_200MHz_3_HOM(self, quantity, fitWakefield=False):
        fileHOM = 'cavities/200MHz/3sections/TWC200_3sections_HOM.dat'
        self.importResonatorFromFile(fileHOM, RshFactor=quantity)
        # Apply damping on HOM
        # ---------------------        
        # -- 630 MHz HOM
        if fitWakefield:
            self.damp_R_resonatorOrTWC(fileHOM,[0.6289e9,0.6299e9],0)
            self.damp_Q_resonatorOrTWC(fileHOM,[0.6289e9,0.6299e9],0)
            self.importResonatorFromFile('cavities/200MHz/3sections/fit_3sec_628_orig.txt', unitFreq=1e6, RshFactor=quantity*self.HOM_630_R_factor, QFactor=self.HOM_630_Q_factor)
        else:
            self.damp_R_resonatorOrTWC(fileHOM,[0.6289e9,0.6299e9],self.HOM_630_R_factor)
            self.damp_Q_resonatorOrTWC(fileHOM,[0.6289e9,0.6299e9],self.HOM_630_Q_factor)
        # -- 915 MHz HOM
        self.damp_R_resonatorOrTWC(fileHOM,[0.914e9,0.9148e9],self.HOM_915_R_factor)
        self.damp_Q_resonatorOrTWC(fileHOM,[0.914e9,0.9148e9],self.HOM_915_Q_factor)        
        # -- 1130 MHz HOM
        self.damp_R_resonatorOrTWC(fileHOM,[1.1306e9,1.1335e9],self.HOM_1130_R_factor)
        self.damp_Q_resonatorOrTWC(fileHOM,[1.1306e9,1.1335e9],self.HOM_1130_Q_factor)        
        # -- 1500 MHz HOM
        self.damp_R_resonatorOrTWC(fileHOM,[1.5073e9,1.5076e9],self.HOM_1500_R_factor)
        self.damp_Q_resonatorOrTWC(fileHOM,[1.5073e9,1.5076e9],self.HOM_1500_Q_factor)

    def import_200MHz_3_HOM_postLS2(self, quantity):
        self.importResonatorFromFile('cavities/200MHz/3sections/3Section_HOM_PostLS2.txt', RshFactor=quantity, unitFreq=1e6)

    def import_200MHz_3_main_dome(self, quantity):
        self.importTWCFromFile('cavities/200MHz/3sections/TWC200_3sections_dome_MAIN.dat',RshFactor=quantity * 10**(self.FB_attenuation/20.))

    def import_800MHz_main(self, quantity):
        self.importResonatorFromFile('cavities/800MHz/TWC800_MAIN.dat', RshFactor=quantity)
        self.importTWCFromFile('cavities/800MHz/TWC800_main_I.dat', RshFactor=quantity)

    def import_800MHz_HOM(self, quantity):
        self.importResonatorFromFile('cavities/800MHz/TWC800_HOM.dat', RshFactor=quantity)
    
    
    def import_crab_ychamber(self, quantity):
        self.importInputTableFromFile('cavities/crab/YChamberTable.dat', unitFreq=1e9, ZFactor=quantity)
    def import_crab_cavity(self, quantity):
        self.importResonatorFromFile('cavities/crab/CrabCavitySPS.dat',unitFreq=1e9, RshFactor=quantity)


# -----------------
# FLANGES
# --------
    def import_QDQD_enamelled(self, quantity):
        self.importResonatorFromFile('flanges/QD/QD-QD-Enam-Flange.dat', RshFactor=quantity)
    def import_QDQD_closedflange(self, quantity):
        self.importResonatorFromFile('flanges/QD/QD-QD-Closed-Flange.dat', RshFactor=quantity)
    def import_QDLongBellow_shieldedPumpingPort(self, quantity):
        self.importResonatorFromFile('flanges/QD/Shielded_Pumping_ports.dat', RshFactor=quantity)
    def import_QDLongBellow_shieldedPumpingPort_VVSA(self, quantity):
        self.importResonatorFromFile('flanges/QD/Shielded_Pumping_ports_VVSA.dat', RshFactor=quantity)
    def import_QD_BPV(self, quantity):
        self.importResonatorFromFile('flanges/QD/BPV-QD-Flanges.dat',RshFactor=self.Flange_BPVQD_R_factor*quantity)
    def import_QDQD_enamelled_shielded(self, quantity):
        self.importResonatorFromFile('flanges/QD/QD-QD-Enam-Flange-Shielded.dat', RshFactor=quantity)
    def import_QDQD_closedflange_shielded(self, quantity):
        self.importResonatorFromFile('flanges/QD/QD-QD-Closed-Flange-Shielded.dat', RshFactor=quantity)

    def import_QFQF_closedflange_noBellow(self, quantity):
        self.importResonatorFromFile('flanges/QF/QF-QF-Closed-Flange-noBellow.dat', RshFactor=quantity)
    def import_QFMBA_closedflange_noBellow(self, quantity):
        self.importResonatorFromFile('flanges/QF/QF-MBA-Closed-Flange-noBellow.dat', RshFactor=quantity)
    def import_QFQF_closedflange_noBellow_shielded(self, quantity):
        self.importResonatorFromFile('flanges/QF/QF-QF-Closed-Flange-noBellow-Shielded.dat', RshFactor=quantity)
    def import_QFMBA_closedflange_noBellow_shielded(self, quantity):
        self.importResonatorFromFile('flanges/QF/QF-MBA-Closed-Flange-noBellow-Shielded.dat', RshFactor=quantity)

    def import_QFQF_closedflange_bellow(self, quantity):
        self.importResonatorFromFile('flanges/QF/QF-QF-Closed-Flange_bellow.dat', RshFactor=quantity)
    def import_QFQF_closedflange_bellow_shielded(self, quantity):
        self.importResonatorFromFile('flanges/QF/QF-QF-Closed-Flange-Shielded.dat', RshFactor=quantity*self.Flange_QFQFWB_R_factor)
    def import_QFMBA_closedflange_bellow_shielded(self, quantity):
        self.importResonatorFromFile('flanges/QF/QF-MBA-Closed-Flange-Shielded.dat', RshFactor=quantity*self.Flange_QFQFWB_R_factor)
    def import_QFQF_closedflange_bellow_dampingResistor(self, quantity):
        self.importResonatorFromFile('flanges/QF/QF-QF-Closed-Flange_bellow_dampingResistor.dat', RshFactor=quantity)
    def import_QFMBA_closedflange_bellow_dampingResistor(self, quantity):
        self.importResonatorFromFile('flanges/QF/QF-MBA-Closed-Flange_bellow_dampingResistor.dat', RshFactor=quantity)
    
    def import_QFQF_enamelled_shielded(self, quantity):
        self.importResonatorFromFile('flanges/QF/QF-MBA_MBA-MBA-Flanges-Shielded.dat',RshFactor=self.Flange_MBAMBA_R_factor*quantity)
    def import_QFQF_doubleTubeShielded(self, quantity):
        self.importResonatorFromFile('/flanges/groupe2/QF-MBA_MBA-MBA-Flanges-DoubleTubeShielded.dat',RshFactor=quantity)
    def import_QFQF_springDrivenRFFingers(self,quantity):
        self.importResonatorFromFile('/flanges/groupe2/QF-MBA_MBA-MBA-Flanges-SpringDrivenRFFingers',RshFactor=quantity)
    
    def import_QF_BPH(self, quantity):
        self.importResonatorFromFile('flanges/QF/BPH-QF-Flanges.dat',RshFactor=self.Flange_BPHQF_R_factor*quantity)
    def import_QF_BPH_dampingResistor(self, quantity):
        self.importResonatorFromFile('flanges/QF/BPH-QF-Flanges_dampingResistor.dat',RshFactor=self.Flange_BPHQF_R_factor*quantity)
    def import_QFMBA(self, quantity):
        self.importResonatorFromFile('flanges/QF/QF-MBA-Flanges.dat', RshFactor=self.Flange_QFMBA_R_factor*quantity)
    def import_QFMBA_dampingResistor(self, quantity):
        self.importResonatorFromFile('flanges/QF/QF-MBA-Flanges_dampingResistor.dat', RshFactor=self.Flange_QFMBA_R_factor*quantity)
    def import_MBAMBA(self, quantity):
        self.importResonatorFromFile('flanges/QF/MBA-MBA-Flanges.dat', RshFactor= self.Flange_MBAMBA_R_factor*quantity)
    def import_MBAMBA_dampingResistor(self, quantity):
        self.importResonatorFromFile('flanges/QF/MBA-MBA-Flanges_dampingResistor.dat', RshFactor= self.Flange_MBAMBA_R_factor*quantity)
    def import_QFMBA_unshieldedPumpingPort(self, quantity):
        self.importResonatorFromFile('flanges/QF/QF-MBA_Unshielded_Pumping_ports.dat',RshFactor=quantity)
    
    def import_Ychamber(self, quantity):
        self.importResonatorFromFile('flanges/chambers/Y_chambers.dat', RshFactor=quantity)
    def import_beamScrappersTank_empty(self, quantity):
        self.importResonatorFromFile('flanges/chambers/Empty_Beam_scrapers_tank.dat', RshFactor=quantity)
    def import_beamScrappers(self, quantity):
        self.importResonatorFromFile('flanges/chambers/Beam_scrapers.dat', RshFactor=quantity)

    def import_VVSA(self, quantity):
        self.importResonatorFromFile('flanges/sector_valves/VVSA.dat', RshFactor= quantity * self.Flange_VVSA_R_factor, QFactor= self.Flange_VVSA_Q_factor)
    def import_VVSA_shielded(self, quantity):
        self.importResonatorFromFile('flanges/sector_valves/VVSA_shielded.dat', RshFactor= quantity * self.Flange_VVSA_R_factor, QFactor= self.Flange_VVSA_Q_factor)
    def import_VVSB(self, quantity):
        self.importResonatorFromFile('flanges/sector_valves/VVSB.dat', RshFactor= quantity * self.Flange_VVSB_R_factor, QFactor = self.Flange_VVSB_Q_factor)
    def import_VVSB_shielded(self, quantity):
        self.importResonatorFromFile('flanges/sector_valves/VVSB_shielded.dat', RshFactor= quantity * self.Flange_VVSB_R_factor, QFactor = self.Flange_VVSB_Q_factor)
# -----------------
# KICKERS
# --------

    def import_MKP(self, quantity):
        #Kicker magnet, proton injection
        #MKP 11955
        self.importResonatorFromFile('kickers/MKP_11955_res.dat', RshFactor=self.mkpAttenuation_R*quantity)
    def import_MKDH_A(self, quantity):
        #Kicker magnet, horizontal sweeping for dumping, A-type
        #MKDH 11751 11754
        self.importResonatorFromFile('kickers/MKDH_11751and11754_res.dat', RshFactor=quantity)
    def import_MKDH_B(self,quantity):
        #Kicker magnet, horizontal sweeping for dumping, B-type
        #MKDH 11757
        self.importResonatorFromFile('kickers/MKDH_11757_res.dat', RshFactor=quantity)
    def import_MKDV_A(self,quantity):
        #Kicker magnet, vertical deflection for dumping, A-type
        #MKDV 11731
        self.importResonatorFromFile('kickers/MKDV_11731_res.dat', RshFactor=quantity)
    def import_MKDV_B(self, quantity):
        #Kicker magnet, vertical deflection for dumping, B-type
        #MKDV 11736
        self.importResonatorFromFile('kickers/MKDV_11736_res.dat', RshFactor=quantity)
    def import_MKE_L(self, quantity):
        #Kicker magnet, extraction, L-type, KickerFastExtraction
        #MKE 41631 41634 61361 61634
        self.importResonatorFromFile('kickers/MKELser_41631and41634and41654and61631and61634_res.dat', RshFactor=quantity)
    def import_MKE_S(self, quantity):
        #Kicker magnet, extraction, S-type, KickerFastExtraction
        #MKE 41637 41651 61637
        self.importResonatorFromFile('kickers/MKESser_41637and41651and61637_res.dat', RshFactor=quantity)
    def import_MKE_180mm_mario(self, quantity):
        #New MKE model with 180mm serigraphy from Mario Beck
        #should be 4 of them in the machine, to be confirmed (janvier 2018)
        self.importResonatorFromFile('kickers/MKE_ser180_mario.dat', unitRsh=1., RshFactor=quantity)
    def import_MKE_200mm_mario(self, quantity):
        #New MKE model with 200mm serigraphy from Mario Beck
        #should be 3 of them in the machine, to be confirmed (janvier 2018)
        self.importResonatorFromFile('kickers/MKE_ser200_mario.dat', unitRsh=1., RshFactor=quantity)
    def import_MKE_L_180mm_mario(self, quantity):
        self.importResonatorFromFile('kickers/MBMKEL_ser180.txt', RshFactor=quantity)
    def import_MKE_S_180mm_mario(self, quantity):
        self.importResonatorFromFile('kickers/MBMKES_ser180.txt', RshFactor=quantity)
    def import_MKE_L_200mm_mario(self, quantity):
        self.importResonatorFromFile('kickers/MBMKEL_ser200.txt', RshFactor=quantity)
    def import_MKE_S_200mm_mario(self, quantity):
        self.importResonatorFromFile('kickers/MBMKES_ser200.txt', RshFactor=quantity)
    def import_MKPA(self, quantity):
        #Kicker magnet, Inject. Kicker Hadron
        #MKPA 11931 11936
        self.importResonatorFromFile('kickers/MKPA_11931and11936_res.dat', RshFactor=self.mkpAttenuation_R*quantity)
    def import_MKPC(self, quantity):
        #Kicker magnet, proton injection, short MKP
        #MKPC 11952
        self.importResonatorFromFile('kickers/MKPC_11952_res.dat', RshFactor=self.mkpAttenuation_R*quantity)
    def import_MKQH(self, quantity):
        #Kicker magnet, Q-measurement, horizontal
        #MKQH 11653
        self.importResonatorFromFile('kickers/MKQH_11653_res.dat', RshFactor=quantity)
    def import_MKQV(self, quantity):
        #Kicker magnet, Q-measurement, vertical
        #MKQH 11679
        self.importResonatorFromFile('kickers/MKQV_11679_res.dat', RshFactor=quantity)
    def import_slottedLineKicker(self, quantity):
        self.importInputTableFromFile('kickers/SLK_Full_200m.txt', unitFreq=1e9, RshFactor=quantity)

# -------------------
# Instrumentation
# ---------
    def import_BPH(self, quantity):
        self.importResonatorFromFile('beam_instrumentations/BPHs.dat', RshFactor=quantity)
    def import_BPV(self, quantity):
        self.importResonatorFromFile('beam_instrumentations/BPVs.dat', RshFactor=quantity)
    def import_wirescanner(self, quantity):
        self.importResonatorFromFile('beam_instrumentations/PS_SPS_wire_scanner.dat', RshFactor=quantity)
# ----------------------------------------------------------------------------
#
# Methods to import various SPS impedance at once
#
# ----------------------------------------------------------------------------

# -----------------
# CAVITIES
# --------
    def importCavities800MHz(self):
        self.import_800MHz_main(2)
        self.import_800MHz_HOM(2)

    def importCavities200MHz_5sections(self, impType='resonator', fitWakefield=False):
        if impType=='resonator':
            self.import_200MHz_5_main(2)
        elif impType=='dome':
            self.import_200MHz_5_main_dome(2)
        self.import_200MHz_5_HOM(1, fitWakefield=fitWakefield)

    def importCavities200MHz_4sections(self, impType='dome', fitWakefield=False):
        if impType == 'dome':
            self.import_200MHz_4_main_dome(2)
        else:
            self.import_200MHz_4_main(2)
        self.import_200MHz_4_HOM(2, fitWakefield=fitWakefield)

    def importCavities200MHz_3sections(self, fitWakefield=False):
        self.import_200MHz_3_main_dome(4)
        self.import_200MHz_3_HOM(4, fitWakefield=fitWakefield)

    def importCavities200MHz(self, fffb_reduction=False, fitWakefield=False, domeOnly=False):
        if self.MODEL == 'present':
            if domeOnly:
                self.importCavities200MHz_5sections(impType='dome', fitWakefield = fitWakefield)
                self.importCavities200MHz_4sections(impType='dome', fitWakefield=fitWakefield)
            else:
                self.importCavities200MHz_5sections(fitWakefield = fitWakefield)
                self.importCavities200MHz_4sections(impType='resonator', fitWakefield=fitWakefield)
        elif self.MODEL == 'future':
            self.importCavities200MHz_3sections(fitWakefield=fitWakefield)
            self.importCavities200MHz_4sections(impType='dome', fitWakefield=fitWakefield)
                    
    def import_200MHzCavities_withNewHOMDampingScheme(self):
        
        self.import_200MHz_4_main_dome(2)
        self.import_200MHz_4_HOM_postLS2(2)

        self.import_200MHz_3_main_dome(4)
        self.import_200MHz_3_HOM_postLS2(4)

    def importCrabSPS(self):
        self.import_crab_ychamber(4)
        self.import_crab_cavity(2)
# -----------------
# FLANGES
# --------
    def importFlanges(self, VVSA_shielded=False, VVSB_shielded=False, QDshield=False, BPH_shield=False):
        self.import_QFMBA_unshieldedPumpingPort(25*self.UPP_R_factor)
        self.import_QDLongBellow_shieldedPumpingPort(71)
        self.import_QD_BPV(90)
        
        if BPH_shield:
            self.import_QF_BPH(1)
            self.import_QF_BPH_dampingResistor(5)
        else:
            self.import_QF_BPH(12)
            self.import_QF_BPH_dampingResistor(25)
        if QDshield:
            self.import_QDQD_closedflange_shielded(1)
            self.import_QDQD_enamelled_shielded(1)
        else:
            self.import_QDQD_closedflange(75)
            self.import_QDQD_enamelled(99)
        if self.Gr2flangesShield == 'Shield':
            self.import_QFQF_enamelled_shielded(94)
            self.import_QFQF_closedflange_bellow_shielded(25)
            self.import_QFMBA_closedflange_bellow_shielded(1)
            self.import_QFQF_closedflange_noBellow_shielded(18)
            self.import_QFMBA_closedflange_noBellow_shielded(2)
        else:
            self.import_QFMBA(2)
            self.import_QFMBA_dampingResistor(78)
            self.import_MBAMBA(2)
            self.import_MBAMBA_dampingResistor(12)
            self.import_QFQF_closedflange_noBellow(18)
            self.import_QFMBA_closedflange_noBellow(2)
            self.import_QFQF_closedflange_bellow(3)
            self.import_QFQF_closedflange_bellow_dampingResistor(22)
            self.import_QFMBA_closedflange_bellow_dampingResistor(1)

        if self.ThomasVVSA:
            if VVSA_shielded:
                self.import_VVSA_shielded(31)
            else:
                self.import_VVSA(31)
        else:
            self.import_QDLongBellow_shieldedPumpingPort_VVSA(17)

        if self.ThomasVVSB:
            if VVSB_shielded:
                self.import_VVSB_shielded(33)
            else:
                self.import_VVSB(33)

    def importFutureQFDoubleTubeShield(self, fffb_reduction=False):
        self.importCavities800MHz()
        self.importCavities200MHz(fffb_reduction=fffb_reduction)
        self.importMeasurementDevices()
        self.importKickers()
        self.importMiscellaneous()
        self.importFlanges()
        typeOfFlangeToModify = 'flanges/QF/QF-MBA_MBA-MBA-Flanges-Shielded.dat'
        self.damp_R_resonatorOrTWC(typeOfFlangeToModify, [0,6e9], R_factor=0.)
        self.import_QFQF_doubleTubeShielded(96)

    def importFutureQFQDshield(self, fffb_reduction=False):
        self.importCavities800MHz()
        self.importCavities200MHz(fffb_reduction=fffb_reduction)
        self.importMeasurementDevices()
        self.importKickers()
        self.importMiscellaneous()
        self.importFlanges()
#        typeOfFlangeToModify = 'flanges/QF/QF-MBA_MBA-MBA-Flanges-Shielded.dat'
#        self.damp_R_resonatorOrTWC(typeOfFlangeToModify, [0,6e9], R_factor=0.)
#        self.import_QFQF_doubleTubeShielded(96)
        
        typeOfFlangeToModify = 'flanges/QD/QD-QD-Closed-Flange.dat'
        typeOfFlangeToModify2 = 'flanges/QD/QD-QD-Enam-Flange.dat'
        self.damp_R_resonatorOrTWC(typeOfFlangeToModify, [0,6e9], R_factor=0.)
        self.damp_R_resonatorOrTWC(typeOfFlangeToModify2, [0,6e9], R_factor=0.)

        self.import_QDQD_closedflange_shielded(75)
        self.import_QDQD_enamelled_shielded(99)

# -----------------
# KICKERS
# --------
    def importKickers(self):
        self.import_MKDH_A(2)
        self.import_MKDH_B(1)
        self.import_MKDV_A(1)
        self.import_MKDV_B(1)
        self.import_MKE_L(4)
        self.import_MKE_S(3)
        self.import_MKP(1)
        self.import_MKPA(2)
        self.import_MKPC(1)
        self.import_MKQH(1)
        self.import_MKQV(1)

    def importKickers_mario(self,noMKP=False):
        self.import_MKDH_A(2)
        self.import_MKDH_B(1)
        self.import_MKDV_A(1)
        self.import_MKDV_B(1)
#        self.import_MKE_180mm_mario(4)
#        self.import_MKE_200mm_mario(3)
        self.import_MKE_L_180mm_mario(4)
        self.import_MKE_S_180mm_mario(3)
        if noMKP:
            pass
        else:
            self.import_MKP(1)
            self.import_MKPA(2)
            self.import_MKPC(1)
        self.import_MKQH(1)
        self.import_MKQV(1)
    def importSlottedLineKicker(self):
        self.import_slottedLineKicker(1)
        
# -----------------
# MISCELLANEOUS
# --------
    def importInjectionPipe(self):
        self.importResonatorFromFile('miscellaneous/injection_pipe.dat', RshFactor=3)

    def importInjectionPipeonlyone(self):
        self.importResonatorFromFile('miscellaneous/injection_pipe.dat', RshFactor=1)

    def importMeasurementDevices(self,wirescanner=False):
        #--------------------------------------------------------------------
        # BEAM POSITION MONITOR
        #--------------------------------------------------------------------

        self.import_BPH(106)
        self.import_BPV(99)
        if wirescanner:
            self.import_wirescanner(1)


    def importMiscellaneous(self, ychamber=True):

        if ychamber:
            self.import_Ychamber(3)
        else:
            pass
        self.import_beamScrappersTank_empty(2)
        self.import_beamScrappers(3)
        # Z given as Z/n: must be multiplied by f/frev
        frev = 43271.480274305672
        self.importInputTableFromFile('SC_resistive_wall/Resistive_wall.dat', ZFactor=(1./frev))
        self.table_impedance['SC_resistive_wall/Resistive_wall.dat']['ReZ'] *=\
            self.table_impedance['SC_resistive_wall/Resistive_wall.dat']['fr']
        self.table_impedance['SC_resistive_wall/Resistive_wall.dat']['ImZ'] *=\
            self.table_impedance['SC_resistive_wall/Resistive_wall.dat']['fr']
        self.importInputTableFromFile('miscellaneous/totWrongZSMSE_imp.dat')

# ----------------------------------------------------------------------------
#
# Import everything
#
#-----------------------------------------------------------------------------

    def importImpedanceSPS(self, fffb_reduction=False, VVSA_shielded=False, VVSB_shielded=False,
                           newDampingScheme=False, fitWakefield=False, kickerMario=False,
                           cavity800MHz=True, nokicker=False, noflange=False,
                           domeOnly=False, noMKP=False, QDshield=False,ychamber=True, BPH_shield=False):
        if cavity800MHz:
            self.importCavities800MHz()
        if newDampingScheme:
            self.import_200MHzCavities_withNewHOMDampingScheme()
        else:
            self.importCavities200MHz(fffb_reduction=fffb_reduction, fitWakefield=fitWakefield, domeOnly=domeOnly)
        if noflange == False:
            self.importFlanges(VVSA_shielded=VVSA_shielded, VVSB_shielded=VVSB_shielded, QDshield=QDshield, BPH_shield=BPH_shield)
        self.importMeasurementDevices()
        if nokicker == False:
            if kickerMario:
                self.importKickers_mario(noMKP=noMKP)
            else:
                self.importKickers()
        self.importMiscellaneous(ychamber=ychamber)


class impedance2blond(handleImpedance):

    """Class used to convert the impedance scenario to object usable by BLonD


    Attributes
    ----------
    table_impedance : dict
        Dictionary generated by the scenario class containing all the impedance
    wakeList : list
        list containing the BLonD impedance sources to be solved in time
    impedanceList : list
            list containing the BLonD impedance sources to be solved in freq
    impedanceListToPlot : list
            list containing all the BLonD impedance sources (to be plotted)

    Examples
    --------
    >>> from blabla import blibli
    >>> 
    """

    def __init__(self, table_impedance):
        self.table_impedance = table_impedance
        
        self.wakeAndImpListProcess()
    
    
    def generateResonator(self, fr, Rsh, Q):

        """
        Generates a BLonD Resonator based on the input data
                
        Parameters
        ----------
        fr : float,array
            array (or float) of resonant frequency.
        Rsh : float,array
            array (or float) of shunt impedance.
        Q : float,array
            array (or float) of quality factor Q.
        
        Returns
        -------
        Resonators : Resonators
            BLonD resonators
        """

        return Resonators(Rsh,fr,Q)

    def get_impedance(self, key, freq_array):
        imp = self.table_impedance[key]['impedance']
        imp.imped_calc(freq_array)
        return imp.impedance

    def generateInputTable(self, fr, ReZ, ImZ):

        """
        Generates a BLonD InputTable based on the input data
                
        Parameters
        ----------
        fr : float,array
            array (or float) of resonant frequency.
        ReZ : float,array
            array (or float) of the real part of the impedance.
        Q : float,array
            array (or float) of the imaginary part of the impedance.
        
        Returns
        -------
        InputTable : InputTable
            BLonD InputTable
        """

        return InputTable(fr, ReZ, ImZ)
        
    def generateTWC(self, fr, Rsh, alpha):

        """
        Generates a BLonD TravelingWaveCavity based on the input data
                
        Parameters
        ----------
        fr : float,array
            array (or float) of resonant frequency.
        Rsh : float,array
            array (or float) of shunt impedance.
        alpha : float,array
            array (or float) of time factor alpha.
        
        Returns
        -------
        TravelingWaveCavity : TravelingWaveCavity
            BLonD TravelingWaveCavity
        """

        return TravelingWaveCavity(Rsh, fr, alpha)

    def wakeAndImpListProcess(self):

        """
        Reprocess the wake and impedance list from the table_impedance
        """

        self.wakeList = list()
        self.impedanceList = list()
        self.impedanceListToPlot = list()
        
        for key in self.table_impedance:
            imp = self.table_impedance[key]
            if imp['type'] == 'resonator':
                imp['impedance'] = self.generateResonator(imp['fr'], imp['Rsh'], imp['Q'])
                self.wakeList.append(imp['impedance'])
                self.impedanceListToPlot.append(imp['impedance'])
            elif imp['type'] == 'inputtable':
                imp['impedance'] = self.generateInputTable(imp['fr'], imp['ReZ'], imp['ImZ'])
                self.impedanceList.append(imp['impedance'])
                self.impedanceListToPlot.append(imp['impedance'])
            elif imp['type'] == 'twc':
                imp['impedance'] = self.generateTWC(imp['fr'], imp['Rsh'], imp['alpha'])
                self.wakeList.append(imp['impedance'])
                self.impedanceListToPlot.append(imp['impedance'])
            else:
                SystemExit('type of impedance not recognized when importing in BLonD')
    