# coding: utf-8
# Copyright 2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Module containing the fundamental beam class with methods to compute beam 
statistics

:Authors: **Danilo Quartullo**, **Helga Timko**, **ALexandre Lasheen**

"""

from __future__ import division
from builtins import object
import numpy as np
from scipy.constants import m_p, m_e, e, c
from ..trackers.utilities import is_in_separatrix
from ..utils import bmath as bm


class Particle(object):

    def __init__(self, user_mass, user_charge):

        if user_mass > 0.:
            self.mass = float(user_mass)
            self.charge = float(user_charge)
        else:
            # MassError
            raise RuntimeError('ERROR: Particle mass not recognized!')


class Proton(Particle):

    def __init__(self):

        Particle.__init__(self, float(m_p*c**2/e), np.float(1))


class Electron(Particle):

    def __init__(self):
        self.mass = float(m_e*c**2/e)
        self.charge = float(-1)


class Beam(object):
    """Class containing the beam properties.

    This class containes the beam coordinates (dt, dE) and the beam properties.

    The beam coordinate 'dt' is defined as the particle arrival time to the RF 
    station w.r.t. the reference time that is the sum of turns. The beam
    coordiate 'dE' is defined as the particle energy offset w.r.t. the
    energy of the synchronous particle.

    The class creates a beam with zero dt and dE, see distributions to match
    a beam with respect to the RF and intensity effects.

    Parameters
    ----------
    Ring : Ring
        Used to import different quantities such as the mass and the energy.
    n_macroparticles : int
        total number of macroparticles.
    intensity : float
        total intensity of the beam (in number of charge).

    Attributes
    ----------
    mass : float
        mass of the particle [eV].
    charge : int
        integer charge of the particle [e].
    beta : float
        relativistic velocity factor [].
    gamma : float
        relativistic mass factor [].
    energy : float
        energy of the synchronous particle [eV].
    momentum : float
        momentum of the synchronous particle [eV].
    dt : numpy_array, float
        beam arrival times with respect to synchronous time [s].
    dE : numpy_array, float
        beam energy offset with respect to the synchronous particle [eV].
    mean_dt : float
        average beam arrival time [s].
    mean_dE : float
        average beam energy offset [eV].
    sigma_dt : float
        standard deviation of beam arrival time [s].
    sigma_dE : float
        standard deviation of beam energy offset [eV].
    intensity : float
        total intensity of the beam in number of charges [].
    n_macroparticles : int
        total number of macroparticles in the beam [].
    ratio : float
        ratio intensity per macroparticle [].
    n_macroparticles_lost : int
        number of macro-particles marked as 'lost' [].
    id : numpy_array, int
        unique macro-particle ID number; zero if particle is 'lost'.

    See Also
    ---------
    distributions.matched_from_line_density:
        match a beam with a given bunch profile.
    distributions.matched_from_distribution_function:
        match a beam with a given distribution function in phase space.

    Examples
    --------
    >>> from input_parameters.ring import Ring
    >>> from beam.beam import Beam
    >>>
    >>> n_turns = 10
    >>> C = 100
    >>> eta = 0.03
    >>> momentum = 26e9
    >>> ring = Ring(n_turns, C, eta, momentum, 'proton')
    >>> n_macroparticle = 1e6
    >>> intensity = 1e11
    >>>
    >>> my_beam = Beam(ring, n_macroparticle, intensity)
    """

    def __init__(self, Ring, n_macroparticles, intensity):

        self.Particle = Ring.Particle
        self.beta = Ring.beta[0][0]
        self.gamma = Ring.gamma[0][0]
        self.energy = Ring.energy[0][0]
        self.momentum = Ring.momentum[0][0]
        self.dt = np.zeros([int(n_macroparticles)], dtype=bm.precision.real_t)
        self.dE = np.zeros([int(n_macroparticles)], dtype=bm.precision.real_t)

        self.mean_dt = 0.
        self.sigma_dt = 0.
        self.min_dt = 0.
        self.max_dt = 0.

        self.mean_dE = 0.
        self.sigma_dE = 0.
        self.min_dE = 0.
        self.max_dE = 0.

        self.intensity = float(intensity)
        self.n_macroparticles = int(n_macroparticles)
        self.ratio = self.intensity/self.n_macroparticles
        self.id = np.arange(1, self.n_macroparticles + 1, dtype=int)
        self.n_total_macroparticles_lost = 0

    @property
    def n_macroparticles_lost(self):
        '''Number of lost macro-particles, defined as @property.

        Returns
        -------        
        n_macroparticles_lost : int
            number of macroparticles lost.

        '''

        return len(np.where(self.id == 0)[0])

    @property
    def n_macroparticles_alive(self):
        '''Number of transmitted macro-particles, defined as @property.

        Returns
        -------        
        n_macroparticles_alive : int
            number of macroparticles not lost.

        '''

        return self.n_macroparticles - self.n_macroparticles_lost

    def eliminate_lost_particles(self):
        """Eliminate lost particles from the beam coordinate arrays
        """

        indexalive = np.where(self.id == 0)[0]
        if len(indexalive) < self.n_macroparticles:
            self.dt = np.ascontiguousarray(
                self.beam.dt[indexalive], dtype=bm.precision.real_t, order='C')
            self.dE = np.ascontiguousarray(
                self.beam.dE[indexalive], dtype=bm.precision.real_t, order='C')
            self.n_macroparticles = len(self.beam.dt)
        else:
            # AllParticlesLost
            raise RuntimeError("ERROR in Beams: all particles lost and" +
                               " eliminated!")

    def statistics(self):
        '''
        Calculation of the mean and standard deviation of beam coordinates,
        as well as beam emittance using different definitions.
        Take no arguments, statistics stored in

        - mean_dt
        - mean_dE
        - sigma_dt
        - sigma_dE
        '''

        # Statistics only for particles that are not flagged as lost
        itemindex = np.where(self.id != 0)[0]
        # itemindex = bm.where(self.id, 0)
        self.mean_dt = bm.mean(self.dt[itemindex])
        self.sigma_dt = bm.std(self.dt[itemindex])
        self.min_dt = np.min(self.dt[itemindex])
        self.max_dt = np.max(self.dt[itemindex])

        self.mean_dE = bm.mean(self.dE[itemindex])
        self.sigma_dE = bm.std(self.dE[itemindex])
        self.min_dE = np.min(self.dE[itemindex])
        self.max_dE = np.max(self.dE[itemindex])

        # R.m.s. emittance in Gaussian approximation
        self.epsn_rms_l = np.pi*self.sigma_dE*self.sigma_dt  # in eVs

    def losses_separatrix(self, Ring, RFStation):
        '''Beam losses based on separatrix.

        Set to 0 all the particle's id not in the separatrix anymore.

        Parameters
        ----------
        Ring : Ring
            Used to call the function is_in_separatrix.
        RFStation : RFStation
            Used to call the function is_in_separatrix.
        '''

        itemindex = np.where(is_in_separatrix(Ring, RFStation, self,
                                              self.dt, self.dE) == False)[0]

        if itemindex.size != 0:
            self.id[itemindex] = 0

    def losses_longitudinal_cut(self, dt_min, dt_max):
        '''Beam losses based on longitudinal cuts.

        Set to 0 all the particle's id with dt not in the interval 
        (dt_min, dt_max).

        Parameters
        ----------
        dt_min : float
            minimum dt.
        dt_max : float
            maximum dt.
        '''

        itemindex = np.where((self.dt - dt_min)*(dt_max - self.dt) < 0)[0]

        if itemindex.size != 0:
            self.id[itemindex] = 0

    def losses_energy_cut(self, dE_min, dE_max):
        '''Beam losses based on energy cuts, e.g. on collimators.

        Set to 0 all the particle's id with dE not in the interval (dE_min, dE_max).

        Parameters
        ----------
        dE_min : float
            minimum dE.
        dE_max : float
            maximum dE.
        '''

        itemindex = np.where((self.dE - dE_min)*(dE_max - self.dE) < 0)[0]

        if itemindex.size != 0:
            self.id[itemindex] = 0

    def losses_below_energy(self, dE_min):
        '''Beam losses based on lower energy cut.

        Set to 0 all the particle's id with dE below dE_min.

        Parameters
        ----------
        dE_min : float
            minimum dE.
        '''

        itemindex = np.where((self.dE - dE_min) < 0)[0]

        if itemindex.size != 0:
            self.id[itemindex] = 0

    def split(self, random=False, fast=False):
        '''
        MPI ONLY ROUTINE: Splits the beam equally among the workers for
        MPI processing.
        Parameters
        ----------
        random : boolean
            Shuffle the beam before splitting, to be used with the
            approximation methonds.
        fast : boolean
            If true, it assumes that every worker has already a copy of the
            beam so only the particle ids are distributed.
            If false, all the coordinates are distributed by the master to all
            the workers.
        '''

        if not bm.mpiMode():
            raise RuntimeError(
                'ERROR: Cannot use this routine unless in MPI Mode')

        from ..utils.mpi_config import worker
        if worker.isMaster and random:
            import random
            random.shuffle(self.id)

        self.id = worker.scatter(self.id)

        if fast:
            self.dt = np.ascontiguousarray(self.dt[self.id-1])
            self.dE = np.ascontiguousarray(self.dE[self.id-1])
        else:
            self.dt = worker.scatter(self.dt)
            self.dE = worker.scatter(self.dE)
        assert (len(self.dt) == len(self.dE) and len(self.dt) == len(self.id))

        self.n_macroparticles = len(self.dt)
        # self.is_splitted = True

    # Split and gather should be going in pairs, undefined behavior in case
    # of split split or gather gather sequence
    # def split(self):
    #     from ..utils.mpi_config import worker
    #     if len(worker.indices) == 0 or worker.isMaster:
    #         start, size = worker.split(self.n_macroparticles)
    #         self.dt = self.dt[start: start + size]
    #         self.dE = self.dE[start: start + size]
    #         self.id = self.id[start: start + size]
    #         worker.indices['beam'] = {'start': start,
    #                                   'size': size,
    #                                   'stride': 1,
    #                                   'total_size': self.n_macroparticles}
    #         self.n_macroparticles = size

    # def split_random(self):
    #     from ..utils.mpi_config import worker
    #     if len(worker.indices) == 0 or worker.isMaster:
    #         # start, size = worker.split(self.n_macroparticles)
    #         start = worker.rank
    #         stride = worker.workers
    #         self.dt = np.ascontiguousarray(self.dt[start:: stride])
    #         self.dE = np.ascontiguousarray(self.dE[start:: stride])
    #         self.id = np.ascontiguousarray(self.id[start:: stride])
    #         size = len(self.dt)
    #         worker.indices['beam'] = {'start': start,
    #                                   'size': size,
    #                                   'stride': stride,
    #                                   'total_size': self.n_macroparticles}
    #         self.n_macroparticles = size

    # def split_random(self):
    #     from ..utils.mpi_config import worker
    #     import random

    #     ids = np.arange(self.n_macroparticles)
    #     random.shuffle(ids)
    #     ids = worker.scatter(ids, self.n_macroparticles)
    #     self.dt = np.ascontiguousarray(
    #         self.dt[ids], dtype=bm.precision.real_t, order='C')
    #     self.dE = np.ascontiguousarray(
    #         self.dE[ids], dtype=bm.precision.real_t, order='C')
    #     self.id = np.ascontiguousarray(
    #         self.id[ids], dtype=bm.precision.real_t, order='C')
    #     size = len(self.dt)
    #     worker.indices['beam'] = {'start': 0,
    #                               'stride': 0,
    #                               'size': size,
    #                               'total_size': self.n_macroparticles}
    #     self.n_macroparticles = size

    def gather(self, all=False):
        '''
        MPI ONLY ROUTINE: Gather the beam coordinates to the master or all workers.
        Parameters
        ----------
        all : boolean
            If true, every worker will get a copy of the whole beam coordinates.
            If false, only the master will gather the coordinates. 
        '''
        if not bm.mpiMode():
            raise RuntimeError(
                'ERROR: Cannot use this routine unless in MPI Mode')
        from ..utils.mpi_config import worker

        if all:
            self.dt = worker.allgather(self.dt)
            self.dE = worker.allgather(self.dE)
            self.id = worker.allgather(self.id)
            self.is_splitted = False
        else:
            self.dt = worker.gather(self.dt)
            self.dE = worker.gather(self.dE)
            self.id = worker.gather(self.id)
            if worker.isMaster:
                self.is_splitted = False

        self.n_macroparticles = len(self.dt)

    # def gather(self):
    #     from ..utils.mpi_config import worker

    #     total_size = worker.indices['beam']['total_size']
    #     self.dt = worker.gather(self.dt, total_size)
    #     self.dE = worker.gather(self.dE, total_size)
    #     self.id = worker.gather(self.id, total_size)
    #     self.n_macroparticles = total_size

    def gather_statistics(self, all=False):
        '''
        MPI ONLY ROUTINE: Gather beam statistics. 
        Parameters
        ----------
        all : boolean
            if true, all workers will gather the beam stats.
            If false, only the master will get the beam stats.
        '''
        if not bm.mpiMode():
            raise RuntimeError(
                'ERROR: Cannot use this routine unless in MPI Mode')

        from ..utils.mpi_config import worker
        if all:
            # temp = worker.allgather(np.array([self.mean_dt]))
            # self.mean_dt = bm.mean(temp)

            self.mean_dt = worker.allreduce(
                np.array([self.mean_dt]), operator='mean')[0]
            
            self.min_dt = worker.allreduce(
                np.array([np.min(self.dt)]), operator='min')[0]

            self.max_dt = worker.allreduce(
                np.array([np.max(self.dt)]), operator='max')[0]

            self.std_dt = worker.allreduce(
                np.array([self.mean_dt, self.sigma_dt, self.n_macroparticles_alive]),
                operator='std')[0]

            self.mean_dE = worker.allreduce(
                np.array([self.mean_dE]), operator='mean')[0]

            self.min_dE = worker.allreduce(
                np.array([np.min(self.dE)]), operator='min')[0]

            self.max_dE = worker.allreduce(
                np.array([np.max(self.dE)]), operator='max')[0]

            self.std_dE = worker.allreduce(
                np.array([self.mean_dE, self.sigma_dE, self.n_macroparticles_alive]),
                operator='std')[0]

            self.n_total_macroparticles_lost = worker.allreduce(
                np.array([self.n_macroparticles_lost]), operator='sum')[0]


        else:
            self.mean_dt = worker.reduce(
                np.array([self.mean_dt]), operator='mean')[0]
            
            self.min_dt = worker.reduce(
                np.array([np.min(self.dt)]), operator='min')[0]

            self.max_dt = worker.reduce(
                np.array([np.max(self.dt)]), operator='max')[0]

            self.std_dt = worker.reduce(
                np.array([self.mean_dt, self.sigma_dt, self.n_macroparticles_alive]),
                operator='std')[0]

            self.mean_dE = worker.reduce(
                np.array([self.mean_dE]), operator='mean')[0]

            self.min_dE = worker.reduce(
                np.array([np.min(self.dE)]), operator='min')[0]

            self.max_dE = worker.reduce(
                np.array([np.max(self.dE)]), operator='max')[0]

            self.std_dE = worker.reduce(
                np.array([self.mean_dE, self.sigma_dE, self.n_macroparticles_alive]),
                operator='std')[0]

            self.n_total_macroparticles_lost = worker.reduce(
                np.array([self.n_macroparticles_lost]), operator='sum')[0]


            # temp = worker.gather(np.array([self.mean_dt]))
            # self.mean_dt = bm.mean(temp)

            # temp = worker.gather(np.array([self.mean_dE]))
            # self.mean_dE = bm.mean(temp)

            # temp = worker.gather(np.array([self.n_macroparticles_lost]))
            # self.n_total_macroparticles_lost = np.sum(temp)

            # temp = worker.gather(np.array([np.min(self.dt)]))
            # self.min_dt = np.min(temp)

            # temp = worker.gather(np.array([np.max(self.dt)]))
            # self.max_dt = np.max(temp)

            # temp = worker.gather(np.array([np.max(self.dt)]))
            # self.max_dt = np.max(temp)

            # temp = worker.gather(np.array([np.max(self.dt)]))
            # self.max_dt = np.max(temp)

        #     temp = worker.gather(np.array([self.mean_dt]))
        #     self.mean_dt = bm.mean(temp)
        #     temp = worker.gather(np.array([self.mean_dE]))
        #     self.mean_dE = bm.mean(temp)
        #     temp = worker.gather(np.array([self.n_macroparticles_lost]))
        #     self.n_total_macroparticles_lost = np.sum(temp)

        #     total_size = worker.workers
        #     mean_dt_arr = worker.gather(np.array([self.mean_dt]), total_size)
        #     mean_dE_arr = worker.gather(np.array([self.mean_dE]), total_size)
        #     losses_arr = worker.gather(np.array([self.n_macroparticles_lost]),
        #                                total_size)

        #     self.mean_dt = np.mean(mean_dt_arr)
        #     self.min_dt = np.min(min_dt_arr)
        #     self.max_dt = np.max(max_dt_arr)

        #     self.mean_dE = np.mean(mean_dE_arr)
        #     self.min_dE = np.min(min_dE_arr)
        #     self.max_dE = np.max(max_dE_arr)

        #     self.losses = np.sum(losses_arr)
        # else:

    # def gather_mean_dE(self):
    #     from ..utils.mpi_config import worker

    #     total_size = worker.workers
    #     self.mean_dE = np.mean(self.dE)
    #     mean_dE_arr = worker.gather(np.array([self.mean_dE]), total_size)
    #     self.mean_dE = np.mean(mean_dE_arr)
    #     return self.mean_dE

    # def gather_mean_dt(self):
    #     from ..utils.mpi_config import worker

    #     total_size = worker.workers
    #     self.mean_dt = np.mean(self.dt)
    #     mean_dt_arr = worker.gather(np.array([self.mean_dt]), total_size)
    #     self.mean_dt = np.mean(mean_dt_arr)
    #     return self.mean_dt

    def gather_losses(self, all=False):
        '''
        MPI ONLY ROUTINE: Gather beam losses. 
        Parameters
        ----------
        all : boolean
            if true, all workers will gather the beam stats.
            If false, only the master will get the beam stats.
        '''
        if not bm.mpiMode():
            raise RuntimeError(
                'ERROR: Cannot use this routine unless in MPI Mode')

        from ..utils.mpi_config import worker

        if all:
            temp = worker.allgather(np.array([self.n_macroparticles_lost]))
            self.n_total_macroparticles_lost = np.sum(temp)
        else:
            temp = worker.gather(np.array([self.n_macroparticles_lost]))
            self.n_total_macroparticles_lost = np.sum(temp)
