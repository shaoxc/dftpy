import numpy as np
from typing import Dict, List, Union

from dftpy.mpi import MPIFile
from dftpy.td.hamiltonian import Hamiltonian
from dftpy.functional.total_functional import TotalFunctional
from dftpy.functional import Functional
from dftpy.field import BaseField
from dftpy.interface import GetEnergyPotential, PrintEnergy
from dftpy.mpi import sprint, mp
from dftpy.system import System
from dftpy.constants import ENERGY_CONV
from copy import deepcopy
from dftpy.optimize import Dynamics
from dftpy.formats import npy
from dftpy.grid import DirectGrid


class KPointSCFRunner(Dynamics):

    psi_list: List[Union[BaseField, None]]

    def __init__(self, system: System, config: Dict, boson_functionals: TotalFunctional):
        Dynamics.__init__(self, system, None)
        self.scipy = True
        self.max_steps = 100
        self.beta = 0.5
        self.nk = [1,1,3]
        self.nnk = self.nk[0] * self.nk[1] * self.nk[2]
        self.outfile = 'test113'
        self.tol = 1.0e-12
        self.boson_functionals = boson_functionals
        vW = Functional(type='KEDF', name='vW')
        self.functionals = deepcopy(boson_functionals)
        self.functionals.UpdateFunctional(newFuncDict={'vW': vW})
        self.n_elec = self.system.field.integral()
        self.rho = self.system.field
        self.old_rho = np.zeros_like(self.rho)
        self.new_rho = None
        self.psi_list = [None] * self.nnk
        self.hamiltonian = Hamiltonian()
        self.hamiltonian_0 = Hamiltonian()
        self.hamiltonian_0.potential = np.zeros_like(self.rho)
        self.k_point_list = self.system.field.grid.get_reciprocal().calc_k_points(self.nk)
        self.numk = len(self.k_point_list)
        self.ke = 0
        self.diff = 1e6
        self.grid = DirectGrid(lattice=self.rho.grid.lattice, nr=self.rho.grid.nr, full=True)

        #self.attach(self.compare, before_log=True)
        self.attach(self.mixing, before_log=True)
        self.attach(self.debug)

    def step(self):
        self.old_rho = self.rho
        self.hamiltonian.potential = self.boson_functionals(self.rho, calcType={'V'}).potential
        self.new_rho = np.zeros_like(self.rho)
        self.ke = 0
        x0 = np.sqrt(self.rho)
        for i_k, k_point in enumerate(self.k_point_list):
            Es, psis = self.hamiltonian.diagonalize(grid=self.grid, numeig=1, return_eigenvectors=True, k_point=k_point, scipy=self.scipy, x0=x0, sigma=0.025)
            self.psi_list[i_k] = psis[0]
            self.new_rho += np.real(self.psi_list[i_k] * np.conj(self.psi_list[i_k])) * self.n_elec
            self.ke += np.real(np.conj(self.psi_list[i_k]) * self.hamiltonian_0(self.psi_list[i_k], sigma=0.025)).integral() * self.n_elec

        self.new_rho /= self.numk
        self.ke /= self.numk

    def debug(self):
        if self.psi_list[0] is not None:
            vW = Functional(type='KEDF', name='vW')
            print((self.psi_list[0]*self.hamiltonian_0(self.psi_list[0])).integral() * self.n_elec - vW(rho=self.psi_list[0] * self.psi_list[0] * self.n_elec).energy)

    def converged(self):
        self.diff = np.max(np.abs(self.old_rho-self.rho))
        return self.diff < self.tol

    def log(self):
        sprint(self.nsteps, self.diff)
        PrintEnergy(GetEnergyPotential(self.system.ions, self.rho, EnergyEvaluator=self.functionals), self.system.ions.nat, self.rho.mp)
        sprint('KE: ', self.ke * ENERGY_CONV['Hartree']['eV'])

    def mixing(self):
        if self.nsteps > 0:
            self.rho = self.beta * self.new_rho + (1.0 - self.beta) * self.rho

    def compare(self):
        if self.nsteps > 0:
            self.alt_rho = np.zeros_like(self.rho)
            self.alt_ke = 0
            for k_point in self.k_point_list:
                Es, psis = self.hamiltonian.diagonalize(1, return_eigenvectors=True, k=k_point)
                psi = psis[0]
                self.alt_rho += np.real(psi * np.conj(psi)) * self.n_elec
                self.alt_ke += np.real(np.conj(psi) * self.hamiltonian_0(psi)).integral() * self.n_elec

            self.alt_rho /= self.numk
            self.alt_ke /= self.numk
            PrintEnergy(GetEnergyPotential(self.system.ions, self.alt_rho, EnergyEvaluator=self.functionals),
                        self.system.ions.nat, self.rho.mp)
            sprint('Alt KE: ', self.alt_ke * ENERGY_CONV['Hartree']['eV'])
            sprint('diff:', np.abs(self.alt_rho-self.new_rho).integral())
            sprint('Alt rho: ', self.alt_rho.integral())
            sprint('rho: ', self.new_rho.integral())

    def save(self):
        sprint('Save wavefunction data.')
        fname = './{0:s}.npy'.format(self.outfile, self.nsteps)
        if mp.size > 1:
            f = MPIFile(fname, mp, amode=mp.MPI.MODE_CREATE | mp.MPI.MODE_WRONLY)
        else:
            f = open(fname, "wb")
        if mp.is_root:
            npy.write(f, self.nk, single=True)
        for psi in self.psi_list:
            npy.write(f, psi, grid=psi.grid)

        f.close()

    def run(self):

        converged = super(KPointSCFRunner, self).run()
        self.save()
        return converged









    # rho = rho0
    # nelec = rho0.integral()
    # converged = False
    # step = 0
    # while not converged:
    #     hamiltonian.v = E_v_Evaluator(rho, calcType={'V'}).potential - 50 * np.ones_like(rho)
    #     new_rho = np.zeros_like(rho0)
    #     ke = 0
    #     x0 = np.sqrt(rho0)
    #     for k_point in k_point_list:
    #         Es, psis = hamiltonian.diagonalize(1, return_eigenvectors=True, k=k_point, scipy=False, x0=x0)
    #         psi = psis[0]
    #         new_rho += np.real(psi * np.conj(psi)) * nelec
    #         ke += np.real(np.conj(psi) * hamiltonian_0(psi)).integral() * nelec
    #
    #     new_rho /= numk
    #     ke /= numk
    #     print(new_rho.integral())
    #     diff = np.max(np.abs(new_rho-rho))
    #     if diff < tol:
    #         converged = True
    #     step += 1
    #     sprint(step, diff)
    #     #sprint(GetEnergyPotential(ions, new_rho, EnergyEvaluator=E_v_Evaluator_total)['TOTAL'].energy*ENERGY_CONV['Hartree']['eV'])
    #     rho = beta * new_rho + (1.0 - beta) * rho
    #     PrintEnergy(GetEnergyPotential(ions, rho, EnergyEvaluator=E_v_Evaluator_total), ions.nat, rho.mp)
    #     print('KE: ', ke*ENERGY_CONV['Hartree']['eV'])


