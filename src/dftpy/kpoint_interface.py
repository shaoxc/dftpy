import numpy as np
from typing import Dict
from dftpy.td.hamiltonian import Hamiltonian
from dftpy.functional.total_functional import TotalFunctional
from dftpy.functional import Functional
from dftpy.field import DirectField
from dftpy.interface import GetEnergyPotential, PrintEnergy
from dftpy.mpi.utils import sprint
from dftpy.system import System
from dftpy.constants import ENERGY_CONV
from copy import deepcopy
from dftpy.optimize import Dynamics


class KPointSCFRunner(Dynamics):

    def __init__(self, system: System, config: Dict, boson_functionals: TotalFunctional):
        Dynamics.__init__(self, system, None)
        self.scipy = True
        self.max_steps = 100
        self.beta = 1
        self.nk = [3,3,3]
        self.tol = 1.0e-12
        self.boson_functionals = boson_functionals
        vW = Functional(type='KEDF', name='vW')
        self.functionals = deepcopy(boson_functionals)
        self.functionals.UpdateFunctional(newFuncDict={'vW': vW})
        self.n_elec = self.system.field.integral()
        self.rho = self.system.field
        self.old_rho = np.zeros_like(self.rho)
        self.new_rho = None
        self.hamiltonian = Hamiltonian()
        self.hamiltonian_0 = Hamiltonian()
        self.hamiltonian_0.potential = np.zeros_like(self.rho)
        self.k_point_list = self.system.field.grid.get_reciprocal().calc_k_points(self.nk)
        self.numk = len(self.k_point_list)
        self.ke = 0
        self.diff = 1e6

        #self.attach(self.compare, before_log=True)
        self.attach(self.mixing, before_log=True)

    def step(self):
        self.old_rho = self.rho
        self.hamiltonian.potential = self.boson_functionals(self.rho, calcType={'V'}).potential
        self.new_rho = np.zeros_like(self.rho)
        self.ke = 0
        x0 = np.sqrt(self.system.field)
        for k_point in self.k_point_list:
            Es, psis = self.hamiltonian.diagonalize(grid=self.rho.grid, numeig=1, return_eigenvectors=True, k_point=k_point, scipy=self.scipy, x0=x0)
            psi = psis[0]
            self.new_rho += np.real(psi * np.conj(psi)) * self.n_elec
            self.ke += np.real(np.conj(psi) * self.hamiltonian_0(psi)).integral() * self.n_elec

        self.new_rho /= self.numk
        self.ke /= self.numk

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


