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

def KPointSCFRunner(config: Dict, struct: System, E_v_Evaluator: TotalFunctional):

    beta = 0.1

    rho0 = struct.field
    ions = struct.ions
    nk = [1,1,1]
    tol = 1.0e-12

    #E_v_Evaluator.UpdateFunctional(keysToRemove=['KineticEnergyFunctional'])
    vW = Functional(type='KEDF', name='vW')
    E_v_Evaluator_total = deepcopy(E_v_Evaluator)
    E_v_Evaluator_total.UpdateFunctional(newFuncDict={'vW': vW})


    k_point_list = rho0.grid.get_reciprocal().calc_k_points(nk)
    numk = len(k_point_list)
    print(k_point_list)
    print(numk)
    hamiltonian = Hamiltonian()
    hamiltonian_0 = Hamiltonian()
    hamiltonian_0.v = np.zeros_like(rho0)
    sprint(E_v_Evaluator)

    rho = rho0
    nelec = rho0.integral()
    converged = False
    step = 0
    while not converged:
        hamiltonian.v = E_v_Evaluator(rho, calcType={'V'}).potential - 50 * np.ones_like(rho)
        new_rho = np.zeros_like(rho0)
        ke = 0
        x0 = np.sqrt(rho0)
        for k_point in k_point_list:
            Es, psis = hamiltonian.diagonalize(1, return_eigenvectors=True, k=k_point, scipy=False, x0=x0)
            psi = psis[0]
            new_rho += np.real(psi * np.conj(psi)) * nelec
            ke += np.real(np.conj(psi) * hamiltonian_0(psi)).integral() * nelec

        new_rho /= numk
        ke /= numk
        print(new_rho.integral())
        diff = np.max(np.abs(new_rho-rho))
        if diff < tol:
            converged = True
        step += 1
        sprint(step, diff)
        #sprint(GetEnergyPotential(ions, new_rho, EnergyEvaluator=E_v_Evaluator_total)['TOTAL'].energy*ENERGY_CONV['Hartree']['eV'])
        rho = beta * new_rho + (1.0 - beta) * rho
        PrintEnergy(GetEnergyPotential(ions, rho, EnergyEvaluator=E_v_Evaluator_total), ions.nat, rho.mp)
        print('KE: ', ke*ENERGY_CONV['Hartree']['eV'])


