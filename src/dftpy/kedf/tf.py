# Collection of local and semilocal functionals

import numpy as np
from dftpy.field import DirectField, ReciprocalField
from dftpy.functional_output import Functional
from dftpy.math_utils import TimeData, PowerInt


def ThomasFermiPotential(rho):
    '''
    The Thomas-Fermi Potential
    '''
    factor = (3.0 / 10.0) * (5.0 / 3.0) * (3.0 * np.pi**2)**(2.0 / 3.0)
    # pot = np.cbrt(rho*rho)
    # pot = factor * np.cbrt(rho * rho)
    # pot = rho * rho
    # pot = np.cbrt(pot, out = pot)
    # pot = np.multiply(factor, pot, out = pot)
    pot = rho**(2.0 / 3.0) * factor
    # return (3.0/10.0)*(5.0/3.0)*(3.0*np.pi**2)**(2.0/3.0)*np.abs(rho)**(2.0/3.0)
    return pot


def ThomasFermiEnergy(rho):
    '''
    The Thomas-Fermi Energy
    '''
    # edens = (3.0/10.0)*(3.0*np.pi**2)**(2.0/3.0)*np.abs(rho)**(5.0/3.0)
    # edens = np.cbrt(rho * rho * rho * rho * rho)
    edens = PowerInt(rho, 5, 3)
    ene = np.einsum('ijkl->', edens)
    ene *= (3.0 / 10.0) * (3.0 * np.pi**2)**(2.0 / 3.0) * rho.grid.dV
    return ene


def ThomasFermiStress(rho, x=1.0, energy=None):
    '''
    The Thomas-Fermi Stress
    '''
    if energy is None:
        energy = TF(rho, x=x, calcType='Energy').energy
    Etmp = -2.0 / 3.0 * energy / rho.grid.volume
    stress = np.zeros((3, 3))
    for i in range(3):
        stress[i, i] = Etmp
    return stress


def TF(rho, x=1.0, calcType='Both', split=False, **kwargs):
    TimeData.Begin('TF')
    if calcType == 'Energy':
        ene = ThomasFermiEnergy(rho)
        pot = np.empty_like(rho)
    elif calcType == 'Potential':
        pot = ThomasFermiPotential(rho)
        ene = 0
    else:
        pot = ThomasFermiPotential(rho)
        ene = ThomasFermiEnergy(rho)
    OutFunctional = Functional(name='TF')
    OutFunctional.potential = pot * x
    OutFunctional.energy = ene * x
    TimeData.End('TF')
    return OutFunctional
