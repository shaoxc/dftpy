# Collection of local and semilocal functional

import numpy as np

from dftpy.functional.functional_output import FunctionalOutput
from dftpy.math_utils import PowerInt
from dftpy.time_data import TimeData

__all__ = ['TF', 'ThomasFermiStress']


def ThomasFermiPotential(rho):
    """
    The Thomas-Fermi Potential
    """
    factor = (3.0 / 10.0) * (5.0 / 3.0) * (3.0 * np.pi ** 2) ** (2.0 / 3.0)
    pot = factor * np.cbrt(rho * rho)
    # pot = rho * rho
    # pot = np.cbrt(pot, out = pot)
    # pot = np.multiply(factor, pot, out = pot)
    # pot = rho ** (2.0 / 3.0) * factor
    # return (3.0/10.0)*(5.0/3.0)*(3.0*np.pi**2)**(2.0/3.0)*np.abs(rho)**(2.0/3.0)
    return pot


def ThomasFermiEnergyDensity(rho):
    """
    The Thomas-Fermi EnergyDensity
    """
    # edens = (3.0/10.0)*(3.0*np.pi**2)**(2.0/3.0)*np.abs(rho)**(5.0/3.0)
    # edens = np.cbrt(rho * rho * rho * rho * rho)
    edens = PowerInt(rho, 5, 3)
    edens *= (3.0 / 10.0) * (3.0 * np.pi ** 2) ** (2.0 / 3.0)
    return edens


def ThomasFermiEnergy(rho):
    """
    The Thomas-Fermi Energy
    """
    edens = ThomasFermiEnergyDensity(rho)
    ene = edens.sum() * rho.grid.dV
    return ene


def ThomasFermiF(rho):
    ctf = 0.3 * (3.0 * np.pi * np.pi) ** (2.0 / 3.0)
    return 10.0 / 9.0 * ctf / np.cbrt(rho)


def ThomasFermiStress(rho, x=1.0, energy=None, **kwargs):
    """
    The Thomas-Fermi Stress
    """
    if energy is None:
        energy = TF(rho, x=x, calcType={"E"}).energy
    Etmp = -2.0 / 3.0 * energy / rho.grid.volume
    stress = np.zeros((3, 3))
    for i in range(3):
        stress[i, i] = Etmp
    return stress


def TF(rho, x=1.0, calcType={"E", "V"}, split=False, **kwargs):
    TimeData.Begin("TF")
    OutFunctional = FunctionalOutput(name="TF")
    if "E" in calcType or "D" in calcType:
        energydensity = ThomasFermiEnergyDensity(rho)
        ene = energydensity.sum() * rho.grid.dV
        OutFunctional.energy = ene * x
        if 'D' in calcType:
            OutFunctional.energydensity = energydensity
    if "V" in calcType:
        pot = ThomasFermiPotential(rho)
        OutFunctional.potential = pot * x
    if "V2" in calcType:
        v2rho2 = ThomasFermiF(rho)
        OutFunctional.v2rho2 = v2rho2 * x
    TimeData.End("TF")
    return OutFunctional
