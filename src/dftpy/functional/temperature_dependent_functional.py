import numpy as np

from dftpy.field import DirectField
from dftpy.functional.kedf.tf import ThomasFermiPotential, ThomasFermiEnergy
from dftpy.math_utils import PowerInt


def TemperatureTFPotential(rho: DirectField, temperature: float) -> DirectField:
    '''
    Temperature dependent Thomas-Fermi potential
    Parameters
    ----------
    rho
    temperature: in Hartree

    Returns
    -------

    '''
    v_tf = ThomasFermiPotential(rho)
    theta = temperature / v_tf
    theta_cut = 1.36
    a = [
        0.016,
        -0.957,
        -0.293,
        0.209
    ]
    c = 0.752252778063675
    mask = theta >= theta_cut
    potential = 0.0 * theta
    temp = np.ones_like(theta)
    for i in range(4):
        temp *= theta
        potential += a[i] * temp
    potential *= v_tf

    temp = 1.0 / PowerInt(theta, 3, 2)
    potential[mask] = temperature * (np.log(c * temp[mask]) + np.log(1 + c * temp[mask] / 2.0 ** 1.5))
    potential[mask] -= v_tf[mask]

    return potential


def TemperatureTFEnergy(rho: DirectField, temperature: float) -> float:
    '''
    Temperature dependent Thomas-Fermi energy
    Parameters
    ----------
    rho
    temperature: in Hartree

    Returns
    -------

    '''
    v_tf = ThomasFermiPotential(rho)
    theta = temperature / v_tf
    v_tf_t = TemperatureTFPotential(rho, temperature)
    ene_den = v_tf_t * rho
    ene_den += -2 * np.sqrt(2) / 3 / np.pi ** 2 * temperature ** 2.5 / PowerInt(theta, 5, 2) * np.sqrt(
        0.16 + PowerInt(theta, 2)) / (1.0 - 0.14 * (np.exp(-theta) - np.exp(-3.68 * theta)))
    energy = ene_den.integral()

    energy -= ThomasFermiEnergy(rho)

    return energy
