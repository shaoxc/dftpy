import numpy as np
from dftpy.functional_output import Functional
from dftpy.field import DirectField
from typing import List


def WCDHCPotential(rho: DirectField, j: DirectField, rhotwothirds_cutoff: float = 0, **kwargs) -> DirectField:
    """
    current-dependent dynamic kinetic energy potential
    Eq. 3 of PRL 121, 145001 (2018)
    """
    rhotwothirds = np.cbrt(rho * rho)
    rhotwothirds[rhotwothirds<rhotwothirds_cutoff] = rhotwothirds_cutoff
    reciprocal_grid = j.grid.get_reciprocal()
    g = reciprocal_grid.g
    sqrt_gg = np.sqrt(reciprocal_grid.gg)
    sqrt_gg[0, 0, 0] = 1.0
    #temp = 1j * j.fft().dot(g) / np.sqrt(gg) * np.exp(-0.5 * gg)
    temp = 1j * j.fft().dot(g) / sqrt_gg

    return np.pi ** (5.0 / 3.0) / (2.0 * 3.0 ** (2.0 / 3.0) * rhotwothirds) * temp.ifft(force_real=True)


def WCDHC(rho: DirectField, j: DirectField, calcType: List[str], **kwargs) -> Functional:
    functional = Functional(name = "WCDHC")
    if "V" in calcType:
        functional.potential = WCDHCPotential(rho, j, **kwargs)
    return functional

