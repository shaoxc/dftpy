import numpy as np
from dftpy.field import DirectField
from dftpy.optimization import Optimization
from dftpy.functional import TotalFunctional


def saw(emaxpos, eopreg, x):
    sawout = np.zeros_like(x)
    z = x - emaxpos
    y = z - np.floor(z)
    mask = y <= eopreg
    mask2 = np.invert(mask)
    sawout[mask] = (0.5-y[mask]/eopreg)*(1.0-eopreg)
    sawout[mask2] = (-0.5+(y[mask2]-eopreg)/(1.0-eopreg))*(1.0-eopreg)
    return sawout


def dipole(drho: DirectField, direction: int) -> float:
    grid = drho.grid
    dip = (drho * grid.r[direction]).integral()
    return dip


def quadrupole(drho: DirectField, direction1: int, direction2: int) -> float:
    grid = drho.grid
    tmp = 3 * grid.r[direction1] * grid.r[direction2]
    if direction1 == direction2:
        tmp -= grid.rr
    quadp = (drho * tmp).integral()
    return quadp


def main(rho_ini: DirectField, functionals: TotalFunctional):
