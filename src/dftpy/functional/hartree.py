# Hartree functional

import numpy as np
from dftpy.functional.functional_output import FunctionalOutput
from dftpy.functional.abstract_functional import AbstractFunctional
from dftpy.time_data import TimeData


class Hartree(AbstractFunctional):

    def __init__(self):
        self.type = 'HARTREE'
        self.name = 'HARTREE'

    def compute(self, density, calcType={"E", "V"}, **kwargs):
        TimeData.Begin("Hartree_Func")
        invgg = density.grid.get_reciprocal().invgg
        if density.rank > 1:
            rho = np.sum(density, axis=0)
        else:
            rho = density
        if np.isrealobj(rho):
            force_real = True
        else:
            force_real = False
        rho_of_g = rho.fft()
        v_h = rho_of_g * invgg * 4 * np.pi
        v_h_of_r = v_h.ifft(force_real=force_real)
        if 'E' in calcType:
            e_h = np.einsum("ijk, ijk->", v_h_of_r, rho) * density.grid.dV / 2.0
        else:
            e_h = 0
        if density.rank > 1:
            v_h_of_r = np.tile(v_h_of_r, (density.rank, 1, 1, 1))
        TimeData.End("Hartree_Func")
        return FunctionalOutput(name="Hartree", potential=v_h_of_r, energy=e_h)


def HartreePotentialReciprocalSpace(density):
    invgg = density.grid.get_reciprocal().invgg
    rho_of_g = density.fft()
    v_h = rho_of_g.copy()
    v_h = rho_of_g * invgg * 4 * np.pi
    return v_h


def HartreeFunctionalStress(density, energy=None):
    TimeData.Begin("Hartree_Stress")
    if energy is None:
        hartree = Hartree()
        energy = hartree(density, calcType={"E"}).energy

    if density.rank > 1:
        rho = np.sum(density, axis=0)
    else:
        rho = density
    g = rho.grid.get_reciprocal().g
    invgg = rho.grid.get_reciprocal().invgg
    mask = rho.grid.get_reciprocal().mask

    rhoG = rho.fft()
    stress = np.zeros((3, 3))
    rhoG2 = rhoG * np.conjugate(rhoG) * invgg * invgg
    for i in range(3):
        for j in range(i, 3):
            den = (g[i][mask] * g[j][mask]) * rhoG2[mask]
            Etmp = np.sum(den)
            stress[i, j] = stress[j, i] = Etmp.real * 8.0 * np.pi / rho.grid.volume ** 2
            if i == j:
                stress[i, j] -= energy / rho.grid.volume
    TimeData.End("Hartree_Stress")
    return stress
