# Nonadaibatic functional

import numpy as np
from dftpy.functional_output import Functional
from dftpy.math_utils import PowerInt
from dftpy.time_data import TimeData


def NonadiabaticFunctional(density, calcType=["V"], **kwargs):
    TimeData.Begin("Nonadiabatic_Func")
    kf = kwargs['kf']
    omega = kwargs['omega']
    #q = density.grid.get_reciprocal().q
    if density.rank > 1 :
        rho = np.sum(density, axis = 0)
    else :
        rho = density
    #if np.isrealobj(rho):
    #    force_real = True
    #else:
    #    force_real = False
    #rho_of_g = rho.fft()
    # v_h = rho_of_g.copy()
    # mask = gg != 0
    # v_h[mask] = rho_of_g[mask]*gg[mask]**(-1)*4*np.pi
    #q[0, 0, 0] = 1.0
    #v_p = np.pi * np.pi / kf * (1 + 25.0/24.0 * omega ** 2 / kf ** 4) * rho_of_g
    v_p_of_r = np.pi * np.pi / kf * (1 + 25.0/24.0 * omega ** 2 / PowerInt(kf, 4)) * rho
    #q[0, 0, 0] = 0.0
    #v_q[0, 0, 0] = 0.0
    #v_p_of_r = v_p.ifft(force_real=force_real)
    #if 'E' in calcType:
    #    e_h = np.einsum("ijk, ijk->", v_h_of_r, rho) * density.grid.dV / 2.0
    #else:
    #    e_h = 0
    if density.rank > 1 :
        v_p_of_r = np.tile(v_p_of_r, (density.rank, 1, 1, 1))
    TimeData.End("Nonadiabatic_Func")
    return Functional(name="Nonadiabiatic", potential=v_p_of_r)


