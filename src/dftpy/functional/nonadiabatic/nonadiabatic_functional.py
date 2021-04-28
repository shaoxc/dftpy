# Nonadaibatic functional

import numpy as np

from dftpy.functional.functional_output import FunctionalOutput
from dftpy.time_data import TimeData


def NonadiabaticFunctional(density, calcType=["V"], **kwargs):
    TimeData.Begin("Nonadiabatic_Func")
    kf = kwargs['kf']
    kf_square = kf * kf
    kf_cube = kf_square * kf
    kf_threehalves = np.sqrt(kf_cube)
    omega = kwargs['omega']
    q = density.grid.get_reciprocal().q
    if density.rank > 1:
        rho = np.sum(density, axis=0)
    else:
        rho = density
    if np.isrealobj(rho):
        force_real = True
    else:
        force_real = False
    rho_over_kf_of_g = (rho / kf).fft()
    rho_over_kf_square_of_g = (rho / kf_square).fft()
    # rho_over_kf_cube_of_g = (rho / kf_cube).fft()
    rho_over_kf_threehalves_of_g = (rho / kf_threehalves).fft()
    q[0, 0, 0] = 1.0
    v_p_kf = 1.0j / 2.0 * np.pi ** 3 * omega / q * rho_over_kf_of_g
    v_p_kf_square = 1.0j / 12.0 * np.pi ** 3 * omega * q * rho_over_kf_square_of_g
    # v_p_kf_cube = -1.0j/3.0 * np.pi * omega ** 2 / q  * rho_over_kf_cube_of_g
    v_p_kf_threehalves = 1.0 / 4.0 * np.pi ** 2 * omega ** 2 * (
                16.0 - np.pi ** 2) / q / q * rho_over_kf_threehalves_of_g
    q[0, 0, 0] = 0.0
    v_p_kf[0, 0, 0] = 0.0
    v_p_kf_square[0, 0, 0] = 0.0
    # v_p_kf_cube[0, 0, 0] = 0.0
    v_p_kf_threehalves[0, 0, 0] = 0.0
    # if 'E' in calcType:
    #    e_h = np.einsum("ijk, ijk->", v_h_of_r, rho) * density.grid.dV / 2.0
    # else:
    #    e_h = 0
    v_p_of_r = v_p_kf.ifft(force_real=force_real) / kf + v_p_kf_square.ifft(
        force_real=force_real) / kf_square + v_p_kf_threehalves.ifft(
        force_real=force_real) / kf_threehalves  # + v_p_kf_cube.ifft(force_real = force_real) / kf_cube)
    v_p_of_r += np.pi * np.pi / 48.0 * (16.0 - 3.0 * np.pi * np.pi) * omega ** 2 / kf_square / kf_cube * rho
    # v_p_of_r *= 0.01
    if density.rank > 1:
        v_p_of_r = np.tile(v_p_of_r, (density.rank, 1, 1, 1))
    TimeData.End("Nonadiabatic_Func")
    return FunctionalOutput(name="Nonadiabiatic", potential=v_p_of_r)
