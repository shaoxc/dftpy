import numpy as np


def DynamicPotential(rho, j, rhotwothirds_cutoff = 1.0e-4):
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
