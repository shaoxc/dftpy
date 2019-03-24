# Hartree functional

import numpy as np
from .functional_output import Functional

def HartreeFunctional(density):
    gg=density.grid.get_reciprocal().gg
    rho_of_g = density.fft()
    v_h = rho_of_g.copy()
    v_h[0,0,0,0] = np.float(0.0)
    mask = gg != 0
    v_h[mask] = rho_of_g[mask]*gg[mask]**(-1)*4*np.pi
    v_h_of_r = v_h.ifft(force_real=True)
    e_h = v_h_of_r*density/2.0
    return Functional(name='Hartree', potential=v_h_of_r, energydensity=e_h)


def HartreePotentialReciprocalSpace(density):
    gg=density.grid.get_reciprocal().gg
    rho_of_g = density.fft()
    v_h = rho_of_g.copy()
    v_h[0,0,0,0] = np.float(0.0)
    mask = gg != 0
    v_h[mask] = rho_of_g[mask]*gg[mask]**(-1)*4*np.pi
    return v_h

