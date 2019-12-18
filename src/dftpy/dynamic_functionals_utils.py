
import numpy as np
from .field import DirectField,ReciprocalField
from .grid import DirectGrid, ReciprocalGrid

def DynamicPotential(rho, j):
    '''
    current-dependent dynamic kinetic energy potential
    Eq. 3 of PRL 121, 145001 (2018)
    '''

    reciprocal_grid = j.grid.get_reciprocal()
    g = reciprocal_grid.g
    gg = reciprocal_grid.gg
    gg[gg == 0] = 1e-10
    rhotwothirds = np.cbrt(rho*rho)
    #rhotwothirds[rhotwothirds<1e-10] = 1e-10
    #rho[rho < 1e-10] = 1e-10
    #temp = np.einsum('ijkl,ijkl->ijk',g,j.fft())
    #temp = np.expand_dims(temp, axis=3)
    #temp = j.fft().dot(g)
    temp = 1j * j.fft().dot(g) / np.sqrt(gg) * np.exp(-0.5*gg)


    return np.pi**(5.0/3.0)/(2.0*3.0**(2.0/3.0)*rhotwothirds) * np.real(temp.ifft())
