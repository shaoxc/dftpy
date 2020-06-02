# Collection of local and semilocal functionals

import numpy as np
from dftpy.field import DirectField, ReciprocalField
from dftpy.functional_output import Functional
from dftpy.math_utils import PowerInt
from dftpy.time_data import TimeData
from dftpy.kedf.tf import TF


def vonWeizsackerPotentialCplx(wav, grid, sigma=0.025):
    """
    The von Weizsacker Potential for complex pseudo-wavefunction
    """
    if not isinstance(sigma, (np.generic, int, float)):
        print("Bad type for sigma")
        return Exception
    wav = DirectField(grid=grid, griddata_3d=wav, cplx=True)
    gg = grid.get_reciprocal().ggF
    potG = wav.fft() * np.exp(-gg * (sigma) ** 2 / 4.0) * gg
    potG = ReciprocalField(grid=grid, griddata_3d=wav, cplx=True)
    a = potG.ifft(force_real=True)
    np.multiply(0.5, a, out=a)
    return DirectField(grid=grid, griddata_3d=a)


def vonWeizsackerPotential(rho, sigma=None, phi = None, lphi = False, **kwargs):
    """
    The von Weizsacker Potential
    """

    gg = rho.grid.get_reciprocal().gg
    if lphi and phi is not None :
        sq_dens = phi
    else :
        sq_dens = np.sqrt(rho)
    if sigma is None :
        n2_sq_dens = sq_dens.fft() * gg
    else :
        n2_sq_dens = sq_dens.fft()*np.exp(-gg*(sigma)**2/4.0)*gg
    a = n2_sq_dens.ifft(force_real=True)
    a *= 0.5
    sq_dens[np.abs(sq_dens) < 1E-300] = 1E-300 # for safe
    return DirectField(grid=rho.grid, griddata_3d=np.divide(a, sq_dens, out=a))


def vonWeizsackerEnergy(rho, potential=None, sigma=None, **kwargs):
    """
    The von Weizsacker Energy Density
    """
    if potential is None :
        edens = vonWeizsackerPotential(rho, sigma = sigma, **kwargs)
    else :
        edens = potential
    ene = np.einsum("ijk, ijk->", rho, edens) * rho.grid.dV
    return ene


def vonWeizsackerStress(rho, y=1.0, energy=None, **kwargs):
    """
    The von Weizsacker Stress
    """
    g = rho.grid.get_reciprocal().g
    rhoG = rho.fft()
    dRho_ij = []
    stress = np.zeros((3, 3))
    for i in range(3):
        dRho_ij.append((1j * g[i] * rhoG).ifft(force_real=True))
    for i in range(3):
        for j in range(i, 3):
            Etmp = -0.25 / rho.grid.volume * rho.grid.dV * np.einsum("ijk -> ", dRho_ij[i] * dRho_ij[j] / rho)
            stress[i, j] = stress[j, i] = Etmp.real * y
    return stress


def vW(rho, y=1.0, sigma=None, calcType=["E","V"], split=False, **kwargs):
    TimeData.Begin("vW")
    pot = vonWeizsackerPotential(rho, sigma, **kwargs)
    if "E" in calcType:
        ene = vonWeizsackerEnergy(rho, pot, **kwargs)
    else:
        ene = 0.0

    OutFunctional = Functional(name="vW")
    OutFunctional.potential = pot * y
    OutFunctional.energy = ene * y
    TimeData.End("vW")
    if split:
        return {"vW": OutFunctional}
    else:
        return OutFunctional


def x_TF_y_vW(rho, x=1.0, y=1.0, sigma=None, calcType=["E","V"], split=False, **kwargs):
    xTF = TF(rho, x=x, calcType=calcType)
    yvW = vW(rho, y=y, sigma=sigma, calcType=calcType, **kwargs)
    pot = xTF.potential + yvW.potential
    ene = xTF.energy + yvW.energy
    OutFunctional = Functional(name=str(x) + "_TF_" + str(y) + "_vW")
    OutFunctional.potential = pot
    OutFunctional.energy = ene
    if split:
        return {"TF": xTF, "vW": yvW}
    else:
        return OutFunctional
