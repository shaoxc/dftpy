import numpy as np

from dftpy.functional.functional_output import FunctionalOutput
from dftpy.functional.kedf.kernel import SMKernel, WTKernel
from dftpy.math_utils import PowerInt
from dftpy.mpi import sprint
from dftpy.time_data import timer

"""
F. Perrot : Hydrogen-hydrogen interaction in an electron gas.
J. Phys. : Condens. Matter 6, 431 (1994).
"""

__all__ = ["FP", "FPStress"]


def FPPotentialEnergy(rho, rho0, Kernel, alpha=1.0, beta=1.0):
    nu = 5.0 / np.sqrt(32.0)
    rhoFiveSixth = PowerInt(rho, 5, 6)
    nuMinus1 = nu - 1.0
    Prho = (nu - (nuMinus1 * rho0) / rho) * (rhoFiveSixth - rho0 ** (5.0 / 6.0))
    dPrho = (
            rho0 ** (11.0 / 6.0) * (1 - nu)
            + (1.0 / 6.0 * rho0 * nuMinus1 * rhoFiveSixth)
            + (5.0 / 6.0 * nu * rho * rhoFiveSixth)
    )
    dPrho /= rho * rho
    pot = (Prho.fft() * Kernel).ifft(force_real=True)
    ene = np.einsum("ijk, ijk->", pot, Prho) * rho.grid.dV
    pot *= 2.0 * dPrho
    return pot, ene


def FPPotential(rho, rho0, Kernel, alpha=1.0, beta=1.0):
    nu = 5.0 / np.sqrt(32.0)
    rhoFiveSixth = PowerInt(rho, 5, 6)
    nuMinus1 = nu - 1.0
    Prho = (nu - (nuMinus1 * rho0) / rho) * (rhoFiveSixth - rho0 ** (5.0 / 6.0))
    dPrho = (
            rho0 ** (11.0 / 6.0) * (1 - nu)
            + (1.0 / 6.0 * rho0 * nuMinus1 * rhoFiveSixth)
            + (5.0 / 6.0 * nu * rho * rhoFiveSixth)
    )
    dPrho /= rho * rho
    pot = 2.0 * dPrho * (Prho.fft() * Kernel).ifft(force_real=True)
    return pot


def FPEnergy(rho, rho0, Kernel, alpha=1.0, beta=1.0):
    # rhoD = rho - rho0
    nu = 5.0 / np.sqrt(32.0)
    rhoFiveSixth = PowerInt(rho, 5, 6)
    nuMinus1 = nu - 1.0
    Prho = (nu - (nuMinus1 * rho0) / rho) * (rhoFiveSixth - rho0 ** (5.0 / 6.0))
    pot = (Prho.fft() * Kernel).ifft(force_real=True)
    ene = np.einsum("ijk, ijk->", pot, Prho) * rho.grid.dV

    return ene


def FPStress(rho, energy=None):
    pass


@timer()
def FP_origin(rho, x=1.0, y=1.0, sigma=None, alpha=1.0, beta=1.0, calcType={"E", "V"},
              ke_kernel_saved=None, **kwargs):
    q = rho.grid.get_reciprocal().q
    rho0 = rho.amean()
    if ke_kernel_saved is None:
        KE_kernel_saved = {"Kernel": None, "rho0": 0.0, "shape": None}
    else:
        KE_kernel_saved = ke_kernel_saved
    if abs(KE_kernel_saved["rho0"] - rho0) > 1e-10 or np.shape(rho) != KE_kernel_saved["shape"]:
        sprint("Re-calculate KE_kernel", comm=rho.mp.comm, level=1)
        # KE_kernel = SMKernel(q,rho0, alpha = alpha, beta = beta)
        KE_kernel = SMKernel(q, rho0, alpha=1.0, beta=1.0) * 1.6
        KE_kernel_saved["Kernel"] = KE_kernel
        KE_kernel_saved["rho0"] = rho0
        KE_kernel_saved["shape"] = np.shape(rho)
    else:
        KE_kernel = KE_kernel_saved["Kernel"]

    ene = pot = 0
    # -----------------------------------------------------------------------
    nu = 5.0 / np.sqrt(32.0)
    rhoFiveSixth = PowerInt(rho, 5, 6)
    nuMinus1 = nu - 1.0
    Prho = (nu - (nuMinus1 * rho0) / rho) * (rhoFiveSixth - rho0 ** (5.0 / 6.0))
    pot = (Prho.fft() * KE_kernel).ifft(force_real=True)
    # -----------------------------------------------------------------------
    if "E" in calcType:
        ene = np.einsum("ijk, ijk->", pot, Prho) * rho.grid.dV
    if "V" in calcType:
        # dPrho = rho0 **(11.0/6.0) * (1 - nu)  + (1.0/6.0 * rho0 * nuMinus1 * rhoFiveSixth) + \
        # (5.0/6.0 * nu * rho * rhoFiveSixth)
        # dPrho /= (rho * rho)
        dPrho = 5.0 / 6.0 * (nu - (nuMinus1 * rho0) / rho) * rhoFiveSixth / rho
        pot *= 2.0 * dPrho

    OutFunctional = FunctionalOutput(name="FP")
    OutFunctional.potential = pot
    OutFunctional.energy = ene
    return OutFunctional


@timer()
def FP0(rho, x=1.0, y=1.0, sigma=None, alpha=1.0, beta=1.0, calcType={"E", "V"}, split=False,
        ke_kernel_saved=None, **kwargs):
    # Only performed once for each grid
    q = rho.grid.get_reciprocal().q
    rho0 = rho.amean()
    if ke_kernel_saved is None:
        KE_kernel_saved = {"Kernel": None, "rho0": 0.0, "shape": None}
    else:
        KE_kernel_saved = ke_kernel_saved
    if abs(KE_kernel_saved["rho0"] - rho0) > 1e-6 or np.shape(rho) != KE_kernel_saved["shape"]:
        sprint("Re-calculate KE_kernel", comm=rho.mp.comm, level=1)
        # KE_kernel = SMKernel(q,rho0, alpha = 1.0, beta = 1.0) * 1.6
        KE_kernel = WTKernel(q, rho0, alpha=alpha, beta=beta)
        KE_kernel_saved["Kernel"] = KE_kernel
        KE_kernel_saved["rho0"] = rho0
        KE_kernel_saved["shape"] = np.shape(rho)
    else:
        KE_kernel = KE_kernel_saved["Kernel"]

    # -----------------------------------------------------------------------
    nu = 5.0 / np.sqrt(32.0)
    rhoFiveSixth = PowerInt(rho, 5, 6)
    nuMinus1 = nu - 1.0
    drho = np.abs(rho - rho0)
    Mr = 1 + (nuMinus1 * drho) / (rho0 + drho)
    # Mr = (1 + (nuMinus1 * drho)/(rho))
    Prho = Mr * (rhoFiveSixth - rho0 ** (5.0 / 6.0))
    pot = (Prho.fft() * KE_kernel).ifft(force_real=True)
    # -----------------------------------------------------------------------
    ene = 0
    if "E" in calcType:
        ene = np.einsum("ijk, ijk->", pot, Prho) * rho.grid.dV
    if "V" in calcType:
        # dPrho = 5.0/6.0 * Mr * rhoFiveSixth/rho;# pot *= 2.0 * dPrho
        pot *= (5.0 / 3.0) * Mr * rhoFiveSixth / rho

    NL = FunctionalOutput(name="NL", potential=pot, energy=ene)
    return NL


@timer()
def FP(rho, x=1.0, y=1.0, sigma=None, alpha=1.0, beta=1.0, rho0=None, calcType={"E", "V"}, split=False,
       ke_kernel_saved=None, **kwargs):
    q = rho.grid.get_reciprocal().q
    if rho0 is None:
        rho0 = rho.amean()
    if ke_kernel_saved is None:
        KE_kernel_saved = {"Kernel": None, "rho0": 0.0, "shape": None}
    else:
        KE_kernel_saved = ke_kernel_saved
    if abs(KE_kernel_saved["rho0"] - rho0) > 1e-6 or np.shape(rho) != KE_kernel_saved["shape"]:
        sprint("Re-calculate KE_kernel", comm=rho.mp.comm, level=1)
        # KE_kernel = SMKernel(q,rho0, alpha = 1.0, beta = 1.0) * 1.6
        KE_kernel = WTKernel(q, rho0, alpha=alpha, beta=beta)
        KE_kernel_saved["Kernel"] = KE_kernel
        KE_kernel_saved["rho0"] = rho0
        KE_kernel_saved["shape"] = np.shape(rho)
    else:
        KE_kernel = KE_kernel_saved["Kernel"]

    # -----------------------------------------------------------------------
    nu = 5.0 / np.sqrt(32.0)
    rhoFiveSixth = PowerInt(rho, 5, 6)
    # nuMinus1 = nu - 1.0
    drho = np.abs(rho - rho0)
    # drho = rho - rho0
    Mr = (rho0 + nu * drho) / (rho0 + drho)
    # Prho = Mr * (rhoFiveSixth - rho0 ** (5.0/6.0))
    Prho = Mr * rhoFiveSixth
    pot = (Prho.fft() * KE_kernel).ifft(force_real=True)
    # -----------------------------------------------------------------------
    NL = FunctionalOutput(name="NL")

    if "E" in calcType or "D" in calcType:
        energydensity = pot * Prho
        if 'D' in calcType:
            NL.energydensity = energydensity
        NL.energy = energydensity.sum() * rho.grid.dV

    if "V" in calcType:
        # dPrho = 5.0/6.0 * Mr * rhoFiveSixth/rho;# pot *= 2.0 * dPrho
        pot *= (5.0 / 3.0) * Mr * rhoFiveSixth / rho
        NL.potential = pot

    return NL
