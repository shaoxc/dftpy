import numpy as np

from dftpy.functional.functional_output import FunctionalOutput
from dftpy.functional.kedf.kernel import SMKernel
from dftpy.mpi import sprint
from dftpy.time_data import timer

"""
E. Smargiassi and P.A. Madden : Orbital-free kinetic-energy functionals for first-principles molecular dynamics.
Phys.Rev.B 49,  5220 (1994).
Tips : In the SM paper, $\Delta\rho = \rho - \rho_{0}$, but $\Delta\rho^{\alpha} = ?$?
       I think it should be $\Delta\rho^{\alpha} = \rho^{\alpha} - \rho_{0}^{\alpha}$.
"""

__all__ = ["SM", "SMStress"]


def SMPotential2(rho, rho0, Kernel, alpha=0.5, beta=0.5):
    # alpha equal beta
    tol = 1e-10
    # rhoD = np.abs(rho - rho0)
    fac = 2.0 * alpha
    if abs(alpha - 0.5) < tol:
        rhoDBeta = np.sqrt(rho)
        rhoDAlpha1 = 1.0 / rhoDBeta
        rhoDBeta -= np.sqrt(rho0)
    elif abs(alpha - 1.0) < tol:
        rhoDBeta = rho - rho0
        rhoDAlpha1 = 1.0
    else:
        rhoDBeta = rho ** beta
        rhoDAlpha1 = rhoDBeta / rho
        rhoDBeta -= rho0 ** beta

    pot = fac * rhoDAlpha1 * (rhoDBeta.fft() * Kernel).ifft(force_real=True)
    return pot


def SMEnergy2(rho, rho0, Kernel, alpha=0.5, beta=0.5):
    tol = 1e-10
    if abs(alpha - 0.5) < tol:
        rhoDAlpha = np.sqrt(rho) - np.sqrt(rho0)
    elif abs(alpha - 1.0) < tol:
        rhoDAlpha = rho - rho0
    else:
        rhoDAlpha = rho ** alpha - rho0 ** alpha

    rhoDBeta = rhoDAlpha

    pot = (rhoDBeta.fft() * Kernel).ifft(force_real=True)
    ene = np.einsum("ijk, ijk->", pot, rhoDAlpha) * rho.grid.dV

    return ene


def SMPotential(rho, rho0, Kernel, alpha=0.5, beta=0.5):
    # alpha equal beta
    fac = 2.0 * alpha
    rhoDBeta = rho ** beta
    rhoDAlpha1 = rhoDBeta / rho
    rhoDBeta -= rho0 ** beta

    pot = fac * rhoDAlpha1 * (rhoDBeta.fft() * Kernel).ifft(force_real=True)
    return pot


def SMEnergyDensity(rho, rho0, Kernel, alpha=0.5, beta=0.5):
    rhoDAlpha = rho ** alpha - rho0 ** alpha
    rhoDBeta = rhoDAlpha

    pot = (rhoDBeta.fft() * Kernel).ifft(force_real=True)
    energydensity = pot * rhoDAlpha

    return energydensity


def SMEnergy(rho, rho0, Kernel, alpha=0.5, beta=0.5):
    rhoDAlpha = rho ** alpha - rho0 ** alpha
    rhoDBeta = rhoDAlpha

    pot = (rhoDBeta.fft() * Kernel).ifft(force_real=True)
    ene = np.einsum("ijk, ijk->", pot, rhoDAlpha) * rho.grid.dV

    return ene


def SMStress(rho, energy=None):
    pass


@timer()
def SM(rho, x=1.0, y=1.0, sigma=None, alpha=0.5, beta=0.5, rho0=None, calcType={"E", "V"}, split=False,
       ke_kernel_saved=None, **kwargs):
    # alpha = beta = 5.0/6.0
    q = rho.grid.get_reciprocal().q
    if rho0 is None:
        rho0 = rho.amean()
    if ke_kernel_saved is None:
        KE_kernel_saved = {"Kernel": None, "rho0": 0.0, "shape": None}
    else:
        KE_kernel_saved = ke_kernel_saved
    if abs(KE_kernel_saved["rho0"] - rho0) > 1e-6 or np.shape(rho) != KE_kernel_saved["shape"]:
        sprint("Re-calculate KE_kernel", comm=rho.mp.comm, level=1)
        KE_kernel = SMKernel(q, rho0, alpha=alpha, beta=beta)
        KE_kernel_saved["Kernel"] = KE_kernel
        KE_kernel_saved["rho0"] = rho0
        KE_kernel_saved["shape"] = np.shape(rho)
    else:
        KE_kernel = KE_kernel_saved["Kernel"]

    NL = FunctionalOutput(name="NL")

    if "E" in calcType or "D" in calcType:
        energydensity = SMEnergyDensity(rho, rho0, KE_kernel, alpha, beta)
        if 'D' in calcType:
            NL.energydensity = energydensity
        NL.energy = energydensity.sum() * rho.grid.dV

    if "V" in calcType:
        pot = SMPotential(rho, rho0, KE_kernel, alpha, beta)
        NL.potential = pot

    return NL
