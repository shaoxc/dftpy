import numpy as np
import scipy.special as sp
from scipy.interpolate import interp1d, splrep, splev
from dftpy.mpi import sprint
from dftpy.functional_output import Functional
from dftpy.field import DirectField
from dftpy.kedf.tf import TF
from dftpy.kedf.vw import vW
from dftpy.kedf.wt import WTPotential, WTEnergy
from dftpy.kedf.kernel import MGPKernel, MGPOmegaE, LindhardDerivative
from dftpy.time_data import TimeData

__all__ = ["MGP", "MGPStress", "MGPA", "MGPG"]


def MGPStress(rho, x=1.0, y=1.0, sigma=None, alpha=5.0 / 6.0, beta=5.0 / 6.0, calcType=["E","V"]):
    pass


def MGP(
    rho,
    x=1.0,
    y=1.0,
    sigma=None,
    alpha=5.0 / 6.0,
    beta=5.0 / 6.0,
    lumpfactor=0.2,
    maxpoint=1000,
    symmetrization=None,
    calcType=["E","V"],
    split=False,
    ke_kernel_saved = None,
    **kwargs
):
    TimeData.Begin("MGP")
    q = rho.grid.get_reciprocal().q
    rho0 = rho.amean()
    # sprint('rho0000', rho0)
    if ke_kernel_saved is None :
        KE_kernel_saved = {"Kernel": None, "rho0": 0.0, "shape": None}
    else :
        KE_kernel_saved = ke_kernel_saved
    # if abs(KE_kernel_saved['rho0']-rho0) > 1E-6 or np.shape(rho) != KE_kernel_saved['shape'] :
    if abs(KE_kernel_saved["rho0"] - rho0) > 1e-2 or np.shape(rho) != KE_kernel_saved["shape"]:
        sprint("Re-calculate KE_kernel")
        KE_kernel = MGPKernel(q, rho0, maxpoints=maxpoint, symmetrization=symmetrization)
        if lumpfactor is not None:
            Ne = rho0 * rho.grid.Volume
            KE_kernel += MGPOmegaE(q, Ne, lumpfactor)
        # -----------------------------------------------------------------------
        # rh0 = 0.03;lumpfactor = 0.0;q = np.linspace(1E-3, 8, 10000).reshape((1, 1, 1, -1))
        # mgp = MGPKernel(q,rho0,  maxpoints = maxpoint, symmetrization = None, KernelTable = None)
        # mgpa = MGPKernel(q,rho0, maxpoints = maxpoint, symmetrization = 'Arithmetic', KernelTable = None)
        # mgpg = MGPKernel(q,rho0, maxpoints = maxpoint, symmetrization = 'Geometric', KernelTable = None)
        # np.savetxt('mgp.dat', np.c_[q.ravel()/2.0, mgp.ravel(), mgpa.ravel(), mgpg.ravel()])
        # stop
        # -----------------------------------------------------------------------
        KE_kernel_saved["Kernel"] = KE_kernel
        KE_kernel_saved["rho0"] = rho0
        KE_kernel_saved["shape"] = np.shape(rho)
    else:
        KE_kernel = KE_kernel_saved["Kernel"]

    if "E" in calcType:
        ene = WTEnergy(rho, rho0, KE_kernel, alpha, beta)
    else:
        ene = 0.0
    if "V" in calcType:
        pot = WTPotential(rho, rho0, KE_kernel, alpha, beta)
    else:
        pot = np.empty_like(rho)
    NL = Functional(name="NL", potential=pot, energy=ene)
    return NL


def MGPA(
    rho,
    x=1.0,
    y=1.0,
    sigma=None,
    alpha=5.0 / 6.0,
    beta=5.0 / 6.0,
    lumpfactor=0.2,
    maxpoint=1000,
    symmetrization="Arithmetic",
    calcType=["E","V"],
    split=False,
    **kwargs
):
    return MGP(rho, x, y, sigma, alpha, beta, lumpfactor, maxpoint, "Arithmetic", calcType, split, **kwargs)


def MGPG(
    rho,
    x=1.0,
    y=1.0,
    sigma=None,
    alpha=5.0 / 6.0,
    beta=5.0 / 6.0,
    lumpfactor=0.2,
    maxpoint=1000,
    symmetrization="Geometric",
    calcType=["E","V"],
    split=False,
    **kwargs
):
    return MGP(rho, x, y, sigma, alpha, beta, lumpfactor, maxpoint, "Geometric", calcType, split, **kwargs)
