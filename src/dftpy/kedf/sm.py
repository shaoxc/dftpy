import numpy as np
import scipy.special as sp
from scipy.interpolate import interp1d, splrep, splev
from dftpy.functional_output import Functional
from dftpy.field import DirectField
from dftpy.kedf.tf import TF
from dftpy.kedf.vw import vW
from dftpy.kedf.kernel import SMKernel, LindhardDerivative, WTKernel
from dftpy.math_utils import TimeData
'''
E. Smargiassi and P.A. Madden : Orbital-free kinetic-energy functionals for first-principles molecular dynamics. 
Phys.Rev.B 49,  5220 (1994).
Tips : In the SM paper, $\Delta\rho = \rho - \rho_{0}$, but $\Delta\rho^{\alpha} = ?$?
       I think it should be $\Delta\rho^{\alpha} = \rho^{\alpha} - \rho_{0}^{\alpha}$.
'''

__all__  =  ['SM', 'SMStress']

KE_kernel_saved ={'Kernel':None, 'rho0':0.0, 'shape':None, \
        'KernelTable':None, 'etaMax':None, 'KernelDeriv':None}

def SMPotential2(rho, rho0, Kernel, alpha = 0.5, beta = 0.5):
    # alpha equal beta
    tol = 1E-10
    alphaMinus1 = alpha - 1.0
    # rhoD = np.abs(rho - rho0)
    fac = 2.0 * alpha
    if abs(alpha - 0.5) < tol :
        rhoDBeta = np.sqrt(rho)
        rhoDAlpha1 = 1.0 / rhoDBeta
        rhoDBeta -= np.sqrt(rho0)
    elif abs(alpha - 1.0) < tol :
        rhoDBeta = rho - rho0
        rhoDAlpha1 = 1.0 
    else :
        rhoDBeta = rho ** beta
        rhoDAlpha1 = rhoDBeta / rho
        rhoDBeta -= rho0 ** beta

    pot = fac * rhoDAlpha1 * (rhoDBeta.fft() * Kernel).ifft(force_real = True)
    return pot

def SMEnergy2(rho, rho0, Kernel, alpha = 0.5, beta = 0.5):
    tol = 1E-10
    if abs(alpha - 0.5) < tol :
        rhoDAlpha = np.sqrt(rho) - np.sqrt(rho0)
    elif abs(alpha - 1.0) < tol :
        rhoDAlpha = rho - rho0
    else :
        rhoDAlpha = rho ** alpha - rho0 ** alpha

    rhoDBeta = rhoDAlpha

    pot = (rhoDBeta.fft() * Kernel).ifft(force_real = True)
    ene = np.einsum('ijkl, ijkl->', pot, rhoDAlpha) * rho.grid.dV

    return ene

def SMPotential(rho, rho0, Kernel, alpha = 0.5, beta = 0.5):
    # alpha equal beta
    tol = 1E-10
    alphaMinus1 = alpha - 1.0
    fac = 2.0 * alpha
    rhoDBeta = rho ** beta
    rhoDAlpha1 = rhoDBeta / rho
    rhoDBeta -= rho0 ** beta

    pot = fac * rhoDAlpha1 * (rhoDBeta.fft() * Kernel).ifft(force_real = True)
    return pot

def SMEnergy(rho, rho0, Kernel, alpha = 0.5, beta = 0.5):
    tol = 1E-10
    rhoDAlpha = rho ** alpha - rho0 ** alpha
    rhoDBeta = rhoDAlpha

    pot = (rhoDBeta.fft() * Kernel).ifft(force_real = True)
    ene = np.einsum('ijkl, ijkl->', pot, rhoDAlpha) * rho.grid.dV

    return ene

def SMStress(rho, energy=None):
    pass

def SM(rho,x=1.0,y=1.0,Sigma=0.025, alpha = 0.5, beta = 0.5, calcType='Both', split = False, **kwargs):
    TimeData.Begin('SM')
    # alpha = beta = 5.0/6.0
    global KE_kernel_saved
    #Only performed once for each grid
    q = rho.grid.get_reciprocal().q
    rho0 = np.einsum('ijkl -> ', rho) / np.size(rho)
    rho1 = 0.5 * (np.max(rho) + np.min(rho))
    # print(rho0, rho1, rho1 / rho0)
    # rho0 *= 1.3
    # rho0 *= 1.27
    # rho0 *= 0.62
    # rho0 = rho1
    if abs(KE_kernel_saved['rho0']-rho0) > 1E-6 or np.shape(rho) != KE_kernel_saved['shape'] :
        print('Re-calculate KE_kernel')
        KE_kernel = SMKernel(q,rho0, alpha = alpha, beta = beta)
        KE_kernel_saved['Kernel'] = KE_kernel
        KE_kernel_saved['rho0'] = rho0
        KE_kernel_saved['shape'] = np.shape(rho)
    else :
        KE_kernel = KE_kernel_saved['Kernel']

    if calcType == 'Energy' :
        ene = SMEnergy(rho, rho0, KE_kernel, alpha, beta)
        pot = np.empty_like(rho)
    elif calcType == 'Potential' :
        pot = SMPotential(rho, rho0, KE_kernel, alpha, beta)
        ene = 0
    else :
        pot = SMPotential(rho, rho0, KE_kernel, alpha, beta)
        ene = SMEnergy(rho, rho0, KE_kernel, alpha, beta)
        # ene = np.einsum('ijkl, ijkl->', pot, rho) * rho.grid.dV / (2 * alpha)

    NL = Functional(name='NL', potential = pot, energy= ene)
    xTF = TF(rho, x = x,  calcType = calcType)
    yvW = vW(rho, y = y, Sigma = Sigma, calcType = calcType)
    OutFunctional = NL + xTF + yvW
    OutFunctional.name = 'SM'
    TimeData.End('SM')
    if split :
        return {'TF': xTF, 'vW': yvW, 'NL' : NL}
    else :
        return OutFunctional