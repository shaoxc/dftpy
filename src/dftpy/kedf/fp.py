import numpy as np
import scipy.special as sp
from scipy.interpolate import interp1d, splrep, splev
from dftpy.math_utils import PowerInt
from dftpy.functional_output import Functional
from dftpy.field import DirectField
from dftpy.kedf.tf import TF
from dftpy.kedf.vw import vW
from dftpy.kedf.kernel import SMKernel, LindhardDerivative, WTKernel
from dftpy.math_utils import TimeData
'''
F. Perrot : Hydrogen-hydrogen interaction in an electron gas.
J. Phys. : Condens. Matter 6, 431 (1994).
'''

__all__  =  ['FP', 'FPStress']

KE_kernel_saved ={'Kernel':None, 'rho0':0.0, 'shape':None, \
        'KernelTable':None, 'etaMax':None, 'KernelDeriv':None}

def FPPotentialEnergy(rho, rho0, Kernel, alpha = 1.0, beta = 1.0):
    nu = 5.0/np.sqrt(32.0)
    rhoFiveSixth = PowerInt(rho, 5, 6)
    nuMinus1 = nu - 1.0
    Prho = (nu - (nuMinus1 * rho0)/rho) * (rhoFiveSixth - rho0 ** (5.0/6.0))
    dPrho = rho0 **(11.0/6.0) * (1 - nu)  + (1.0/6.0 * rho0 * nuMinus1 * rhoFiveSixth) + \
            (5.0/6.0 * nu * rho * rhoFiveSixth)
    dPrho /= (rho * rho)
    pot = (Prho.fft() * Kernel).ifft(force_real = True)
    ene = np.einsum('ijkl, ijkl->', pot, Prho) * rho.grid.dV
    pot *= 2.0 * dPrho
    return pot, ene

def FPPotential(rho, rho0, Kernel, alpha = 1.0, beta = 1.0):
    nu = 5.0/np.sqrt(32.0)
    rhoFiveSixth = PowerInt(rho, 5, 6)
    nuMinus1 = nu - 1.0
    Prho = (nu - (nuMinus1 * rho0)/rho) * (rhoFiveSixth - rho0 ** (5.0/6.0))
    dPrho = rho0 **(11.0/6.0) * (1 - nu)  + (1.0/6.0 * rho0 * nuMinus1 * rhoFiveSixth) + \
            (5.0/6.0 * nu * rho * rhoFiveSixth)
    dPrho /= (rho * rho)
    pot = 2.0 * dPrho * (Prho.fft() * Kernel).ifft(force_real = True)
    return pot

def FPEnergy(rho, rho0, Kernel, alpha = 1.0, beta = 1.0):
    rhoD = rho - rho0
    nu = 5.0/np.sqrt(32.0)
    rhoFiveSixth = PowerInt(rho, 5, 6)
    nuMinus1 = nu - 1.0
    Prho = (nu - (nuMinus1 * rho0)/rho) * (rhoFiveSixth - rho0 ** (5.0/6.0))
    pot = (Prho.fft() * Kernel).ifft(force_real = True)
    ene = np.einsum('ijkl, ijkl->', pot, Prho) * rho.grid.dV

    return ene

def FPStress(rho, energy=None):
    pass

def FP_origin(rho,x=1.0,y=1.0,Sigma=0.025, alpha = 1.0, beta = 1.0, calcType='Both'):
    TimeData.Begin('FP')
    global KE_kernel_saved
    #Only performed once for each grid
    q = rho.grid.get_reciprocal().q
    rho0 = np.einsum('ijkl -> ', rho) / np.size(rho)
    if abs(KE_kernel_saved['rho0']-rho0) > 1E-10 or np.shape(rho) != KE_kernel_saved['shape'] :
        print('Re-calculate KE_kernel')
        # KE_kernel = SMKernel(q,rho0, alpha = alpha, beta = beta)
        KE_kernel = SMKernel(q,rho0, alpha = 1.0, beta = 1.0) * 1.6
        KE_kernel_saved['Kernel'] = KE_kernel
        KE_kernel_saved['rho0'] = rho0
        KE_kernel_saved['shape'] = np.shape(rho)
    else :
        KE_kernel = KE_kernel_saved['Kernel']

    ene = pot = 0
    #-----------------------------------------------------------------------
    nu = 5.0/np.sqrt(32.0)
    rhoFiveSixth = PowerInt(rho, 5, 6)
    nuMinus1 = nu - 1.0
    Prho = (nu - (nuMinus1 * rho0)/rho) * (rhoFiveSixth - rho0 ** (5.0/6.0))
    pot = (Prho.fft() * KE_kernel).ifft(force_real = True)
    #-----------------------------------------------------------------------
    if calcType == 'Energy' or calcType == 'Both' :
        ene = np.einsum('ijkl, ijkl->', pot, Prho) * rho.grid.dV
    if calcType == 'Potential' or calcType == 'Both' :
        # dPrho = rho0 **(11.0/6.0) * (1 - nu)  + (1.0/6.0 * rho0 * nuMinus1 * rhoFiveSixth) + \
                # (5.0/6.0 * nu * rho * rhoFiveSixth)
        # dPrho /= (rho * rho)
        # print('dd', dPrho[:3, 0, 0, 0])
        dPrho = 5.0/6.0 * (nu  -  (nuMinus1  *  rho0)/rho) * rhoFiveSixth/rho
        # print('dd2', dPrho[:3, 0, 0, 0])
        pot *= 2.0 * dPrho

    xTF = TF(rho, calcType)
    yvW = vW(rho, Sigma, calcType)
    pot += x * xTF.potential + y * yvW.potential
    ene += x * xTF.energy + y * yvW.energy

    OutFunctional = Functional(name='FP')
    OutFunctional.potential = pot
    OutFunctional.energy= ene
    TimeData.End('FP')
    return OutFunctional

def FP0(rho,x=1.0,y=1.0,Sigma=0.025, alpha = 1.0, beta = 1.0, calcType='Both', split = False, **kwargs):
    TimeData.Begin('FP')
    global KE_kernel_saved
    #Only performed once for each grid
    q = rho.grid.get_reciprocal().q
    rho0 = np.einsum('ijkl -> ', rho) / np.size(rho)
    if abs(KE_kernel_saved['rho0']-rho0) > 1E-6 or np.shape(rho) != KE_kernel_saved['shape'] :
        print('Re-calculate KE_kernel')
        # KE_kernel = SMKernel(q,rho0, alpha = 1.0, beta = 1.0) * 1.6
        KE_kernel = WTKernel(q,rho0, alpha = alpha, beta = beta)
        KE_kernel_saved['Kernel'] = KE_kernel
        KE_kernel_saved['rho0'] = rho0
        KE_kernel_saved['shape'] = np.shape(rho)
    else :
        KE_kernel = KE_kernel_saved['Kernel']

    #-----------------------------------------------------------------------
    nu = 5.0/np.sqrt(32.0)
    rhoFiveSixth = PowerInt(rho, 5, 6)
    nuMinus1 = nu - 1.0
    drho = np.abs(rho - rho0)
    Mr = (1 + (nuMinus1 * drho)/(rho0 + drho))
    # Mr = (1 + (nuMinus1 * drho)/(rho))
    Prho = Mr * (rhoFiveSixth - rho0 ** (5.0/6.0))
    pot = (Prho.fft() * KE_kernel).ifft(force_real = True)
    #-----------------------------------------------------------------------
    ene = 0
    if calcType == 'Energy' or calcType == 'Both' :
        ene = np.einsum('ijkl, ijkl->', pot, Prho) * rho.grid.dV
    if calcType == 'Potential' or calcType == 'Both' :
        # dPrho = 5.0/6.0 * Mr * rhoFiveSixth/rho;# pot *= 2.0 * dPrho
        pot *= (5.0/3.0) * Mr * rhoFiveSixth/rho

    NL = Functional(name='NL', potential = pot, energy= ene)
    xTF = TF(rho, x = x,  calcType = calcType)
    yvW = vW(rho, y = y, Sigma = Sigma, calcType = calcType)
    OutFunctional = NL + xTF + yvW
    OutFunctional.name = 'FP'
    TimeData.End('FP')
    if split :
        return {'TF': xTF, 'vW': yvW, 'NL' : NL}
    else :
        return OutFunctional

def FP(rho,x=1.0,y=1.0,Sigma=0.025, alpha = 1.0, beta = 1.0, rho0 = None, calcType='Both', split = False, **kwargs):
    TimeData.Begin('FP')
    global KE_kernel_saved
    #Only performed once for each grid
    q = rho.grid.get_reciprocal().q
    if rho0 is None :
        rho0 = np.einsum('ijkl -> ', rho) / np.size(rho)
    if abs(KE_kernel_saved['rho0']-rho0) > 1E-6 or np.shape(rho) != KE_kernel_saved['shape'] :
        print('Re-calculate KE_kernel')
        # KE_kernel = SMKernel(q,rho0, alpha = 1.0, beta = 1.0) * 1.6
        KE_kernel = WTKernel(q,rho0, alpha = alpha, beta = beta)
        KE_kernel_saved['Kernel'] = KE_kernel
        KE_kernel_saved['rho0'] = rho0
        KE_kernel_saved['shape'] = np.shape(rho)
    else :
        KE_kernel = KE_kernel_saved['Kernel']

    #-----------------------------------------------------------------------
    nu = 5.0/np.sqrt(32.0)
    rhoFiveSixth = PowerInt(rho, 5, 6)
    nuMinus1 = nu - 1.0
    drho = np.abs(rho - rho0)
    # drho = rho - rho0
    Mr = (rho0 + nu * drho)/(rho0 + drho)
    # Prho = Mr * (rhoFiveSixth - rho0 ** (5.0/6.0))
    Prho = Mr * rhoFiveSixth
    pot = (Prho.fft() * KE_kernel).ifft(force_real = True)
    #-----------------------------------------------------------------------
    ene = 0
    if calcType == 'Energy' or calcType == 'Both' :
        ene = np.einsum('ijkl, ijkl->', pot, Prho) * rho.grid.dV
    if calcType == 'Potential' or calcType == 'Both' :
        # dPrho = 5.0/6.0 * Mr * rhoFiveSixth/rho;# pot *= 2.0 * dPrho
        pot *= (5.0/3.0) * Mr * rhoFiveSixth/rho

    NL = Functional(name='NL', potential = pot, energy= ene)
    xTF = TF(rho, x = x,  calcType = calcType)
    yvW = vW(rho, y = y, Sigma = Sigma, calcType = calcType)
    OutFunctional = NL + xTF + yvW
    OutFunctional.name = 'FP'
    TimeData.End('FP')
    if split :
        return {'TF': xTF, 'vW': yvW, 'NL' : NL}
    else :
        return OutFunctional
