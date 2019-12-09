import numpy as np
import scipy.special as sp
from scipy.interpolate import interp1d, splrep, splev
from dftpy.functional_output import Functional
from dftpy.field import DirectField
from dftpy.kedf.tf import TF
from dftpy.kedf.vw import vW
from dftpy.kedf.kernel import WTKernel, LindhardDerivative
from dftpy.kedf.kernel import MGPKernel
from dftpy.math_utils import TimeData

__all__  =  ['WT', 'WTStress']

KE_kernel_saved ={'Kernel':None, 'rho0':0.0, 'shape':None, \
        'KernelTable':None, 'etaMax':None, 'KernelDeriv':None}

def WTPotential(rho, rho0, Kernel, alpha, beta):
    alphaMinus1 = alpha - 1.0
    betaMinus1 = beta - 1.0
    if abs(beta - alpha) < 1E-9 :
        rhoBeta = rho ** beta
        rhoAlpha1 = rhoBeta / rho
        fac = 2.0 * alpha
        pot = fac * rhoAlpha1 * (rhoBeta.fft() * Kernel).ifft(force_real = True)
    else :
        pot = alpha * rho ** alphaMinus1 * ((rho ** beta).fft() * Kernel).ifft(force_real = True)
        pot += beta * rho ** betaMinus1 * ((rho ** alpha).fft() * Kernel).ifft(force_real = True)

    return pot

def WTPotentialEdens(rho, rho0, Kernel, alpha, beta):
    mask = rho < 0.0
    rho[mask] = 0.0
    edens = rho ** alpha * ((rho ** beta).fft() * Kernel).ifft(force_real = True)
    return edens

def WTEnergy(rho, rho0, Kernel, alpha, beta):
    rhoBeta = rho ** beta
    if abs(beta - alpha) < 1E-9 :
        rhoAlpha = rhoBeta
    else :
        rhoAlpha = rho ** alpha
    pot1 = (rhoBeta.fft() * Kernel).ifft(force_real = True)
    ene = np.einsum('ijkl, ijkl->', pot1, rhoAlpha) * rho.grid.dV

    return ene

def WTStress(rho,x=1.0,y=1.0,Sigma=0.025, alpha = 5.0/6.0, beta = 5.0/6.0, energy=None):
    rho0 = np.sum(rho)/np.size(rho)
    g = rho.grid.get_reciprocal().g
    gg = rho.grid.get_reciprocal().gg
    q = rho.grid.get_reciprocal().q
    if energy is None :
        global KE_kernel_saved
        if abs(KE_kernel_saved['rho0']-rho0) > 1E-6 or np.shape(rho) != KE_kernel_saved['shape'] :
            print('Re-calculate KE_kernel')
            KE_kernel = WTkernel(q, rho0, alpha = alpha, beta = beta)
            KE_kernel_saved['Kernel'] = KE_kernel
            KE_kernel_saved['rho0'] = rho0
            KE_kernel_saved['shape'] = np.shape(rho)
        else :
            KE_kernel = KE_kernel_saved['Kernel']
        energy = WTEnergy(rho, rho0, KE_kernel, alpha, beta)
    mask = rho.grid.get_reciprocal().mask
    factor = 5.0 / (9.0 * alpha * beta * rho0 ** (alpha + beta - 5.0/3.0))
    tkf = 2.0 * (3.0 * rho0 * np.pi**2)**(1.0/3.0)
    tkf = float(tkf)
    rhoG_A = (rho ** alpha).fft()/ rho.grid.volume
    rhoG_B = np.conjugate((rho ** beta).fft())/ rho.grid.volume
    DDrho = LindhardDerivative(q/tkf, y) * rhoG_A * rhoG_B
    stress = np.zeros((3, 3))
    gg[0, 0, 0, 0] = 1.0
    mask2 = mask[..., np.newaxis]
    for i in range(3):
        for j in range(i, 3):
            if i == j :
                fac = 1.0/3.0
            else :
                fac = 0.0
            # den = (g[..., i] * g[..., j]/gg[..., 0]-fac)[..., np.newaxis] * DDrho
            den = (g[..., i][mask] * g[..., j][mask]/gg[mask2]-fac) * DDrho[mask2]
            stress[i, j] = stress[j, i] = (np.einsum('i->', den)).real
    stress *= np.pi ** 2 /(alpha*beta*rho0**(alpha+beta-2)*tkf/2.0)
    for i in range(3):
        stress[i, i] -= 2.0/3.0 * energy/rho.grid.volume
    gg[0, 0, 0, 0] = 0.0

    return stress

def WT(rho,x=1.0,y=1.0,Sigma=0.025, alpha = 5.0/6.0, beta = 5.0/6.0, rho0 = None, calcType='Both', split = False, **kwargs):
    TimeData.Begin('WT')
    global KE_kernel_saved
    #Only performed once for each grid
    q = rho.grid.get_reciprocal().q
    if rho0 is None :
        rho0 = np.einsum('ijkl -> ', rho) / np.size(rho)
    # print('rho0', rho0)

    if abs(KE_kernel_saved['rho0']-rho0) > 1E-6 or np.shape(rho) != KE_kernel_saved['shape'] :
        print('Re-calculate KE_kernel')
        KE_kernel = WTKernel(q,rho0, x = x, y = y, alpha = alpha, beta = beta)
        KE_kernel_saved['Kernel'] = KE_kernel
        KE_kernel_saved['rho0'] = rho0
        KE_kernel_saved['shape'] = np.shape(rho)
    else :
        KE_kernel = KE_kernel_saved['Kernel']


    if calcType == 'Energy' :
        ene = WTEnergy(rho, rho0, KE_kernel, alpha, beta)
        pot = np.empty_like(rho)
    elif calcType == 'Potential' :
        pot = WTPotential(rho, rho0, KE_kernel, alpha, beta)
        ene = 0
    else :
        pot = WTPotential(rho, rho0, KE_kernel, alpha, beta)
        if abs(beta - alpha) < 1E-9 :
            ene = np.einsum('ijkl, ijkl->', pot, rho) * rho.grid.dV / (2 * alpha)
        else :
            ene = WTEnergy(rho, rho0, KE_kernel, alpha, beta)

    NL = Functional(name='NL', potential = pot, energy= ene)
    xTF = TF(rho, x = x,  calcType = calcType)
    yvW = vW(rho, y = y, Sigma = Sigma, calcType = calcType)
    OutFunctional = NL + xTF + yvW
    OutFunctional.name = 'WT'
    TimeData.End('WT')
    if split :
        return {'TF': xTF, 'vW': yvW, 'NL' : NL}
    else :
        return OutFunctional
