import numpy as np
import scipy.special as sp
from scipy.interpolate import interp1d, splrep, splev
from dftpy.functional_output import Functional
from dftpy.field import DirectField
from dftpy.kedf.tf import TF
from dftpy.kedf.vw import vW
from dftpy.kedf.wt import WTPotential, WTEnergy
from dftpy.kedf.kernel import MGPKernel, MGPOmegaE, LindhardDerivative
from dftpy.math_utils import TimeData

__all__  =  ['MGP', 'MGPStress', 'MGPA', 'MGPG']

KE_kernel_saved ={'Kernel':None, 'rho0':0.0, 'shape':None, \
        'KernelTable':None, 'etaMax':None, 'KernelDeriv':None, \
        'MGPKernelE' :None}

def MGPStress(rho,x=1.0,y=1.0,Sigma=0.025, alpha = 5.0/6.0, beta = 5.0/6.0, calcType='Both'):
    pass

def MGP(rho,x=1.0,y=1.0,Sigma=0.025, alpha = 5.0/6.0, beta = 5.0/6.0, lumpfactor = 0.2, \
        maxpoint = 1000, symmetrization = None, calcType='Both', split = False, **kwargs):
    TimeData.Begin('MGP')
    global KE_kernel_saved
    #Only performed once for each grid
    q = rho.grid.get_reciprocal().q
    rho0 = np.einsum('ijkl -> ', rho) / np.size(rho)
    # if abs(KE_kernel_saved['rho0']-rho0) > 1E-6 or np.shape(rho) != KE_kernel_saved['shape'] :
    if abs(KE_kernel_saved['rho0']-rho0) > 1E-2 or np.shape(rho) != KE_kernel_saved['shape'] :
        print('Re-calculate KE_kernel')
        KE_kernel = MGPKernel(q,rho0, maxpoints = maxpoint, symmetrization = symmetrization)
        if lumpfactor is not None :
            Ne = rho0 * np.size(rho) * rho.grid.dV
            KE_kernel += MGPOmegaE(q, Ne, lumpfactor)
        #-----------------------------------------------------------------------
        # rh0 = 0.03;lumpfactor = 0.0;q = np.linspace(1E-3, 8, 10000).reshape((1, 1, 1, -1))
        # mgp = MGPKernel(q,rho0,  maxpoints = maxpoint, symmetrization = None, KernelTable = None)
        # mgpa = MGPKernel(q,rho0, maxpoints = maxpoint, symmetrization = 'Arithmetic', KernelTable = None)
        # mgpg = MGPKernel(q,rho0, maxpoints = maxpoint, symmetrization = 'Geometric', KernelTable = None)
        # np.savetxt('mgp.dat', np.c_[q.ravel()/2.0, mgp.ravel(), mgpa.ravel(), mgpg.ravel()])
        # stop
        #-----------------------------------------------------------------------
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
    return NL

def MGPA(rho,x=1.0,y=1.0,Sigma=0.025, alpha = 5.0/6.0, beta = 5.0/6.0, lumpfactor = 0.2, \
        maxpoint = 1000, symmetrization = 'Arithmetic', calcType='Both', split = False, **kwargs):
    return MGP(rho,x,y,Sigma, alpha, beta, lumpfactor, maxpoint, 'Arithmetic', calcType, split, **kwargs)

def MGPG(rho,x=1.0,y=1.0,Sigma=0.025, alpha = 5.0/6.0, beta = 5.0/6.0, lumpfactor = 0.2, \
        maxpoint = 1000, symmetrization = 'Geometric', calcType='Both', split = False, **kwargs):
    return MGP(rho,x,y,Sigma, alpha, beta, lumpfactor, maxpoint, 'Geometric', calcType, split, **kwargs)
