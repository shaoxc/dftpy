import numpy as np
import scipy.special as sp
from scipy.interpolate import interp1d, splrep, splev
from dftpy.functional_output import Functional
from dftpy.field import DirectField
from dftpy.kedf.tf import TF
from dftpy.kedf.vw import vW
from dftpy.kedf.kernel import WTKernelTable, WTKernelDerivTable, LWTKernel, LWTKernelKf
from dftpy.kedf.kernel import MGPKernelTable,  MGPOmegaE
from dftpy.math_utils import TimeData

__all__  =  ['LWT', 'LWTStress', 'LMGP', 'LMGPA', 'LMGPG']

KE_kernel_saved ={'Kernel':None, 'rho0':0.0, 'shape':None, \
        'KernelTable':None, 'etamax':None, 'KernelDeriv':None, \
        'MGPKernelE' :None}

def LWTStress(rho, x=1.0,y=1.0, Sigma=0.025, interp = 'linear', kerneltype = 'WT', symmetrization = None, \
        lumpfactor = None, alpha = 5.0/6.0, beta = 5.0/6.0, etamax = 50.0, \
        neta = 50000, order = 3, calcType='Both', energy = None, **kwargs):
    pass

def LWTPotentialEnergy(rho, etamax = 100.0, ratio = 1.1, nsp = None, delta = None, kdd = 1, \
        alpha = 5.0/6.0, beta = 5.0/6.0, interp = 'linear', calcType = 'Both'):

    savetol = 1E-18
    q = rho.grid.get_reciprocal().q
    rho0 = np.mean(rho)
    kf0 = 2.0*(3.0*np.pi**2*rho0)**(1.0/3.0)
    kf = 2.0*(3.0*np.pi**2*rho)**(1.0/3.0)
    kfBound = [1E-3, 100.0]
    #----------------Test WT------------------------------------------------
    # kfBound = [kf0, kf0]
    # nsp = 2
    #-----------------------------------------------------------------------
    #ignore the vacuum density contribution
    #mask1 = kf < kfBound[0]
    #kf[mask1] = kfBound[0]
    #-----------------------------------------------------------------------
    mask2 = kf > kfBound[1]
    kf[mask2] = kfBound[1]
    kfMax = np.max(kf)
    # kfMin = np.min(kf)
    kfMin = max(np.min(kf), kfBound[0])
    ### HEG
    if abs(kfMax - kfMin) < 1E-8 :
        return np.zeros_like(rho), 0.0

    if nsp is None :
        nsp = int(np.ceil(np.log(kfMax/kfMin)/np.log(ratio))) + 1
        # kflists = kfMin * ratio ** np.arange(nsp)
        kflists = np.geomspace(kfMin, kfMax, nsp)
    elif delta is not None :
        # delta = 0.10
        nsp = int(np.ceil((kfMax - kfMin)/delta)) + 1
        kflists = np.linspace(kfMin,kfMax,nsp)
    else :
        # kflists = kfMin + (kfMax - kfMin)/(nsp - 1) * np.arange(nsp)
        if kfMax - kfMin < 1E-3 :
            kflists = [kfMin, kfMax]
        else :
            kflists = kfMin + (kfMax - kfMin)/(nsp - 1) * np.arange(nsp)
    kflists[0] -= 1E-10 # for numerical safe
    kflists[-1] += 1E-10 # for numerical safe
    # print('nsp', nsp, kfMax, kfMin, kf0, np.max(kflists))
    #-----------------------------------------------------------------------
    kernel0 = np.empty_like(q)
    kernel1 = np.empty_like(q)
    kernelDeriv0 = np.empty_like(q)
    kernelDeriv1 = np.empty_like(q)
    Rmask = np.empty_like(rho)
    pot1 = np.zeros_like(rho)
    # pot2G = np.zeros_like(q, dtype = 'complex')
    pot2G = None
    pot3 = np.zeros_like(rho)
    #pot4 = np.zeros_like(rho)
    rhoAlpha = rho**alpha
    rhoAlpha1 = rhoAlpha/rho
    if abs(alpha - beta) < 1E-8 :
        rhoBeta = rhoAlpha
        rhoBeta1 = rhoAlpha1
    else :
        rhoBeta = rho ** beta
        rhoBeta1 = rhoBeta/rho
    rhoBetaG = rhoBeta.fft()
    if abs(alpha - beta) < 1E-8 :
        rhoAlphaG = rhoBetaG
    else :
        rhoAlphaG = rhoAlpha.fft()

    KernelTable = KE_kernel_saved['KernelTable']
    KernelDeriv = KE_kernel_saved['KernelDeriv']
    MGPKernelE = KE_kernel_saved['MGPKernelE']
    if kdd > 1 or interp == 'hermite' :
        mcalc= True
    else :
        mcalc= False
    # print('interp', interp, mcalc)
    # print('interp', interp, np.sum(rhoBeta), alpha, beta)
    #-----------------------------------------------------------------------
    for i in range(nsp - 1):
        if i == 0 :
            kernel0 = LWTKernelKf(q, kflists[i], KernelTable, etamax = etamax, out = kernel0)
            p0 = (kernel0 * rhoBetaG).ifft(force_real = True)
            if mcalc :
                kernelDeriv0 = LWTKernelKf(q, kflists[i], KernelDeriv, etamax = etamax, out = kernelDeriv0) / kflists[i]
                m0 = (kernelDeriv0 * rhoBetaG).ifft(force_real = True)
        else :
            p0, p1 = p1, p0
            kernel0, kernel1 = kernel1, kernel0
            if mcalc :
                m0, m1 = m1, m0
                kernelDeriv0, kernelDeriv1 = kernelDeriv1, kernelDeriv0

        kernel1 = LWTKernelKf(q, kflists[i + 1], KernelTable, etamax = etamax, out = kernel1)
        p1 = (kernel1 * rhoBetaG).ifft(force_real = True)
        if mcalc :
            kernelDeriv1 = LWTKernelKf(q, kflists[i + 1], KernelDeriv, etamax = etamax, out = kernelDeriv1) / kflists[i + 1]
            m1 = (kernelDeriv1 * rhoBetaG).ifft(force_real = True)

        mask = np.logical_and(kf > kflists[i], kf < kflists[i + 1]+1E-18) #  1E-18 for numerical errors, must be very small
        Rmask[:] = 0.0
        Rmask[mask] = 1.0
        rhoU = rhoAlpha * Rmask
        Dkf = kflists[i + 1] - kflists[i]
        t = (kf - kflists[i]) / Dkf
        #-----------------------------------------------------------------------
        # if i == 0 :
            # small = kf < kflists[0]+1E-10
            # if len(small) > 0 :
                # pot1[small] =  p0[small] * t[small]
        #-----------------------------------------------------------------------
        if interp == 'newton' or interp == 'linear' :
            t0 = 1 - t
            pot1[mask] = p0[mask] * t0[mask] + p1[mask] * t[mask]
            if kdd > 1 :
                if pot2G is None :
                    pot2G = kernel0 * (rhoU * t0).fft() + kernel1 * (rhoU * t).fft()
                else :
                    pot2G += kernel0 * (rhoU * t0).fft() + kernel1 * (rhoU * t).fft()
                if kdd  ==  3 :
                    pot3[mask] = m0[mask] * t0[mask] + m1[mask] * t[mask] 

        elif interp == 'hermite' :
            t2 = t * t
            t3 = t2 * t
            h00 = 2.0*t3 - 3.0*t2 + 1.0
            h10 = t3 - 2.0*t2 + t
            h01 = 1.0 - h00
            h11 = t3 - t2
            pot1[mask] = h00[mask] * p0[mask] + h01[mask] * p1[mask] + \
                    Dkf * (h10[mask] * m0[mask] + h11[mask] * m1[mask])

            if kdd > 1 :
                pG = kernel0 * (rhoU * h00).fft() + kernel1 * (rhoU * h01).fft() + \
                        Dkf * (kernelDeriv0 * (rhoU * h10).fft() + kernelDeriv1 * (rhoU * h11).fft())
                if pot2G is None :
                    pot2G = pG
                else :
                    pot2G += pG

                if kdd == 3 :
                    t = t[mask]
                    t2 = t2[mask]
                    t3 = t3[mask]
                    h00D = 6.0 * t2 - 6 * t
                    h10D = 3.0 * t2 - 4 * t + 1.0
                    h01D = -h00D
                    h11D = 3.0 * t2 - 2 * t
                    pot3[mask] = (h00D * p0[mask] + h01D * p1[mask]) /Dkf + \
                            h10D * m0[mask] + h11D * m1[mask]


    if MGPKernelE is not None :
        mgpe = (MGPKernelE * rhoBetaG).ifft(force_real = True)
        pot1 += mgpe

    if pot2G is not None :
        pot2 = pot2G.ifft(force_real = True)
        if MGPKernelE is not None :
            pot2 += mgpe
    else :
        pot2 = pot1.copy()
    #-----------------------------------------------------------------------
    ene = np.einsum('ijkl, ijkl ->', rhoAlpha, pot1) * rho.grid.dV
    #-----------------------------------------------------------------------
    # save = True
    # if save :
        # pot3S = pot3 * (kf/(3.0 * alpha))
        # # pot4S = pot4 * (kf/(3.0 * alpha))
        # # np.savetxt('pot.dat', np.c_[pot1.ravel(), pot3S.ravel(), pot4S.ravel(), rho.ravel()])
        # np.savetxt('pot.dat', np.c_[pot1.ravel(), pot2.ravel(), pot3S.ravel(), rho.ravel()])
        # header = '%10d %10d %10d'%(tuple(rho.grid.nr[:3]))
        # # np.savetxt('rho.dat', rho.ravel(), header = header)
    #-----------------------------------------------------------------------

    pot1 *= alpha * rhoAlpha1
    pot2 *= beta * rhoBeta1
    pot3 *= (kf/3.0) * rhoAlpha1
    pot1 += pot2 + pot3
    pot = pot1

    return pot, ene

def LWT(rho, x=1.0,y=1.0, Sigma=0.025, interp = 'linear', kerneltype = 'WT', symmetrization = None, \
        lumpfactor = None, alpha = 5.0/6.0, beta = 5.0/6.0, etamax = 50.0, maxpoints = 1000, fd = 1, kdd = 1, \
        ratio = 1.2, nsp = None, delta = None, neta = 50000, order = 3, calcType='Both', split = False, **kwargs):
    TimeData.Begin('LWT')
    global KE_kernel_saved
    #Only performed once for each grid
    gg = rho.grid.get_reciprocal().gg
    q = rho.grid.get_reciprocal().q
    rho0 = np.mean(rho)
    if abs(KE_kernel_saved['rho0']-rho0) > 1E-6 or np.shape(rho) != KE_kernel_saved['shape'] :
        #print('Re-calculate %s KernelTable ' %kerneltype)
        eta = np.linspace(0, etamax, neta)
        if kerneltype == 'WT' :
            KernelTable = WTKernelTable(eta, x, y, alpha, beta)
        elif kerneltype == 'MGP' :
            KernelTable = MGPKernelTable(eta, q, maxpoints = maxpoints, symmetrization = symmetrization)
        elif kerneltype == 'MGPA' :
            KernelTable = MGPKernelTable(eta, q, maxpoints = maxpoints, symmetrization = 'Arithmetic')
        elif kerneltype == 'MGPG' :
            KernelTable = MGPKernelTable(eta, q, maxpoints = maxpoints, symmetrization = 'Geometric')
        # Add MGP kinetic electron
        if lumpfactor is not None :
            #print('Calculate MGP kinetic electron(%f)' %lumpfactor)
            Ne = rho0 * np.size(rho) * rho.grid.dV
            MGPKernelE = MGPOmegaE(q, Ne, lumpfactor)
            KE_kernel_saved['MGPKernelE'] = MGPKernelE
        # Different method to interpolate the kernel
        if order > 1 :
            KE_kernel_saved['KernelTable'] = splrep(eta, KernelTable, k=order)
            KernelDerivTable = splev(eta, KE_kernel_saved['KernelTable'], der = 1) * (-1.0 * eta)
            KE_kernel_saved['KernelDeriv'] = splrep(eta, KernelDerivTable, k=order)
        else :
            tck = splrep(eta, KernelTable, k=3)
            KernelDerivTable = splev(eta, tck, der = 1) * (-1.0 * eta)
            KE_kernel_saved['KernelTable'] = KernelTable
            KE_kernel_saved['KernelDeriv'] = KernelDerivTable
        KE_kernel_saved['etamax'] = etamax
        KE_kernel_saved['shape'] = np.shape(rho)
        KE_kernel_saved['rho0'] = rho0
    KE_kernel_func = KE_kernel_saved
    pot, ene = LWTPotentialEnergy(rho, alpha = alpha, beta = beta, etamax = etamax, ratio = ratio, \
        nsp = nsp, kdd = kdd, delta = delta, interp = interp, calcType = calcType)
    NL = Functional(name='NL', potential = pot, energy= ene)
    return NL

def LMGP(rho, x=1.0,y=1.0, Sigma=0.025, interp = 'linear', kerneltype = 'MGP', symmetrization = None, \
        lumpfactor = None, alpha = 5.0/6.0, beta = 5.0/6.0, etamax = 50.0, maxpoints = 1000, fd = 1, kdd = 1, \
        ratio = 1.2, nsp = None, delta = None, neta = 50000, order = 3, calcType='Both', split = False, **kwargs):
    return LWT(rho, x,y, Sigma, interp, 'MGP', symmetrization, \
        lumpfactor, alpha, beta, etamax, maxpoints, fd, kdd, ratio, nsp, delta, neta, order, calcType, split, **kwargs)

def LMGPA(rho, x=1.0,y=1.0, Sigma=0.025, interp = 'linear', kerneltype = 'MGPA', symmetrization = None, \
        lumpfactor = None, alpha = 5.0/6.0, beta = 5.0/6.0, etamax = 50.0, maxpoints = 1000, fd = 1, kdd = 1, \
        ratio = 1.2, nsp = None, delta = None, neta = 50000, order = 3, calcType='Both', split = False, **kwargs):
    return LWT(rho, x,y, Sigma, interp, 'MGPA', symmetrization, \
        lumpfactor, alpha, beta, etamax, maxpoints, fd, kdd, ratio, nsp, delta, neta, order, calcType, split, **kwargs)

def LMGPG(rho, x=1.0,y=1.0, Sigma=0.025, interp = 'linear', kerneltype = 'MGPG', symmetrization = None, \
        lumpfactor = None, alpha = 5.0/6.0, beta = 5.0/6.0, etamax = 50.0, maxpoints = 1000, fd = 1, kdd = 1, \
        ratio = 1.2, nsp = None, delta = None, neta = 50000, order = 3, calcType='Both', split = False, **kwargs):
    return LWT(rho, x,y, Sigma, interp, 'MGPG', symmetrization, \
        lumpfactor, alpha, beta, etamax, maxpoints, fd, kdd, ratio, nsp, delta, neta, order, calcType, split, **kwargs)
