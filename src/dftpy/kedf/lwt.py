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

__all__  =  ['LWT', 'LWTStress']

KE_kernel_saved ={'Kernel':None, 'rho0':0.0, 'shape':None, \
        'KernelTable':None, 'etamax':None, 'KernelDeriv':None, \
        'MGPKernelE' :None}

def LWTStress(rho, x=1.0,y=1.0, Sigma=0.025, interp = 'linear', kerneltype = 'WT', symmetrization = None, \
        lumpfactor = None, alpha = 5.0/6.0, beta = 5.0/6.0, etamax = 50.0, \
        neta = 50000, order = 3, calcType='Both', energy = None, **kwargs):
    pass

def LWTEnergyDensitybak(rho, etamax = 100.0, ratio = 1.1, nsp = None, alpha = 5.0/6.0, beta = 5.0/6.0, interp = 'linear'):
    q = rho.grid.get_reciprocal().q
    rho0 = np.mean(rho)
    kf0 = 2.0*(3.0*np.pi**2*rho0)**(1.0/3.0)
    kf = 2.0*(3.0*np.pi**2*rho)**(1.0/3.0)
    kfBound = [1E-3, 100.0]
    # kfBound = [kf0, kf0]
    mask1 = kf < kfBound[0]
    mask2 = kf > kfBound[1]
    kf[mask1] = kfBound[0]
    kf[mask2] = kfBound[1]

    rhoAlpha = rho**alpha
    kfMax = np.max(kf)
    kfMin = np.min(kf)
    if nsp is None :
        nsp = int(np.log(kfMax/kfMin)/np.log(ratio) + 1)+1
    else :
        kfMin = kfMax / (ratio ** (nsp - 1))
    kflists = kfMin * ratio ** np.arange(nsp)
    kflists[0] -= 1E-10 # for numeric safe
    kflists[-1] += 1E-10 # for numeric safe
    # print('rho', np.max(rho), np.min(rho))
    # print('nsp', nsp, kfMax, kfMin, kf0, np.max(kflists))
    #-----------------------------------------------------------------------
    kernel0 = np.empty_like(q)
    kernel1 = np.empty_like(q)
    kernelDeriv0 = np.empty_like(q)
    kernelDeriv1 = np.empty_like(q)
    pot1 = np.zeros_like(rho)
    if abs(alpha - beta) < 1E-8 :
        rhoBeta = rhoAlpha
    else :
        rhoBeta = rho ** beta
    rhoBetaG = rhoBeta.fft()
    KernelTable = KE_kernel_saved['KernelTable']
    KernelDeriv = KE_kernel_saved['KernelDeriv']
    MGPKernelE = KE_kernel_saved['MGPKernelE']
    #-----------------------------------------------------------------------
    if interp == 'newton' or interp == 'linear' :
        for i in range(nsp - 1):
            if i == 0 :
                kernel0 = LWTKernelKf(q, kflists[i], KernelTable, etamax = etamax, out = kernel0)
                p0 = (kernel0 * rhoBetaG).ifft(force_real = True)
            else :
                p0 = p1
            kernel1 = LWTKernelKf(q, kflists[i + 1], KernelTable, etamax = etamax, out = kernel1)
            p1 = (kernel1 * rhoBetaG).ifft(force_real = True)
            mask = np.logical_and(kf > kflists[i], kf < kflists[i + 1])
            Dkf = kflists[i + 1] - kflists[i]
            t = (kf - kflists[i]) / Dkf
            pot1[mask] = p0[mask] + (p1[mask] - p0[mask]) * t[mask]

    elif interp == 'newton2' :
        p2 = None
        for i in range(nsp - 1):
            if i == 0 :
                kernel0 = LWTKernelKf(q, kflists[i], KernelTable, etamax = etamax, out = kernel0)
                p0 = (kernel0 * rhoBetaG).ifft(force_real = True)
                kernel1 = LWTKernelKf(q, kflists[i + 1], KernelTable, etamax = etamax, out = kernel1)
                p1 = (kernel1 * rhoBetaG).ifft(force_real = True)
                Dkf = kflists[i + 1] - kflists[i]
                f01 = (p1 - p0) / Dkf
            elif i == 1 :
                p0 = p1
                kernel0 = LWTKernelKf(q, kflists[i], KernelTable, etamax = etamax, out = kernel0)
                p1 = (kernel0 * rhoBetaG).ifft(force_real = True)
                kernel1 = LWTKernelKf(q, kflists[i + 1], KernelTable, etamax = etamax, out = kernel1)
                p2 = (kernel1 * rhoBetaG).ifft(force_real = True)
            else :
                p0 = p1
                p1 = p2
                kernel1 = LWTKernelKf(q, kflists[i + 1], KernelTable, etamax = etamax, out = kernel1)
                p2 = (kernel1 * rhoBetaG).ifft(force_real = True)

            mask = np.logical_and(kf > kflists[i], kf < kflists[i + 1])
            t = kf - kflists[i]

            if i < 2 :
                pot1[mask] = p0[mask] + f01[mask] * t[mask]
            else :
                t0 = kf - kflists[i - 1]
                Dkf = kflists[i + 1] - kflists[i]
                f12 = (p2 - p1) / Dkf
                f02 = (f12 - f01) / (kflists[i + 1] - kflists[i - 1])
                pot1[mask] = p0[mask] + f01[mask] * t0[mask] + f02[mask] * t[mask] * t0[mask]
                f01 = f12

    elif interp == 'hermite' :
        for i in range(nsp - 1):
            if i == 0 :
                kernel0 = LWTKernelKf(q, kflists[i], KernelTable, etamax = etamax, out = kernel0)
                p0 = (kernel0 * rhoBetaG).ifft(force_real = True)
                kernelDeriv0 = LWTKernelKf(q, kflists[i], KernelDeriv, etamax = etamax, out = kernelDeriv0)
                kernelDeriv0 /= kflists[i]
                m0 = (kernelDeriv0 * rhoBetaG).ifft(force_real = True)
            else :
                p0 = p1
                m0 = m1
            kernel1 = LWTKernelKf(q, kflists[i + 1], KernelTable, etamax = etamax, out = kernel1)
            p1 = (kernel1 * rhoBetaG).ifft(force_real = True)
            kernelDeriv1 = LWTKernelKf(q, kflists[i + 1], KernelDeriv, etamax = etamax, out = kernelDeriv1)
            kernelDeriv1 /= kflists[i + 1]
            m1 = (kernelDeriv1 * rhoBetaG).ifft(force_real = True)

            mask = np.logical_and(kf > kflists[i], kf < kflists[i + 1])
            Dkf = kflists[i + 1] - kflists[i]
            t = (kf - kflists[i]) / Dkf
            t2 = t * t
            t3 = t2 * t
            h00 = 2.0*t3 - 3.0*t2 + 1.0
            h10 = t3 - 2.0*t2 + t
            h01 = 1.0 - h00
            h11 = t3 - t2
            pot1[mask] = h00[mask] * p0[mask] + h01[mask] * p1[mask] + \
                    Dkf * (h10[mask] * m0[mask] + h11[mask] * m1[mask])

    if MGPKernelE is not None :
        pot1 += (MGPKernelE * rhoBetaG).ifft(force_real = True)

    edens = rhoAlpha * pot1
    return edens, pot1

def LWTPotentialEnergy(rho, etamax = 100.0, ratio = 1.1, nsp = None, delta = None,\
        alpha = 5.0/6.0, beta = 5.0/6.0, interp = 'linear', calcType = 'Both', save = False):

    q = rho.grid.get_reciprocal().q
    rho0 = np.mean(rho)
    kf0 = 2.0*(3.0*np.pi**2*rho0)**(1.0/3.0)
    kf = 2.0*(3.0*np.pi**2*rho)**(1.0/3.0)
    kfBound = [1E-3, 100.0]
    #----------------Test WT------------------------------------------------
    # kfBound = [kf0, kf0]
    # nsp = 2
    # nsp = 30
    #-----------------------------------------------------------------------
    mask1 = kf < kfBound[0]
    mask2 = kf > kfBound[1]
    kf[mask1] = kfBound[0]
    kf[mask2] = kfBound[1]
    rhoAlpha = rho**alpha
    kfMax = np.max(kf)+1E-10
    # kfMin = np.min(kf)
    kfMin = max(np.min(kf), kfBound[0])
    if nsp is None :
        nsp = int(np.ceil(np.log(kfMax/kfMin)/np.log(ratio))) + 1
        kflists = kfMin * ratio ** np.arange(nsp)
    elif delta is not None :
        # delta = 0.10
        nsp = int(np.ceil((kfMax - kfMin)/delta)) + 1
        kflists = np.linspace(kfMin,kfMax,nsp)
    else :
        kfMin = kfMax / (ratio ** (nsp - 1))
        kflists = kfMin * ratio ** np.arange(nsp)
    kflists[0] -= 1E-10 # for numeric safe
    kflists[-1] += 1E-10 # for numeric safe
    print('nsp', nsp, kfMax, kfMin, kf0, np.max(kflists))
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
    pot4 = np.zeros_like(rho)
    rhoAlpha1 = rhoAlpha/rho
    if abs(alpha - beta) < 1E-8 :
        rhoBeta = rhoAlpha
        rhoBeta1 = rhoAlpha1
    else :
        rhoBeta = rho ** beta
        rhoBeta1 = rhoBeta/rho
    rhoBetaG = rhoBeta.fft()
    KernelTable = KE_kernel_saved['KernelTable']
    KernelDeriv = KE_kernel_saved['KernelDeriv']
    MGPKernelE = KE_kernel_saved['MGPKernelE']
    # interp = 'hermite'
    # interp = 'linear'
    # print('interp', interp)
    #-----------------------------------------------------------------------
    for i in range(nsp - 1):
        if i == 0 :
            kernel0 = LWTKernelKf(q, kflists[i], KernelTable, etamax = etamax, out = kernel0)
            p0 = (kernel0 * rhoBetaG).ifft(force_real = True)
            kernelDeriv0 = LWTKernelKf(q, kflists[i], KernelDeriv, etamax = etamax, out = kernelDeriv0) / kflists[i]
            m0 = (kernelDeriv0 * rhoBetaG).ifft(force_real = True)
        else :
            p0, p1 = p1, p0
            m0, m1 = m1, m0
            kernel0, kernel1 = kernel1, kernel0
            kernelDeriv0, kernelDeriv1 = kernelDeriv1, kernelDeriv0
            # p0 = p1.copy()
            # m0 = m1.copy()
            # kernel0 = kernel1.copy()
            # kernelDeriv0 = kernelDeriv1.copy()
        kernel1 = LWTKernelKf(q, kflists[i + 1], KernelTable, etamax = etamax, out = kernel1)
        p1 = (kernel1 * rhoBetaG).ifft(force_real = True)
        kernelDeriv1 = LWTKernelKf(q, kflists[i + 1], KernelDeriv, etamax = etamax, out = kernelDeriv1) / kflists[i + 1]
        m1 = (kernelDeriv1 * rhoBetaG).ifft(force_real = True)

        mask = np.logical_and(kf > kflists[i], kf < kflists[i + 1]+1E-10)
        Rmask[:] = 0.0
        Rmask[mask] = 1.0
        rhoU = rhoBeta * Rmask
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
            if pot2G is None :
                pot2G = kernel0 * (rhoU * t0).fft() + kernel1 * (rhoU * t).fft()
            else :
                pot2G += kernel0 * (rhoU * t0).fft() + kernel1 * (rhoU * t).fft()
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

            pG = kernel0 * (rhoU * h00).fft() + kernel1 * (rhoU * h01).fft() + \
                    Dkf * (kernelDeriv0 * (rhoU * h10).fft() + kernelDeriv1 * (rhoU * h11).fft())
            if pot2G is None :
                pot2G = pG
            else :
                pot2G += pG

            t = t[mask]
            t2 = t2[mask]
            t3 = t3[mask]
            h00D = 6.0 * t2 - 6 * t
            h10D = 3.0 * t2 - 4 * t + 1.0
            h01D = -h00D
            h11D = 3.0 * t2 - 2 * t
            pot3[mask] = (h00D * p0[mask] + h01D * p1[mask]) /Dkf + \
                    h10D * m0[mask] + h11D * m1[mask]
            #-----------------------------------------------------------------------
            # t0 = 1 - t
            # pot4[mask] = m0[mask] * (1 - t) + m1[mask] * t 


    pot2 = pot2G.ifft(force_real = True)
    #-------------------Some Trick------------------------------------------
    # rhoCut = 1E-6 * np.max(rho)
    # mask = rho < rhoCut
    # pot3[mask]= 0.0
    # pot1[mask]= 0.0
    # pot2[mask]= 0.0
    #-----------------------------------------------------------------------
    ene = np.einsum('ijkl, ijkl ->', rhoAlpha, pot1) * rho.grid.dV
    # ene2 = np.einsum('ijkl, ijkl ->', rhoAlpha, (pot1 + pot2) * 0.5) * rho.grid.dV
    # print('ene-->', ene, ene2, np.sum(pot1 ** 2), np.sum(pot2 ** 2))
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
    if MGPKernelE is not None :
        pot1 += (MGPKernelE * rhoBetaG).ifft(force_real = True)

    pot1 *= alpha * rhoAlpha1
    pot2 *= beta * rhoBeta1
    pot3 *= (kf/3.0) * rhoAlpha1
    pot1 += pot2 + pot3
    pot = pot1
    # if abs(alpha - beta) < 1E-8 :
        # # pot3 *= (kf/(3.0 * alpha))
        # # pot1 += pot2 + pot3
        # # pot1 *= ( alpha * rhoAlpha1)
        # pot1 *= alpha * rhoAlpha1
        # pot2 *= beta * rhoBeta1
        # pot = pot1
    # else :
        # raise AttributeError("For alpha != beta will implement as soon as possible")

    return pot, ene

def LWTEnergyDensity(rho, etamax = 100.0, ratio = 1.1, nsp = None, delta = None,\
        alpha = 5.0/6.0, beta = 5.0/6.0, interp = 'linear', calcType = 'Both', save = False):

    q = rho.grid.get_reciprocal().q
    rho0 = np.mean(rho)
    kf0 = 2.0*(3.0*np.pi**2*rho0)**(1.0/3.0)
    kf = 2.0*(3.0*np.pi**2*rho)**(1.0/3.0)
    kfBound = [1E-3, 100.0]
    #----------------Test WT------------------------------------------------
    # kfBound = [kf0, kf0]
    # nsp = 2
    # nsp = 30
    #-----------------------------------------------------------------------
    # mask1 = kf < kfBound[0]
    mask2 = kf > kfBound[1]
    # kf[mask1] = kfBound[0]
    kf[mask2] = kfBound[1]
    rhoAlpha = rho**alpha
    kfMax = np.max(kf)+1E-10
    # kfMin = np.min(kf)
    kfMin = max(np.min(kf), kfBound[0])
    if nsp is None :
        nsp = int(np.ceil(np.log(kfMax/kfMin)/np.log(ratio))) + 1
        kflists = kfMin * ratio ** np.arange(nsp)
    elif delta is not None :
        # delta = 0.10
        nsp = int(np.ceil((kfMax - kfMin)/delta)) + 1
        kflists = np.linspace(kfMin,kfMax,nsp)
    else :
        # kfMin = kfMax / (ratio ** (nsp - 1))
        # kflists = kfMin * ratio ** np.arange(nsp)
        kflists = kfMin + (kfMax - kfMin)/(nsp - 1) * np.arange(nsp)
    kflists[0] -= 1E-10 # for numeric safe
    kflists[-1] += 1E-10 # for numeric safe
    print('nsp', nsp, kfMax, kfMin, kf0, np.max(kflists), np.min(rho))
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
    pot4 = np.zeros_like(rho)
    rhoAlpha1 = rhoAlpha/rho
    if abs(alpha - beta) < 1E-8 :
        rhoBeta = rhoAlpha
        rhoBeta1 = rhoAlpha1
    else :
        rhoBeta = rho ** beta
        rhoBeta1 = rhoBeta/rho
    rhoBetaG = rhoBeta.fft()
    KernelTable = KE_kernel_saved['KernelTable']
    KernelDeriv = KE_kernel_saved['KernelDeriv']
    MGPKernelE = KE_kernel_saved['MGPKernelE']
    # interp = 'hermite'
    # interp = 'linear'
    # print('interp', interp)
    #-----------------------------------------------------------------------
    for i in range(nsp - 1):
        if i == 0 :
            kernel0 = LWTKernelKf(q, kflists[i], KernelTable, etamax = etamax, out = kernel0)
            p0 = (kernel0 * rhoBetaG).ifft(force_real = True)
            kernelDeriv0 = LWTKernelKf(q, kflists[i], KernelDeriv, etamax = etamax, out = kernelDeriv0) / kflists[i]
            m0 = (kernelDeriv0 * rhoBetaG).ifft(force_real = True)
        else :
            p0, p1 = p1, p0
            m0, m1 = m1, m0
            kernel0, kernel1 = kernel1, kernel0
            kernelDeriv0, kernelDeriv1 = kernelDeriv1, kernelDeriv0
        kernel1 = LWTKernelKf(q, kflists[i + 1], KernelTable, etamax = etamax, out = kernel1)
        p1 = (kernel1 * rhoBetaG).ifft(force_real = True)
        kernelDeriv1 = LWTKernelKf(q, kflists[i + 1], KernelDeriv, etamax = etamax, out = kernelDeriv1) / kflists[i + 1]
        m1 = (kernelDeriv1 * rhoBetaG).ifft(force_real = True)

        mask = np.logical_and(kf > kflists[i], kf < kflists[i + 1]+1E-10)
        Rmask[:] = 0.0
        Rmask[mask] = 1.0
        rhoU = rhoBeta * Rmask
        Dkf = kflists[i + 1] - kflists[i]
        t = (kf - kflists[i]) / Dkf
        #-----------------------------------------------------------------------
        if i == 0 :
            small = kf < kflists[0]+1E-10
            if len(small) > 0 :
                pot1[small] =  p0[small] * t[small]
        #-----------------------------------------------------------------------
        if interp == 'newton' or interp == 'linear' :
            t0 = 1 - t
            pot1[mask] = p0[mask] * t0[mask] + p1[mask] * t[mask]
        elif interp == 'hermite' :
            t2 = t * t
            t3 = t2 * t
            h00 = 2.0*t3 - 3.0*t2 + 1.0
            h10 = t3 - 2.0*t2 + t
            h01 = 1.0 - h00
            h11 = t3 - t2
            pot1[mask] = h00[mask] * p0[mask] + h01[mask] * p1[mask] + \
                    Dkf * (h10[mask] * m0[mask] + h11[mask] * m1[mask])

    if MGPKernelE is not None :
        pot1 += (MGPKernelE * rhoBetaG).ifft(force_real = True)

    edens = rhoAlpha * pot1
    return edens, pot1

def LWT(rho, x=1.0,y=1.0, Sigma=0.025, interp = 'linear', kerneltype = 'WT', symmetrization = None, \
        lumpfactor = None, alpha = 5.0/6.0, beta = 5.0/6.0, etamax = 50.0, maxpoints = 1000, fd = 1, \
        ratio = 1.2, nsp = None, delta = None, neta = 50000, order = 3, calcType='Both', split = False, **kwargs):
    TimeData.Begin('LWT')
    global KE_kernel_saved
    #Only performed once for each grid
    gg = rho.grid.get_reciprocal().gg
    q = rho.grid.get_reciprocal().q
    rho0 = np.mean(rho)
    if abs(KE_kernel_saved['rho0']-rho0) > 1E-6 or np.shape(rho) != KE_kernel_saved['shape'] :
        print('Re-calculate %s KernelTable ' %kerneltype)
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
            print('Calculate MGP kinetic electron(%f)' %lumpfactor)
            Ne = rho0 * np.size(rho) * rho.grid.dV
            MGPKernelE = MGPOmegaE(q, Ne, lumpfactor)
            KE_kernel_saved['MGPKernelE'] = MGPKernelE
        # Different method to interpolate the kernel
        if order > 1 :
            KE_kernel_saved['KernelTable'] = splrep(eta, KernelTable, k=order)
            KernelDerivTable = splev(eta, KE_kernel_saved['KernelTable'], der = 1) * (-1.0 * eta)
            # KernelDerivTable = WTKernelDerivTable(eta, x, y, alpha, beta)
            KE_kernel_saved['KernelDeriv'] = splrep(eta, KernelDerivTable, k=order)
        else :
            tck = splrep(eta, KernelTable, k=3)
            KernelDerivTable = splev(eta, tck, der = 1) * (-1.0 * eta)
            # KernelDerivTable = WTKernelDerivTable(eta, x, y, alpha, beta)
            KE_kernel_saved['KernelTable'] = KernelTable
            KE_kernel_saved['KernelDeriv'] = KernelDerivTable
        KE_kernel_saved['etamax'] = etamax
        KE_kernel_saved['shape'] = np.shape(rho)
        KE_kernel_saved['rho0'] = rho0
    KE_kernel_func = KE_kernel_saved
    if fd < 1 :
        pot, ene = LWTPotentialEnergy(rho, etamax = etamax, ratio = ratio, nsp = nsp, delta = delta, interp = interp, calcType = calcType)
    else :
        edens, pot1 = LWTEnergyDensity(rho, etamax = etamax, ratio = ratio, nsp = nsp, delta = delta, interp = interp, calcType = calcType)
        tol = 1E-9
        ene = 0.0
        if calcType == 'Energy' or calcType == 'Both' :
            ene = np.einsum('ijkl ->', edens) * rho.grid.dV
            pot = pot1
        if calcType == 'Potential' or calcType == 'Both' :
            if fd == 1 :
                pot = ((rho + tol) ** alpha - rho ** alpha) * pot1
                pot /= (0.5 * tol)
            elif fd == 2 :
                rho1 = rho - tol
                mask = rho1 < 0.0
                rho1[mask] = 0.0
                pot = ((rho + tol) ** alpha - rho1 ** alpha) * pot1
                pot[mask] = 0.0
                pot /= tol
            else : 
                coefs = [-25.0/12.0, 4.0, -3.0, 4.0/3.0, -1.0/4.0]
                coefs = np.asarray(coefs)/tol * 2.0
                for i, c in enumerate(coefs) :
                    if i == 0 :
                        pot = rho ** alpha * c
                    else :
                        pot += (rho + i * tol) ** alpha * c
                pot *= pot1
    #-----------------------------------------------------------------------
    NL = Functional(name='NL', potential = pot, energy= ene)
    xTF = TF(rho, x = x,  calcType = calcType)
    yvW = vW(rho, y = y, Sigma = Sigma, calcType = calcType)
    OutFunctional = NL + xTF + yvW
    OutFunctional.name = 'LWT'
    TimeData.End('LWT')
    if split :
        return {'TF': xTF, 'vW': yvW, 'NL' : NL}
    else :
        return OutFunctional
