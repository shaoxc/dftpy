import numpy as np
import scipy.special as sp
from scipy.interpolate import interp1d, splrep, splev
from dftpy.time_data import TimeData
from dftpy.constants import ZERO


# def LindhardFunctionSeries(eta,lbda,mu):
def LindhardFunction(eta, lbda, mu):
    """
    The Inverse Lindhard Function

    Attributes
    ----------
    eta: numpy array
    lbda, mu: floats (TF and vW contributions)

    """
    TimeData.Begin("Lindhard")
    if isinstance(eta, (np.ndarray, np.generic)):
        #
        cond0 = (eta > 0.0) & (np.abs(eta - 1.0) > 1.0e-10)
        cond1 = eta < 1.0e-10
        cond2 = np.abs(eta - 1.0) < 1.0e-10
        cond3 = eta > 3.65
        cond11 = eta > 1.0e-10

        invEta2 = eta.copy()
        invEta2[cond11] = 1.0 / eta[cond11] ** 2

        #
        indx0 = cond0
        indx1 = np.where(cond1)
        indx2 = np.where(cond2)
        indx3 = np.where(cond3)

        LindG = eta.copy()

        LindG[indx0] = (
            1.0
            / (
                0.5
                + 0.25 * (1.0 - eta[indx0] ** 2) * np.log((1.0 + eta[indx0]) / np.abs(1.0 - eta[indx0])) / eta[indx0]
            )
            - 3.0 * mu * eta[indx0] ** 2
            - lbda
        )

        LindG[indx1] = 1.0 - lbda + eta[indx1] ** 2 * (1.0 / 3.0 - 3.0 * mu)
        LindG[indx2] = 2.0 - lbda - 3.0 * mu + 20.0 * (eta[indx2] - 1.0)
        LindG[indx3] = (
            3.0 * (1.0 - mu) * eta[indx3] ** 2
            - lbda
            - 0.6
            + invEta2[indx3]
            * (
                -0.13714285714285712
                + invEta2[indx3]
                * (
                    -6.39999999999999875e-2
                    + invEta2[indx3]
                    * (
                        -3.77825602968460128e-2
                        + invEta2[indx3]
                        * (
                            -2.51824061652633074e-2
                            + invEta2[indx3]
                            * (
                                -1.80879839616166146e-2
                                + invEta2[indx3]
                                * (
                                    -1.36715733124818332e-2
                                    + invEta2[indx3]
                                    * (
                                        -1.07236045520990083e-2
                                        + invEta2[indx3]
                                        * (
                                            -8.65192783339199453e-3
                                            + invEta2[indx3]
                                            * (
                                                -7.1372762502456763e-3
                                                + invEta2[indx3]
                                                * (
                                                    -5.9945117538835746e-3
                                                    + invEta2[indx3]
                                                    * (
                                                        -5.10997527675418131e-3
                                                        + invEta2[indx3]
                                                        * (
                                                            -4.41060829979912465e-3
                                                            + invEta2[indx3]
                                                            * (
                                                                -3.84763737842981233e-3
                                                                + invEta2[indx3]
                                                                * (
                                                                    -3.38745061493813488e-3
                                                                    + invEta2[indx3] * (-3.00624946457977689e-3)
                                                                )
                                                            )
                                                        )
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )

    TimeData.End("Lindhard")
    return LindG


def LindhardFunction2(eta, lbda, mu):
    # def LindhardFunction(eta,lbda,mu):
    """
    (1) for x -> 0.0
            2      4
           x    8⋅x
       1 + ── + ──── + ...
           3     45

    (2)  for x -> 1.0
        2 + (1-x)⋅(2⋅log(1-x) - 2⋅log(2)) + ...
        We use a magic number 48, because 2.0*(log(1E-10)-log(2))~ -47.4

    (3) for y -> 0.0, y = 1/x
                     2      4          6
        3    3   24⋅y    8⋅y    12728⋅y
        ── - ─ - ───── - ──── - ──────── -...
         2   5    175    125     336875
        y
        Actually, if not write the multiplication using C++ or Fortran, numpy.log will be faster.

    The Inverse Lindhard Function

    Attributes
    ----------
    eta: numpy array
    lbda, mu: floats (TF and vW contributions)

    """
    TimeData.Begin("Lindhard")
    if isinstance(eta, (np.ndarray, np.generic)):
        LindG = np.zeros_like(eta)
        atol = 1.0e-10

        cond0 = np.logical_and(eta > atol, np.abs(eta - 1.0) > atol)
        cond1 = eta < atol
        cond2 = np.abs(eta - 1.0) < atol

        LindG[cond0] = (
            1.0
            / (
                0.5
                + 0.25 * (1.0 - eta[cond0] ** 2) * np.log((1.0 + eta[cond0]) / np.abs(1.0 - eta[cond0])) / eta[cond0]
            )
            - 3.0 * mu * eta[cond0] ** 2
            - lbda
        )

        LindG[cond1] = 1.0 + eta[cond1] ** 2 * (1.0 / 3.0 - 3.0 * mu) - lbda
        LindG[cond2] = 2.0 - 48 * np.abs(eta[cond2] - 1.0) - 3.0 * mu * eta[cond2] ** 2 - lbda
        # -----------------------------------------------------------------------
        # LindG += 1.6
        # cond = eta > 20.0
        # LindG[cond] = -0.6 - lbda
        # -----------------------------------------------------------------------
    TimeData.End("Lindhard")
    return LindG


def LindhardDerivative(eta, mu):
    TimeData.Begin("LindDeriv")
    LindDeriv = np.zeros_like(eta)
    atol = 1.0e-10
    cond0 = np.logical_and(eta > atol, np.abs(eta - 1.0) > atol)
    cond1 = eta < atol
    cond2 = np.abs(eta - 1.0) < atol

    TempA = np.log(np.abs((1.0 + eta[cond0]) / (1.0 - eta[cond0])))
    LindDeriv[cond0] = (0.5 / eta[cond0] - 0.25 * (eta[cond0] ** 2 + 1.0) / eta[cond0] ** 2 * TempA) / (
        0.5 + 0.25 * (1 - eta[cond0] ** 2) / eta[cond0] * TempA
    ) ** 2 + 6.0 * eta[cond0] * mu
    LindDeriv[cond1] = -2.0 * eta[cond1] * (1.0 / 3.0 - 3.0 * mu)
    LindDeriv[cond2] = -48

    TimeData.End("LindDeriv")
    return LindDeriv * eta


def MGPKernelOld(q, rho0, lumpfactor, maxpoints):
    """
    The MGP Kernel
    """
    # cTF_WT = 2.87123400018819
    cTF = np.pi ** 2 / (3.0 * np.pi ** 2) ** (1.0 / 3.0)
    tkf = 2.0 * (3.0 * rho0 * np.pi ** 2) ** (1.0 / 3.0)
    t_var = 1.0 / (maxpoints)
    deltat = 1.0 / (maxpoints)
    dt = deltat / 100

    kertmp = np.zeros(np.shape(q))

    for i_var in range(maxpoints):
        kertmp = kertmp + 0.5 * (
            (
                LindhardFunction(q / (tkf * (t_var + dt) ** (1.0 / 3.0)), -0.60, 1.0)
                - LindhardFunction(q / (tkf * (t_var - dt) ** (1.0 / 3.0)), -0.60, 1.0)
            )
            / dt
        ) * t_var ** (5.0 / 6.0)
        #
        t_var = t_var + deltat

    tmpker1 = -1.2 * kertmp * deltat
    indx = np.where(q != 0)
    tmpker2 = kertmp.copy()
    tmpker2[indx] = (
        4 * np.pi * sp.erf(q[indx]) ** 2 * lumpfactor * np.exp(-q[indx] ** 2 * lumpfactor) / q[indx] ** 2 / cTF
    )
    indx = np.where(q == 0)
    tmpker2[indx] = q[indx] ** 2
    tmpker3 = 1.2 * LindhardFunction(q / tkf, 1.0, 1.0)

    return (tmpker1 + tmpker2 + tmpker3) * cTF  # *cTF_WT


def MGPKernel(q, rho0, lumpfactor=0.2, maxpoints=1000, symmetrization=None, KernelTable=None):
    """
    The MGP Kernel
    symmetrization : 'None', 'Arithmetic', 'Geometric'
    """
    tkf = 2.0 * (3.0 * rho0 * np.pi ** 2) ** (1.0 / 3.0)
    eta = q / tkf
    return MGPKernelTable(eta, q, maxpoints, symmetrization, KernelTable)


def MGPKernelTable(eta, q, maxpoints=1000, symmetrization=None, KernelTable=None):
    """
    The MGP Kernel
    symmetrization : 'None', 'Arithmetic', 'Geometric'
    """
    # if symmetrization == "Try":
        # return MGPKernelTableSym(eta, q, maxpoints, symmetrization, KernelTable)
    TimeData.Begin("MGPKernelTable")
    dt = 1.0 / (maxpoints)
    cTF = 0.3 * (3.0 * np.pi ** 2) ** (2.0 / 3.0)
    # factor = 5.0 / (9.0 * alpha * beta * rho0 ** (alpha + beta - 5.0/3.0))*2*alpha
    coe = 4.0 / 5.0 * cTF * 5.0 / 6.0 * dt
    cWT = 4.0 / 5.0 * 0.3 * (3.0 * np.pi ** 2) ** (2.0 / 3.0)
    for i in range(1, maxpoints + 1):
        t = i * dt
        eta2 = eta / np.cbrt(t)
        # t16 = np.cbrt(np.cbrt(t))
        if symmetrization == "Geometric":
            t16 = t ** (-2.0 / 3.0)
        else:
            t16 = t ** (1.0 / 6.0)
        if KernelTable is not None:
            Gt = splev(eta2, KernelTable)
        else:
            Gt = LindhardFunction(eta2, 1.0, 1.0)
        if i == 1:
            kernel = Gt / t16
        else:
            kernel += Gt / t16
    if symmetrization == "Arithmetic":
        # kernel = kernel * (coe * 0.5) + WTKernelTable(eta) * (0.5 / (2.0 * 5.0/6.0))
        kernel = 0.5 * (kernel * coe + WTKernelTable(eta))
    elif symmetrization == "Geometric":
        kernel *= 2.0 * coe
    else:
        kernel *= coe
    TimeData.End("MGPKernelTable")
    return kernel

def WTKernel(q, rho0, x=1.0, y=1.0, alpha=5.0 / 6.0, beta=5.0 / 6.0):
    """
    The WT Kernel
    """
    tkf = 2.0 * (3.0 * rho0 * np.pi ** 2) ** (1.0 / 3.0)
    cTF = 0.3 * (3.0 * np.pi ** 2) ** (2.0 / 3.0)
    factor = 5.0 / (9.0 * alpha * beta * rho0 ** (alpha + beta - 5.0 / 3.0))
    factor *= cTF
    return LindhardFunction(q / tkf, x, y) * factor


def SMKernel(q, rho0, x=1.0, y=1.0, alpha=0.5, beta=0.5):
    """
    The SM Kernel
    """
    return WTKernel(q, rho0, x, y, alpha, beta)
    tkf = 2.0 * (3.0 * rho0 * np.pi ** 2) ** (1.0 / 3.0)
    cTF = 0.3 * (3.0 * np.pi ** 2) ** (2.0 / 3.0)
    factor = 1.0 / (2.0 * alpha ** 2 * rho0 ** (2.0 * alpha - 2))
    factor *= cTF
    return LindhardFunction(q / tkf, x, y) * factor


def WTKernelTable(eta, x=1.0, y=1.0, alpha=5.0 / 6.0, beta=5.0 / 6.0):
    """
    Tip : In this version, this is only work for alpha = beta = 5.0/6.0
    """
    # factor =1.2*np.pi**2/(3.0*np.pi**2)**(1.0/3.0)
    # factor = 5.0 / (9.0 * alpha * beta) * (0.3 * (3.0  *  np.pi ** 2) ** (2.0/3.0))
    factor = 4.0 / 5.0 * 0.3 * (3.0 * np.pi ** 2) ** (2.0 / 3.0)
    return LindhardFunction(eta, x, y) * factor


def WTKernelDerivTable(eta, x=1.0, y=1.0, alpha=5.0 / 6.0, beta=5.0 / 6.0):
    factor = 5.0 / (9.0 * alpha * beta)
    cTF = 0.3 * (3.0 * np.pi ** 2) ** (2.0 / 3.0)
    factor *= cTF
    return LindhardDerivative(eta, y) * factor


def LWTKernel(q, rho0, KernelTable, etamax=1000.0):
    """
    Create the LWT kernel for given rho0 and Kernel Table
    """
    tkf = 2.0 * (3.0 * np.pi ** 2 * rho0) ** (1.0 / 3.0)
    eta = q / tkf
    Kernel = np.empty_like(q)
    cond0 = eta < etamax
    cond1 = np.invert(cond0)
    limit = splev(etamax, KernelTable)
    Kernel[cond0] = splev(eta[cond0], KernelTable)
    Kernel[cond1] = limit
    return Kernel


def LWTKernelKf(q, kf, KernelTable, etamax=1000.0, out=None):
    """
    Create the LWT kernel for given kf and Kernel Table
    """
    TimeData.Begin("LWTKernelKf")
    eta = q / kf
    if out is not None:
        Kernel = out
    else:
        Kernel = np.empty_like(q)
    cond0 = eta < etamax
    cond1 = np.invert(cond0)
    if isinstance(KernelTable, tuple):
        limit = splev(etamax, KernelTable)
        if np.where(cond0)[0].size > 0 :
            Kernel[cond0] = splev(eta[cond0], KernelTable)
    elif isinstance(KernelTable, np.ndarray):
        limit = KernelTable[-1]
        deta = etamax / (np.size(KernelTable) - 1)
        index = np.around(eta[cond0] / deta)
        index = index.astype(np.int)
        Kernel[cond0] = KernelTable[index]
    else:
        raise AttributeError("Wrong type of KernelTable")
    Kernel[cond1] = limit
    if q[0, 0, 0] < ZERO :
        Kernel[0, 0, 0] = 0.0
    TimeData.End("LWTKernelKf")
    return Kernel


def MGPOmegaE(q, Ne=1, lumpfactor=0.2):
    """
    """
    c = 1.0
    b = None
    if isinstance(lumpfactor, list):
        a = lumpfactor[0]
        if len(lumpfactor) > 1 :
            b = lumpfactor[1]
        if len(lumpfactor) > 2 :
            c = lumpfactor[2]
    else :
        a = lumpfactor

    if b is None :
        a = float(a / Ne ** (2.0 / 3.0))
        b = a

    qflag = False
    if q[0, 0, 0] < ZERO :
        qflag = True
        q[0, 0, 0] = 1.0
    gg = q ** 2
    corr = 4 * np.pi * sp.erf(c * gg) ** 2 * a * np.exp(-gg * b) / gg
    # corr = 4 * np.pi * sp.erf(c * q) ** 2* a * np.exp(-gg * b) / gg + 0.001*np.exp(-0.1*(gg-1.5)**2)
    if qflag :
        q[0, 0, 0] = 0.0
        corr[0, 0, 0] = 0.0
    # Same as the formular in MGP
    corr /= 1.2
    return corr

def SmoothKernel(q, rho0, x=1.0, y=1.0, alpha=5.0 / 6.0, beta=5.0 / 6.0):
    """
    The WT Kernel
    """
    tkf = 2.0 * (3.0 * rho0 * np.pi ** 2) ** (1.0 / 3.0)
    cTF = 0.3 * (3.0 * np.pi ** 2) ** (2.0 / 3.0)
    factor = 5.0 / (9.0 * alpha * beta * rho0 ** (alpha + beta - 5.0 / 3.0))
    factor *= cTF
    # -----------------------------------------------------------------------
    coef = 4.0
    factor = factor * np.exp(-(q ** 2) / (coef ** 2 * (tkf / 2.0) ** 2))
    # -----------------------------------------------------------------------
    return LindhardFunction(q / tkf, x, y) * factor
