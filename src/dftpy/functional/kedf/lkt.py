import numpy as np

from dftpy.functional.kedf.vw import vW
from dftpy.functional.kedf.gga import calc_s
from dftpy.constants import C_TF, TKF0
from dftpy.math_utils import PowerInt
from dftpy.functional.functional_output import FunctionalOutput

def LKT(rho, calcType, a = 1.3, y = 1.0, sigma = None, **kwargs):
    rhom = rho.copy()
    tol = 1e-16
    rhom[rhom < tol] = tol
    rho23 = PowerInt(rhom, 2, 3)
    rho43 = rho23 * rho23
    rho53 = rho23 * rhom
    if sigma is None:
        gradient_flag = 'standard'
    else:
        gradient_flag = 'supersmooth'
    rho_sigma = rhom.sigma(flag=gradient_flag, sigma_gradient=sigma)
    rho_sigma[rho_sigma < tol] = tol
    sqrt_sigma = np.sqrt(rho_sigma)
    s = sqrt_sigma / rho43 / TKF0
    exp_1 = np.exp(-a * s)
    exp_2 = np.exp(-2.0 * a * s)
    F = 2.0 * exp_1 / (1.0 + exp_2)
    OutFunctional = FunctionalOutput(name='LKT')
    if 'E' in calcType:
        OutFunctional.energy = (C_TF * rho53 * F).integral(gather=False)

    if 'V' in calcType:
        dFds = - a * (1.0 - exp_2) / (1.0 + exp_2) * F
        dsdrho = -4.0 / 3.0 / TKF0 * sqrt_sigma / rho43 / rhom
        dsdsigma = 0.5 / TKF0 / sqrt_sigma / rho43
        vrho = 5.0 / 3.0 * C_TF * rho23 * F + C_TF * rho53 * dFds * dsdrho
        vsigma = C_TF * rho53 * dFds * dsdsigma
        OutFunctional.potential = vrho - 2 * (vsigma * rhom.gradient(flag=gradient_flag, sigma=sigma)).divergence(flag=gradient_flag, sigma=sigma)

    OutFunctional += vW(rho, y, sigma, calcType)

    return OutFunctional