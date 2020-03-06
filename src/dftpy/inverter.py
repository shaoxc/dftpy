from dftpy.field import DirectField
from dftpy.functionals import FunctionalClass, TotalEnergyAndPotential
from dftpy.external_potential import ExternalPotential
from dftpy.optimization import Optimization
from scipy.linalg import lu_factor, lu_solve
from dftpy.formats.xsf import XSF

import numpy as np

class Inverter(object):
    """
    Class handling invertions

    Attributes
    ----------

    """

    def __init__(self):
        pass

    def __call__(self, rho_in, EnergyEvaluator):

        phi = np.sqrt(rho_in)
        v = phi.laplacian(force_real=True) / phi / 2.0
        v_of = EnergyEvaluator(rho_in, calcType='Potential').potential
        vw = FunctionalClass(type='KEDF', name='vW')
        v_vw = vw(rho_in, calcType='Potential').potential
        v_ext = v - v_of + v_vw
        ext = ExternalPotential(v_ext)
        EnergyEvaluator.UpdateFunctional(newFuncDict={'EXT': ext})
        optimizer = Optimization(EnergyEvaluator=EnergyEvaluator, guess_rho=rho_in, optimization_options={'econv':1e-8})
        rho = optimizer.optimize_rho()

        return ext, rho

def linear_inverter(delta_rho, alpha):
    return alpha * delta_rho

def scaled_linear_inverter(delta_rho, rho_in, alpha=1.0):
    return alpha * delta_rho / rho_in

def build_error_matrix(e_list):
    num = len(e_list)
    b = np.empty((num+1,num+1))
    for i in range(num):
        for j in range(i, num):
            b[i,j] = (e_list[i]*e_list[j]).integral()
            if i != j:
                b[j,i] = b[i,j]

    for i in range(num):
        b[i,num] = 1
        b[num,i] = 1
    b[num,num] = 0
    c = np.zeros(num+1)
    c[num] = 1

    return b, c
