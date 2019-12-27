# Collection of Kinetic Energy Density Functionals
import numpy as np
from dftpy.kedf.tf import *
from dftpy.kedf.vw import *
from dftpy.kedf.wt import  *
from dftpy.kedf.lwt import *
from dftpy.kedf.fp import *
from dftpy.kedf.sm import *
from dftpy.kedf.mgp import *
from dftpy.kedf.gga import *

__all__  =  ['KEDFunctional']

KEDF_Dict = {
        'TF'         : TF,
        'vW'         : vW,
        'GGA'        : GGA,
        }

NLKEDF_Dict = {
        'WT'         : WT,
        'SM'         : SM,
        'FP'         : FP,
        'MGP'        : MGP,
        'MGPA'       : MGPA,
        'MGPG'       : MGPG,
        'LWT'        : LWT,
        'LMGP'       : LMGP,
        'LMGPA'      : LMGPA,
        'LMGPG'      : LMGPG,
        }

def KEDFunctional(rho, name = 'WT', calcType = 'Both', split = False, **kwargs):
    if name[:3] == 'GGA' :
        func = GGA
        k_str = kwargs['k_str'].upper()
        if k_str not in GGA_KEDF_list :
            raise AttributeError("%s GGA KEDF to be implemented" %k_str)
        else :
            OutFunctional = GGA(rho, functional=k_str, calcType = calcType,  **kwargs)
            OutFunctionalDict = {'GGA': OutFunctional}
    elif name in KEDF_Dict :
        func = KEDF_Dict[name]
        OutFunctional = func(rho, calcType = calcType,  **kwargs)
        OutFunctionalDict = {name : OutFunctional}
    elif name == 'x_TF_y_vW' or name == 'xTFyvW' :
        xTF = TF(rho, calcType = calcType,  **kwargs)
        yvW = vW(rho, calcType = calcType,  **kwargs)
        OutFunctional = xTF + yvW
        OutFunctional.name = name
        OutFunctionalDict = {'TF': xTF, 'vW': yvW}
    elif name in NLKEDF_Dict :
        func = NLKEDF_Dict[name]
        NL = func(rho, calcType = calcType,  **kwargs)
        x = kwargs.get('x', 1.0)
        alpha = kwargs.get('alpha', 5.0/6.0)
        beta = kwargs.get('beta', 5.0/6.0)
        if abs(alpha + beta - 5.0/3.0) < 1E-8 :
            s = 1.0 + (x - 1) * 20/(25 - (abs(beta - alpha)*3) ** 2) # only works when \alpha+\beta=5/3
        else :
            s = x
        xTF = TF(rho, x = s,  calcType = calcType)
        yvW = vW(rho, calcType = calcType,  **kwargs)
        OutFunctional = NL + xTF + yvW
        OutFunctional.name = name
        OutFunctionalDict = {'TF': xTF, 'vW': yvW, 'NL' : NL}
    else :
        raise AttributeError("%s KEDF to be implemented" %name)
    
    if split :
        return OutFunctionalDict
    else :
        return OutFunctional
