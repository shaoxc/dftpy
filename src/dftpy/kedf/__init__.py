# Collection of Kinetic Energy Density Functionals
import numpy as np
import copy
from dftpy.field import DirectField
from dftpy.kedf.tf import *
from dftpy.kedf.vw import *
from dftpy.kedf.wt import *
from dftpy.kedf.lwt import *
from dftpy.kedf.fp import *
from dftpy.kedf.sm import *
from dftpy.kedf.mgp import *
from dftpy.kedf.gga import *
from dftpy.functional_output import Functional

__all__ = ["KEDF", "KEDFunctional", "KEDFStress"]

KEDF_Dict = {
    "TF": TF,
    "vW": vW,
    "GGA": GGA,
}

NLKEDF_Dict = {
    "WT": WT,
    "SM": SM,
    "FP": FP,
    "MGP": MGP,
    "MGPA": MGPA,
    "MGPG": MGPG,
    "LWT": LWT,
    "LMGP": LMGP,
    "LMGPA": LMGPA,
    "LMGPG": LMGPG,
}

KEDF_Stress_Dict = {
    "TF": ThomasFermiStress, 
    "vW": vonWeizsackerStress, 
    "WT": WTStress, 
}

class KEDF:
    def __init__(self, name = "WT", **kwargs):
        self.name = name
        self.kwargs = kwargs
        self.ke_kernel_saved = {
                "Kernel": None,
                "rho0": 0.0,
                "shape": None,
                "KernelTable": None,
                "etamax": None,
                "KernelDeriv": None,
                "MGPKernelE": None,
                "kfmin" :None, 
                "kfmax" :None, 
                }

    def __call__(self, density, calcType=["E","V"], name=None, **kwargs):
        if name is None :
            name = self.name
        return self.compute(density, name=name, calcType=calcType, **kwargs)

    def compute(self, density, calcType=["E","V"], name=None, split = False, **kwargs):
        if name is None :
            name = self.name
        ke_kwargs = copy.deepcopy(self.kwargs)
        ke_kwargs.update(kwargs)
        functional = None
        for item in name.split('+'):
            out = KEDFunctional(density, name = item, calcType = calcType, split = split, ke_kernel_saved = self.ke_kernel_saved, **ke_kwargs)
            if functional is None :
                functional = out
            else :
                if split :
                    functional.update(out)
                else :
                    functional += out
        return functional

def KEDFunctional(rho, name="WT", calcType=["E","V"], split=False, nspin = 1, **kwargs):
    """
    KEDF interface
    """
    if rho.ndim > 3 :
        nspin = rho.rank
        ke = KEDFunctional(rho[0] * nspin, name, calcType, split, nspin = nspin, **kwargs)
        if split :
            for i in range(1, nspin):
                ke1 = KEDFunctional(rho[i] * nspin, name, calcType, split, nspin = nspin, **kwargs)
                for k1 in ke :
                    if 'E' in calcType :
                        ke[k1].energy = ke[k1].energy + ke1[k1].energy
                    if 'V' in calcType :
                        ke[k1].potential = np.vstack((ke[k1].potential, ke1[k1].potential))
            for k1 in ke :
                if 'V' in calcType :
                    ke[k1].potential = ke[k1].potential.reshape(rho.shape)
        else :
            for i in range(1, nspin):
                ke1 = KEDFunctional(rho[i] * nspin, name, calcType, split, nspin = nspin, **kwargs)
                if 'E' in calcType :
                    ke.energy = ke.energy + ke1.energy
                if 'V' in calcType :
                    ke.potential = np.vstack((ke.potential, ke1.potential))
            if 'V' in calcType :
                ke.potential = DirectField(grid=rho.grid, griddata_3d=ke.potential, rank=nspin)
        return ke
    #-----------------------------------------------------------------------
    if name[:3] == "GGA":
        func = GGA
        k_str = kwargs["k_str"].upper()
        if k_str not in GGA_KEDF_list:
            raise AttributeError("%s GGA KEDF to be implemented" % k_str)
        else:
            OutFunctional = func(rho, functional=k_str, calcType=calcType, **kwargs)
            OutFunctionalDict = {"GGA": OutFunctional}
    elif name in KEDF_Dict:
        func = KEDF_Dict[name]
        OutFunctional = func(rho, calcType=calcType, **kwargs)
        if nspin > 1 and 'E' in calcType :
            OutFunctional.energy /= nspin
        OutFunctionalDict = {name: OutFunctional}
    elif name == "x_TF_y_vW" or name == "TFvW":
        xTF = TF(rho, calcType=calcType, **kwargs)
        yvW = vW(rho, calcType=calcType, **kwargs)
        if nspin > 1 and 'E' in calcType :
            xTF.energy /= nspin
            yvW.energy /= nspin
        OutFunctional = xTF + yvW
        OutFunctional.name = name
        OutFunctionalDict = {"TF": xTF, "vW": yvW}
    elif name in NLKEDF_Dict:
        func = NLKEDF_Dict[name]
        NL = func(rho, calcType=calcType, **kwargs)
        x = kwargs.get("x", 1.0)
        y = kwargs.get("y", 1.0)
        alpha = kwargs.get("alpha", 5.0 / 6.0)
        beta = kwargs.get("beta", 5.0 / 6.0)
        if abs(alpha + beta - 5.0 / 3.0) < 1e-8:
            s = 1.0 + (x - 1) * 20 / (25 - (abs(beta - alpha) * 3) ** 2)  # only works when \alpha+\beta=5/3
        else:
            s = x
        if abs(s) > 1e-8 :
            xTF = TF(rho, x=s, calcType=calcType)
        else :
            xTF = Functional(name="ZERO", energy = 0.0)

        if abs(y) > 1e-8 :
            yvW = vW(rho, calcType=calcType, **kwargs)
        else :
            yvW = Functional(name="ZERO", energy = 0.0)
        if nspin > 1 and 'E' in calcType :
            xTF.energy /= nspin
            yvW.energy /= nspin
            NL.energy /= nspin
        OutFunctional = NL + xTF + yvW
        OutFunctional.name = name
        OutFunctionalDict = {"TF": xTF, "vW": yvW, "NL": NL}
    elif name[:5] == 'NLGGA' and name[6:] in NLKEDF_Dict:
        func = NLKEDF_Dict[name[6:]]
        NL = func(rho, calcType=calcType, **kwargs)
        k_str = kwargs["k_str"].upper()
        if k_str in GGA_KEDF_list:
            gga = GGA(rho, functional=k_str, calcType=calcType, **kwargs)
        else :
            raise AttributeError("%s GGA KEDF to be implemented" % k_str)
        if nspin > 1 and 'E' in calcType :
            gga.energy /= nspin
            NL.energy /= nspin
        OutFunctional = NL + gga
        OutFunctional.name = name
        OutFunctionalDict = {"GGA": gga, "NL": NL}
    else:
        raise AttributeError("%s KEDF to be implemented" % name)
    #-----------------------------------------------------------------------
    # rho_min = kwargs.get("rho_min", None)
    # if rho_min is not None and 'V' in calcType :
        # OutFunctional.potential[rho < rho_min] = 0.0
    #-----------------------------------------------------------------------

    if split:
        return OutFunctionalDict
    else:
        return OutFunctional


def KEDFStress(rho, name="WT", energy=None, **kwargs):
    """
    KEDF stress interface
    """
    if name in KEDF_Stress_Dict :
        func = KEDF_Stress_Dict[name]

    if rho.ndim > 3 :
        nspin = rho.rank
        stress = func(rho[0] * nspin, energy=energy, **kwargs)
        for i in range(1, nspin):
            stress += func(rho[i] * nspin, energy=energy, **kwargs)
        stress /= nspin
    else :
        stress = func(rho, energy=energy, **kwargs)

    return stress
