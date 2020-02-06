# Class handling functional evaluations
# functional class (output handler) in output

# local imports
from dftpy.field import DirectField
from dftpy.functional_output import Functional
from dftpy.semilocal_xc import PBE, LDA, XC, LIBXC_KEDF
from dftpy.hartree import HartreeFunctional
from dftpy.kedf import KEDFunctional
from dftpy.pseudo import LocalPseudo

# general python imports
from abc import ABC, abstractmethod
import numpy as np


class AbstractFunctional(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, rho, **kwargs):
        # call the XC and such... depending on kwargs
        # Interface for scipy.optimize
        pass

    @abstractmethod
    def ComputeEnergyPotential(self, rho, **kwargs):
        # returns energy and potential
        pass

    def GetName(self):
        return self.name

    def GetType(self):
        return self.type

    def AssignName(self, name):
        self.name = name

    def AssignType(self, type):
        self.type = type

    def CheckFunctional(self):
        if self.type not in self.FunctionalTypeList:
            print(self.type, " is not a valid Functional type")
            print("Valid Functional types are:")
            print(self.FunctionalTypeList)
            return False
        if self.name not in self.FunctionalNameList:
            print(self.name, " is not a valid Functional name")
            print("Valid Functional names are:")
            print(self.FunctionalNameList)
            return False
        return True


class FunctionalClass(AbstractFunctional):
    """
    Object handling evaluation of a DFT functional
    
    Attributes
    ----------
    name: string
        The name of the functional

    type: string
        The functional type (XC, KEDF, HARTREE, PSEUDO) 

    is_nonlocal: logical
        Is the functional a nonlocal functional? 
        
    optional_kwargs: dict
        set of kwargs for the different functional types/names

 
    Example
    -------
     XC = FunctionalClass(type='XC',name='LDA')
     outXC = XC(rho)
     outXC.energy --> the energy
     outXC.potential     --> the pot
    """

    def __call__(self, rho, calcType="Both"):
        """
        Functional class is callable

        Attributes 
        ----------  
          rho: DirectField
             The input density

        Returns
        -------
          Functional: functional output handler
             The output is a Functional class
        """
        self._outfunctional = self.ComputeEnergyPotential(rho, calcType)
        return self._outfunctional

    @property
    def GetFunctional(self, rho, calcType = 'Both'):
        if self._outfunctional is None:
            self._outfunctional = self.ComputeEnergyPotential(rho, calcType)
        return self._outfunctional

    def __init__(self, type=None, name=None, PSEUDO = None, is_nonlocal=None, optional_kwargs=None, **kwargs):
        # init the class

        # This is compatible for PSEUDO FunctionalClass 

        if optional_kwargs is None:
            self.optional_kwargs = {}
        else:
            self.optional_kwargs = optional_kwargs
        self.optional_kwargs.update(kwargs)

        self.FunctionalNameList = []
        self.FunctionalTypeList = []

        self.FunctionalTypeList = ["XC", "KEDF", "PSEUDO", "HARTREE"]
        XCNameList = ["LDA", "PBE", "LIBXC_XC", "CUSTOM_XC"]
        KEDFNameList = ["TF", "vW", "x_TF_y_vW", "LC94", "revAPBEK", "TFvW", "LIBXC_KEDF", "CUSTOM_KEDF"]
        KEDFNLNameList = [
            "WT",
            "MGP",
            "MGP0",
            "WGC2",
            "WGC1",
            "WGC0",
            "LMGP",
            "LMGP0",
            "LWT",
            "FP",
            "SM",
            "MGPA",
            "MGPG",
            "LMGP0",
            "LMGPA",
            "LMGPG",
            "GGA",
        ]
        HNameList = ["HARTREE"]
        PPNameList = ["PSEUDO"]

        self.FunctionalNameList = XCNameList + KEDFNameList + KEDFNLNameList + HNameList + PPNameList

        if type is None:
            raise AttributeError("Must assign type to FunctionalClass")
        else:
            self.type = type

        if name is None:
            if type in ["HARTREE", "PSEUDO"] :
                self.name = self.type
            else :
                raise AttributeError("Must assign name to FunctionalClass")
        else:
            self.name = name

        if not isinstance(self.optional_kwargs, dict):
            raise AttributeError("optional_kwargs must be dict")

        if not self.CheckFunctional():
            raise Exception("Functional check failed")

        if self.name == 'PSEUDO' :
            if PSEUDO is None :
                self.PSEUDO = LocalPseudo(**kwargs)
            else :
                self.PSEUDO = PSEUDO

    def ComputeEnergyPotential(self, rho, calcType="Both", **kwargs):
        self.optional_kwargs.update(kwargs)
        if self.type == "KEDF":
            if self.name != "LIBXC_KEDF":
                return KEDFunctional(rho, self.name, calcType=calcType, **self.optional_kwargs)
            else:
                polarization = self.optional_kwargs.get("polarization", "unpolarized")
                k_str = self.optional_kwargs.get("k_str", "gga_k_lc94")
                return LIBXC_KEDF(density=rho, k_str=k_str, polarization=polarization, calcType=calcType)
        elif self.type == "XC":
            if self.name == "LDA":
                polarization = self.optional_kwargs.get("polarization", "unpolarized")
                return LDA(rho, polarization=polarization, calcType=calcType)
            if self.name == "PBE":
                polarization = self.optional_kwargs.get("polarization", "unpolarized")
                return PBE(density=rho, polarization=polarization, calcType=calcType)
            if self.name == "LIBXC_XC":
                polarization = self.optional_kwargs.get("polarization", "unpolarized")
                x_str = self.optional_kwargs.get("x_str", "gga_x_pbe")
                c_str = self.optional_kwargs.get("c_str", "gga_c_pbe")
                return XC(density=rho, x_str=x_str, c_str=c_str, polarization=polarization, calcType=calcType)
        elif self.type == "HARTREE":
            return HartreeFunctional(density=rho, calcType=calcType)
        elif self.type == "PSEUDO":
            return self.PSEUDO(density=rho, calcType=calcType)

    def force(self, rho, **kwargs):
        if self.type != 'PSEUDO' :
            raise AttributeError("Only PSEUDO Functional have force property")
        return self.PSEUDO.force(rho)

    def stress(self, rho, energy=None, **kwargs):
        if self.type == 'PSEUDO' :
            return self.PSEUDO.stress(rho, energy=energy, **kwargs)
        else :
            raise AttributeError("Only PSEUDO Functional have stress property, others will implemented later")


class TotalEnergyAndPotential(AbstractFunctional):
    """
     Object handling energy evaluation for the 
     purposes of optimizing the electron density
     
     Attributes
     ----------

     KineticEnergyFunctional, XCFunctional, PSEUDO, HARTREE: FunctionalClass
         Instances of functional class needed for the computation
         of the chemical potential, total potential and total energy.

     Example
     -------

     XC = FunctionalClass(type='XC',name='LDA')
     KE = FunctionalClass(type='KEDF',name='TF')
     HARTREE = FunctionalClass(type='HARTREE')
     PSEUDO = FunctionalClass(type='PSEUDO', kwargs)

     EnergyEvaluator = TotalEnergyAndPotential(KEDF,XC,PSEUDO,HARTREE)
     or given a dict
     EnergyEvaluator = TotalEnergyAndPotential()

     [the energy:]
     E = EnergyEvaluator.Energy(rho,ions)
     
     [total energy and potential:]
     out = EnergyEvaluator.ComputeEnergyPotential(rho)

     [time for optimization of density:]
     in_for_scipy_minimize = EnergyEvaluator(phi)
    """

    def __init__(self, KineticEnergyFunctional=None, XCFunctional=None, PSEUDO=None, HARTREE=None, **kwargs):

        self.name = ""
        self.type = ""
        funcDict = {}
        funcDict['KineticEnergyFunctional'] = KineticEnergyFunctional
        funcDict['XCFunctional'] = XCFunctional
        funcDict['PSEUDO'] = PSEUDO
        funcDict['HARTREE'] = HARTREE
        funcDict.update(kwargs)
        # remove useless key
        keys = list(funcDict.keys())
        for key in keys :
            if funcDict[key] is None :
                del funcDict[key]

        self.funcDict = funcDict

        for key, evalfunctional in self.funcDict.items():
            if isinstance(evalfunctional, LocalPseudo):
                # This is a trick for PSEUDO
                if not hasattr(evalfunctional, 'name'):
                    setattr(evalfunctional, 'name', 'PSEUDO')
                if not hasattr(evalfunctional, 'type'):
                    setattr(evalfunctional, 'type', 'PSEUDO')
            elif not isinstance(evalfunctional, FunctionalClass):
                raise AttributeError("{} must be FunctionalClass".format(key))
            setattr(self, key, evalfunctional)
            self.name += getattr(evalfunctional, 'name') + " "
            self.type += getattr(evalfunctional, 'type') + " "

    def __call__(self, rho, calcType="Both"):
        return self.ComputeEnergyPotential(rho, calcType)

    def ComputeEnergyPotential(self, rho, calcType="Both"):
        Obj = None
        for key, evalfunctional in self.funcDict.items():
            if Obj is None :
                Obj = evalfunctional(rho, calcType)
            else :
                Obj += evalfunctional(rho, calcType)
        return Obj

    def Energy(self, rho, ions, usePME=False, calcType="Energy"):
        from .ewald import ewald

        ewald_ = ewald(rho=rho, ions=ions, PME=usePME)
        total_e = self.ComputeEnergyPotential(rho, calcType="Energy")
        return ewald_.energy + total_e.energy

class TotalEnergyAndPotentialOld(AbstractFunctional):

    def __init__(self, KineticEnergyFunctional=None, XCFunctional=None, PSEUDO=None, HARTREE=None):

        self.name = ""
        self.type = ""

        if KineticEnergyFunctional is None:
            raise AttributeError("Must define KineticEnergyFunctional")
        elif not isinstance(KineticEnergyFunctional, FunctionalClass):
            raise AttributeError("KineticEnergyFunctional must be FunctionalClass")
        else:
            self.KineticEnergyFunctional = KineticEnergyFunctional
            self.name += self.KineticEnergyFunctional.name + " "
            self.type += self.KineticEnergyFunctional.type + " "

        if XCFunctional is None:
            raise AttributeError("Must define XCFunctional")
        elif not isinstance(XCFunctional, FunctionalClass):
            raise AttributeError("XCFunctional must be FunctionalClass")
        else:
            self.XCFunctional = XCFunctional
            self.name += self.XCFunctional.name + " "
            self.type += self.XCFunctional.type + " "

        if PSEUDO is None:
            raise AttributeError("Must define PSEUDO")
        # elif not isinstance(PSEUDO, FunctionalClass):
        # raise AttributeError('PSEUDO must be FunctionalClass')
        else:
            self.PSEUDO = PSEUDO
            # self.name += self.PSEUDO.name + ' '
            # self.type += self.PSEUDO.type + ' '

        if HARTREE is None:
            print("WARNING: using FFT Hartree")
            self.HARTREE = HARTREE
        else:
            self.HARTREE = HARTREE
            self.name += self.HARTREE.name + " "
            self.type += self.HARTREE.type + " "

    def __call__(self, rho, calcType="Both"):
        return self.ComputeEnergyPotential(rho, calcType)

    def ComputeEnergyPotential(self, rho, calcType="Both"):
        Obj = (
            self.KineticEnergyFunctional(rho, calcType)
            + self.XCFunctional(rho, calcType)
            + self.PSEUDO(rho, calcType)
            + self.HARTREE(rho, calcType)
        )
        return Obj

    def Energy(self, rho, ions, usePME=False, calcType="Energy"):
        from .ewald import ewald

        ewald_ = ewald(rho=rho, ions=ions, PME=usePME)
        total_e = self.ComputeEnergyPotential(rho, calcType="Energy")
        return ewald_.energy + total_e.energy
