# Class handling functional evaluations
# functional class (output handler) in output

# local imports
from dftpy.field import DirectField
from dftpy.functional_output import Functional
from dftpy.semilocal_xc import PBE, LDA, LibXC
from dftpy.hartree import HartreeFunctional
# from dftpy.kedf import KEDFunctional
from dftpy.kedf import KEDF
from dftpy.pseudo import LocalPseudo
from dftpy.external_potential import ExternalPotential

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
        # if self.name not in self.FunctionalNameList:
        for name in self.name.split('+'):
            if name not in self.FunctionalNameList:
                print(name, " is not a valid Functional name")
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

    def __call__(self, rho, calcType=["E","V"], **kwargs):
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
        self._outfunctional = self.ComputeEnergyPotential(rho, calcType, **kwargs)
        return self._outfunctional

    @property
    def GetFunctional(self, rho, calcType =["E","V"]):
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

        self.FunctionalTypeList = ["XC", "KEDF", "PSEUDO", "HARTREE","EXT"]
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
        NLGGAList = ['NLGGA-' + item for item in KEDFNLNameList]
        KEDFNLNameList.extend(NLGGAList)
        HNameList = ["HARTREE"]
        PPNameList = ["PSEUDO"]
        EXTNameList = ["EXT"]

        self.FunctionalNameList = XCNameList + KEDFNameList + KEDFNLNameList + HNameList + PPNameList + EXTNameList

        if type is None:
            raise AttributeError("Must assign type to FunctionalClass")
        else:
            self.type = type

        if name is None:
            if type in ["HARTREE", "PSEUDO", "EXT"] :
                self.name = self.type
            else :
                raise AttributeError("Must assign name to FunctionalClass")
        else:
            self.name = name

        if not isinstance(self.optional_kwargs, dict):
            raise TypeError("optional_kwargs must be dict")

        if not self.CheckFunctional():
            raise Exception("Functional check failed")

        if self.name == 'PSEUDO' :
            if PSEUDO is None :
                self.PSEUDO = LocalPseudo(**kwargs)
            else :
                self.PSEUDO = PSEUDO
        elif self.name == 'EXT':
            self.EXT = ExternalPotential(**kwargs)

        if self.type == 'KEDF' :
            self.KEDF = KEDF(self.name, **kwargs)

    def ComputeEnergyPotential(self, rho, calcType=["E","V"], **kwargs):
        self.optional_kwargs.update(kwargs)
        if self.type == "KEDF":
            if self.name != "LIBXC_KEDF":
                return self.KEDF(rho, calcType=calcType, **self.optional_kwargs)
                # return KEDFunctional(rho, self.name, calcType=calcType, **self.optional_kwargs)
            else:
                k_str = self.optional_kwargs.get("k_str", "gga_k_lc94")
                return LibXC(density=rho, k_str=k_str, calcType=calcType)
        elif self.type == "XC":
            if self.name == "LDA":
                return LDA(rho, calcType=calcType)
            elif self.name == "PBE":
                return PBE(rho, calcType=calcType)
            elif self.name == "LIBXC_XC":
                x_str = self.optional_kwargs.get("x_str", "gga_x_pbe")
                c_str = self.optional_kwargs.get("c_str", "gga_c_pbe")
                return LibXC(density=rho, x_str=x_str, c_str=c_str, calcType=calcType)
        elif self.type == "HARTREE":
            return HartreeFunctional(density=rho, calcType=calcType)
        elif self.type == "PSEUDO":
            return self.PSEUDO(density=rho, calcType=calcType)
        elif self.type == "EXT":
            return self.EXT(density=rho, calcType=calcType)

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

     KE = FunctionalClass(type='KEDF',name='TF')
     XC = FunctionalClass(type='XC',name='LDA')
     PSEUDO = FunctionalClass(type='PSEUDO', kwargs)
     HARTREE = FunctionalClass(type='HARTREE')

     EnergyEvaluator = TotalEnergyAndPotential(
         KineticEnergyFunctional = KE,
         XCFunctional = XC,
         PSEUDO = PSEUDO,
         HARTREE = HARTREE
     )

     or given a dict
     funcdict = {
         "KineticEnegyFunctional": KE,
         "XCFunctional": XC,
         "PSEUDO": PSEUDO,
         "HARTREE": HARTREE
     }
     EnergyEvaluator = TotalEnergyAndPotential(**funcdict)

     [the energy:]
     E = EnergyEvaluator.Energy(rho,ions)

     [total energy and potential:]
     out = EnergyEvaluator.ComputeEnergyPotential(rho)

     [time for optimization of density:]
     in_for_scipy_minimize = EnergyEvaluator(phi)
    """

    def __init__(self, **kwargs):

        self.funcDict = {}
        self.funcDict.update(kwargs)
        # remove useless key
        for key, evalfunctional in self.funcDict.items():
            if evalfunctional is None:
                del self.funcDict[key]

        self.UpdateNameType()

    def __call__(self, rho, calcType=["E","V"],  **kwargs):
        return self.ComputeEnergyPotential(rho, calcType, **kwargs)

    def UpdateNameType(self):
        self.name = ""
        self.type = ""
        for key, evalfunctional in self.funcDict.items():
            if isinstance(evalfunctional, LocalPseudo):
                # This is a trick for PSEUDO
                if not hasattr(evalfunctional, 'name'):
                    setattr(evalfunctional, 'name', 'PSEUDO')
                if not hasattr(evalfunctional, 'type'):
                    setattr(evalfunctional, 'type', 'PSEUDO')
            elif isinstance(evalfunctional, ExternalPotential):
                pass
            elif not isinstance(evalfunctional, FunctionalClass):
                raise TypeError("{} must be FunctionalClass".format(key))
            setattr(self, key, evalfunctional)
            self.name += getattr(evalfunctional, 'name') + " "
            self.type += getattr(evalfunctional, 'type') + " "

    def UpdateFunctional(self, keysToRemove=[], newFuncDict={}):
        for key in keysToRemove:
            del self.funcDict[key]

        self.funcDict.update(newFuncDict)
        self.UpdateNameType()

    def Subset(self, keys):
        subdict = dict((key, self.funcDict[key]) for key in keys)
        return TotalEnergyAndPotential(**subdict)

    def ComputeEnergyPotential(self, rho, calcType=["E","V"]):
        Obj = None
        for key, evalfunctional in self.funcDict.items():
            if Obj is None :
                Obj = evalfunctional(rho, calcType)
                # print('key', key, Obj.energy)
            else :
                Obj += evalfunctional(rho, calcType)
                # sss = evalfunctional(rho, calcType)
                # print('key', key, sss.energy)
        if Obj is None :
            Obj = Functional(name = 'NONE')
        return Obj

    def Energy(self, rho, ions, usePME=False, calcType=["E"]):
        from .ewald import ewald

        ewald_ = ewald(rho=rho, ions=ions, PME=usePME)
        print('Ewald :', ewald_.energy)
        total_e = self.ComputeEnergyPotential(rho, calcType=["E"])
        return ewald_.energy + total_e.energy

