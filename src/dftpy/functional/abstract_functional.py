# Class handling functional evaluations
# functional class (output handler) in output

# local imports
# from dftpy.mpi import sprint
from dftpy.functional.functional_output import FunctionalOutput

# general python imports
from abc import ABC, abstractmethod


class AbstractFunctional(ABC):
    @abstractmethod
    def __init__(self):
        pass

    def __call__(self, rho, **kwargs):
        return self.compute(rho, **kwargs)

    @abstractmethod
    def compute(self, rho, **kwargs):
        # returns energy and potential
        pass

    # def GetName(self):
    #     return self.name
    #
    # def GetType(self):
    #     return self.type
    #
    # def AssignName(self, name):
    #     self.name = name
    #
    # def AssignType(self, type):
    #     self.type = type

    # def CheckFunctional(self):
    #     if self.type not in self.FunctionalTypeList:
    #         print(self.type, " is not a valid Functional type")
    #         print("Valid Functional types are:")
    #         print(self.FunctionalTypeList)
    #         return False
    #     # if self.name not in self.FunctionalNameList:
    #     for name in self.name.split('+'):
    #         if name not in self.FunctionalNameList:
    #             print(name, " is not a valid Functional name")
    #             print("Valid Functional names are:")
    #             print(self.FunctionalNameList)
    #             return False
    #     return True


# class Functional(AbstractFunctional):
#     """
#     Object handling evaluation of a DFT functional
#
#     Attributes
#     ----------
#     name: string
#         The name of the functional
#
#     type: string
#         The functional type (XC, KEDF, HARTREE, PSEUDO)
#
#     is_nonlocal: logical
#         Is the functional a nonlocal functional?
#
#     optional_kwargs: dict
#         set of kwargs for the different functional types/names
#
#
#     Example
#     -------
#      XC = Functional(type='XC',name='LDA')
#      outXC = XC(rho)
#      outXC.energy --> the energy
#      outXC.potential     --> the pot
#     """
#
#     def __call__(self, rho, calcType={"E","V"}, **kwargs):
#         """
#         Functional class is callable
#
#         Attributes
#         ----------
#           rho: DirectField
#              The input density
#
#         Returns
#         -------
#           Functional: functional output handler
#              The output is a Functional class
#         """
#         self._outfunctional = self.compute(rho, calcType, **kwargs)
#         return self._outfunctional
#
#     @property
#     def GetFunctional(self, rho, calcType ={"E","V"}):
#         if self._outfunctional is None:
#             self._outfunctional = self.compute(rho, calcType)
#         return self._outfunctional
#
#     def __init__(self, type=None, name=None, PSEUDO = None, is_nonlocal=None, optional_kwargs=None, **kwargs):
#         # init the class
#
#         # This is compatible for PSEUDO Functional
#
#         if optional_kwargs is None:
#             self.optional_kwargs = {}
#         else:
#             self.optional_kwargs = optional_kwargs
#         self.optional_kwargs.update(kwargs)
#
#         self.FunctionalNameList = []
#         self.FunctionalTypeList = []
#
#         self.FunctionalTypeList = ["XC", "KEDF", "PSEUDO", "HARTREE","EXT"]
#         XCNameList = ["LDA", "PBE", "LIBXC_XC", "CUSTOM_XC"]
#         KEDFNameList = ["TF", "vW", "x_TF_y_vW", "LC94", "revAPBEK", "TFvW", "LIBXC_KEDF", "CUSTOM_KEDF"]
#         KEDFNLNameList = [
#             "WT",
#             "MGP",
#             "MGP0",
#             "WGC2",
#             "WGC1",
#             "WGC0",
#             "LMGP",
#             "LMGP0",
#             "LWT",
#             "FP",
#             "SM",
#             "MGPA",
#             "MGPG",
#             "LMGP0",
#             "LMGPA",
#             "LMGPG",
#             "GGA",
#         ]
#         NLGGAList = ['NLGGA-' + item for item in KEDFNLNameList]
#         KEDFNLNameList.extend(NLGGAList)
#         HNameList = ["HARTREE"]
#         PPNameList = ["PSEUDO"]
#         EXTNameList = ["EXT"]
#
#         self.FunctionalNameList = XCNameList + KEDFNameList + KEDFNLNameList + HNameList + PPNameList + EXTNameList
#
#         if type is None:
#             raise AttributeError("Must assign type to Functional")
#         else:
#             self.type = type
#
#         if name is None:
#             if type in ["HARTREE", "PSEUDO", "EXT"] :
#                 self.name = self.type
#             else :
#                 raise AttributeError("Must assign name to Functional")
#         else:
#             self.name = name
#
#         if not isinstance(self.optional_kwargs, dict):
#             raise TypeError("optional_kwargs must be dict")
#
#         if not self.CheckFunctional():
#             raise Exception("Functional check failed")
#
#         if self.name == 'PSEUDO' :
#             if PSEUDO is None :
#                 self.PSEUDO = LocalPseudo(**kwargs)
#             else :
#                 self.PSEUDO = PSEUDO
#         elif self.name == 'EXT':
#             self.EXT = ExternalPotential(**kwargs)
#
#         if self.type == 'KEDF' :
#             self.KEDF = KEDF(self.name, **kwargs)
#
#     def compute(self, rho, calcType={"E","V"}, **kwargs):
#         self.optional_kwargs.update(kwargs)
#         if self.type == "KEDF":
#             return self.KEDF(rho, calcType=calcType, **self.optional_kwargs)
#         elif self.type == "XC":
#             if self.name == "LDA":
#                 return LDA(rho, calcType=calcType)
#             elif self.name == "PBE":
#                 return PBE(rho, calcType=calcType)
#             elif self.name == "LIBXC_XC":
#                 x_str = self.optional_kwargs.get("x_str", "gga_x_pbe")
#                 c_str = self.optional_kwargs.get("c_str", "gga_c_pbe")
#                 return LibXC(density=rho, x_str=x_str, c_str=c_str, calcType=calcType)
#         elif self.type == "HARTREE":
#             return HartreeFunctional(density=rho, calcType=calcType)
#         elif self.type == "PSEUDO":
#             return self.PSEUDO(density=rho, calcType=calcType)
#         elif self.type == "EXT":
#             return self.EXT(density=rho, calcType=calcType)
#
#     def force(self, rho, **kwargs):
#         if self.type != 'PSEUDO' :
#             raise AttributeError("Only PSEUDO Functional have force property")
#         return self.PSEUDO.force(rho)
#
#     def stress(self, rho, energy=None, **kwargs):
#         if self.type == 'PSEUDO' :
#             return self.PSEUDO.stress(rho, energy=energy, **kwargs)
#         else :
#             raise AttributeError("Only PSEUDO Functional have stress property, others will implemented later")



