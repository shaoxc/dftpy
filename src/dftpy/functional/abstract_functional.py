# Class handling functional evaluations
# functional class (output handler) in output

# local imports
# from dftpy.mpi import sprint

# general python imports
from abc import ABC, abstractmethod


class AbstractFunctional(ABC):
    @abstractmethod
    def __init__(self):
        pass

    def __repr__(self):
        rep = self.name + ', ' + self.__dict__.__repr__()
        return rep

    def __call__(self, rho, *args, **kwargs):
        return self.compute(rho, *args, **kwargs)

    @abstractmethod
    def compute(self, rho, *args, **kwargs):
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
