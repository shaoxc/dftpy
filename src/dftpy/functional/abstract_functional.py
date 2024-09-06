# Class handling functional evaluations
# functional class (output handler) in output

# local imports
# from dftpy.mpi import sprint

# general python imports
from abc import ABC, abstractmethod
from dftpy.functional.functional_output import FunctionalOutput


class AbstractFunctional(ABC):
    @abstractmethod
    def __init__(self):
        pass

    def __repr__(self):
        name = getattr(self, 'name', self.__class__.__name__)
        rep = name + ', ' + self.__dict__.__repr__()
        return rep

    def __call__(self, rho, *args, **kwargs):
        return self.compute(rho, *args, **kwargs)

    @abstractmethod
    def compute(self, rho, *args, **kwargs):
        # returns energy and potential
        pass


