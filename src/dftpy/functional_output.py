# Class handling output of functional evaluations
from dftpy.field import DirectField, ReciprocalField

# general python imports
import numpy as np


class Functional(object):
    """
    Object handling DFT functional output
    
    Attributes
    ----------
    name: string
        The (optional) name of the functional

    energy: float
        The energy

    potential: DirectField
        The first functional derivative of the functional wrt 
        the electron density 
        
    kernel: ReciprocalField
        The value of the reciprocal space kernel. This will
        be populated only if the functional is nonlocal
    """

    def __init__(self, name=None, energy=None, potential=None, energydensity=None, kernel=None):

        if name is not None:
            self.name = name
        else:
            raise AttributeError("Functional name must be specified")

        if energy is not None:
            self.energy = energy
        if potential is not None:
            # if isinstance(potential, DirectField):
            self.potential = potential
        # if energydensity is not None :
        # self.energydensity = energydensity
        if kernel is not None:
            if isinstance(kernel, (np.ndarray)):
                self.kernel = kernel

    def sum(self, other):
        energy = self.energy + other.energy
        potential = self.potential + other.potential
        name = self.name + other.name
        return Functional(name=name, energy=energy, potential=potential)

    def mul(self, x):
        energy = x * self.energy
        potential = x * self.potential
        name = self.name
        return Functional(name=name, energy=energy, potential=potential)

    def div(self, x):
        energy = self.energy/x
        potential = self.potential/x
        name = self.name
        return Functional(name=name, energy=energy, potential=potential)

    def __add__(self, other):
        return self.sum(other)

    def __mul__(self, x):
        return self.mul(x)

    def __div__(self, x):
        return self.div(x)

    def __truediv__(self, x):
        return self.div(x)

    def copy(self):
        energy = self.energy
        potential = self.potential.copy()
        name = self.name
        return Functional(name=name, energy=energy, potential=potential)
