# Class handling output of functional evaluations
from dftpy.field import DirectField, ReciprocalField

# general python imports
import numpy as np
import copy


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

    def __init__(self, name=None, energy=None, potential=None, energydensity=None, v2rho2=None, v3rho3=None, v4rho4=None, force=None, stress=None):

        args = locals()

        for key, value in args.items():
            if value is not None:
                setattr(self, key, value)
            elif key == "name":
                raise AttributeError("Functional name must be specified")

    def __iter__(self):
        attr_list = [
        'energy',
        'potential',
        'energydensity',
        'v2rho2',
        'v3rho3',
        'v4rho4',
        'force',
        'stress',
        ]
        for key in attr_list:
            if hasattr(self, key):
                yield key, getattr(self, key)

    def sum(self, other):
        if self.name == other.name:
            name = self.name
        else:
            name = self.name + other.name
        result = Functional(name=name)
        for key, value in self:
            if hasattr(other, key):
                setattr(result, key, value + getattr(other, key))
            else:
                setattr(result, key, value)
        for key, value in other:
            if not hasattr(result, key):
                setattr(result, key, value)
        return result

    def mul(self, x):
        result = self.copy()
        for key, value in result:
            setattr(result, key, value*x)
        return result

    def div(self, x):
        if x == 0:
            raise ValueError("Dividing zero")
        result = self.copy()
        for key, value in result:
            setattr(result, key, value/x)
        return result

    def __add__(self, other):
        return self.sum(other)

    def __mul__(self, x):
        return self.mul(x)

    def __truediv__(self, x):
        return self.div(x)

    def copy(self):
        return copy.deepcopy(self)
