import numpy as np
from dftpy.field import DirectField
from dftpy.functional_output import Functional

class ExternalPotential(object):

    def __init__(self, v=None):
        #if v is None:
        #    raise AttributeError("Must specify v")
        #else:
        self.name = 'EXT'
        self.type = 'EXT'
        self._v = v

    @property
    def v(self):
        return self._v

    @v.setter
    def v(self, new_v):
        self._v = new_v

    def __call__(self, density=None, calcType=None):
        pot = self._v
        if 'E' in calcType:
            ene = np.einsum("ijk, ijk->", self._v, density) * self._v.grid.dV
        else:
            ene = 0
        return Functional(name="ext", energy=ene, potential=pot)
