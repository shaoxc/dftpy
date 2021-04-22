import numpy as np
from dftpy.functional.abstract_functional import AbstractFunctional
from dftpy.functional.functional_output import FunctionalOutput


class ExternalPotential(AbstractFunctional):

    def __init__(self, v=None):
        # if v is None:
        #    raise AttributeError("Must specify v")
        # else:
        self.name = 'EXT'
        self.type = 'EXT'
        self._v = v

    @property
    def v(self):
        return self._v

    @v.setter
    def v(self, new_v):
        self._v = new_v

    def compute(self, density, calcType={"E", "V"}):
        pot = self._v
        if 'E' in calcType:
            ene = np.einsum("ijk, ijk->", self._v, density) * self._v.grid.dV
        else:
            ene = 0
        return FunctionalOutput(name="ext", energy=ene, potential=pot)
