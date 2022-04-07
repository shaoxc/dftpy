import numpy as np

from dftpy.functional.abstract_functional import AbstractFunctional
from dftpy.functional.functional_output import FunctionalOutput
from dftpy.field import DirectField
from dftpy.math_utils import PowerInt


class TestPotential(AbstractFunctional):

    def __init__(self, rho0: DirectField=None, w: float = None) -> None:
        # if v is None:
        #    raise AttributeError("Must specify v")
        # else:
        self.name = 'RST'
        self.type = 'EXT'
        self.rho0 = rho0
        self.w = w

    def __repr__(self):
        return 'EXT'

    def compute(self, rho, calcType={"E", "V"}, **kwargs):
        pot = self.w / 2.0 * PowerInt((rho - self.rho0), 2)
        ene = 0
        return FunctionalOutput(name="ext", energy=ene, potential=pot)
