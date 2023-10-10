#!/usr/bin/env python3
import unittest
import numpy as np
import pytest

from dftpy.functional.xc import RVV10, RVV10NL
from dftpy.formats import io
from common import dftpy_data_path


class Test(unittest.TestCase):
    def test_rvv10nl(self):
        rho = io.read_density(dftpy_data_path / "al_random.xsf", ecut=80)
        func = RVV10NL()
        obj = func(rho)
        print(obj.energy, obj.potential[0, 0, 0])
        self.assertTrue(np.isclose(obj.energy*2, 0.146613532321586)) # QE NL energy
        self.assertTrue(np.isclose(obj.potential[0, 0, 0]*2, -1.246297010609865E-002)) # QE NL potential

    def test_rvv10(self):
        pytest.importorskip("pylibxc")
        rho = io.read_density(dftpy_data_path / "al_random.xsf")
        rho.grid.ecut = 80
        func = RVV10()
        obj = func(rho)
        self.assertTrue(np.isclose(obj.energy*2, -31.10112047)) # QE energy


if __name__ == "__main__":
    unittest.main()
