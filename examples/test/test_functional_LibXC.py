#!/usr/bin/env python3
import os
import unittest
import numpy as np
import pytest

from dftpy.functional import Functional
from dftpy.functional.semilocal_xc import LibXC, PBE
from dftpy.formats import io


class Test(unittest.TestCase):
    def test_libxc_lda(self):
        pytest.importorskip("pylibxc")
        dftpy_data_path = os.environ.get('DFTPY_DATA_PATH')
        rho_r = io.read_density(dftpy_data_path + "/Al_fde_rho.pp")
        thefuncclass = Functional(type='XC',
                                  name='LDA',
                                  libxc=False)
        func2 = thefuncclass.compute(rho_r)
        func1 = LibXC(density=rho_r,
                   x_str='lda_x',
                   c_str='lda_c_pz')
        a = func2.energy
        b = func1.energy
        self.assertTrue(np.allclose(a, b))

    def test_libxc_pbe(self):
        pytest.importorskip("pylibxc")
        dftpy_data_path = os.environ.get('DFTPY_DATA_PATH')
        rho_r = io.read_density(dftpy_data_path + "/Al_fde_rho.pp")
        Functional_LibXC = LibXC(density=rho_r,
                              x_str='gga_x_pbe',
                              c_str='gga_c_pbe')
        Functional_LibXC2 = PBE(rho_r)
        self.assertTrue(
            np.isclose(Functional_LibXC2.energy, Functional_LibXC.energy))


if __name__ == "__main__":
    unittest.main()
