#!/usr/bin/env python3
import os
import unittest
import numpy as np
import dftpy.formats.io as dftpy_io
import pytest


class Test(unittest.TestCase):
    def test_io_ase(self):
        pytest.importorskip("ase")
        dftpy_data_path = os.environ.get('DFTPY_DATA_PATH')
        a1 = dftpy_io.read(dftpy_data_path + '/fcc.vasp', driver='ase')
        a2 = dftpy_io.read(dftpy_data_path + '/fcc.vasp')

        self.assertTrue(np.allclose(a1.cell, a2.cell))
        self.assertTrue(np.allclose(a1.positions, a2.positions))

    def test_io_pmg(self):
        pytest.importorskip("pymatgen")
        dftpy_data_path = os.environ.get('DFTPY_DATA_PATH')
        a1 = dftpy_io.read(dftpy_data_path + '/fcc.vasp', driver='pmg')
        a2 = dftpy_io.read(dftpy_data_path + '/fcc.vasp')

        self.assertTrue(np.allclose(a1.cell, a2.cell))
        self.assertTrue(np.allclose(a1.positions, a2.positions))


if __name__ == "__main__":
    unittest.main()
