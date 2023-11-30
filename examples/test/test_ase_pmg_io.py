#!/usr/bin/env python3
import os
import unittest
import numpy as np
from dftpy.formats import io
import pytest
from common import dftpy_data_path


class Test(unittest.TestCase):
    def test_io_ase(self):
        a1 = io.read(dftpy_data_path / 'fcc.vasp', driver='ase')
        a2 = io.read(dftpy_data_path / 'fcc.vasp')

        self.assertTrue(np.allclose(a1.cell, a2.cell))
        self.assertTrue(np.allclose(a1.positions, a2.positions))

    def test_io_pmg(self):
        pytest.importorskip("pymatgen")
        a1 = io.read(dftpy_data_path / 'fcc.vasp', driver='pmg')
        a2 = io.read(dftpy_data_path / 'fcc.vasp')

        self.assertTrue(np.allclose(a1.cell, a2.cell))
        self.assertTrue(np.allclose(a1.positions, a2.positions))


if __name__ == "__main__":
    unittest.main()
