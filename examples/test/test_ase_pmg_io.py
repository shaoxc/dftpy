#!/usr/bin/env python3
import os
import unittest
import numpy as np
# from numpy.testing import assert_allclose
import dftpy.formats.ase_io as ase_io
import dftpy.formats.pmg_io as pmg_io
import dftpy.formats.io as dftpy_io


class Test(unittest.TestCase):
    def test_io(self):
        dftpy_data_path = os.environ.get('DFTPY_DATA_PATH')
        a1 = ase_io.ase_read(dftpy_data_path + '/fcc.vasp')
        a2 = pmg_io.pmg_read(dftpy_data_path + '/fcc.vasp')
        a3 = dftpy_io.read(dftpy_data_path + '/fcc.vasp')

        self.assertTrue(np.allclose(a1.pos.cell.lattice, a2.pos.cell.lattice))
        self.assertTrue(np.allclose(a1.pos.cell.lattice, a3.pos.cell.lattice))
        self.assertTrue(np.allclose(a1.pos, a2.pos))
        self.assertTrue(np.allclose(a1.pos, a3.pos))


if __name__ == "__main__":
    unittest.main()
