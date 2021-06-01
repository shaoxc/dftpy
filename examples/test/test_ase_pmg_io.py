#!/usr/bin/env python3
import os
import unittest
import numpy as np
import dftpy.formats.io as dftpy_io
import importlib.util


class Test(unittest.TestCase):
    def test_io_ase(self):
        ishas = importlib.util.find_spec("ase")
        if not ishas : return
        import dftpy.formats.ase_io as ase_io
        dftpy_data_path = os.environ.get('DFTPY_DATA_PATH')
        a1 = ase_io.ase_read(dftpy_data_path + '/fcc.vasp')
        a2 = dftpy_io.read(dftpy_data_path + '/fcc.vasp')

        self.assertTrue(np.allclose(a1.pos.cell.lattice, a2.pos.cell.lattice))
        self.assertTrue(np.allclose(a1.pos, a2.pos))

    def test_io_pmg(self):
        ishas = importlib.util.find_spec("pymatgen")
        if not ishas : return
        import dftpy.formats.pmg_io as pmg_io
        dftpy_data_path = os.environ.get('DFTPY_DATA_PATH')
        a1 = pmg_io.pmg_read(dftpy_data_path + '/fcc.vasp')
        a2 = dftpy_io.read(dftpy_data_path + '/fcc.vasp')

        self.assertTrue(np.allclose(a1.pos.cell.lattice, a2.pos.cell.lattice))
        self.assertTrue(np.allclose(a1.pos, a2.pos))


if __name__ == "__main__":
    unittest.main()
