#!/usr/bin/env python3
# coding: utf-8

from dftpy.formats import io, snpy
from dftpy.mpi import MP
import numpy as np
import pytest
import os
import unittest

import warnings
warnings.filterwarnings('ignore')

import tempfile

class Test(unittest.TestCase):
    def test_0_serial(self):
        dftpy_data_path = os.environ.get('DFTPY_DATA_PATH')
        infile=dftpy_data_path+'/GaAs_random.xsf'
        ions, field, _ = io.read_all(infile)
        try:
            mp = MP(parallel = True, decomposition = 'Pencil')
        except Exception:
            mp = MP()
        outfile = tempfile.gettempdir() + '/serial.snpy'
        if mp.rank == 0 :
            snpy.write(outfile, ions, field)
        if hasattr(mp.comm, 'Barrier'):
            mp.comm.Barrier()
        ions2, field2, _ = snpy.read(outfile)

        self.assertTrue(np.allclose(ions.cell, ions2.cell))
        self.assertTrue(np.allclose(ions.positions, ions2.positions))
        self.assertTrue(np.allclose(ions.numbers, ions2.numbers))
        self.assertTrue(np.allclose(field, field2))
        return

    def test_1_mpi(self):
        pytest.importorskip("mpi4py")
        pytest.importorskip("mpi4py_fft")
        mp = MP(parallel = True, decomposition = 'Pencil')
        mp.comm.Barrier()

        infile = tempfile.gettempdir() + '/serial.snpy'
        outfile = tempfile.gettempdir() + '/mpi.snpy'

        ions, field, _ = snpy.read(infile, mp=mp)
        data = field.gather()
        if mp.rank == 0 :
            _, data2, _ = snpy.read(infile)
            self.assertTrue(np.allclose(data, data2))

        snpy.write(outfile, ions = ions, data = field, mp = mp)
        mp.comm.Barrier()
        ions, field, _ = snpy.read(infile)
        ions2, field2, _ = snpy.read(infile)

        self.assertTrue(np.allclose(ions.cell, ions2.cell))
        self.assertTrue(np.allclose(ions.positions, ions2.positions))
        self.assertTrue(np.allclose(ions.numbers, ions2.numbers))
        self.assertTrue(np.allclose(field, field2))

        mp.comm.Barrier()
        return


if __name__ == "__main__":
    unittest.main()
