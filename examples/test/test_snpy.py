#!/usr/bin/env python
# coding: utf-8

from dftpy.formats import io, qepp, xsf, snpy, npy
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
        struct = xsf.read_xsf(infile)
        try:
            mp = MP(parallel = True, decomposition = 'Pencil')
        except :
            mp = MP()
        outfile = tempfile.gettempdir() + '/serial.snpy'
        if mp.rank == 0 :
            snpy.write(outfile,struct)
        if hasattr(mp.comm, 'Barrier'):
            mp.comm.Barrier()
        struct2 = snpy.read(outfile)
        self.check_system(struct, struct2)
        return

    def test_1_mpi(self):
        pytest.importorskip("mpi4py")
        pytest.importorskip("mpi4py_fft")
        mp = MP(parallel = True, decomposition = 'Pencil')
        mp.comm.Barrier()

        infile = tempfile.gettempdir() + '/serial.snpy'
        outfile = tempfile.gettempdir() + '/mpi.snpy'

        struct=snpy.read(infile, mp)
        data = struct.field.gather()
        if mp.rank == 0 :
            struct2 = snpy.read(infile)
            data2 = struct2.field
            self.assertTrue(np.allclose(data, data2))

        snpy.write(outfile, struct, mp)
        mp.comm.Barrier()
        struct = snpy.read(infile)
        struct2 = snpy.read(outfile)
        self.check_system(struct, struct2)
        mp.comm.Barrier()
        return

    def check_system(self, struct, struct2):
        self.assertTrue(np.allclose(struct.ions.pos.cell.lattice, struct2.ions.pos.cell.lattice))
        self.assertTrue(np.allclose(struct.ions.pos, struct2.ions.pos))
        self.assertTrue(np.allclose(struct.ions.Z, struct2.ions.Z))
        self.assertTrue(np.allclose(struct.field, struct2.field))


if __name__ == "__main__":
    unittest.main()
