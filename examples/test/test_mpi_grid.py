#!/usr/bin/env python3
import unittest
import numpy as np

from dftpy.grid import DirectGrid
from dftpy.field import DirectField
from dftpy.mpi import mp

class TestGrid(unittest.TestCase):

    def test(self):
        try :
            from mpi4py import MPI
            mp.comm = MPI.COMM_WORLD
        except :
            pass

        lattice = np.asarray([[1,0,0],[0,1,0],[0,0,1]])
        nr = np.asarray([2,2,2])
        grid = DirectGrid(lattice, nr, cplx=True, mp=mp)
        field = DirectField(grid, cplx=True)
        k = 2.0
        field[:] = np.exp(1j * k * grid.r[0])
        sumf = field.asum()
        self.assertEqual(sumf.real, 6.161209223472559)

        lap = field.laplacian(sigma = 0)
        suml = (lap * np.conj(lap)).asum()
        self.assertEqual(suml, 2865.839010291929)


if __name__ == "__main__":
    unittest.main()
