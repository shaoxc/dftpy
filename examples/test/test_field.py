#!/usr/bin/env python3
import unittest
import numpy as np

from dftpy.grid import DirectGrid, ReciprocalGrid
from dftpy.field import DirectField, ReciprocalField
from dftpy.constants import Units

from .common import run_test_orthorombic, run_test_triclinic, make_orthorombic_cell, make_triclinic_cell

class TestField(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
          This setUp is in common for all the test cases below, and it's only execuded once
        """
        # Test a constant scalar field
        N = 8
        A, B, C = 5, 10, 6
        nr = np.array([A*20, B*20, C*20])
        A, B, C = A/Units.Bohr, B/Units.Bohr, C/Units.Bohr
        grid = make_orthorombic_cell(A=A,B=B,C=C,CellClass=DirectGrid, nr=nr)
        d = N/grid.volume
        initial_vals = np.ones(nr)*d
        cls.constant_field = DirectField(grid=grid, griddata_3d=initial_vals)
        cls.N = N

    def test_direct_field(self):
        print()
        print("*"*50)
        print("Testing DirectField")
        #print(initial_vals[0,0,:])
        #print(field[0,0,:])
        field = self.constant_field
        N = self.N

        self.assertTrue(type(field) is DirectField)
        N1 = field.integral()
        self.assertAlmostEqual(N,N1)

        # fft
        reciprocal_field = field.fft()
        self.assertAlmostEqual(N, reciprocal_field[0,0,0])

        # ifft
        field1 = reciprocal_field.ifft(check_real=True)
        N1 = field1.integral()
        self.assertAlmostEqual(N,N1)

        # gradient
        gradient = field.gradient()
        self.assertTrue(isinstance(gradient, DirectField))
        self.assertEqual(gradient.rank, 3)

    def test_direct_field_interpolation(self):
        field = self.constant_field
        nr = field.grid.nr
        # interpolate up
        field1 = field.get_3dinterpolation(np.array(nr*1.5,dtype=int))
        N1 = field1.integral()
        self.assertAlmostEqual(self.N,N1)

        # interpolate down
        field2 = field.get_3dinterpolation(nr//2)
        N1 = field2.integral()
        self.assertAlmostEqual(self.N,N1)

    def test_direct_field_cut(self):
        field = self.constant_field
        nr = field.grid.nr
        x0 = [0,0,0]
        r0 = [1,0,0]
        field_cut = field.get_cut(origin=x0, r0=r0, nr=nr[0])
        self.assertTrue(np.isclose(field_cut[0,0,:], field[0,0,:]).all())

    def test_fft_ifft(self):
        nr=(51,51,51)
        A=10.0
        B=10.0
        C=10.0

        dgrid = make_orthorombic_cell(A=A, B=B, C=C, CellClass=DirectGrid, nr=nr)
        rgrid=dgrid.get_reciprocal()

        def ReciprocalSpaceGaussian(sigma,mu,grid):
            if not isinstance(grid,(ReciprocalGrid)):
                raise Exception()
            a = np.einsum('lijk,l->ijk',grid.g,mu)
            b = np.exp(-sigma**2*grid.gg/2.0)
            c = np.exp(-1j*a)
            d=np.einsum('ijk,ijk->ijk',b,c)
            return ReciprocalField(grid=grid,rank=1,griddata_3d=d)

        def DirectSpaceGaussian(sigma,mu,grid):
            if not isinstance(grid,(DirectGrid)):
                raise Exception()
            a = grid.r-mu[:, None, None, None]
            b = (sigma*np.sqrt(2.0*np.pi))**(-3.0)*np.exp( - 0.5 * np.einsum('lijk,lijk->ijk', a, a) / sigma**2.0 )
            return DirectField(grid=grid,rank=1,griddata_3d=b)

        center = dgrid.r[:, 25,25,25]
        sigma = 0.5

        rf = ReciprocalSpaceGaussian(sigma,center,rgrid)
        df = DirectSpaceGaussian(sigma,center,dgrid)

        df_dftpy=rf.ifft()
        rf_dftpy=df.fft()

        self.assertTrue(np.isclose(df_dftpy.integral(),1.0))
        self.assertTrue(np.isclose(df.integral(),1.0))

        self.assertTrue(np.isclose(rf,rf_dftpy).all())
        self.assertTrue(np.isclose(df,df_dftpy).all())


if __name__ == "__main__":
    unittest.main()
