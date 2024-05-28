#!/usr/bin/env python3
import unittest
import numpy as np

from dftpy.grid import DirectGrid

from common import run_test_orthorombic, run_test_triclinic, make_orthorombic_cell, make_triclinic_cell

class TestCell(unittest.TestCase):
    
    def test_orthorombic_cell(self):
        print()
        print("*"*50)
        print("Testing orthorombic DirectGrid")
        # run the same tests we ran on Cell onto Grid
        nr = 100
        a, b, c = 10.,12.,14.
        run_test_orthorombic(self, DirectGrid, nr=[nr,nr,nr])
        grid = make_orthorombic_cell(a,b,c,nr=[nr,nr,nr],CellClass=DirectGrid)

        # calculate grid points
        r = grid.r
        self.assertTrue(np.isclose(r[:,0,0,0],np.array([0,0,0])).all())
        #self.assertTrue(np.isclose(r[:, nr-1,nr-1,nr-1],np.array([a-a/nr,b-b/nr,c-c/nr]),rtol=1.e-3).all())
        # calculate crystal grid points
        s = grid.s
        self.assertTrue(np.isclose(s[:, 0,0,0],np.array([0,0,0])).all())
        self.assertTrue(np.isclose(s[:, nr-1,nr-1,nr-1],np.array([1.-1./nr,1.-1./nr,1.-1./nr])).all())

    def test_triclinic_cell(self):
        print()
        print("*"*50)
        print("Testing triclinic DirectGrid")
        run_test_triclinic(self, DirectGrid, nr=[10,10,10])
