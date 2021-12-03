#!/usr/bin/env python3
import unittest
import numpy as np

from dftpy.linear_solver import cg, bicg, bicgstab

def A(x):
    mat_A = np.asarray([[4.0,1.0,2.0],[1.0,3.0,4.0],[2.0,4.0,0.0]])
    return np.einsum('ij,j', mat_A, x)

class TestGrid(unittest.TestCase):

    def setUp(self):
        self.b = np.asarray([1.0,2.0,3.0])
        self.x0 = np.asarray([3.0,5.0,6.0])
        self.tol = 1e-8
        self.maxiter = 100
        self.result = np.asarray([0.1,0.7,-0.05])

    def test_cg(self):
        x, info = cg(A, self.b, self.x0, self.tol, self.maxiter)
        self.assertTrue(np.allclose(x, self.result, 1e-8, 1e-8))
        self.assertEqual(info, 0)

    def test_bicg(self):
        x, info = bicg(A, self.b, self.x0, self.tol, self.maxiter)
        self.assertTrue(np.allclose(x, self.result, 1e-8, 1e-8))
        self.assertEqual(info, 0)

    def test_bicgstab(self):
        x, info = bicgstab(A, self.b, self.x0, self.tol, self.maxiter)
        self.assertTrue(np.allclose(x, self.result, 1e-8, 1e-8))
        self.assertEqual(info, 0)

if __name__ == '__main__':
    unittest.main()
