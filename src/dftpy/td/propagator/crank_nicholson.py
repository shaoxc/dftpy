from typing import Union, Tuple, Callable

import numpy as np
import scipy.sparse.linalg as linalg
from scipy.sparse.linalg import LinearOperator

import dftpy.linear_solver
from dftpy.field import DirectField, ReciprocalField
from dftpy.mpi.utils import sprint
from dftpy.td.hamiltonian import Hamiltonian
from dftpy.td.propagator.abstract_propagator import AbstractPropagator
from dftpy.time_data import timer


class CrankNicholson(AbstractPropagator):
    """
    Crank-Nicholson propagator for real-time propagation
    Eq. (4.23) of C. A. Ullrich, Time-dependent density-functional theory: concepts and applications (OUP Oxford,
2011)
    """
    LinearSolverDict = {
        "bicg_scipy": {"func": linalg.bicg, "scipy": True},
        "bicgstab_scipy": {"func": linalg.bicgstab, "scipy": True},
        "cg_scipy": {"func": linalg.cg, "scipy": True},
        "cgs_scipy": {"func": linalg.cgs, "scipy": True},
        "gmres_scipy": {"func": linalg.gmres, "scipy": True},
        "lgmres_scipy": {"func": linalg.lgmres, "scipy": True},
        "minres_scipy": {"func": linalg.minres, "scipy": True},
        "qmr_scipy": {"func": linalg.qmr, "scipy": True},
        "bicg": {"func": dftpy.linear_solver.bicg, "scipy": False},
        "bicgstab": {"func": dftpy.linear_solver.bicgstab, "scipy": False},
        "cg": {"func": dftpy.linear_solver.cg, "scipy": False},
    }

    def __init__(self, hamiltonian: Hamiltonian, interval: float, linear_solver: str = "cg", tol: float = 1e-8,
                 maxiter: int = 100, atol: Union[float, None] = None, **kwargs) -> None:
        """

        Parameters
        ----------
        hamiltonian: the time-dependent Hamiltonian
        interval: the time interval for one time step
        linear_solver: the name of the linear solver to solve the Ax=b problem
        tol: the tolerance of the linear solver
        maxiter: the max number of iterations of the linear solver
        atol: the absolute tolerance of the linear solver

        """
        super(CrankNicholson, self).__init__(hamiltonian, interval)
        if linear_solver not in self.LinearSolverDict:
            raise AttributeError("{0:s} is not a linear solver".format(linear_solver))
        self.linear_solver = self.LinearSolverDict[linear_solver]['func']
        self.scipy = self.LinearSolverDict[linear_solver]['scipy']
        self.tol = tol
        self.maxiter = maxiter
        self.atol = atol

    def _scipy_matvec_util(self, dt: float, reciprocal: bool = False) -> Callable:
        if reciprocal:
            reci_grid = self.hamiltonian.grid.get_reciprocal()

        def _scipy_matvec(psi_: np.ndarray) -> np.ndarray:
            if reciprocal:
                psi = ReciprocalField(
                    reci_grid, rank=1, griddata_3d=np.reshape(psi_, reci_grid.nr), cplx=True
                )
            else:
                psi = DirectField(
                    self.hamiltonian.grid, rank=1, griddata_3d=np.reshape(psi_, self.hamiltonian.grid.nr),
                    cplx=True
                )
            prod = psi + 1j * self.hamiltonian(psi) * dt / 2.0
            return prod.ravel()

        return _scipy_matvec

    def _dftpy_matvec_util(self, dt: float) -> Callable:
        def dftpy_matvec(psi: Union[DirectField, ReciprocalField]) -> Union[DirectField, ReciprocalField]:
            return psi + 1j * self.hamiltonian(psi) * dt / 2.0

        return dftpy_matvec

    def _calc_b(self, psi: Union[DirectField, ReciprocalField]) -> Union[DirectField, ReciprocalField]:
        return psi - 1j * self.hamiltonian(psi) * self.interval / 2.0

    @timer('Crank-Nicholson-Propagator')
    def __call__(self, psi0: Union[DirectField, ReciprocalField]) -> Tuple:
        """
        Perform one step of propagation.

        Parameters
        ----------
        psi0: the initial wavefunction.

        Returns
        -------
        A tuple (psi1, status)
        psi1: the final wavefunction.
        status: 0: converged, others: not converged

        """

        if self.scipy:
            size = self.hamiltonian.grid.nnr
            b = self._calc_b(psi0).ravel()
            mat = LinearOperator(shape=(size, size), dtype=psi0.dtype, matvec=self._scipy_matvec_util(self.interval))
            psi1_, status = self.linear_solver(mat, b, x0=psi0.ravel(), tol=self.tol, maxiter=self.maxiter,
                                               atol=self.atol)
            psi1 = DirectField(grid=self.hamiltonian.grid, rank=1,
                               griddata_3d=np.reshape(psi1_, self.hamiltonian.grid.nr), cplx=True)
        else:
            b = self._calc_b(psi0)
            psi1, status = self.linear_solver(self._dftpy_matvec_util(self.interval), b, psi0, self.tol, self.maxiter,
                                              atol=self.atol, mp=psi0.mp)

        if status:
            sprint("Linear solver did not converge. Info: {0:d}".format(status))

        return psi1, status
