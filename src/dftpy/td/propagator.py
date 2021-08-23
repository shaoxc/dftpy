import numpy as np
import scipy.sparse.linalg as linalg
from scipy.sparse.linalg import LinearOperator

import dftpy.linear_solver as linear_solver
from dftpy.field import DirectField, ReciprocalField
from dftpy.mpi import sprint
from dftpy.td.hamiltonian import Hamiltonian
from dftpy.time_data import TimeData


class Propagator(object):
    """
    Class handling propagating the wavefunctions in time

    Attributes
    ----------
    interval: float
        The time interval of each propagation

    propagator: function
        The type of propagator

    kwargs:
        Optional kwargs
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
        "bicg": {"func": linear_solver.bicg, "scipy": False},
        "bicgstab": {"func": linear_solver.bicgstab, "scipy": False},
        "cg": {"func": linear_solver.cg, "scipy": False},
    }

    def __init__(self, hamiltonian, propagator="crank-nicolson", **kwargs):
        # init the class

        TypeList = [
            "taylor",
            "crank-nicolson",
            "rk4",
        ]

        if propagator in TypeList:
            self.type = propagator
        else:
            raise AttributeError("{0:s} is not a supported propagator type".format(propagator))

        if isinstance(hamiltonian, Hamiltonian):
            self.hamiltonian = hamiltonian
        else:
            raise TypeError("hamiltonian must be a DFTpy Hamiltonian.")

        self.optional_kwargs = kwargs

    def __call__(self, psi, interval):

        if self.type == "taylor":
            order = self.optional_kwargs.get("order", 1)
            return self.Taylor(psi, interval, order)
        elif self.type == "crank-nicolson":
            linearsolver = self.optional_kwargs.get("linearsolver", "cg")
            tol = self.optional_kwargs.get("tol", 1e-8)
            atol = self.optional_kwargs.get("atol", None)
            maxiter = self.optional_kwargs.get("maxiter", 100)
            return self.CrankNicolson(psi, interval, linearsolver=linearsolver, tol=tol, maxiter=maxiter, atol=atol)
        elif self.type == "rk4":
            return self.RK4(psi, interval)

    def Taylor(self, psi0, interval, order=1):

        TimeData.Begin('Taylor')

        N0 = (psi0 * np.conj(psi0)).integral()
        psi1 = psi0

        new_psi = psi0
        for i_order in range(order):
            new_psi = -1j * interval / (i_order + 1) * self.hamiltonian(new_psi)
            if np.isnan(new_psi).any():
                sprint("Warning: taylor propagator exits on order {0:d} due to NaN in new psi.".format(i_order))
                psi1 = psi1 + new_psi
                return psi1, 1
            psi1 = psi1 + new_psi

        N1 = (psi1 * np.conj(psi1)).integral()
        psi1 = psi1 * np.sqrt(N0 / N1)

        TimeData.End('Taylor')

        return psi1, 0

    def cnMatvecUtil(self, dt, reciprocal=False):
        if reciprocal:
            reci_grid = self.hamiltonian.grid.get_reciprocal()

        def cnMatvec(psi_):
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

        return cnMatvec

    def A_MatvecUtil(self, dt):
        def A_Matvec(psi):
            return psi + 1j * self.hamiltonian(psi) * dt / 2.0

        return A_Matvec

    def CrankNicolson(self, psi0, interval, linearsolver="cg", tol=1e-8, maxiter=100, atol=None):

        TimeData.Begin('Crank-Nicolson')

        if linearsolver not in self.LinearSolverDict:
            raise AttributeError("{0:s} is not a linear solver".format(linearsolver))

        if self.LinearSolverDict[linearsolver]["scipy"]:
            size = self.hamiltonian.grid.nnr
            b = (psi0 - 1j * self.hamiltonian(psi0) * interval / 2.0).ravel()
            A = LinearOperator((size, size), dtype=psi0.dtype, matvec=self.cnMatvecUtil(interval))
            psi1_, info = self.LinearSolverDict[linearsolver]["func"](A, b, x0=psi0.ravel(), tol=tol, maxiter=maxiter,
                                                                      atol=atol)
            psi1 = DirectField(grid=self.hamiltonian.grid, rank=1,
                               griddata_3d=np.reshape(psi1_, self.hamiltonian.grid.nr), cplx=True)
        else:
            b = psi0 - 1j * self.hamiltonian(psi0) * interval / 2.0
            psi1, info = self.LinearSolverDict[linearsolver]["func"](self.A_MatvecUtil(interval), b, psi0, tol, maxiter,
                                                                     atol=atol, mp=psi0.mp)

        if info:
            sprint("Linear solver did not converge. Info: {0:d}".format(info))

        TimeData.End('Crank-Nicolson')

        return psi1, info

    def RK4(self, psi0, interval):

        k1 = -1j * interval * self.hamiltonian(psi0)
        k2 = -1j * interval * self.hamiltonian(psi0 + k1 / 2.0)
        k3 = -1j * interval * self.hamiltonian(psi0 + k2 / 2.0)
        k4 = -1j * interval * self.hamiltonian(psi0 + k3)
        psi1 = psi0 + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

        return psi1, 0
