import numpy as np
from scipy.sparse.linalg import LinearOperator
import scipy.sparse.linalg as linalg
from dftpy.field import DirectField, ReciprocalField
from dftpy.td.hamiltonian import Hamiltonian

class Propagator(object):
    """
    Class handling propagating the wavefunctions in time

    Attributes
    ----------
    interval: float
        The time interval of each propagation

    type: function
        The type of propagator

    optional_kwargs: dict
        set of optional kwargs
    """

    def __init__(self, hamiltonian, interval=1.0e-3, type="taylor", **kwargs):
        # init the class

        TypeList = [
            "taylor",
            "crank-nicolson",
            "rk4",
        ]

        if not type in TypeList:
            raise AttributeError("{0:s} is not a supported propagator type".format(type))
        else:
            self.type = type

        if isinstance(hamiltonian, Hamiltonian):
            self.hamiltonian = hamiltonian
        else:
            raise TypeError("hamiltonian must be a DFTpy Hamiltonian.")

        self.optional_kwargs = kwargs
        self.interval = interval


    def __call__(self, psi):

        if self.type == "taylor":
            order = self.optional_kwargs.get("order", 1)
            return self.Taylor(psi, self.interval, order)
        elif self.type == "crank-nicolson":
            linearsolver = self.optional_kwargs.get("linearsolver", "bicgstab")
            tol = self.optional_kwargs.get("tol", 1e-8)
            maxiter = self.optional_kwargs.get("maxiter", 100)
            return self.CrankNicolson(psi, self.interval, linearsolver, tol, maxiter)
        elif self.type == "rk4":
            return self.RK4(psi, self.interval)


    def Taylor(self, psi0, interval, order=1):
        N0 = (psi0 * np.conj(psi0)).integral()
        psi1 = psi0

        new_psi = psi0
        for i_order in range(order):
            new_psi = 1j * interval / (i_order + 1) * self.hamiltonian(new_psi)
            if np.isnan(new_psi).any():
                print("Warning: taylor propagator exits on order {0:d} due to NaN in new psi.".format(i_order))
                psi1 = psi1 + new_psi
                return psi1, 1
            psi1 = psi1 + new_psi

        N1 = (psi1 * np.conj(psi1)).integral()
        psi1 = psi1 * np.sqrt(N0 / N1)

        return psi1, 0


    def cnMatvecUtil(self, dt, reciprocal = False):
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
            prod = psi - 1j * self.hamiltonian(psi) * dt / 2.0
            return prod.ravel()

        return cnMatvec


    def CrankNicolson(self, psi0, interval, linearsolver="bicgstab", tol=1e-8, maxiter=100):

        LinearSolverDict = {
            "bicg": linalg.bicg,
            "bicgstab": linalg.bicgstab,
            "cg": linalg.cg,
            "cgs": linalg.cgs,
            "gmres": linalg.gmres,
            "lgmres": linalg.lgmres,
            "minres": linalg.minres,
            "qmr": linalg.qmr,
        }

        size = self.hamiltonian.grid.nnr
        b = (psi0 + 1j * self.hamiltonian(psi0) * interval / 2.0).ravel()
        A = LinearOperator((size, size), dtype=psi0.dtype, matvec=self.cnMatvecUtil(interval))
        try:
            psi1_, info = LinearSolverDict[linearsolver](A, b, x0=psi0.ravel(), tol=tol, maxiter=maxiter, atol=0)
        except KeyError:
            raise AttributeError("{0:s} is not a linear solver".format(linearsolver))

        if info:
            print(info)
        psi1 = DirectField(grid=self.hamiltonian.grid, rank=1, griddata_3d=np.reshape(psi1_, self.hamiltonian.grid.nr), cplx=True)

        return psi1, info


    def RK4(self, psi0, interval):

        k1 = -1j * interval * self.hamiltonian(psi0)
        k2 = -1j * interval * self.hamiltonian(psi0 + k1 / 2.0)
        k3 = -1j * interval * self.hamiltonian(psi0 + k2 / 2.0)
        k4 = -1j * interval * self.hamiltonian(psi0 + k3)
        psi1 = psi0 + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

        return psi1, 0
