import numpy as np
from scipy.sparse.linalg import LinearOperator
import scipy.sparse.linalg as linalg
from .field import DirectField, ReciprocalField


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

    def __init__(self, interval=1.0e-3, type="taylor", optional_kwargs=None):
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

        if optional_kwargs is None:
            self.optional_kwargs = {}
        else:
            self.optional_kwargs = optional_kwargs

        self.interval = interval

    def __call__(self, psi, v):

        sigma = self.optional_kwargs.get("sigma", 0.025)
        if self.type == "taylor":
            order = self.optional_kwargs.get("order", 1)
            return taylor(psi, v, self.interval, sigma, order)
        elif self.type == "crank-nicolson":
            linearsolver = self.optional_kwargs.get("linearsolver", "bicgstab")
            tol = self.optional_kwargs.get("tol", 1e-8)
            maxiter = self.optional_kwargs.get("maxiter", 100)
            return CrankNicolson(psi, v, self.interval, sigma, linearsolver, tol, maxiter)
        elif self.type == "rk4":
            return RK4(psi, v, self.interval, sigma)


def hamiltonian(psi, v, sigma=0.025):
    return -0.5 * psi.laplacian(sigma) + v * psi
    # return v*psi


def hamiltonian_fft(psi_fft, v):
    return 0.5 * psi_fft.grid.gg * psi_fft + (v * psi_fft.ifft()).fft()


def taylor(psi0, v, interval, sigma=0.025, order=1):
    N0 = (psi0 * np.conj(psi0)).integral()
    psi1 = psi0

    new_psi = psi0
    for i_order in range(order):
        new_psi = 1j * interval / (i_order + 1) * hamiltonian(new_psi, v, sigma)
        # new_psi = 1j * interval / (i_order+1) * hamiltonian_fft(new_psi, v)
        if np.isnan(new_psi).any():
            print("Warning: taylor propagator exits on order {0:d} due to NaN in new psi.".format(i_order))
            break
        psi1 = psi1 + new_psi

    N1 = (psi1 * np.conj(psi1)).integral()
    psi1 = psi1 * np.sqrt(N0 / N1)

    return psi1, 0


def cnMatvecUtil(v, dt, sigma):
    def cnMatvec(psi_):
        psi = DirectField(grid=v.grid, rank=1, griddata_3d=np.reshape(psi_, np.shape(v)), cplx=True)
        prod = psi - 1j * hamiltonian(psi, v, sigma) * dt / 2.0
        return prod.ravel()

    return cnMatvec


def cnMatvecUtil_fft(v, dt):
    def cnMatvec_fft(psi_fft_):
        reci_grid = v.grid.get_reciprocal()
        psi_fft = ReciprocalField(
            reci_grid, rank=1, griddata_3d=np.reshape(psi_fft_, np.shape(reci_grid.gg)), cplx=True
        )
        prod = psi_fft - 1j * hamiltonian_fft(psi_fft, v) * dt / 2.0
        return prod.ravel()

    return cnMatvec_fft


def CrankNicolson(psi0, v, interval, sigma=0.025, linearsolver="bicgstab", tol=1e-8, maxiter=100):

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

    # tmp = hamiltonian(psi0, v, sigma)/psi0
    # print(np.std(tmp))
    # N0 = (np.real(psi0 * np.conj(psi0))).integral()
    b = (psi0 + 1j * hamiltonian(psi0, v, sigma) * interval / 2.0).ravel()
    # b = (psi0 + 1j * hamiltonian_fft(psi0, v) * interval/2.0).ravel()
    size = np.size(b)
    A = LinearOperator((size, size), dtype=psi0.dtype, matvec=cnMatvecUtil(v, interval, sigma))
    # A = LinearOperator((size,size), dtype=psi0.dtype, matvec=cnMatvecUtil_fft(v, interval))
    try:
        psi1_, info = LinearSolverDict[linearsolver](A, b, x0=psi0.ravel(), tol=tol, maxiter=maxiter)
    except KeyError:
        raise AttributeError("{0:s} is not a linear solver".format(linearsolver))

    if info:
        print(info)
    psi1 = DirectField(grid=psi0.grid, rank=1, griddata_3d=np.reshape(psi1_, np.shape(psi0)), cplx=True)
    # psi1 = ReciprocalField(grid=psi0.grid, rank=1, griddata_3d=np.reshape(psi1_, np.shape(psi0)))
    # N1 = (np.real(psi1 * np.conj(psi1))).integral()
    # psi1 = psi1 * np.sqrt(N0/N1)
    return psi1, info


def RK4(psi0, v, interval, sigma=0.025):
    # N0 = (psi0 * np.conj(psi0)).integral()

    k1 = -1j * interval * hamiltonian(psi0, v, sigma)
    k2 = -1j * interval * hamiltonian(psi0 + k1 / 2.0, v, sigma)
    k3 = -1j * interval * hamiltonian(psi0 + k2 / 2.0, v, sigma)
    k4 = -1j * interval * hamiltonian(psi0 + k3, v, sigma)
    psi1 = psi0 + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    # N1 = (psi1 * np.conj(psi1)).integral()
    # psi1 = psi1 * np.sqrt(N0/N1)

    return psi1, 0
