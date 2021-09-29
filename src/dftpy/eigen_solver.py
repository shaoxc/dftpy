import numpy as np

from dftpy.functional.abstract_functional import AbstractFunctional
from dftpy.functional.functional_output import FunctionalOutput
from dftpy.linear_solver import cg
from dftpy.optimization import Optimization


def power_iter(A, x0, tol: float = 1.0e-15, maxiter: int = 10000):
    x = x0 / x0.norm()
    k = 0
    old_mu = 1.0e6
    while k < maxiter:
        Ax = A(x)
        mu = np.real(np.conj(x) * Ax).integral()
        new_x = Ax / Ax.norm()
        res = np.abs(mu - old_mu)
        # print(res)
        if res < tol:
            return mu, new_x
        x = new_x
        old_mu = mu
        k += 1

    return mu, x


def minimization(A, x0, tol: float = 1.0e-15, maxiter: int = 10000):
    x = x0
    k = 0
    Ax = A(x)
    old_mu = np.real(np.conj(x) * Ax).integral() / x.norm()
    while k < maxiter:
        b = old_mu * x
        new_x, _ = cg(A, b, x, tol, maxiter=100)
        x = new_x
        Ax = A(x)
        mu = np.real(np.conj(x) * Ax).integral() / x.norm()
        res = np.abs(mu - old_mu)
        if res < tol:
            return mu, x
        old_mu = mu
        k += 1

    return mu, x


def minimization2(A, x0, tol: float = 1.0e-15, maxiter: int = 10000):
    evaluator = DiagFunctional(A)
    optimizer = Optimization(EnergyEvaluator=evaluator, optimization_options={'econv': tol, 'maxiter': maxiter})
    optimizer(guess_rho=x0*x0, guess_phi=x0, lphi=True)

    x = optimizer.phi.normalize()
    mu = x * A(x) / x.norm()

    return mu, x


class DiagFunctional(AbstractFunctional):

    def __init__(self, hamiltonian):
        self.hamiltonian = hamiltonian

    def compute(self, rho, calcType=None, phi=None, lphi=True):
        result = FunctionalOutput(name='a')
        h_phi = self.hamiltonian(phi)
        phi_norm = phi.norm()
        if 'E' in calcType:
            result.energy = (phi * h_phi).integral() / phi_norm
        if 'V' in calcType:
            result.potential = (h_phi / phi_norm + (phi * h_phi).integral() / phi_norm ** 2.0 * phi) / phi

        return result
