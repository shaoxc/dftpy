import numpy as np

from dftpy.field import DirectField

def power_iter(A, x0, tol: float = 1.0e-15, maxiter: int = 10000):
    x = x0 / x0.norm()
    k = 0
    old_mu = 1.0e6
    while k < maxiter:
        Ax = A(x)
        mu = np.real(np.conj(x)*Ax).integral()
        new_x = Ax / Ax.norm()
        res = np.abs(mu - old_mu)
        #print(res)
        if res < tol:
            return 0, new_x
        x = new_x
        old_mu = mu
        k += 1

    return 1, x