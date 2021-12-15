import numpy as np

from dftpy.mpi import MP, sprint

__all__ = ['cg', 'bicg', 'bicgstab']


def _get_atol(tol, atol, bnrm2):
    tol = float(tol)

    if atol is None:
        if bnrm2 == 0:
            return tol
        else:
            return tol * float(bnrm2)

    return max(float(atol), tol * float(bnrm2))


def _get_norm(x, mp):
    return np.sqrt(np.real(mp.asum(np.conj(x) * x)))


def cg(A, b, x0, tol, maxiter, atol=None, mp=None):
    if mp is None:
        mp = MP()

    atol = _get_atol(tol, atol, _get_norm(b, mp))
    res = b - A(x0)
    if _get_norm(res, mp) < atol:
        return x0, 0
    p = res
    k = 0
    x = x0
    while k < maxiter:
        v = A(p)
        alpha = mp.asum(res * res) / mp.asum(p * v)
        x = x + alpha * p
        old_res = res
        res = old_res - alpha * v
        # sprint(k, mp.amax(np.abs(res[-1])))
        if _get_norm(res, mp) < atol:
            return x, 0
        beta = mp.asum(res * res) / mp.asum(old_res * old_res)
        p = res + beta * p
        k += 1
    return x, 1


def bicg(A, b, x0, tol, maxiter, atol=None, mp=None):
    if mp is None:
        mp = MP()

    atol = _get_atol(tol, atol, _get_norm(b, mp))
    res = [b - A(x0)]
    if _get_norm(res[0], mp) < atol:
        return x0, 0
    p = [res[0]]
    k = 0
    x = x0
    while k < maxiter:
        v = A(p[-1])
        alpha = mp.asum(np.conj(res[-1]) * res[-1]) / mp.asum(np.conj(p[-1]) * v)
        x = x + alpha * p[-1]
        res.append(res[-1] - alpha * v)
        # sprint(k, mp.amax(np.abs(res[-1])))
        if _get_norm(res[-1], mp) < atol:
            return x, 0
        beta = mp.asum(np.conj(res[-1]) * res[-1]) / mp.asum(np.conj(res[-2]) * res[-2])
        p.append(res[-1] + beta * p[-1])
        k += 1
    return x, 1


def bicgstab(A, b, x0, tol, maxiter, atol=None, mp=None):
    if mp is None:
        mp = MP()

    atol = _get_atol(tol, atol, _get_norm(b, mp))
    r0 = b - A(x0)
    if _get_norm(r0, mp) < atol:
        return x0, 0
    rho = [1.0]
    omega = 1.0
    alpha = 1.0
    v = 0 * b
    p = v
    k = 0
    x = x0
    res = r0
    r = r0
    while k < maxiter:
        rho.append(mp.asum(r0 * r))
        beta = rho[-1] / rho[-2] * alpha / omega
        p = r + beta * (p - omega * v)
        v = A(p)
        alpha = rho[-1] / mp.asum(r0 * v)
        h = x + alpha * p
        res = res - alpha * v
        # sprint(k, _get_norm(res, mp))
        if _get_norm(res, mp) < atol:
            return h, 0
        s = r - alpha * v
        t = A(s)
        omega = mp.asum(t * s) / mp.asum(t * t)
        x = h + omega * s
        res = res - omega * t
        # sprint(k, _get_norm(res, mp))
        if _get_norm(res, mp) < atol:
            return x, 0
        r = s - omega * t
        k += 1
    return x, 1
