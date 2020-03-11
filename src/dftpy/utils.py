import warnings
import numpy as np


def dipole_moment(rho, ions = None, center = [0.0, 0.0, 0.0]):
    rcenter = np.asarray(center)
    rcenter = np.dot(rho.grid.lattice, rcenter)
    r = rho.grid.r - rcenter[:, None, None, None]
    dm = np.einsum('lijk, ijk -> l', r, rho) * rho.grid.dV
    if ions is not None :
        for i in range(ions.nat) :
            z = ions.Zval[ions.labels[i]]
            dm -= z * (ions.pos[i] - rcenter)
    return dm


def dipole_correction(rho, axis=2, ions=None, center = [0.0, 0.0, 0.0], coef=10.0):
    rcenter = np.asarray(center)
    rcenter = np.dot(rho.grid.lattice, rcenter)
    r = rho.grid.r[axis,...] - rcenter[axis]
    dm = np.einsum('ijk, ijk ->', r, rho) * rho.grid.dV
    if ions is not None :
        for i in range(ions.nat) :
            z = ions.Zval[ions.labels[i]]
            dm -= z * (ions.pos[i][axis] - rcenter[axis])

    s = rho.grid.s[axis,...] - center[axis]
    rho_add = s * np.exp(coef * s * s)
    dm_add = np.einsum('ijk, ijk ->', r, rho_add)
    factor = -dm/dm_add
    rho_add *= factor
    return rho_add


def hamiltonian(psi, v):
    return -0.5 * psi.laplacian() + v * psi


def hamiltonian_fft(psi_fft, v):
    return 0.5 * psi_fft.grid.gg * psi_fft + (v * psi_fft.ifft()).fft()


def calc_rho(psi, N=1):
    return np.real(psi * np.conj(psi)) * N

def calc_drho(psi, dpsi, N=1):
    return 2.0*np.real(np.conj(dpsi)*psi) * N

def calc_j(psi, N=1):
    psi_conj = DirectField(psi.grid, rank=1, griddata_3d=np.conj(psi), cplx = True)
    j = np.real(-0.5j * (psi_conj * psi.gradient(flag='standard', force_real=False) - psi * psi_conj.gradient(flag='standard', force_real=False))) * N
    return j
