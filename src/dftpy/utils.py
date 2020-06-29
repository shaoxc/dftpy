import warnings
import numpy as np
from dftpy.field import DirectField, ReciprocalField
from dftpy.grid import ReciprocalGrid


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


def calc_rho(psi, N=1):
    return np.real(psi * np.conj(psi)) * N

def calc_drho(psi, dpsi, N=1):
    return 2.0*np.real(np.conj(dpsi)*psi) * N

def calc_j(psi, N=1):
    psi_conj = DirectField(psi.grid, rank=1, griddata_3d=np.conj(psi), cplx = True)
    j = np.real(-0.5j * (psi_conj * psi.gradient(flag='standard', force_real=False) - psi * psi_conj.gradient(flag='standard', force_real=False))) * N
    return j

def grid_map_index(nr, nr2, full = False):
    nr_coarse = np.array(nr, dtype = int)
    nr_fine = np.array(nr2, dtype = int)
    if np.all(nr_fine < nr_coarse):
        nr_coarse, nr_fine = nr_fine, nr_coarse
    elif np.all(nr_fine > nr_coarse):
        pass
    else :
        print('!WARN : Two grids are similar, here set the second grid as fine grid')
    dnr = np.ones(3, dtype = int)
    dnr[:] = nr_coarse[:3]
    if full :
        dnr[:3] = nr_coarse[:3] // 2
    else :
        dnr[:2] = nr_coarse[:2] // 2
    index = np.mgrid[:nr_coarse[0], :nr_coarse[1], :nr_coarse[2]].reshape((3, -1))
    for i in range(3):
        mask = index[i] > dnr[i]
        index[i][mask] += nr_fine[i]-nr_coarse[i]
    return index

def grid_map_data(data, nr2, direct = True, index = None):
    if hasattr(data, 'fft'):
        value = data.fft()
    else :
        value = data
    nr = np.array(value.shape, dtype = int)
    nr2_g = np.array(nr2, dtype = int)
    nr2_g[2] = nr2_g[2]//2+1
    if index is None :
        index = grid_map_index(nr, nr2_g)
    grid = ReciprocalGrid(value.grid.lattice, nr2)
    value2= ReciprocalField(grid)
    if np.all(nr2_g < nr):
        value2[:] = value[index[0], index[1], index[2]].reshape(nr2_g)
    else :
        value2[index[0], index[1], index[2]] = value.ravel()
    if direct :
        results = value2.ifft(force_real=True)
    else :
        results = value2
    return results


def coarse_to_fine(data, nr_fine, direct = True, index = None):
    if hasattr(data, 'fft'):
        value = data.fft()
    else :
        value = data
    nr = value.shape
    nr2 = nr_fine.copy()
    nr2[2] = nr2[2]//2+1
    if index is None :
        index = grid_map_index(nr, nr2)
    grid = ReciprocalGrid(value.grid.lattice, nr_fine)
    value_fine = ReciprocalField(grid)
    value_fine[index[0], index[1], index[2]] = value.ravel()
    if direct :
        results = value_fine.ifft(force_real=True)
    else :
        results = value_fine
    return results

def fine_to_coarse(data, nr_coarse, direct = True, index = None):
    if hasattr(data, 'fft'):
        value = data.fft()
    else :
        value = data
    nr = value.shape
    nr2 = nr_coarse.copy()
    nr2[2] = nr2[2]//2+1
    if index is None :
        index = grid_map_index(nr2, nr)
    value_coarse = value[index[0], index[1], index[2]]
    grid = ReciprocalGrid(value.grid.lattice, nr_coarse)
    value_g = ReciprocalField(grid, griddata_3d=value_coarse)
    if direct :
        results = value_g.ifft(force_real=True)
    else :
        results = value_g
    return results
