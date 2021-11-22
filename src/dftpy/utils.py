import numpy as np
import gc
import os
import importlib.util
import resource
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
    lfine = True
    if np.all(nr_fine < nr_coarse):
        nr_coarse, nr_fine = nr_fine, nr_coarse
        lfine = False
    elif np.all(nr_fine > nr_coarse):
        pass
    elif np.prod(nr_fine) > np.prod(nr_coarse):
        pass
        # print('!WARN : grid {} and {} are similar, here set the second grid as fine grid'.format(nr, nr2))
    else :
        lfine = False
        nr_coarse, nr_fine = nr_fine, nr_coarse
        # print('!WARN : grid {} and {} are similar, here set the first grid as fine grid'.format(nr, nr2))
    nr_coarse = np.minimum(nr_fine, nr_coarse)
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
    return index, lfine

def grid_map_data(data, nr = None, direct = True, index = None, grid = None):
    """
    Only support for serial.
    """
    if hasattr(data, 'fft'):
        value = data.fft()
    else :
        value = data
    nr1_g = np.array(value.shape, dtype = int)
    if grid is not None :
        if not isinstance(grid, ReciprocalGrid):
            grid2 = grid.get_reciprocal()
        else :
            grid2 = grid
        nr2_g = grid2.nr
    else :
        nr2_g = np.array(nr, dtype = int)
        nr2_g[2] = nr2_g[2]//2+1
        grid2 = ReciprocalGrid(value.grid.lattice, nr)

    value2= ReciprocalField(grid2)

    index, lfine = grid_map_index(nr1_g, nr2_g)

    bd = np.minimum(nr1_g, nr2_g)
    if lfine :
        value2[index[0], index[1], index[2]] = value[:bd[0], :bd[1], :bd[2]].ravel()
    else :
        value2[:bd[0], :bd[1], :bd[2]] = value[index[0], index[1], index[2]].reshape(bd)

    if direct :
        results = value2.ifft(force_real=True)
    else :
        results = value2

    if grid is not None :
        if isinstance(grid, ReciprocalGrid) and not direct :
            results.grid = grid
        elif not isinstance(grid, ReciprocalGrid) and direct :
            results.grid = grid
    return results


def coarse_to_fine(data, nr_fine, direct = True, index = None):
    """
    Only support for serial.
    """
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
    """
    Only support for serial.
    """
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

def clean_variables(*args):
    for item in args :
        del item
    gc.collect()

def bytes2human(n):
    symbols = ('B', 'K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')
    if isinstance(n, (int, float)):
        ns = len(symbols) - 1
        for i, s in enumerate(reversed(symbols)):
            unit = 1 << (ns - i) * 10
            if n >= unit :
                value = float(n) / unit
                return f'{value:.1f}{s:s}'
        return f'{n}B'
    return f'{n}U'

def get_mem_info(pid = None, width = 8):
    # return "PID", "USS", "PSS", "Swap", "RSS"
    if not pid : pid = os.getpid()
    templ = "{:<{width}s} {:>{width}s} {:>{width}s} {:>{width}s} {:>{width}s}"
    is_psutil = importlib.util.find_spec("psutil")
    if is_psutil :
        import psutil
        mem = psutil.Process(pid).memory_full_info()
        uss = mem.uss
        rss = mem.rss
        pss = getattr(mem, "pss", "0U")
        swap = getattr(mem, "swap", "0U")
        line = templ.format(str(pid), bytes2human(uss), bytes2human(pss), bytes2human(swap), bytes2human(rss), width = width)
    else :
        try:
            mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if os.uname().sysname == 'Linux' : # kB
                mem = bytes2human(mem*1024)
            else : # Darwin
                mem = bytes2human(mem)
        except Exception :
            mem = '0U'
        line = templ.format(str(pid), mem, mem, '0U', mem, width = width)
    return line
