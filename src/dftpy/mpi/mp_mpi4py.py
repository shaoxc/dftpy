import numpy as np
from mpi4py_fft import PFFT, newDistArray
from mpi4py import MPI

from dftpy.mpi import smpi
import dftpy.mpi.mp_serial as mps

from dftpy.constants import FFTLIB

# Global variables
fft_saved = None
#-----------------------------------------------------------------------
def get_local_fft_shape(nr, realspace = True, decomposition = 'Slab', backend = None, **kwargs):
    """
    TIP :
        When the environment variable LD_PRELOAD is defined, backend = 'fftw' will give a wrong results
        for mpi4py-fft==2.0.3
    """
    fft = get_mpi4py_fft(nr, decomposition=decomposition, backend=backend, **kwargs)
    s = fft.local_slice(not realspace)
    shape = fft.shape(not realspace)
    offsets = np.zeros_like(s, dtype = np.int)
    for i, item in enumerate(s):
        if item.start is not None :
            offsets[i] = item.start
    shape = np.asarray(shape)
    return (s, shape, offsets)

def get_mpi4py_fft(nr, decomposition = 'Slab', backend = None, grid = None, **kwargs):
    fft_support = ['pyfftw', 'numpy','scipy', 'mkl_fft']
    # fft_support = ['fftw', 'pyfftw', 'numpy','scipy', 'mkl_fft']
    if backend not in fft_support :
        backend = FFTLIB
    if backend not in fft_support :
        backend = 'numpy'
    global fft_saved
    if fft_saved is not None and fft_saved.global_shape() == tuple(nr) :
        fft = fft_saved
    else :
        if decomposition == 'Slab' :
            # fft = PFFT(smpi.comm, nr, axes=(0, 1, 2), dtype=np.float, grid=(-1,), backend = backend)
            fft = PFFT(smpi.comm, nr, axes=(0, 1, 2), dtype=np.float, backend = backend)
        else :
            fft = PFFT(smpi.comm, nr, axes=(0, 1, 2), dtype=np.float, backend = backend)
        fft_saved = fft
    return fft


class MPI4PYFFTRUN :
    def __init__(self, nr, forward = True, decomposition = 'Slab', backend = None, **kwargs):
        fft = get_mpi4py_fft(nr, decomposition=decomposition, backend=backend, **kwargs)
        # self.arr = newDistArray(fft, False, view = True)
        # self.arr_g = newDistArray(fft, True, view = True)
        self.arr = newDistArray(fft, False)
        self.arr_g = newDistArray(fft, True)
        self.forward = forward
        self.fft = fft

    def __call__(self, input_array, *args, **kwargs):
        if self.forward :
            value = self.arr
            results = self.arr_g
            value[:] = input_array
            results = self.fft.forward(value, results, normalize=False)
        else :
            value = self.arr_g
            results = self.arr
            value[:] = input_array
            # results = self.fft.backward(value, results, normalize=False)
            results = self.fft.backward(value, results, normalize=True)
        return results

def mpi_fft(grid, **kwargs):
    fft_object = MPI4PYFFTRUN(grid.nrR, forward = True, **kwargs)
    return fft_object

def mpi_ifft(grid, **kwargs):
    fft_object = MPI4PYFFTRUN(grid.nrR, forward = False, **kwargs)
    return fft_object

def to_mpi_type(s, dtype = None, **kwargs):
    if dtype is None :
        s = np.array(s)
    else :
        s = np.array(s, dtype = dtype)
    return s

def sum(*args):
    s = mps.sum(*args)
    s = to_mpi_type(s)
    s = smpi.comm.allreduce(s)
    return s

def einsum(*args, **kwargs):
    s = mps.einsum(*args, **kwargs)
    shape = s.shape
    if len(shape) < 2 :
        s = to_mpi_type(s)
        s = smpi.comm.allreduce(s)
    return s

def vsum(a, *args, **kwargs):
    s = mps.vsum(a)
    s = to_mpi_type(s)
    s = smpi.comm.allreduce(s, *args, **kwargs)
    return s

def asum(a, *args, **kwargs):
    s = mps.asum(a)
    s = to_mpi_type(s)
    s = smpi.comm.allreduce(s, *args, **kwargs)
    return s

def amin(a):
    s = mps.amin(a)
    s = to_mpi_type(s)
    s = smpi.comm.allreduce(s, op=MPI.MIN)
    return s

def amax(a):
    s = mps.amax(a)
    s = to_mpi_type(s)
    s = smpi.comm.allreduce(s, op=MPI.MAX)
    return s

def amean(a):
    if not hasattr(a, 'size'):
        a = np.array(a)
    s = mps.asum(a)
    s = to_mpi_type([s, a.size])
    s = smpi.comm.allreduce(s)
    s = s[0]/s[1]
    return s

def split_number(n):
    av = n//smpi.comm.size
    res = n - av * smpi.comm.size
    if smpi.comm.rank < res :
        lb = (av + 1) * smpi.comm.rank
        ub = lb + av + 1
    else :
        lb = av * smpi.comm.rank + res
        ub = lb + av
    return lb, ub

def best_fft_size(n, opt = True, **kwargs):
    if opt :
        n_l = n/smpi.comm.size
    else :
        n_l = n
    n_l = mps.best_fft_size(n_l, **kwargs)

    if opt :
        n_best = n_l * smpi.comm.size
    else :
        n_best = n_l
    return n_best
