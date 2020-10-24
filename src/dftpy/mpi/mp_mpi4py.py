import numpy as np
from mpi4py_fft import PFFT, newDistArray

from dftpy.constants import FFTLIB

# Global variables
fft_saved = None
#-----------------------------------------------------------------------
class MPI4PYFFTRUN :
    def __init__(self, grid, forward = True, decomposition = 'Slab', backend = None, fft = None, **kwargs):
        if fft is None :
            fft = get_mpi4py_fft(grid.smpi.comm, grid.nrR, decomposition=decomposition, backend=backend, **kwargs)
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


def get_mpi4py_fft(comm, nr, decomposition = 'Slab', backend = None, grid = None, **kwargs):
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
            fft = PFFT(comm, nr, axes=(0, 1, 2), dtype=np.float, grid=(-1,), backend = backend)
        else :
            fft = PFFT(comm, nr, axes=(0, 1, 2), dtype=np.float, backend = backend)
        fft_saved = fft
    return fft

def mpi_fft(grid, **kwargs):
    fft_object = MPI4PYFFTRUN(grid, forward = True, **kwargs)
    return fft_object

def mpi_ifft(grid, **kwargs):
    fft_object = MPI4PYFFTRUN(grid, forward = False, **kwargs)
    return fft_object
