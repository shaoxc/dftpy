import numpy as np
import os

class SerialComm :
    def __init__(self, *args, **kwargs):
        self.rank = 0
        self.size = 1
        self.root = True

    def __getattr__(self, attr):
        if attr in self.__dir__():
            return getattr(self, attr)
        else :
            return self.nothing

    def nothing(self, *args, **kwargs):
        pass

class MP :
    def __init__(self, comm = None, parallel = False, decomposition = 'Slab', **kwargs):
        MPI = None
        if comm is None :
            if parallel :
                from mpi4py import MPI
                comm = MPI.COMM_WORLD
            else :
                comm = SerialComm(**kwargs)
        self._comm = comm
        self._is_mpi = parallel
        self._MPI = MPI
        self._is_root = False
        self.decomposition = decomposition

    @property
    def is_mpi(self):
        # if isinstance(self.comm, SerialComm) or self.comm.size == 1:
        if self.comm.size == 1:
            self._is_mpi = False
        else :
            self._is_mpi = True
        return self._is_mpi

    @property
    def is_root(self):
        self._is_root = self.comm.rank == 0
        return self._is_root

    @property
    def comm(self):
        return self._comm

    @property
    def rank(self):
        return self.comm.rank

    @property
    def size(self):
        return self.comm.size

    @comm.setter
    def comm(self, value):
        self._comm = value

    @property
    def MPI(self):
        if self.is_mpi :
            if self._MPI is None :
                from mpi4py import MPI
                self._MPI = MPI
            return self._MPI
        else :
            raise AttributeError("Only works for parallel version")

    def _get_local_fft_shape_mpi4py(self, nr, realspace = True, decomposition = None, backend = None, fft = None, **kwargs):
        """
        TIP :
            When the environment variable LD_PRELOAD is defined, backend = 'fftw' will give a wrong results
            for mpi4py-fft==2.0.3
        """
        decomposition = decomposition or self.decomposition
        if fft is None :
            from .mp_mpi4py import get_mpi4py_fft
            fft = get_mpi4py_fft(self.comm, nr, decomposition=decomposition, backend=backend, **kwargs)
        s = fft.local_slice(not realspace)
        shape = fft.shape(not realspace)
        offsets = np.zeros_like(s, dtype = np.int)
        for i, item in enumerate(s):
            if item.start is not None :
                offsets[i] = item.start
        shape = np.asarray(shape)
        return (s, shape, offsets)

    def _get_local_fft_shape_serial(self, nr, realspace = True, full = False, **kwargs):
        s = []
        for item in nr :
            s.append(slice(None))
        s = tuple(s)
        shape = np.array(nr)
        if not full and not realspace :
            shape[-1] = shape[-1]//2 + 1
        offsets = np.zeros_like(nr, dtype = np.int)
        return (s, shape, offsets)

    def get_local_fft_shape(self, nr, **kwargs):
        if self.is_mpi :
            return self._get_local_fft_shape_mpi4py(nr, **kwargs)
        else :
            return self._get_local_fft_shape_serial(nr, **kwargs)

    def to_mpi_type(self, s, dtype = None, **kwargs):
        if dtype is None :
            s = np.array(s)
        else :
            s = np.array(s, dtype = dtype)
        return s

    def einsum(self, *args, **kwargs):
        s = np.einsum(*args, **kwargs)
        if not self.is_mpi : return s
        shape = s.shape
        if len(shape) < 2 :
            s = self.to_mpi_type(s)
            s = self.comm.allreduce(s)
        return s

    def vsum(self, a, *args, **kwargs):
        s = a
        if not self.is_mpi : return s
        s = self.to_mpi_type(s)
        s = self.comm.allreduce(s, *args, **kwargs)
        return s

    def asum(self, a, *args, **kwargs):
        s = np.sum(a)
        if not self.is_mpi : return s
        s = self.to_mpi_type(s)
        s = self.comm.allreduce(s, *args, **kwargs)
        return s

    def sum(self,a, *args, **kwargs):
        return self.asum(a, *args, **kwargs)

    def amin(self, a):
        s = np.amin(a)
        if not self.is_mpi : return s
        s = self.to_mpi_type(s)
        s = self.comm.allreduce(s, op=self.MPI.MIN)
        return s

    def amax(self, a):
        s = np.amax(a)
        if not self.is_mpi : return s
        s = self.to_mpi_type(s)
        s = self.comm.allreduce(s, op=self.MPI.MAX)
        return s

    def amean(self, a):
        if not hasattr(a, 'size'):
            a = np.array(a)
        if not self.is_mpi :
            s = np.mean(a)
            return s
        s = np.sum(a)
        s = self.to_mpi_type([s, a.size])
        s = self.comm.allreduce(s)
        s = s[0]/s[1]
        return s

    def split_number(self, n):
        if not self.is_mpi : return 0, n
        av = n//self.comm.size
        res = n - av * self.comm.size
        if self.comm.rank < res :
            lb = (av + 1) * self.comm.rank
            ub = lb + av + 1
        else :
            lb = av * self.comm.rank + res
            ub = lb + av
        return lb, ub

    def _sum_1(self, a):
        s = np.sum(a)
        return s

    def _add_2(self, a, b):
        s = a + b
        return s

    def _mul_2(self, a, b):
        s = a * b
        return s

    def _add(self, *args):
        for i, item in enumerate(args):
            if i == 0 :
                a = item
            else :
                a = self._add_2(a, item)
        return a

    def _mul(self, *args):
        for i, item in enumerate(args):
            if i == 0 :
                a = item
            else :
                a = self._mul_2(a, item)
        return a

    def _sum_mul(self, *args):
        a = self._mul(*args)
        s = self._sum_1(a)
        return s

class PMI :
    """
    Detect mpi
    ref :
        https://www.open-mpi.org/faq/?category=running#mpi-environmental-variables
        https://docs.microsoft.com/en-us/powershell/high-performance-computing/environment-variables-for-the-mpiexec-command
    """
    def __init__(self):
        self.comm = None
        self.size = self._get_size()
        self.rank = self._get_rank()

    @staticmethod
    def _get_size():
        psize0 = int(os.environ.get('OMPI_COMM_WORLD_SIZE', 0)) # OpenMPI
        psize1 = int(os.environ.get('PMI_SIZE', 0))  # Intel MPI and MVAPICH2
        pmi_size = max(psize0, psize1)
        return pmi_size

    @staticmethod
    def _get_rank():
        prank0 = int(os.environ.get('OMPI_COMM_WORLD_RANK', 0))
        prank1 = int(os.environ.get('PMI_RANK', 0))
        pmi_rank = max(prank0, prank1)
        return pmi_rank

class MPIFile(object):
    def __init__(self, fname, mp, **kwargs):
        if mp is None :
            mp = MP()
        self.mp = mp
        if isinstance(fname, str):
            if mp.size > 1 :
                self.fh = mp.MPI.File.Open(mp.comm, fname, **kwargs)
            else :
                self.fh = open(fname, **kwargs)
        else :
            self.fh = fname

    @property
    def is_mpi(self):
        if hasattr(self.fh, 'Close'):
            return True
        else :
            return False

    # def __enter__(self):
        # return self.fh

    def __exit__(self, *args, **kwargs):
        if hasattr(self.fh, '__exit__'):
            return self.fh.__exit__(*args, **kwargs)
        if hasattr(self.fh, 'close'):
            return self.fh.close()
        elif hasattr(self.fh, 'Close'):
            return self.fh.Close()

    def close(self, *args, **kwargs):
        return self.__exit__(*args, **kwargs)

    def __getattr__(self, attr):
        if self.is_mpi :
            if attr == 'read' :
                return self._read_bytes
            elif attr == 'write' :
                return self._write_bytes

        return getattr(self.fh, attr)

    def __iter__(self):
        return iter(self.fh)

    def _read_bytes(self, n, data=None):
        if data is None :
            data = bytearray(n)
        self.fh.Read(data)
        return data

    def _read_bytes_2(self, n, data=None):
        if data is None :
            data = np.empty(n, dtype = np.byte)
        self.fh.Read(data)
        return data.tobytes()

    def _write_bytes(self, data):
        self.fh.Write(data)
        return data
