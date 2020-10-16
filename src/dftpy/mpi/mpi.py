import sys
import numpy as np

class SMPI:
    def __init__(self, comm = None, mpi = False, **kwargs):
        if comm is None :
            if mpi :
                from mpi4py import MPI
                comm = MPI.COMM_WORLD
            else :
                comm = SerialComm(**kwargs)
        self._comm = comm
        self._default_vars()

    def _default_vars(self):
        self._is_mpi = False
        self._is_root = False

    @property
    def is_mpi(self):
        self._is_mpi = True
        if isinstance(self._comm, SerialComm):
            self._is_mpi = False
        return self._is_mpi

    @property
    def is_root(self):
        self._is_root = self.comm.rank == 0
        return self._is_root

    @property
    def comm(self):
        return self._comm

    @comm.setter
    def comm(self, value):
        self._comm = value


class SerialComm :
    def __init__(self):
        self.rank = 0
        self.size = 1
        self.root = True

