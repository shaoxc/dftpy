from .mpi import SMPI

__all__ = ["smpi", "sprint"]

smpi = SMPI()

def sprint(*args, comm = None, **kwargs):
    kwargs['flush'] = True
    if (comm is None and smpi.is_root) or (comm is not None and comm.rank == 0) :
        print(*args, **kwargs)
