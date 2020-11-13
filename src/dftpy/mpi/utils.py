from .mpi import MP

__all__ = ["mp", "sprint"]

mp = MP()

def sprint(*args, comm = None, **kwargs):
    kwargs['flush'] = True
    if (comm is None and mp.is_root) or (comm is not None and comm.rank == 0) :
        print(*args, **kwargs)
