from .mpi import MP

__all__ = ["mp", "sprint"]

mp = MP()

def sprint(*args, comm = None, lprint = False, debug = False, **kwargs):
    kwargs['flush'] = True
    if comm is not None :
        if comm.rank == 0:
            lprint = True
    elif mp.is_root :
        lprint = True

    if debug :
        comm = comm or mp.comm
        print('rank -> ', comm.rank, ' -> ', *args, **kwargs)
    elif lprint :
        print(*args, **kwargs)
