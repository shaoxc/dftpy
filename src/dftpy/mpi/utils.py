from .mpi import MP
import sys
import dftpy.constants

__all__ = ["mp", "sprint"]

mp = MP()

def sprint(*args, comm = None, lprint = False, debug = False, fileobj = None, **kwargs):
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
        if fileobj :
            stdout = sys.stdout
            sys.stdout = fileobj
        else :
            sys.stdout = dftpy.constants.STDOUT
        print(*args, **kwargs)
        if fileobj :
            sys.stdout = stdout
