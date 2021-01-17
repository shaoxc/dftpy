from .mpi import MP, PMI
import sys
from dftpy.constants import environ

__all__ = ["mp", "sprint"]

mp = MP()
pmi = PMI()

def sprint(*args, comm = None, lprint = False, debug = False, level = 2, fileobj = None, **kwargs):
    kwargs['flush'] = True
    if comm is not None :
        if comm.rank == 0:
            lprint = True
    elif mp.is_root :
        lprint = True

    if level < environ['LOGLEVEL'] :
        lprint = False
    elif level == 0 :
        debug = True

    if debug :
        comm = comm or mp.comm
        print('rank -> ', comm.rank, ' -> ', *args, **kwargs)
    elif lprint :
        if fileobj :
            stdout = sys.stdout
            sys.stdout = fileobj
        else :
            sys.stdout = environ['STDOUT']
        print(*args, **kwargs)
        if fileobj :
            sys.stdout = stdout
