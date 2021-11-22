from .mpi import MP, PMI
import sys
import os
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
        nodename = os.uname().nodename
        header = f'{nodename[:30]}->{comm.rank:<7d} -> '
        args = funcs2args(*args)
        comm.Barrier()
        print(header, *args, **kwargs)
        comm.Barrier()
    elif lprint :
        if fileobj :
            stdout = sys.stdout
            sys.stdout = fileobj
        else :
            sys.stdout = environ['STDOUT']
        args = funcs2args(*args)
        print(*args, **kwargs)
        if fileobj :
            sys.stdout = stdout

def funcs2args(*args, **kwargs):
    newargs = None
    for i, item in enumerate(args) :
        if hasattr(item, '__call__') :
            if not newargs :
                newargs = list(args)
            newargs[i] = item()
    if not newargs :newargs = args
    return newargs
