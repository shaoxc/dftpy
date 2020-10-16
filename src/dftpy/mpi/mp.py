import sys
if 'mpi4py' in sys.modules:
    from .mp_mpi4py import * 
else :
    from .mp_serial import *
