from .mpi import SMPI

__all__ = ["smpi", "sprint"]

smpi = SMPI()

def sprint(*args, **kwargs):
    if smpi.is_root :
        print(*args, **kwargs)
