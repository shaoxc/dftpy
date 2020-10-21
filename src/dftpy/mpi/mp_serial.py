import numpy as np

# Global variables
#-----------------------------------------------------------------------
def get_local_fft_shape(nr, realspace = True, full = False, **kwargs):
    s = []
    for item in nr :
        s.append(slice(None))
    s = tuple(s)
    shape = np.array(nr)
    if not full and not realspace :
        shape[-1] = shape[-1]//2 + 1
    offsets = np.zeros_like(nr, dtype = np.int)
    return (s, shape, offsets)

def _sum_1(a):
    s = np.sum(a)
    return s

def _add_2(a, b):
    s = a + b
    return s

def _mul_2(a, b):
    s = a * b
    return s

def add(*args):
    for i, item in enumerate(args):
        if i == 0 :
            a = item
        else :
            a = _add_2(a, item)
    return a

def sum(*args):
    a = add(*args)
    s = _sum_1(a)
    return s

def einsum(*args, **kwargs):
    s = np.einsum(*args, **kwargs)
    return s

def mul(*args):
    for i, item in enumerate(args):
        if i == 0 :
            a = item
        else :
            a = _mul_2(a, item)
    return a

def sum_mul(*args):
    a = mul(*args)
    s = _sum_1(a)
    return s

def allreduce(a, *args, **kwargs):
    return a

def vsum(a):
    return a

def asum(a):
    s = np.sum(a)
    return s

def amin(a):
    s = np.amin(a)
    return s

def amax(a):
    s = np.amax(a)
    return s

def amean(a):
    s = np.mean(a)
    return s

def split_number(n):
    return 0, n
