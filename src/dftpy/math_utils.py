import numpy as np
import scipy.special as sp
from scipy import ndimage
from scipy.optimize import minpack2
import time
from dftpy.constants import FFTLIB

if FFTLIB == 'pyfftw' :
    import pyfftw
    # pyfftw.config.NUM_THREADS  =  multiprocessing.cpu_count()
    # print('threads', pyfftw.config.NUM_THREADS)

# Global variables
FFT_SAVE = {
        'FFT_Grid' : [np.zeros(3), np.zeros(3)],
        'IFFT_Grid' : [np.zeros(3), np.zeros(3)],
        'FFT_OBJ' : [None, None], 
        'IFFT_OBJ' : [None, None]}

def LineSearchDcsrch(func, derfunc, alpha0 = None, func0=None, derfunc0=None,
        c1=1e-4, c2=0.9, amax=1.0, amin=0.0, xtol=1e-14, maxiter = 100):

    isave = np.zeros((2,), np.intc)
    dsave = np.zeros((13,), float)
    task = b'START'

    if alpha0 is None :
        alpha0 = 0.0
        func0 = func(alpha0)
        derfunc0 = derfunc(alpha0)

    alpha1 = alpha0
    func1 = func0
    derfunc1 = derfunc0

    for i in range(maxiter):
        alpha1, func1, derfunc1, task = minpack2.dcsrch(alpha1, func1, derfunc1,
                                                   c1, c2, xtol, task,
                                                   amin, amax, isave, dsave)
        if task[:2] == b'FG':
            func1 = func(alpha1)
            derfunc1 = derfunc(alpha1)
        else:
            break
    else:
        alpha1 = None

    if task[:5] == b'ERROR' or task[:4] == b'WARN':
        alpha1 = None  # failed

    return alpha1, func1, derfunc1, task, i

def LineSearchDcsrch2(func,alpha0 = None, func0=None, \
        c1=1e-4, c2=0.9, amax=1.0, amin=0.0, xtol=1e-14, maxiter = 100):

    isave = np.zeros((2,), np.intc)
    dsave = np.zeros((13,), float)
    task = b'START'

    if alpha0 is None :
        alpha0 = 0.0
        func0 = func(alpha0)

    alpha1 = alpha0
    x1 = func0[0]
    g1 = func0[1]
    alists = [alpha1]
    func1 = func0
    tol = 1E-5
    # tol = 0.0

    for i in range(maxiter):
        alpha1, x1, g1, task = minpack2.dcsrch(alpha1, x1, g1, c1, c2, xtol, task,
                                                   amin, amax, isave, dsave)
        alists.append(alpha1)
        if len(alists) > 3 :
            sum0 = 0.0
            for a in alists[-4:] :
                sum0 += abs(a)
            if abs(sum0/4.0 - alists[-1]) < tol :
                # task = b'WARNING : THE STEP CHANGE TOO SMALL'
                task = b'CONV : THE STEP CHANGE TOO SMALL'
                break
        if task[:2] == b'FG':
            func1 = func(alpha1)
            x1 = func1[0]
            g1 = func1[1]
        else:
            break
    else:
        alpha1 = None

    if task[:5] == b'ERROR' or task[:4] == b'WARN':
        alpha1 = None  # failed

    # return alpha1, x1, g1, task, i, func1[2], func1[3]
    return alpha1, x1, g1, task, i, func1

class TimeObj(object):
    '''
    '''
    def __init__(self,  **kwargs):
        self.reset(**kwargs)

    def reset(self,  **kwargs):
        self.labels = []
        self.tic = {}
        self.toc = {}
        self.cost = {}
        self.number = {}

    def Begin(self, label):
        if label in self.tic :
            self.number[label] += 1
        else :
            self.labels.append(label)
            self.number[label] = 1
            self.cost[label] = 0.0

        self.tic[label] = time.time()

    def Time(self, label):
        if label not in self.tic :
            print(' !!! ERROR : You should add "Begin" before this')
        else :
            t = time.time() - self.tic[label]
        return t

    def End(self, label):
        if label not in self.tic :
            print(' !!! ERROR : You should add "Begin" before this')
        else :
            self.toc[label] = time.time()
            t = time.time() - self.tic[label]
            self.cost[label] += t
        return t
TimeData = TimeObj()

def PYfft(grid, cplx = False, threads = 1):
    global FFT_SAVE
    if FFTLIB == 'pyfftw' :
        nr = grid.nr
        if np.all(nr == FFT_SAVE['FFT_Grid'][cplx]):
            fft_object = FFT_SAVE['FFT_OBJ'][cplx]
        else :
            if cplx :
                rA = pyfftw.empty_aligned(tuple(nr), dtype='complex128')
                cA = pyfftw.empty_aligned(tuple(nr), dtype='complex128')
            else :
                nrc = grid.nrG
                rA = pyfftw.empty_aligned(tuple(nr), dtype='float64')
                cA = pyfftw.empty_aligned(tuple(nrc), dtype='complex128')
            fft_object = pyfftw.FFTW(rA, cA, axes = (0, 1, 2),\
                    flags=('FFTW_MEASURE',), direction='FFTW_FORWARD', threads = threads)
            FFT_SAVE['FFT_Grid'][cplx] = nr
            FFT_SAVE['FFT_OBJ'][cplx] = fft_object
        return fft_object

def PYifft(grid, cplx = False, threads = 1):
    global FFT_SAVE
    if FFTLIB == 'pyfftw' :
        nr = grid.nrR
        if np.all(nr == FFT_SAVE['IFFT_Grid'][cplx]):
            fft_object = FFT_SAVE['IFFT_OBJ'][cplx]
        else :
            if cplx :
                rA = pyfftw.empty_aligned(tuple(nr), dtype='complex128')
                cA = pyfftw.empty_aligned(tuple(nr), dtype='complex128')
            else :
                nrc = grid.nr
                rA = pyfftw.empty_aligned(tuple(nr), dtype='float64')
                cA = pyfftw.empty_aligned(tuple(nrc), dtype='complex128')
                fft_object = pyfftw.FFTW(cA, rA, axes = (0, 1, 2), \
                        flags=('FFTW_MEASURE',), direction='FFTW_BACKWARD', threads = threads)
            FFT_SAVE['IFFT_Grid'][cplx] = nr
            FFT_SAVE['IFFT_OBJ'][cplx] = fft_object
        return fft_object

def PowerInt(x, numerator, denominator = 1):
    y = x.copy()
    for i in range(numerator-1):
        np.multiply(y, x, out = y)
    if denominator == 1 :
        return y
    elif denominator == 2 :
        np.sqrt(y, out = y)
    elif denominator == 3 :
        np.cbrt(y, out = y)
    elif denominator == 4 :
        np.sqrt(y, out = y)
        np.sqrt(y, out = y)
    else :
        np.power(y, 1.0/denominator, out = y)
    return y

def bestFFTsize(N):
    '''
    http ://www.fftw.org/fftw3_doc/Complex-DFTs.html#Complex-DFTs
    "FFTW is best at handling sizes of the form 2^a 3^b 5^c 7^d 11^e 13^f,  where e+f is either 0 or 1,  and the other exponents are arbitrary."
    '''
    a = int(np.log2(N)) + 2                                                                                                                      
    b = int(np.log(N)/np.log(3)) + 2                                                                                                             
    c = int(np.log(N)/np.log(5)) + 2                                                                                                             
    d = int(np.log(N)/np.log(7)) + 2                                                                                                             
    even = True
    if even :
        mgrid = np.mgrid[1:a, :b, :c, :d].reshape(4, -1)
    else :
        mgrid = np.mgrid[:a, :b, :c, :d].reshape(4, -1)
    arr0 = 2 ** mgrid[0] * 3 ** mgrid[1] * 5 ** mgrid[2] * 7 ** mgrid[3]
    if N < 100 :
        arr1=arr0[np.logical_and(arr0>N/14, arr0<2*N)]
    else :
        arr1=arr0[np.logical_and(arr0>N/14, arr0<1.2*N)]
    arrAll=[]
    arrAll.extend(arr1)
    arrAll.extend(arr1*11)
    arrAll.extend(arr1*13)
    arrAll = np.asarray(arrAll)                                                                                                                  
    # bestN = np.min(arrAll[arrAll > N-1])   
    bestN = np.min(arrAll[arrAll > 0.99*N])   
    return bestN

def interpolation_3d(arr, nr_new, interp = 'map'):
    nr = np.array(arr.shape)
    values = np.pad(arr, ((0, 1), (0, 1), (0, 1)), mode='wrap')
    # values = arr
    x = np.linspace(0, 1, nr_new[0], endpoint=False) * nr[0]
    y = np.linspace(0, 1, nr_new[1], endpoint=False) * nr[1]
    z = np.linspace(0, 1, nr_new[2], endpoint=False) * nr[2]
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    if interp == 'map' :
        new_values = ndimage.map_coordinates(values, (X, Y, Z), mode='wrap')
        new_values[np.isnan(new_values)] = 0.0
    else :
        values = np.pad(arr, ((0, 1), (0, 1), (0, 1)), mode='wrap')
        x0= np.linspace(0, 1, self.grid.nr[0]+1, endpoint=True) 
        y0= np.linspace(0, 1, self.grid.nr[1]+1, endpoint=True) 
        z0= np.linspace(0, 1, self.grid.nr[2]+1, endpoint=True) 
        X0, Y0, Z0 = np.meshgrid(x0, y0, z0, indexing='ij')
        points = np.c_[X0.ravel(), Y0.ravel(), Z0.ravel()]
        new_values = interpolate.griddata(points, values.ravel(), (X, Y, Z), method = 'linear')
    return new_values

def restriction(arr, scheme = 'bilinear'):
    '''
    For speed, here the coarse gride size must be twice of the dense grid size.
    '''
    nr = np.array(arr.shape)
    # f  = arr.copy()
    # f[1:nr[0], :, :] = 0.25 * (f[:nr[0]-1, :, :]+ 2.0 * f[1:nr[0], :, :]+ f[2:, :, :])
    # f[0, :, :] = 0.25 * (f[0, :, :]+ 2.0 * f[1, :, :]+ f[nr[0]:, :, :])
    # f[nr[0], :, :] = 0.25 * (f[0, :, :]+ 2.0 * f[nr[0]-1, :, :]+ f[nr[0]:, :, :])
    f = np.pad(arr, ((1, 1), (1, 1), (1, 1)), mode='wrap')
    f[1:nr[0]+1, :, :] = 0.25 * (f[:nr[0], :, :]+ 2.0 * f[1:nr[0]+1, :, :]+ f[2:, :, :])
    f[:, 1:nr[0]+1, :] = 0.25 * (f[:, :nr[0], :]+ 2.0 * f[:, 1:nr[0]+1, :]+ f[:, 2:, :])
    f[:, :, 1:nr[2]+1] = 0.25 * (f[:, :, :nr[2]]+ 2.0 * f[:, :, 1:nr[2]+1]+ f[:, :, 2:])
    nr2 = nr//2 + 1
    new_values = f[0:nr2[0]:2, 0:nr2[1]:2, 0:nr2[2]:2]
    # for safe and memory
    new_values = new_values.copy()
    return new_values

def prolongation(arr, scheme = 'bilinear'):
    nr = np.array(arr.shape)
    nr2 = nr + 1
    print(nr2, nr)
    f1 = np.pad(arr, ((0, 1), (0, 1), (0, 1)), mode='wrap')
    f2 = f1 * 0.5
    f4 = f1 * 0.25
    f8 = f1 * 0.125
    nrN = nr * 2
    new_values = np.zeros(nrN)
    new_values[::2, ::2, ::2] = arr
    #line
    new_values[1::2, ::2, ::2] = f2[:nr[0], :nr[1], :nr[2]] + f2[1:, :nr[1], :nr[2]] 
    new_values[::2, 1::2, ::2] = f2[:nr[0], :nr[1], :nr[2]] + f2[:nr[0], 1:, :nr[2]] 
    new_values[::2, ::2, 1::2] = f2[:nr[0], :nr[1], :nr[2]] + f2[:nr[0], :nr[1], 1:] 
    #plane
    new_values[1::2, 1::2, ::2] = f4[:nr[0], :nr[1], :nr[2]] + f4[1:, :nr[1], :nr[2]]\
            + f4[:nr[0], 1:, :nr[2]] + f4[1:, 1:, :nr[2]]
    new_values[1::2, ::2, 1::2] = f4[:nr[0], :nr[1], :nr[2]] + f4[1:, :nr[1], :nr[2]]\
            + f4[:nr[0], :nr[1], 1:] + f4[1:, :nr[1], 1:]
    new_values[::2, 1::2, 1::2] = f4[:nr[0], :nr[1], :nr[2]] + f4[:nr[0], 1:, :nr[2]]\
            + f4[:nr[0], :nr[1], 1:] + f4[:nr[0], 1:, 1:]
    #box
    new_values[1::2, 1::2, 1::2] = f8[:nr[0], :nr[1], :nr[2]]\
            + f8[1:, :nr[1], :nr[2]] + f8[:nr[0], 1:, :nr[2]] + f8[:nr[0], :nr[1], 1:]\
            + f8[1:, 1:, :nr[2]]+ f8[1:, :nr[1], 1:] + f8[:nr[0], 1:, 1:]\
            + f8[1:, 1:, 1:]

    return new_values

def spacing2ecut(spacing, ecut):
    '''
    Ecut = pi^2/(2 * h^2)
    Ref : Briggs, E. L., D. J. Sullivan, and J. Bernholc. "Real-space multigrid-based approach to large-scale electronic structure calculations." Physical Review B 54.20 (1996): 14362.
    '''
    return np.pi ** 2/ (2*spacing**2)

def ecut2spacing(ecut, spacing):
    return np.sqrt(np.pi ** 2/ ecut * 0.5)

class FDcoef(object):
    def __init__(self, deriv = 2, order = 4, **kwargs):
        self.deriv = deriv
        self.order = order
