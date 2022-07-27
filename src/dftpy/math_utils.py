import numpy as np
from scipy import ndimage, interpolate
from scipy.optimize import minpack2
from scipy import optimize as sopt
import dftpy.mpi.mp_serial as mps
from dftpy.constants import environ

if environ["FFTLIB"] == "pyfftw":
    """
    pyfftw.config.NUM_THREADS  =  multiprocessing.cpu_count()
    print('threads', pyfftw.config.NUM_THREADS)
    """
    import pyfftw
    FFTWObj = pyfftw.FFTW
else :
    FFTWObj = object


# Global variables
FFT_SAVE = {
    "FFT_Grid": [np.zeros(3), np.zeros(3)],
    "IFFT_Grid": [np.zeros(3), np.zeros(3)],
    "FFT_OBJ": [None, None],
    "IFFT_OBJ": [None, None],
}


def partial_return(func, n=0, *args, **kwargs):
    def newfunc(*args, **keywords):
        return func(*args, **kwargs)[0]

    newfunc.func = func
    newfunc.args = args
    newfunc.kwargs = kwargs
    return newfunc


def LineSearchDcsrch(
    func, derfunc, alpha0=None, func0=None, derfunc0=None, c1=1e-4, c2=0.9, amax=1.0, amin=0.0, xtol=1e-14, maxiter=100
):

    isave = np.zeros((2,), np.intc)
    dsave = np.zeros((13,), float)
    task = b"START"

    if alpha0 is None:
        alpha0 = 0.0
        func0 = func(alpha0)
        derfunc0 = derfunc(alpha0)

    alpha1 = alpha0
    func1 = func0
    derfunc1 = derfunc0

    for i in range(maxiter):
        alpha1, func1, derfunc1, task = minpack2.dcsrch(
            alpha1, func1, derfunc1, c1, c2, xtol, task, amin, amax, isave, dsave
        )
        if task[:2] == b"FG":
            func1 = func(alpha1)
            derfunc1 = derfunc(alpha1)
        else:
            break
    else:
        alpha1 = None

    if task[:5] == b"ERROR" or task[:4] == b"WARN":
        alpha1 = None  # failed

    return alpha1, func1, derfunc1, task, i


def LineSearchDcsrch2(func, alpha0=None, func0=None, c1=1e-4, c2=0.9, amax=1.0, amin=0.0, xtol=1e-14, maxiter=100):
    isave = np.zeros((2,), np.intc)
    dsave = np.zeros((13,), float)
    task = b"START"

    if alpha0 is None:
        alpha0 = 0.0
        func0 = func(alpha0)

    alpha1 = alpha0
    x1 = func0[0]
    g1 = func0[1]
    alists = [alpha1]
    func1 = func0

    for i in range(maxiter):
        alpha1, x1, g1, task = minpack2.dcsrch(alpha1, x1, g1, c1, c2, xtol, task, amin, amax, isave, dsave)
        alists.append(alpha1)
        if task[:2] == b"FG":
            func1 = func(alpha1)
            x1 = func1[0]
            g1 = func1[1]
        else:
            break
    else:
        alpha1 = None

    if task[:5] == b"ERROR" or task[:4] == b"WARN":
        alpha1 = None  # failed
    # return alpha1, x1, g1, task, i, func1[2], func1[3]
    return alpha1, x1, g1, task, i, func1


def LineSearchDcsrchVector(func, alpha0=None, func0=None, c1=1e-4, c2=0.9, amax=1.0, amin=0.0, xtol=1e-14, maxiter=100):
    isave = np.zeros((2,), np.intc)
    dsave = np.zeros((13,), float)
    econv = 1E-5

    alpha1 = alpha0.copy()
    x1 = func0[0]
    g1 = func0[1]
    func1 = func0

    resA = []
    dirA = []
    it = 0
    for i in range(1, 5):
        alpha0 = alpha1.copy()
        resA.append(g1)
        direction = get_direction_CG(resA, dirA=dirA, method="CG-PR")
        dirA.append(direction)
        grad = (g1 * direction).sum()
        if grad > 0.0 :
            direction = -g1
            grad = (g1 * direction).sum()

        task = b"START"
        factor = (np.abs(direction).max())
        beta = min(0.1, 0.1 * np.pi/factor)
        # stop
        for j in range(maxiter):
            it += 1
            beta, x1, grad, task = minpack2.dcsrch(beta, x1, grad, c1, c2, xtol, task, amin/factor, amax/factor, isave, dsave)
            if task[:2] == b"FG":
                alpha1 = alpha0 + beta * direction
                func1 = func(alpha1)
                x1 = func1[0]
                g1 = func1[1]
                grad = (g1 * direction).sum()
            else:
                break
        else:
            alpha1 = None
        if task[:5] == b"ERROR" or task[:4] == b"WARN":
            alpha1 = None  # failed
        if alpha1 is None or (g1 * g1).sum() < econv :
            break
    return alpha1, x1, g1, task, it, func1


def Brent(func, alpha0=None, brack=(0.0, 1.0), tol=1e-8, full_output=1):
    f = partial_return(func, 0)
    alpha1, x1, _, i = sopt.brent(f, alpha0, brack=brack, tol=tol, full_output=full_output)
    return alpha1, x1, None, "CONV", i, func(alpha1)

class pyfftwFFTW(FFTWObj):
    '''
    https://github.com/pyFFTW/pyFFTW/issues/139
    '''
    def __new__(cls, arr, out, *args, **kwargs):
        obj = super().__new__(cls, arr, out, *args, **kwargs)
        obj.arr = arr
        obj.out = out
        return obj

    def __call__(self, input_array, *args, **kwargs):
        '''
        Here, copy the array into the input_array of FFTW instance. It takes some time, but safe.
        There may be a better way.
        '''
        value = self.arr
        value[:] = input_array
        results = super().__call__(value, *args, **kwargs)
        return results

def PYfft(grid, cplx=False, threads=1):
    global FFT_SAVE
    if environ["FFTLIB"] == "pyfftw":
        nr = grid.nr
        if np.all(nr == FFT_SAVE["FFT_Grid"][cplx]):
            fft_object = FFT_SAVE["FFT_OBJ"][cplx]
        else:
            if cplx:
                rA = pyfftw.empty_aligned(tuple(nr), dtype="complex128")
                cA = pyfftw.empty_aligned(tuple(nr), dtype="complex128")
            else:
                nrc = grid.nrG
                rA = pyfftw.empty_aligned(tuple(nr), dtype="float64")
                cA = pyfftw.empty_aligned(tuple(nrc), dtype="complex128")
            fft_object = pyfftwFFTW(
                rA, cA, axes=(0, 1, 2), flags=("FFTW_MEASURE",), direction="FFTW_FORWARD", threads=threads
            )
            FFT_SAVE["FFT_Grid"][cplx] = nr
            FFT_SAVE["FFT_OBJ"][cplx] = fft_object
        return fft_object


def PYifft(grid, cplx=False, threads=1):
    global FFT_SAVE
    if environ["FFTLIB"] == "pyfftw":
        nr = grid.nrR
        if np.all(nr == FFT_SAVE["IFFT_Grid"][cplx]):
            fft_object = FFT_SAVE["IFFT_OBJ"][cplx]
        else:
            if cplx:
                rA = pyfftw.empty_aligned(tuple(nr), dtype="complex128")
                cA = pyfftw.empty_aligned(tuple(nr), dtype="complex128")
            else:
                nrc = grid.nr
                rA = pyfftw.empty_aligned(tuple(nr), dtype="float64")
                cA = pyfftw.empty_aligned(tuple(nrc), dtype="complex128")
            fft_object = pyfftwFFTW(
                cA, rA, axes=(0, 1, 2), flags=("FFTW_MEASURE",), direction="FFTW_BACKWARD", threads=threads
            )
            FFT_SAVE["IFFT_Grid"][cplx] = nr
            FFT_SAVE["IFFT_OBJ"][cplx] = fft_object
        return fft_object


def PowerInt(x, numerator, denominator=1):
    y = x.copy()
    for i in range(numerator - 1):
        np.multiply(y, x, out=y)
    if denominator == 1:
        return y
    elif denominator == 2:
        np.sqrt(y, out=y)
    elif denominator == 3:
        np.cbrt(y, out=y)
    elif denominator == 4:
        np.sqrt(y, out=y)
        np.sqrt(y, out=y)
    else:
        np.power(y, 1.0 / denominator, out=y)
    return y


def bestFFTsize(n, nproc = 1, opt = True, **kwargs):
    if opt :
        n_l = n/nproc
    else :
        n_l = n
    n_l = mps.best_fft_size(n_l, **kwargs)

    if opt :
        n_best = n_l * nproc
    else :
        n_best = n_l
    return n_best

def interpolation_3d(arr, nr_new, interp="map"):
    nr = np.array(arr.shape)
    values = np.pad(arr, ((0, 1), (0, 1), (0, 1)), mode="wrap")
    # values = arr
    x = np.linspace(0, 1, nr_new[0], endpoint=False) * nr[0]
    y = np.linspace(0, 1, nr_new[1], endpoint=False) * nr[1]
    z = np.linspace(0, 1, nr_new[2], endpoint=False) * nr[2]
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    if interp == "map":
        new_values = ndimage.map_coordinates(values, (X, Y, Z), mode="wrap")
        new_values[np.isnan(new_values)] = 0.0
    else:
        values = np.pad(arr, ((0, 1), (0, 1), (0, 1)), mode="wrap")
        x0 = np.linspace(0, 1, nr[0] + 1, endpoint=True)
        y0 = np.linspace(0, 1, nr[1] + 1, endpoint=True)
        z0 = np.linspace(0, 1, nr[2] + 1, endpoint=True)
        X0, Y0, Z0 = np.meshgrid(x0, y0, z0, indexing="ij")
        points = np.c_[X0.ravel(), Y0.ravel(), Z0.ravel()]
        new_values = interpolate.griddata(points, values.ravel(), (X, Y, Z), method="linear")
    return new_values


def restriction(arr, scheme="bilinear"):
    """
    For speed, here the coarse gride size must be twice of the dense grid size.
    """
    nr = np.array(arr.shape)
    # f  = arr.copy()
    # f[1:nr[0], :, :] = 0.25 * (f[:nr[0]-1, :, :]+ 2.0 * f[1:nr[0], :, :]+ f[2:, :, :])
    # f[0, :, :] = 0.25 * (f[0, :, :]+ 2.0 * f[1, :, :]+ f[nr[0]:, :, :])
    # f[nr[0], :, :] = 0.25 * (f[0, :, :]+ 2.0 * f[nr[0]-1, :, :]+ f[nr[0]:, :, :])
    f = np.pad(arr, ((1, 1), (1, 1), (1, 1)), mode="wrap")
    f[1 : nr[0] + 1, :, :] = 0.25 * (f[: nr[0], :, :] + 2.0 * f[1 : nr[0] + 1, :, :] + f[2:, :, :])
    f[:, 1 : nr[0] + 1, :] = 0.25 * (f[:, : nr[0], :] + 2.0 * f[:, 1 : nr[0] + 1, :] + f[:, 2:, :])
    f[:, :, 1 : nr[2] + 1] = 0.25 * (f[:, :, : nr[2]] + 2.0 * f[:, :, 1 : nr[2] + 1] + f[:, :, 2:])
    nr2 = nr // 2 + 1
    new_values = f[0 : nr2[0] : 2, 0 : nr2[1] : 2, 0 : nr2[2] : 2]
    # for safe and memory
    new_values = new_values.copy()
    return new_values


def prolongation(arr, scheme="bilinear"):
    nr = np.array(arr.shape)
    # nr2 = nr + 1
    # print(nr2, nr)
    f1 = np.pad(arr, ((0, 1), (0, 1), (0, 1)), mode="wrap")
    f2 = f1 * 0.5
    f4 = f1 * 0.25
    f8 = f1 * 0.125
    nrN = nr * 2
    new_values = np.zeros(nrN)
    new_values[::2, ::2, ::2] = arr
    # line
    new_values[1::2, ::2, ::2] = f2[: nr[0], : nr[1], : nr[2]] + f2[1:, : nr[1], : nr[2]]
    new_values[::2, 1::2, ::2] = f2[: nr[0], : nr[1], : nr[2]] + f2[: nr[0], 1:, : nr[2]]
    new_values[::2, ::2, 1::2] = f2[: nr[0], : nr[1], : nr[2]] + f2[: nr[0], : nr[1], 1:]
    # plane
    new_values[1::2, 1::2, ::2] = (
        f4[: nr[0], : nr[1], : nr[2]] + f4[1:, : nr[1], : nr[2]] + f4[: nr[0], 1:, : nr[2]] + f4[1:, 1:, : nr[2]]
    )
    new_values[1::2, ::2, 1::2] = (
        f4[: nr[0], : nr[1], : nr[2]] + f4[1:, : nr[1], : nr[2]] + f4[: nr[0], : nr[1], 1:] + f4[1:, : nr[1], 1:]
    )
    new_values[::2, 1::2, 1::2] = (
        f4[: nr[0], : nr[1], : nr[2]] + f4[: nr[0], 1:, : nr[2]] + f4[: nr[0], : nr[1], 1:] + f4[: nr[0], 1:, 1:]
    )
    # box
    new_values[1::2, 1::2, 1::2] = (
        f8[: nr[0], : nr[1], : nr[2]]
        + f8[1:, : nr[1], : nr[2]]
        + f8[: nr[0], 1:, : nr[2]]
        + f8[: nr[0], : nr[1], 1:]
        + f8[1:, 1:, : nr[2]]
        + f8[1:, : nr[1], 1:]
        + f8[: nr[0], 1:, 1:]
        + f8[1:, 1:, 1:]
    )

    return new_values


def spacing2ecut(spacing):
    """
    Ecut = pi^2/(2 * h^2)
    Ref : Briggs, E. L., D. J. Sullivan, and J. Bernholc. Physical Review B 54.20 (1996): 14362.
    """
    return np.pi ** 2 / (2 * spacing ** 2)


def ecut2spacing(ecut):
    return np.sqrt(np.pi ** 2 / ecut * 0.5)

def ecut2nr(ecut = None, lattice = None, optfft = True, spacing = None, nproc = 1, **kwargs):
    spacings = np.ones(3)
    optffts = np.ones(3, dtype = 'bool')
    optffts[:] = optfft
    nr = np.zeros(3, dtype = 'int32')
    if spacing is None :
        spacings[:] = ecut2spacing(ecut)
    else :
        spacings[:] = spacing
    metric = np.dot(lattice, lattice.T)
    for i in range(3):
        # nr[i] = np.ceil(np.sqrt(metric[i, i])/spacings[i])
        nr[i] = int(np.sqrt(metric[i, i])/spacings[i])
        nr[i] += nr[i] %2
        if optffts[i] :
            nr[i] = bestFFTsize(nr[i], nproc = nproc, **kwargs)
    return nr


class FDcoef(object):
    def __init__(self, deriv=2, order=4, **kwargs):
        self.deriv = deriv
        self.order = order


class LBFGS(object):
    def __init__(self, H0=1.0, Bound=5):
        self.Bound = Bound
        self.H0 = H0
        self.s = []
        self.y = []
        self.rho = []

    def update(self, dx, dg):
        if len(self.s) > self.Bound:
            self.s.pop(0)
            self.y.pop(0)
            self.rho.pop(0)
        self.s.append(dx)
        self.y.append(dg)
        asum = dx.grid.mp.asum if hasattr(dx, 'grid') else np.sum
        rho = 1.0 / asum(dg*dx)

        self.rho.append(rho)


def get_direction_CG(resA, dirA=None, method="CG-HS", **kwargs):
    """
    https ://en.wikipedia.org/wiki/Conjugate_gradient_method
    HS->DY->CD

    """
    asum = resA[-1].grid.mp.asum if hasattr(resA[-1], 'grid') else np.sum
    if len(resA) == 1:
        beta = 0.0
    elif method == "CG-HS" and len(dirA) > 0:  # Maybe the best of the CG.
        beta = asum(resA[-1] * (resA[-1] - resA[-2])) / asum(dirA[-1] * (resA[-1] - resA[-2]))
    elif method == "CG-FR":
        beta = asum(resA[-1] ** 2) / asum(resA[-2] ** 2)
    elif method == "CG-PR":
        beta = asum(resA[-1] * (resA[-1] - resA[-2])) / asum(resA[-2] ** 2)
        beta = max(beta, 0.0)
    elif method == "CG-DY" and len(dirA) > 0:
        beta = asum(resA[-1] ** 2) / asum(dirA[-1] * (resA[-1] - resA[-2]))
    elif method == "CG-CD" and len(dirA) > 0:
        beta = -(resA[-1] ** 2) / asum(dirA[-1] * resA[-2])
    elif method == "CG-LS" and len(dirA) > 0:
        beta = asum(resA[-1] * (resA[-1] - resA[-2])) / asum(dirA[-1] * resA[-2])
    else:
        beta = asum(resA[-1] ** 2) / asum(resA[-2] ** 2)

    if dirA is None or len(dirA) == 0:
        direction = -resA[-1]
    else:
        direction = -resA[-1] + beta * dirA[-1]

    return direction


def get_direction_GD(resA, dirA=None, method="GD", **kwargs):
    direction = -resA[-1]
    return direction


def get_direction_LBFGS(resA, lbfgs=None, **kwargs):
    direction = np.zeros_like(resA[-1])
    q = -resA[-1]
    alphaList = np.zeros(len(lbfgs.s))
    asum = resA[-1].grid.mp.asum if hasattr(resA[-1], 'grid') else np.sum
    for i in range(len(lbfgs.s) - 1, 0, -1):
        # alpha = lbfgs.rho[i] * (lbfgs.s[i] * q).asum()
        alpha = lbfgs.rho[i] * asum(lbfgs.s[i] * q)
        alphaList[i] = alpha
        q -= alpha * lbfgs.y[i]

    if not lbfgs.H0:
        if len(lbfgs.s) < 1:
            gamma = 1.0
        else:
            gamma = asum(lbfgs.s[-1] * lbfgs.y[-1]) / asum(lbfgs.y[-1] * lbfgs.y[-1])
        direction = gamma * q
    else:
        direction = lbfgs.H0 * q

    for i in range(len(lbfgs.s)):
        beta = lbfgs.rho[i] * asum(lbfgs.y[i] * direction)
        direction += lbfgs.s[i] * (alphaList[i] - beta)

    return direction

def quartic_interpolation(f, dx):
    if len(f) == 5 :
        t0 = f[2]
        t1 = 2.0 * f[0] - 16.0 * f[1] + 16.0 * f[3] - 2.0 * f[4]
        t2 = -f[0] + 16.0 * f[1] - 30.0 * f[2] + 16.0 * f[3] - f[4]
        t3 = -2.0 * f[0] + 4.0 * f[1] - 4.0 * f[3] + 2.0 * f[4]
        t4 = f[0] - 4.0 * f[1] + 6.0 * f[2] - 4.0 * f[3] + f[4]
        results = t0 + dx * (t1 + dx * (t2 + dx * (t3 + dx * t4)))/24.0
    else :
        raise AttributeError("Error : Not implemented yet")
    return results
