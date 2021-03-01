import warnings
import numpy as np
from scipy import ndimage
from scipy import signal
from dftpy.grid import DirectGrid, ReciprocalGrid
from dftpy.constants import environ
from dftpy.math_utils import PYfft, PYifft
from dftpy.time_data import TimeData
from dftpy.base import Coord


class BaseField(np.ndarray):
    """
    Extended numpy array representing a field on a grid
    (Cell (lattice) plus discretization)

    Attributes
    ----------
    self : np.ndarray
        the values of the field

    grid : Grid
        Represent the domain of the function

    span : number of directions for which we have more than 1 point
            e.g.: for np.zeros((5,5,1)) -> ndim = 3, span = 2

    rank : rank of the field. default = 1, i.e. scalar field

    memo : optional string to label the field

    """

    def __new__(cls, grid, memo="", rank=1, griddata_F=None, griddata_C=None, griddata_3d=None, fft_data = None, cplx = False):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type

        if griddata_3d is not None:
            if isinstance(griddata_3d, list):
                rank = len(griddata_3d)

        if rank == 1:
            nr = grid.nr
        else:
            nr = rank, *grid.nr
        if griddata_F is None and griddata_C is None and griddata_3d is None:
            if cplx :
                input_values = np.zeros(nr, dtype ='complex128')
            else :
                input_values = np.zeros(nr)
        elif griddata_F is not None:
            input_values = np.reshape(griddata_F, nr, order="F")
        elif griddata_C is not None:
            input_values = np.reshape(griddata_C, nr, order="C")
        elif griddata_3d is not None:
            input_values = np.asarray(griddata_3d)
            input_values = np.reshape(input_values, nr)
            # input_values = np.reshape(griddata_3d, nr)

        obj = np.asarray(input_values).view(cls)
        # add the new attribute to the created instance
        obj.grid = grid
        obj.span = (grid.nr > 1).sum()
        obj.rank = rank
        obj.memo = str(memo)
        # add fft data
        obj._fft_data = fft_data
        obj.mp = obj.grid.mp
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # Restore attributes when we are taking a slice
        # getting the rank right - or at least trying to do so....
        # self.rank = getattr(obj, 'rank', None)
        a = np.shape(np.shape(self))[0]
        if a == 4:
            rank = np.shape(self)[0]
        else:
            rank = 1
        self.rank = rank
        if obj is None:
            return
        self.grid = getattr(obj, "grid", None)
        self.span = getattr(obj, "span", None)
        self.memo = getattr(obj, "memo", None)
        self.mp = getattr(obj, "mp", None)
        if np.prod(self.shape) != np.prod(obj.shape):
            #clean saved fft_data
            self._fft_data = None
        else :
            self._fft_data = getattr(obj, "_fft_data", None)

    @property
    def fft_data(self):
        if self._fft_data is None :
            return None
        else :
            result = self._fft_data.copy()
            result._fft_data = self
            return result
            self.grid = new_grid

    def __array_wrap__(self, obj, context=None):
        """wrap it up"""
        b = np.ndarray.__array_wrap__(self, obj, context)
        a = np.shape(np.shape(b))[0]
        # a = np.shape(np.shape(self))[0]
        # self.rank = np.max([self.rank, obj.rank])
        if a == 4:
            rank = np.shape(b)[0]
            #rank = np.shape(self)[0]
        else:
            rank = 1
        # if rank == 1:
        # b = np.reshape(b, self.grid.nr)
        # b = np.reshape(b,nr)
        b.rank = rank
        #clean saved fft_data
        b._fft_data = None
        return b

    def __setitem__(self,*args,**kwargs):
        #clean saved fft_data
        self._fft_data = None
        return super().__setitem__(*args,**kwargs)

    def dot(self, obj):
        """ Returns the dot product of vector fields self and obj """
        if np.shape(self) != np.shape(obj):
            raise ValueError("Shape incompatible")  # to be specified

        prod = self.mp.einsum("ijkl,ijkl->jkl", self, obj)
        prod = np.expand_dims(prod, axis=3)

        return type(self)(self.grid, rank=1, griddata_3d=prod)

    def project(self, obj):
        """ Returns the field that self projects on obj """
        return obj*(np.conj(obj.normalize())*self).integral()

    def norm(self):
        return np.sqrt((np.real(np.conj(self) * self)).integral())

    def normalize(self, N=1.0):
        """ Normalize the field to N """
        return self / self.norm() * np.sqrt(N)

    def asum(self):
        return self.mp.asum(self)

    def amean(self):
        return self.mp.amean(self)

    def amax(self):
        return self.mp.amax(self)

    def amin(self):
        return self.mp.amin(self)


class DirectField(BaseField):
    spl_order = 3

    def __new__(cls, grid, memo="", rank=1, griddata_F=None, griddata_C=None, griddata_3d=None, cplx=False, fft_data = None):
        if not isinstance(grid, DirectGrid):
            raise TypeError("the grid argument is not an instance of DirectGrid")
        obj = super().__new__(
            cls, grid, memo="", rank=rank, griddata_F=griddata_F, griddata_C=griddata_C, griddata_3d=griddata_3d,
            cplx = cplx, fft_data = fft_data)
        obj._N = None
        obj.spl_coeffs = None
        obj._cplx = cplx
        if obj.mp.is_mpi :
            from dftpy.mpi.mp_mpi4py import mpi_fft
            obj.fft_object = mpi_fft(obj.grid)
        elif not obj._cplx and np.all(obj.grid.nr == obj.grid.nrG):  # Can only use numpy.fft
            obj.fft_object = np.fft.fftn
        elif environ["FFTLIB"] == "pyfftw":
            obj.fft_object = PYfft(obj.grid, obj._cplx)
        elif environ["FFTLIB"] == "numpy":
            if obj._cplx:
                obj.fft_object = np.fft.fftn
            else:
                obj.fft_object = np.fft.rfftn
        else :
            obj.fft_object = np.fft.fftn
        return obj

    def __array_finalize__(self, obj):
        # Restore attributes when we are taking a slice
        super().__array_finalize__(obj)
        if obj is None:
            return
        if isinstance(obj, (DirectField)):
            self.fft_object = getattr(obj, 'fft_object', None)
            self.cplx = getattr(obj, 'cplx', None)
        self.spl_coeffs = None
        self._N = None

    # def __array_wrap__(self,obj,context=None):
        # b = super().__array_wrap__(obj, context)
        # if isinstance(obj, (DirectField)):
            # b.fft_object = obj.fft_object
            # b.cplx = obj.cplx
        # else :
            # b.fft_object = self.fft_object
            # b.cplx = self.cplx
        # return b

    def integral(self, gather = True):
        """ Returns the integral of self """
        mp = self.mp if gather else np
        if self.rank == 1:
            return mp.einsum("ijk->", self) * self.grid.dV
        else:
            return mp.einsum("ijkl->i", self) * self.grid.dV

    def _calc_spline(self):
        padded_values = np.pad(self, ((self.spl_order,)), mode="wrap")
        self.spl_coeffs = ndimage.spline_filter(padded_values, order=self.spl_order)
        return

    def window_functions(self, window="hann"):
        """Only support for serial"""
        nr = self.nr
        Wx = signal.get_window(window, nr[0])
        Wy = signal.get_window(window, nr[1])
        Wz = signal.get_window(window, nr[2])
        Wxyz = self.mp.einsum("i, j, k -> ijk", Wx, Wy, Wz)
        return Wxyz

    def numerically_smooth_gradient(self, ipol=None):
        sq_self = np.sqrt(np.abs(self))
        grad = sq_self.standard_gradient(ipol)
        if ipol is None:
            final = np.empty(np.shape(grad), dtype=float)
            dim = np.shape(np.shape(sq_self))[0]
            if dim == 4:
                a = sq_self[0]
            if dim == 3:
                a = sq_self[:]
            for ipol in np.arange(grad.rank):
                final[ipol] = 2.0 * a * grad[ipol]
            return DirectField(grid=grad.grid, rank=grad.rank, griddata_3d=final)
        else:
            if grad.rank != 1:
                raise ValueError("Gradient rank incompatible with shape")
            return DirectField(grid=grad.grid, rank=1, griddata_3d=2.0 * sq_self * np.reshape(grad, np.shape(sq_self)))

    def super_smooth_gradient(self, ipol=None, force_real=True):
        reciprocal_self = self.fft()
        imag = 0 + 1j
        nr = 3, *reciprocal_self.grid.nr
        grad_g = np.empty(nr, dtype='complex128')
        if ipol is None:
            # FFT(\grad A) = i \vec(G) * FFT(A)
            grad_g = (
                reciprocal_self.grid.g
                * (reciprocal_self * imag)
                * np.exp(-reciprocal_self.grid.gg * (0.1 / 2.0) ** 2)
            )
            grad_g = ReciprocalField(grid=self.grid.get_reciprocal(), rank=3, griddata_3d=grad_g)
            grad = grad_g.ifft(force_real=force_real)
            if grad.rank != np.shape(grad)[0]:
                raise ValueError("Standard Gradient: Gradient rank incompatible with shape")
            return grad
        else:
            i = ipol - 1
            grad_g = (
                reciprocal_self.grid.g[i]
                * (reciprocal_self * imag)
                * np.exp(-reciprocal_self.grid.gg * (0.1 / 2.0) ** 2)
            )
            grad_g = ReciprocalField(grid=self.grid.get_reciprocal(), rank=1, griddata_3d=grad_g)
            grad = grad_g.ifft(force_real=force_real)
            if grad.rank != np.shape(grad)[0]:
                raise ValueError("Standard Gradient: Gradient rank incompatible with shape")
            return grad

    def standard_gradient(self, ipol=None, force_real=True):
        reciprocal_self = self.fft()
        imag = 0 + 1j
        # nr = 3, *reciprocal_self.grid.nr
        # grad_g = np.empty(nr, dtype='complex128')
        if ipol is None:
            # FFT(\grad A) = i \vec(G) * FFT(A)
            grad_g = reciprocal_self.grid.g * (reciprocal_self * imag)
            grad_g = ReciprocalField(grid=self.grid.get_reciprocal(), rank=3, griddata_3d=grad_g)
        elif ipol > 3:
            raise ValueError("Standard Gradient: ipol can not large than 3")
        else:
            i = ipol - 1
            grad_g = reciprocal_self.grid.g[i] * (reciprocal_self * imag)
            grad_g = ReciprocalField(grid=self.grid.get_reciprocal(), rank=1, griddata_3d=grad_g)
        grad = grad_g.ifft(force_real=force_real)
        return grad

    def divergence(self, flag="smooth", force_real=True):
        if self.rank != 3:
            raise ValueError("Divergence: Rank incompatible ", self.rank)
        div = self[0].gradient(flag=flag, ipol=1, force_real=force_real)
        div += self[1].gradient(flag=flag, ipol=2, force_real=force_real)
        div += self[2].gradient(flag=flag, ipol=3, force_real=force_real)
        div.rank = 1
        return div

    def gradient(self, flag="smooth", ipol=None, force_real=True):
        if self.rank > 1 and ipol is None:
            raise Exception("gradient is only implemented for scalar fields")
        if flag == "standard":
            return self.standard_gradient(ipol, force_real)
        elif flag == "smooth":
            if not force_real:
                raise Exception("Smooth gradient is not implemented for complex fields")
            return self.numerically_smooth_gradient(ipol)
        elif flag == "supersmooth":
            return self.super_smooth_gradient(ipol, force_real)
        else :
            raise Exception("Incorrect flag")

    def laplacian(self, check_real = False, force_real = False, sigma = 0.025):
        self_fft = self.fft()
        gg = self_fft.grid.gg
        if sigma is None or sigma == 0:
            self_fft = -self_fft.grid.gg*self_fft
        else :
            self_fft = -gg*self_fft*np.exp(-gg*(sigma*sigma)/4.0)
        return self_fft.ifft(check_real = check_real, force_real = force_real)

    #def laplacian(self, flag="smooth"):
        #return self.gradient(flag=flag).divergence(flag=flag)

    def sigma(self, flag="smooth"):
        """
        \sigma(r) = |\grad rho(r)|^2
        """
        if self.rank > 1 :
            vs = []
            for i in range(0, self.rank):
                gradrho = self[i].gradient(flag=flag)
                vs.append(gradrho)
            sigma = []
            for i in range(0, self.rank):
                for j in range(i, self.rank):
                    s = self.mp.einsum("lijk,lijk->ijk", vs[i], vs[j])
                    sigma.append(s)
            rank = (self.rank * (self.rank + 1))//2
        else :
            gradrho = self.gradient(flag=flag)
            sigma = self.mp.einsum("lijk,lijk->ijk", gradrho, gradrho)
            rank = 1
        return DirectField(grid=self.grid, rank=rank, griddata_3d=sigma)

    def fft(self):
        if self.fft_data is not None and environ["SAVEFFT"] :
            return self.fft_data
        TimeData.Begin("FFT")
        """ Implements the Discrete Fourier Transform
        Tips : If you use pyfft to perform fft, you should copy the input_array sometime. Becuase
        the input_array may be overwite.
        """
        reciprocal_grid = self.grid.get_reciprocal()
        dim = np.shape(np.shape(self))[0]
        if dim == 3:
            self.rank = 1
        else:
            nr = self.rank, *reciprocal_grid.nr
        if self.rank == 1:
            griddata_3d = self.fft_object(self) * self.grid.dV
        else:
            griddata_3d = np.empty(nr, dtype='complex128')
            for i in range(self.rank):
                griddata_3d[i] = self.fft_object(self[i]) * self.grid.dV
        TimeData.End("FFT")
        # print('shape0', self.grid.nr, griddata_3d.shape, reciprocal_grid.nr)
        fft_data=ReciprocalField(
            grid=reciprocal_grid, memo=self.memo, rank=self.rank, griddata_3d=griddata_3d, cplx=self.cplx
        )
        if environ["SAVEFFT"] :
            self._fft_data = fft_data
            fft_data = self.fft_data
        return fft_data

    def get_value_at_points(self, points):
        """
        Only support for serial.
        points is in crystal coordinates
        """
        if self.spl_coeffs is None:
            self._calc_spline()
        for ipol in range(3):
            # restrict crystal coordinates to [0,1)
            points[:, ipol] = (points[:, ipol] % 1) * self.grid.nr[ipol] + self.spl_order
        values = ndimage.map_coordinates(self.spl_coeffs, [points[:, 0], points[:, 1], points[:, 2]], mode="wrap")
        return values

    def get_values_flatarray(self, pad=0, order="F"):
        """
        Only support for serial.
        """
        if pad > 0:
            nr0 = self.shape
            p = []
            for i in nr0:
                if i > 1:
                    p.append([0, pad])
                else:
                    p.append([0, 0])
            vals = np.pad(self, p, mode="wrap")
        else:
            vals = np.asarray(self)
        nr = vals.shape
        nnr = 1
        for n in nr:
            nnr *= n
        return np.reshape(vals, nnr, order=order)

    def get_3dinterpolation(self, nr_new):
        """
        Only support for serial.
        Interpolates the values of the function on a cell with a different number
        of points, and returns a new Grid_Function_Base object.
        """
        if self.rank > 1:
            raise Exception("get_3dinterpolation is only implemented for scalar fields")

        if self.spl_coeffs is None:
            self._calc_spline()
        x = np.linspace(0, 1, nr_new[0], endpoint=False) * self.grid.nr[0] + self.spl_order
        y = np.linspace(0, 1, nr_new[1], endpoint=False) * self.grid.nr[1] + self.spl_order
        z = np.linspace(0, 1, nr_new[2], endpoint=False) * self.grid.nr[2] + self.spl_order
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        new_values = ndimage.map_coordinates(self.spl_coeffs, [X, Y, Z], mode="wrap")
        new_lattice = self.grid.lattice  # *LEN_CONV["Bohr"][self.grid.units]
        new_grid = DirectGrid(new_lattice, nr_new, units=self.grid.units)
        return DirectField(new_grid, self.memo, griddata_3d=new_values)

    def get_3dinterpolation_map(self, nr_new):
        """
        Only support for serial.
        Interpolates the values of the function on a cell with a different number
        of points, and returns a new Grid_Function_Base object.
        """
        if self.rank > 1:
            raise Exception("get_3dinterpolation is only implemented for scalar fields")

        x = np.linspace(0, 1, nr_new[0], endpoint=False) * self.grid.nr[0]
        y = np.linspace(0, 1, nr_new[1], endpoint=False) * self.grid.nr[1]
        z = np.linspace(0, 1, nr_new[2], endpoint=False) * self.grid.nr[2]
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        new_values = ndimage.map_coordinates(self[0], (X, Y, Z), mode="wrap")
        new_lattice = self.grid.lattice  # *LEN_CONV["Bohr"][self.grid.units]
        new_grid = DirectGrid(new_lattice, nr_new, units=self.grid.units)
        return DirectField(new_grid, self.memo, griddata_3d=new_values)

    def get_cut(self, r0, r1=None, r2=None, origin=None, center=None, nr=10, basis = 'Crystal'):
        """
        Only support for serial.
        general routine to get the arbitrary cuts of a Grid_Function_Base object in 1,2,
        or 3 dimensions. spline interpolation will be used.
            r0 = first vector (always required)
            r1 = second vector (required for 2D and 3D cuts)
            r2 = third vector (required for 3D cuts)
            origin = origin of the cut (don't specify center)
            center = center of the cut (don't specify origin)
            nr[i] = number points to discretize each direction ; i = 0,1,2
        r0, r1, r2, origin, center are instances of Coord
        """
        if not isinstance(r0, Coord): r0 = Coord(r0, self.grid, basis = basis).to_cart()
        if not isinstance(r1, (Coord, type(None))): r1 = Coord(r1, self.grid, basis = basis).to_cart()
        if not isinstance(r2, (Coord, type(None))): r2 = Coord(r2, self.grid, basis = basis).to_cart()
        if not isinstance(origin, (Coord, type(None))): origin = Coord(origin, self.grid, basis = basis).to_cart()
        if not isinstance(center, (Coord, type(None))): center = Coord(center, self.grid, basis = basis).to_cart()

        if self.rank > 1:
            raise Exception("get_cut is only implemented for scalar fields")

        span = 1

        do_center = False
        if origin is None and center is None:
            raise AttributeError("Specify either origin or center")
        elif origin is not None and center is not None:
            warnings.warn("Specified both origin and center, center will be ignored", DeprecationWarning)
        elif center is not None:
            do_center = True

        if do_center:
            x0 = center.to_crys()
        else:
            x0 = origin.to_crys()

        r0 = r0.to_crys()
        if do_center:
            x0 = x0 - 0.5 * r0

        if r1 is not None:
            r1 = r1.to_crys()
            if do_center:
                x0 = x0 - 0.5 * r1
            span += 1
            if r2 is not None:
                r2 = r2.to_crys()
                if do_center:
                    x0 = x0 - 0.5 * r2
                span += 1
        nrx = np.ones(3, dtype=int)
        if isinstance(nr, (np.ndarray, list, tuple)):
            nrx[0:span] = np.asarray(nr, dtype=int)
        # elif isinstance(nr, (int, float)):
        else:
            nrx[0:span] = nr

        dr = np.zeros((3, 3), dtype=float)
        dr[0, :] = (r0) / nrx[0]
        if span > 1:
            dr[1, :] = (r1) / nrx[1]
            if span == 3:
                dr[2, :] = (r2) / nrx[2]
        axis = []
        for ipol in range(3):
            axis.append(np.zeros((nrx[ipol], 3)))
            for ir in range(nrx[ipol]):
                axis[ipol][ir, :] = ir * dr[ipol]

        # points = np.zeros((nrx[0], nrx[1], nrx[2], 3))
        # for i in range(nrx[0]):
        # for j in range(nrx[1]):
        # for k in range(nrx[2]):
        # points[i, j, k, :] = x0 + axis[0][i, :] + axis[1][j, :] + axis[2][k, :]
        points = axis[0].reshape((nrx[0], 1, 1, 3)) + axis[1].reshape((1, nrx[1], 1, 3)) + axis[2].reshape((1, 1, nrx[2], 3))
        points += x0[:]

        a, b, c = nrx[0], nrx[1], nrx[2]
        points = points.reshape((nrx[0] * nrx[1] * nrx[2], 3))

        values = self.get_value_at_points(points)

        # generate a new grid (possibly 1D/2D/3D)
        origin = x0.to_cart()
        at = np.identity(3)
        v0 = r0.to_cart()
        v1 = np.zeros(3)
        v2 = np.zeros(3)
        # We still need to define 3 lattice vectors even if the plot is in 1D/2D
        # Here we ensure the 'ficticious' vectors are orthonormal to the actual ones
        # so that units of length/area are correct.
        if span == 1:
            for i in range(3):
                if abs(v0[i]) > 1e-4:
                    j = i - 1
                    v1[j] = v0[i]
                    v1[i] = -v0[j]
                    v1 = v1 / np.sqrt(np.dot(v1, v1))
                    break
            v2 = np.cross(v0, v1)
            v2 = v2 / np.sqrt(np.dot(v2, v2))
        elif span == 2:
            v1 = r1.to_cart()
            v2 = np.cross(v0, v1)
            v2 = v2 / np.sqrt(np.dot(v2, v2))
        elif span == 3:
            v1 = r1.to_cart()
            v2 = r2.to_cart()
        at[:, 0] = v0
        at[:, 1] = v1
        at[:, 2] = v2

        cut_grid = DirectGrid(lattice=at, nr=nrx, origin=origin, units=x0.cell.units)

        if span == 1:
            values = values.reshape((a,))
        elif span == 2:
            values = values.reshape((a, b))
        elif span == 3:
            values = values.reshape((a, b, c))

        return DirectField(grid=cut_grid, memo=self.memo, griddata_3d=values)

    @property
    def N(self):
        if self._N is None:
            self._N = self.integral()
        return self._N

    @property
    def cplx(self):
        return self._cplx

    @cplx.setter
    def cplx(self, value):
        self._cplx = value
        if self._cplx:
            self.grid.full = True
            self.grid.cplx = True

    def repeat(self, reps=1):
        reps = np.ones(3, dtype='int')*reps
        data = np.tile(np.array(self), reps)
        grid = self.grid.repeat(reps)
        results = self.__class__(grid=grid, rank=self.rank, griddata_3d=data, cplx=self.cplx, fft_data=None)
        return results

    def gather(self, grid = None, out = None):
        if out is None :
            value = self.grid.gather(self)
            if self.grid.mp.rank == 0 :
                if grid is None :
                    grid = DirectGrid(self.grid.lattice, self.grid.nrR, units=self.grid.units, full=self.grid.full)
                value = self.__class__(grid=grid, rank=self.rank, griddata_3d=value, cplx=self.cplx, fft_data=None)
        else :
            value = self.grid.gather(self, out = out)
        return value

    def __scatter(self, grid, data = None):
        pass
        if data is None :
            if self.grid.mp.is_mpi :
                data = self.grid.gather(self)
            else :
                data = self
        value = grid.scatter(data)
        value = self.__class__(grid=grid, rank=self.rank, griddata_3d=value, cplx=self.cplx, fft_data=None)
        return value


class ReciprocalField(BaseField):
    def __new__(cls, grid, memo="", rank=1, griddata_F=None, griddata_C=None, griddata_3d=None, cplx=False, fft_data = None):
        if not isinstance(grid, ReciprocalGrid):
            raise TypeError("the grid argument is not an instance of ReciprocalGrid")
        if griddata_F is None and griddata_C is None and griddata_3d is None :
            griddata_3d = np.zeros(grid.nr, dtype=np.complex128)
        obj = super().__new__(
            cls, grid, memo="", rank=rank, griddata_F=griddata_F, griddata_C=griddata_C, griddata_3d=griddata_3d, fft_data = fft_data
        )
        obj.spl_coeffs = None
        obj._cplx = cplx
        if obj.mp.is_mpi :
            from dftpy.mpi.mp_mpi4py import mpi_ifft
            obj.ifft_object = mpi_ifft(obj.grid)
        elif not obj._cplx and np.all(obj.grid.nrG == obj.grid.nrR):  # Can only use numpy.fft
            obj.ifft_object = np.fft.ifftn
            # if environ["FFTLIB"] != 'numpy' :
            # print('!WARN : For full G-space, you can only use numpy.fft.\n So here we reset the FFT as numpy.fft.')
        elif environ["FFTLIB"] == "pyfftw":
            obj.ifft_object = PYifft(obj.grid, obj._cplx)
        elif environ["FFTLIB"] == "numpy":
            if obj._cplx:
                obj.ifft_object = np.fft.ifftn
            else:
                obj.ifft_object = np.fft.irfftn
        else :
            obj.ifft_object = np.fft.ifftn
        return obj

    def __array_finalize__(self, obj):
        # Restore attributes when we are taking a slice
        if obj is None:
            return
        super().__array_finalize__(obj)
        if isinstance(obj, (ReciprocalField)):
            self.ifft_object = obj.ifft_object
            self.cplx = obj.cplx
        self.spl_coeffs = None

    # def __array_wrap__(self,obj,context=None):
    #    '''wrap it up'''
    #    b = np.ndarray.__array_wrap__(self, obj, context)
    #    #b.rank = self.rank * obj.rank
    #    rank = 1
    #    a=np.shape(np.shape(self))[0]
    #    if a == 4:
    #        rank = np.shape(self)[3]
    #    nr = *self.grid.nrR, self.rank
    #    b = np.reshape(b,nr)
    #    return ReciprocalField(grid=self.grid,rank=rank,griddata_3d=b)

    # def __mul__(self, other):
    # return np.multiply(self,other)
    # try:
    # prod = np.multiply(self, other)
    # except:
    # s_sh = np.shape(self)
    # o_sh = np.shape(other)
    # nr = self.grid.nrR
    # a = np.reshape(self, nr)
    # b = np.reshape(other, nr)
    # prod = ReciprocalField(grid=self.grid, rank=self.rank, griddata_3d=np.multiply(a, b))
    # return prod

    # if np.sum(o_sh) == 1: return np.multiply(self,other)
    # if s_sh != o_sh:
    #    nr = self.grid.nrR
    #    a=np.reshape(self,nr)
    #    b=np.reshape(other,nr)
    #    return ReciprocalField(grid=self.grid,rank=self.rank,griddata_3d=np.multiply(a,b))
    # else:
    #    return np.multiply(self,other)
    # if flag is not None:
    #    s_npol = 1
    #    o_npol = 1
    #    if np.sum(np.shape(s_sh)) == 4: s_npol = s_sh[3]
    #    if np.sum(np.shape(o_sh)) == 4: o_npol = s_sh[3]
    #    prod = np.empty((s_sh[0:-1],s_pol*o_pol))
    #    for i_pol in np.arange(s_npol):
    #        for j_pol in np.arange(o_npol):
    #            prod[o_pol*i_pol+j_pol] = self[i_pol]*other[o_pol]
    #    return ReciprocalField(grid=self.grid,rank=s_pol*o_pol,griddata_3d=prod)
    # else:
    #    return self*other

    def integral(self):
        """ Returns the integral of self """
        if self.rank == 1:
            return self.mp.einsum("ijk->", self) * self.grid.dV * self.grid.nnrG / (2.0 * np.pi) ** 3
        else:
            return self.mp.einsum("lijk->l", self) * self.grid.dV * self.grid.nnrG / (2.0 * np.pi) ** 3

    def ifft(self, check_real=False, force_real=False):
        """
        Implements the Inverse Discrete Fourier Transform
        """
        # if self.fft_data is not None :
            # return self.fft_data
        TimeData.Begin("InvFFT")
        direct_grid = self.grid.get_direct()
        nr = self.rank, *direct_grid.nr
        if self.rank == 1:
            if environ["FFTLIB"] == "numpy":
                griddata_3d = self.ifft_object(self, s=self.grid.nrR) / direct_grid.dV
            else:
                griddata_3d = self.ifft_object(self) / direct_grid.dV
        else:
            if self.cplx or np.all(self.grid.nrG == self.grid.nrR):  # Can only use numpy.fft
                griddata_3d = np.empty(nr, dtype="complex128")
            else:
                griddata_3d = np.empty(nr)
            for i in range(self.rank):
                griddata_3d[i] = self.ifft_object(self[i]) / direct_grid.dV
        if check_real:
            if np.isclose(np.imag(griddata_3d), 0.0, atol=1.0e-16).all():
                griddata_3d = np.real(griddata_3d)
        if force_real:
            griddata_3d = np.real(griddata_3d)
        TimeData.End("InvFFT")
        if environ["SAVEFFT"] :
            fft_data=DirectField(grid=direct_grid, memo=self.memo, rank=self.rank, griddata_3d=griddata_3d, cplx=self.cplx, fft_data=self)
        else :
            fft_data=DirectField(grid=direct_grid, memo=self.memo, rank=self.rank, griddata_3d=griddata_3d, cplx=self.cplx)
        # self._fft_data = fft_data
        # return self.fft_data
        return fft_data

    @property
    def cplx(self):
        return self._cplx

    @cplx.setter
    def cplx(self, value):
        self._cplx = value
        if self._cplx and not self.grid.full :
            self.grid.full = True
