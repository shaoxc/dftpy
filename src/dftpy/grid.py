import numpy as np
from dftpy.base import BaseCell, DirectCell, ReciprocalCell, Coord, s2r


class BaseGrid(BaseCell):
    """
    Object representing a grid (Cell (lattice) plus discretization)
    extends Cell

    Attributes
    ----------
    nr : array of numbers used for discretization

    nnr : total number of grid points

    dV : volume of a grid point

    Node:
    Virtual class, DirectGrid and ReciprocalGrid should be used in actual applications

    """

    def __init__(self, lattice, nr, origin=np.array([0.0, 0.0, 0.0]), units="Bohr", convention="mic",
            full = False, realspace = True, cplx = False, mp = None, **kwargs):
        if mp is None :
            from dftpy.mpi import MP
            mp = MP()
        super().__init__(lattice=lattice, origin=origin, units=units, **kwargs)
        self._nrR = np.array(nr, dtype = np.int32)
        self._nnrR = np.prod(self._nrR)
        self._dV = np.abs(self._volume) / self._nnrR
        self._nrG = self._nrR.copy()
        if not full :
            self._nrG[-1] = self._nrG[-1] // 2 + 1
            self._nnrG = np.prod(self._nrG)
        metric = np.dot(lattice.T, lattice)
        latparas = np.zeros(3)
        for i in range(3):
            latparas[i] = np.sqrt(metric[i, i])
        self._spacings = latparas / self._nrR
        self._latparas= latparas
        self._mp = mp
        self.cplx = cplx
        if cplx :
            full = True
        self.local_slice(nr, realspace = realspace, full = full, cplx = cplx, **kwargs)
        self._nnr = np.prod(self._nr)
        # print('nr_local', self.mp.comm.rank, self._nr, realspace, self.mp.comm.size, flush = True)
        self._full = full

    def __eq__(self, other):
        if not super().__eq__(other):
            return False
        for ilat in range(3):
            if self._nr[ilat] != other._nr[ilat]:
                return False

        return True

    @property
    def mp(self):
        return self._mp

    @mp.setter
    def mp(self, value):
        self._mp = value

    @property
    def nr(self):
        return self._nr

    @property
    def nnr(self):
        return self._nnr

    @property
    def nrR(self):
        return self._nrR

    @property
    def nnrR(self):
        return self._nnrR

    @property
    def nrG(self):
        return self._nrG

    @property
    def nnrG(self):
        return self._nnrG

    @property
    def dV(self):
        return self._dV

    @property
    def Volume(self):
        return self._volume

    @property
    def spacings(self):
        return self._spacings

    @property
    def latparas(self):
        return self._latparas

    @property
    def full(self):
        return self._full

    def repeat(self, reps=1):
        reps = np.ones(3, dtype='int')*reps
        lattice = self.lattice.copy()
        for i in range(3):
            lattice[:, i] *= reps[i]
        nr = self.nr * reps
        results = self.__class__(lattice, nr, origin=self.origin, units=self.units, full=self.full, cplx=self.cplx)
        return results

    def local_slice(self, nr, **kwargs):
        self._slice, self._nr, self._offsets = self.mp.get_local_fft_shape(nr, **kwargs)
        if self.mp.is_mpi :
            self.slice_all = self.mp.comm.gather(self._slice)
            self.nr_all = self.mp.comm.gather(self._nr)
            self.offsets_all = self.mp.comm.gather(self._offsets)
        else :
            self.slice_all = self._slice
            self.nr_all = self._nr
            self.offsets_all = self._offsets

    @property
    def slice(self):
        return self._slice

    @property
    def offsets(self):
        return self._offsets

    def gather(self, data, nr, out = None):
        if self.mp.is_mpi :
            reqs = []
            bufs = []
            if self.mp.is_root :
                if out is None :
                    out = np.empty(nr, dtype = data.dtype)
                out[self.slice_all[self.mp.rank]] = data
                for i in range(1, self.mp.comm.size):
                    buf = np.empty(self.nr_all[i], dtype = data.dtype)
                    bufs.append(buf)
                    req = self.mp.comm.Irecv(buf, source = i, tag = i)
                    reqs.append(req)
            else :
                req = self.mp.comm.Isend(data, dest = 0, tag = self.mp.rank)
                reqs.append(req)
                out = np.ones((1, 1, 1))
            self.mp.MPI.Request.Waitall(reqs)
            if self.mp.is_root :
                for i in range(1, self.mp.comm.size):
                    out[self.slice_all[i]] = bufs[i - 1]
            self.mp.comm.Barrier()
        else :
            out = data.copy()
        return out

    def scatter(self, data, out = None):
        if self.mp.is_mpi :
            reqs = []
            if out is None :
                out = np.empty(self.nr, dtype = data.dtype)
            if self.mp.is_root :
                out[:] = data[self.slice_all[self.mp.rank]]
                for i in range(1, self.mp.comm.size):
                    buf = np.empty(self.nr_all[i], dtype = data.dtype)
                    buf[:] = data[self.slice_all[i]]
                    req = self.mp.comm.Isend(buf, dest = i, tag = i)
                    reqs.append(req)
            else :
                req = self.mp.comm.Irecv(out, source = 0, tag = self.mp.rank)
                reqs.append(req)
            self.mp.MPI.Request.Waitall(reqs)
            self.mp.comm.Barrier()
        else :
            if out is None :
                out = data.copy()
            else :
                out[:] = data.copy()
        return out


class DirectGrid(BaseGrid, DirectCell):
    """
        Attributes:
        ----------
        All of BaseGrid and DirectCell

        r : cartesian coordinates of each grid point

        s : crystal coordinates of each grid point
    """

    def __init__(self, lattice, nr, origin=np.array([0.0, 0.0, 0.0]), units=None, full=True, uppergrid = None, **kwargs):
        """
        Parameters
        ----------
        lattice : array_like[3,3]
            matrix containing the direct lattice vectors (as its colums)
        units : {'Bohr', 'Angstrom', 'nm', 'm'}, optional
            length units of the lattice vectors.
        """
        # internally always convert the units to Bohr
        # print("DirectGrid __init__")
        # lattice is already scaled inside the super()__init__, no need to do it here
        # lattice *= LEN_CONV[units]["Bohr"]
        super().__init__(lattice=lattice, nr=nr, origin=origin, units=units, full = full, realspace = True, **kwargs)
        self._r = None
        self._rr = None
        self._s = None
        self.RPgrid = uppergrid
        self._Rtable = None

    def __eq__(self, other):
        """
        Implement the == operator in the DirectGrid class.
        Refer to the __eq__ method of Grid for more information.
        """
        if not isinstance(other, (BaseGrid, DirectGrid)):
            if isinstance(other, DirectCell):
                return DirectCell.__eq__(self, other)
            raise TypeError("You can only compare a DirectGrid with another DirectGrid")
        return BaseGrid.__eq__(self, other)

    def _calc_grid_crys_points(self):
        if self._s is None:
            # s0 = np.linspace(0, 1, self.nr[0], endpoint=False)
            # s1 = np.linspace(0, 1, self.nr[1], endpoint=False)
            # s2 = np.linspace(0, 1, self.nr[2], endpoint=False)
            # S0, S1, S2 = np.meshgrid(s0, s1, s2, indexing="ij")
            # self._s = np.asarray([S0, S1, S2])
            ax = []
            for i in range(3):
                s0 = np.linspace(0, 1, self.nrR[i], endpoint=False)
                ax.append(s0)
            AX = [a[sl] for a, sl in zip(ax, self.slice)]
            S = np.meshgrid(*AX, indexing="ij")
            self._s = np.asarray(S)

    def _calc_grid_cart_points(self):
        if self._r is None:
            # self._r = self.s.to_cart()
            self._r = np.einsum("j...,kj->k...", self.s, self.lattice)

    @property
    def r(self):
        if self._r is None:
            self._calc_grid_cart_points()
        return self._r

    @property
    def rr(self):
        if self._rr is None:
            rr = np.einsum("lijk,lijk->ijk", self.r, self.r)
            # self._rr = np.reshape(rr, [self.nr[0], self.nr[1], self.nr[2], 1])
            self._rr = rr
        return self._rr

    @property
    def s(self):
        if self._s is None:
            self._calc_grid_crys_points()
        return self._s

    @property
    def full(self):
        return self._full

    @full.setter
    def full(self, value):
        if self._full != value :
            '''
            Clean stored information of reciprocal grid.
            '''
            self._full = value
            self.RPgrid = None
            self._nrG = self.nr.copy()
            if not self._full:
                self._nrG[-1] = self._nrG[-1] // 2 + 1

    def get_reciprocal(self, scale=None, convention="physics"):
        """
            Returns a new ReciprocalCell, the reciprocal cell of self
            The ReciprocalCell is scaled properly to include
            the scaled (*self.nr) reciprocal grid points
            -----------------------------
            Note1: We need to use the 'physics' convention where bg^T = 2 \pi * at^{-1}
            physics convention defines the reciprocal lattice to be
            exp^{i G \cdot R} = 1
            Now we have the following "crystallographer's" definition ('crystallograph')
            which comes from defining the reciprocal lattice to be
            e^{2\pi i G \cdot R} =1
            In this case bg^T = at^{-1}
            -----------------------------
            Note2: We have to use 'Bohr' units to avoid changing hbar value
        """
        # TODO define in constants module hbar value for all units allowed
        if self.RPgrid is None or scale is not None:
            if scale is None :
                scale=[1.0, 1.0, 1.0]
            scale = np.array(scale)
            fac = 1.0
            if convention == "physics" or convention == "p":
                fac = 2 * np.pi
            fac = 2 * np.pi
            bg = fac * np.linalg.inv(self.lattice)
            bg = bg.T
            # bg = bg/LEN_CONV["Bohr"][self.units]
            reciprocal_lat = np.einsum("ij,j->ij", bg, scale)

            self.RPgrid = ReciprocalGrid(lattice=reciprocal_lat, nr=self.nrR, units=self.units, full=self.full,
                    uppergrid=self, cplx=self.cplx, mp=self.mp)
        return self.RPgrid

    def get_Rtable(self, rcut=10):
        '''Only support for serial'''
        if self._Rtable is None:
            self._Rtable = {}
            metric = np.dot(self.lattice.T, self.lattice)
            latticeConstants = np.sqrt(np.diag(metric))
            gaps = latticeConstants / self.nr
            Nmax = np.ceil(rcut / gaps).astype(np.int) + 1
            # print('lc', latticeConstants)
            # print('gaps', gaps)
            # print(Nmax)
            # mgrid = np.mgrid[0:Nmax[0], 0:Nmax[0], 0:Nmax[0]].reshape((3, -1))
            # array = np.einsum('jk,ij->ik',gridpos,self.lattice)
            # dists = np.einsum('ij,ij->j', array, array)
            # index = np.arange(0, Nmax[0] * Nmax[1] * Nmax[2]).reshape(Nmax)
            # mgrid = np.mgrid[0:Nmax[0], 0:Nmax[1], 0:Nmax[2]].astype(np.float)
            mgrid = np.mgrid[1 - Nmax[0] : Nmax[0], 1 - Nmax[1] : Nmax[1], 1 - Nmax[2] : Nmax[2]].astype(np.float)
            mgrid[0] /= self.nr[0]
            mgrid[1] /= self.nr[1]
            mgrid[2] /= self.nr[2]
            gridpos = mgrid.astype(np.float)
            array = np.einsum("jklm,ij->iklm", gridpos, self.lattice)
            dists = np.sqrt(np.einsum("ijkl,ijkl->jkl", array, array))
            self._Rtable["Nmax"] = Nmax
            self._Rtable["table"] = dists
        return self._Rtable

    def gather(self, data, out = None):
        value = super().gather(data, self.nrR, out = out)
        return value


class ReciprocalGrid(BaseGrid, ReciprocalCell):
    """
        Attributes:
        ----------
        All of BaseGrid and DirectCell

        g : coordinates of each point in the reciprocal cell

        gg : square of each g vector
    """

    def __init__(self, lattice, nr, units=None, origin=np.array([0.0, 0.0, 0.0]), full=False, uppergrid = None, **kwargs):
        """
        Parameters
        ----------
        lattice : array_like[3,3]
            matrix containing the direct lattice vectors (as its colums)
        units : {'Bohr', 'Angstrom', 'nm', 'm'}, optional
            length units of the lattice vectors.
        """
        # internally always convert the units to Bohr
        # print("ReciprocalGrid __init__")
        # lattice is already scaled inside the super()__init__, no need to do it here
        # lattice /= LEN_CONV[units]["Bohr"]
        super().__init__(lattice=lattice, nr=nr, origin=origin, units=units, full = full, realspace = False, **kwargs)
        self._g = None
        self._gg = None
        self.Dgrid = uppergrid
        self._q = None
        self._mask = None
        self._gF = None
        self._ggF = None
        self._invgg = None
        self._invq = None

    def __eq__(self, other):
        """
        Implement the == operator in the ReciprocalGrid class.
        Refer to the __eq__ method of Grid for more information.
        """
        if not isinstance(other, (BaseGrid, ReciprocalGrid)):
            if isinstance(other, ReciprocalCell):
                return ReciprocalCell.__eq__(self, other)
            raise TypeError("You can only compare a ReciprocalGrid with another ReciprocalGrid")
        return BaseGrid.__eq__(self, other)

    @property
    def g(self):
        if self._g is None:
            self._g = self._calc_grid_points()
        return self._g

    @property
    def q(self):
        if self._q is None:
            self._q = np.sqrt(self.gg)
        return self._q

    @property
    def gg(self):
        if self._gg is None:
            if self._g is None:
                self._g = self._calc_grid_points()
            gg = np.einsum("lijk,lijk->ijk", self._g, self._g)
            self._gg = gg
        return self._gg

    @property
    def invgg(self):
        if self._invgg is None:
            if self.mp.is_root :
                self.gg[0, 0, 0] = 1.0
            invgg = 1.0/self.gg
            if self.mp.is_root :
                self.gg[0, 0, 0] = 0.0
                invgg[0, 0, 0] = 0.0
            self._invgg = invgg
        return self._invgg

    @property
    def invq(self):
        if self._invq is None:
            if self.mp.is_root :
                self.q[0, 0, 0] = 1.0
            invq = 1.0/self.q
            if self.mp.is_root :
                self.q[0, 0, 0] = 0.0
            invq[0, 0, 0] = 0.0
        # self._invq = invq
        # return self._invq
        return invq

    def get_direct(self, scale= None, convention="physics"):
        """
            Returns a new DirectCell, the direct cell of self
            The DirectCell is scaled properly to include
            the scaled (*self.nr) reciprocal grid points
            -----------------------------
            Note1: We need to use the 'physics' convention where bg^T = 2 \pi * at^{-1}
            physics convention defines the reciprocal lattice to be
            exp^{i G \cdot R} = 1
            Now we have the following "crystallographer's" definition ('crystallograph')
            which comes from defining the reciprocal lattice to be
            e^{2\pi i G \cdot R} =1
            In this case bg^T = at^{-1}
            -----------------------------
            Note2: We have to use 'Bohr' units to avoid changing hbar value
        """
        # TODO define in constants module hbar value for all units allowed
        if self.Dgrid is None or scale is not None:
            if scale is None :
                scale=[1.0, 1.0, 1.0]
            scale = np.array(scale)
            fac = 1.0
            if convention == "physics" or convention == "p":
                fac = 1.0 / (2 * np.pi)
            at = np.linalg.inv(self.lattice.T * fac)
            # at = at*LEN_CONV["Bohr"][self.units]
            direct_lat = np.einsum("ij,j->ij", at, 1.0 / scale)
            self.Dgrid = DirectGrid(lattice=direct_lat, nr=self.nrR, units=self.units, full=self.full,
                    uppergrid=self, cplx=self.cplx,  mp=self.mp)
        return self.Dgrid

    def _calc_grid_points(self, full=None):
        ax = []
        for i in range(3):
            # use fftfreq function so we don't have to
            # worry about odd or even number of points
            # dd: this choice of "spacing" is due to the
            # definition of real and reciprocal space for
            # a grid (which is not exactly a conventional
            # lattice), specifically:
            #    1) the real-space points go from 0 to 1 in
            #       crystal coords in n steps of length 1/n
            #    2) thus the reciprocal space (g-space)
            #       crystal coords go from 0 to n in n steps
            #    3) the "physicists" 2*np.pi factor is
            #       included in the definition of reciprocal
            #       lattice vectors in the "grid" class and
            #       is applied with s2r in going from crystal
            #       to Cartesian g-space
            dd = 1 / self.nrR[i]
            if full is None:
                full = self.full
            if i == 2 and not full:
                ax.append(np.fft.rfftfreq(self.nrR[i], d=dd))
            else:
                freq = np.fft.fftfreq(self.nrR[i], d=dd)
                # if freq.size % 2 == 0 :
                    # freq[freq.size//2] *= -1
                    # ax.append(freq)
                # else :
                    # ax.append(freq)
                ax.append(freq)
        AX = [a[sl] for a, sl in zip(ax, self.slice)]
        S = np.meshgrid(*AX, indexing="ij")
        S_cart = np.asarray(S)
        S_cart = np.einsum("j...,kj->k...", S_cart, self.lattice)

        return S_cart

    @property
    def mask_serial(self):
        if self._mask is None:
            nrR = self.nrR[:3]
            # Dnr = nr[:3]//2
            # Dmod = nr[:3]%2
            # mask = np.ones((nr[0], nr[1], Dnr[2]+1), dtype = bool)
            Dnr = nrR[:3] // 2
            Dmod = nrR[:3] % 2
            mask = np.ones(self.nr[:3], dtype=bool)
            if np.all(self.nr == self.nrR):
                mask[:, :, Dnr[2] + 1 :] = False

            mask[0, 0, 0] = False
            mask[0, Dnr[1] + 1 :, 0] = False
            mask[Dnr[0] + 1 :, :, 0] = False
            if Dmod[2] == 0:
                mask[0, 0, Dnr[2]] = False
                mask[0, Dnr[1] + 1 :, Dnr[2]] = False
                mask[Dnr[0] + 1 :, :, Dnr[2]] = False
                if Dmod[1] == 0:
                    mask[0, Dnr[1], Dnr[2]] = False
                if Dmod[0] == 0:
                    mask[Dnr[0], 0, Dnr[2]] = False
                    mask[Dnr[0], Dnr[1] + 1 :, Dnr[2]] = False
            if Dmod[0] == 0:
                mask[Dnr[0], Dnr[1] + 1 :, 0] = False
                if Dmod[1] == 0:
                    mask[Dnr[0], Dnr[1], 0] = False
            if Dmod[1] == 0:
                mask[0, Dnr[1], 0] = False
            if all(Dmod == 0):
                mask[Dnr[0], Dnr[1], Dnr[2]] = False
            self._mask = mask
        return self._mask

    @property
    def mask(self):
        if self._mask is None:
            nrR = self.nrR[:3]
            Dnr = nrR[:3] // 2 - self.offsets
            Dnr = np.where(Dnr > 0, Dnr, 0)
            Dmod = nrR[:3] % 2
            mask = np.ones(self.nr[:3], dtype=bool)
            if np.all(self.nrG == self.nrR):
                mask[:, :, Dnr[2] + 1 :] = False

            if np.all(self.offsets == 0):
                mask[0, 0, 0] = False
            if self.offsets[0] == self.offsets[2] == 0 :
                mask[0, Dnr[1] + 1 :, 0] = False
            if self.offsets[2] == 0 :
                mask[Dnr[0] + 1 :, :, 0] = False
            if Dmod[2] == 0:
                if self.offsets[0] == 0 :
                    if self.offsets[1] == 0 :
                        mask[0, 0, Dnr[2]:Dnr[2]+1] = False
                    mask[0, Dnr[1] + 1 :, Dnr[2]:Dnr[2]+1] = False
                mask[Dnr[0] + 1 :, :, Dnr[2]:Dnr[2]+1] = False
                if Dmod[1] == 0 and self.offsets[0] == 0 :
                    mask[0, Dnr[1]:Dnr[1]+1, Dnr[2]:Dnr[2]+1] = False
                if Dmod[0] == 0:
                    if self.offsets[1] == 0 :
                        mask[Dnr[0]:Dnr[0]+1, 0, Dnr[2]:Dnr[2]+1] = False
                    mask[Dnr[0]:Dnr[0]+1, Dnr[1] + 1 :, Dnr[2]:Dnr[2]+1] = False
            if Dmod[0] == 0 and self.offsets[2] == 0 :
                mask[Dnr[0]:Dnr[0]+1, Dnr[1] + 1 :, 0] = False
                if Dmod[1] == 0:
                    mask[Dnr[0]:Dnr[0]+1, Dnr[1]:Dnr[1]+1, 0] = False
            if Dmod[1] == 0 and self.offsets[2] == 0 :
                mask[0, Dnr[1]:Dnr[1]+1, 0] = False
            if all(Dmod == 0):
                mask[Dnr[0]:Dnr[0]+1, Dnr[1]:Dnr[1]+1, Dnr[2]:Dnr[2]+1] = False
            self._mask = mask
        return self._mask

    @property
    def gF(self):
        if self._gF is None:
            self._gF = self._calc_grid_points(full=True)
        return self._gF

    @property
    def ggF(self):
        if self._ggF is None:
            if self._gF is None:
                self._gF = self._calc_grid_points(full=True)
            ggF = np.einsum("lijk,lijk->ijk", self._gF, self._gF)
            self._ggF = ggF
            # self._ggF = np.reshape(gg, (*self._gF.shape, 1))
        return self._ggF

    @property
    def full(self):
        return self._full

    @full.setter
    def full(self, value):
        if self._full != value :
            self._full = value
