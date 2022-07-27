import numpy as np
from numpy import linalg as LA
from dftpy.field import DirectField

class AtomicDensity(object):
    """
    The densities for atomic atoms.
    """
    def __init__(self, grid, ions, key=None, rcut=5.0, **kwargs):
        self.grid = grid
        self.ions = ions
        self.key = key
        self.rcut = rcut
        self.dnr = (1.0 / self.grid.nr).reshape((3, 1))
        border = (rcut / grid.spacings).astype(np.int32) + 1
        border = np.minimum(border, grid.nr // 2)
        self.ixyzA = np.mgrid[-border[0]:border[0] + 1, -border[1]:border[1] + 1, -border[2]:border[2] + 1].reshape((3, -1))
        self.prho = np.zeros((2 * border[0] + 1, 2 * border[1] + 1, 2 * border[2] + 1))

    # def _init_data(self, files = None, pseudo = None, **kwargs):
    #     readpp = None
    #     if files :
    #         for key, infile in files.items() :
    #             if not os.path.isfile(infile):
    #                 raise Exception("Density file " + infile + " for atom type " + str(key) + " not found")
    #             else:
    #                 if infile[-4:].lower() == "list" :
    #                     try :
    #                         self._r[key], self._arho[key] = self.read_density_list(infile)
    #                     except Exception :
    #                         raise Exception("density file '{}' has some problems".format(infile))
    #                 else :
    #                     readpp = ReadPseudo(files)
    #     elif pseudo is not None :
    #         readpp = pseudo.readpp
    #
    #     if readpp :
    #         if self.is_core :
    #             self._r = readpp._core_density_grid
    #             self._arho = readpp._core_density
    #         else :
    #             self._r = readpp._atomic_density_grid
    #             self._arho = readpp._atomic_density

#     def read_density_list(self, infile):
#         with open(infile, "r") as fr:
#             lines = []
#             for i, line in enumerate(fr):
#                 lines.append(line)
#
#         ibegin = 0
#         iend = len(lines)
#         data = [line.split()[0:2] for line in lines[ibegin:iend]]
#         data = np.asarray(data, dtype = float)
#
#         r = data[:, 0]
#         v = data[:, 1]
#         return r, v
#
#     def guess_rho(self, ions, grid = None, ncharge = None, rho = None, dtol=1E-30, nspin = 1, **kwargs):
#         ncharge = ncharge or self.ncharge
#         rho_s = rho
#         if rho_s is not None :
#             nspin = rho.rank
#             if nspin > 1 : rho = rho_s[0]
#
#         if grid is None :
#             if rho_s is not None :
#                 grid = rho_s.grid
#             else :
#                 raise AttributeError("Please give one grid.")
#
#         rho_unspin = self.guess_rho_unspin(ions, grid = grid, ncharge=ncharge, rho=rho, dtol=dtol, **kwargs)
#
#         if self.is_core :
#             if rho_s is not None :
#                 rho_s[:] = rho_unspin
#             else :
#                 rho_s = rho_unspin
#         else :
#             if rho_s is not None :
#                 rho_s[:] = rho_unspin/nspin
#             else :
#                 if nspin > 1 :
#                     rho_s = Field(grid, rank = nspin)
#                     rho_s[:] = rho_unspin / nspin
#                 else :
#                     rho_s = rho_unspin
#         return rho_s
#
#     def guess_rho_unspin(self, ions, grid = None, ncharge = None, rho = None, dtol=1E-30, **kwargs):
#         ncharge = ncharge or self.ncharge
#         if len(self._r) == 0 :
#             if self.is_core : # no core density
#                 return None
#             new_rho = self.guess_rho_heg(ions, grid, ncharge, rho, dtol = dtol, **kwargs)
#         elif self.direct :
#             new_rho = self.guess_rho_atom(ions, grid, ncharge, rho, dtol = dtol, **kwargs)
#         else :
#             new_rho = self.get_3d_value_recipe(ions, grid.get_reciprocal(), ncharge, rho, dtol = dtol, **kwargs)
#
#         if rho is not None :
#             rho[:] = new_rho
#         else :
#             rho = new_rho
#         return rho
#
#     def guess_rho_heg(self, ions, grid, ncharge = None, rho = None, dtol=1E-30, **kwargs):
#         """
#         """
#         if ncharge is None :
#             ncharge = 0.0
#             for i in range(ions.nat) :
#                 ncharge += ions.charges[i]]
#         if rho is None :
#             rho = DirectField(grid=grid)
#             rho[:] = 1.0
#         rho[:] = ncharge / (rho.integral())
#         return rho
#
#     def guess_rho_atom(self, ions, grid, ncharge = None, rho = None, dtol=1E-30, **kwargs):
#         """
#         """
#         nr = grid.nr
#         dnr = (1.0/nr).reshape((3, 1))
#         if rho is None :
#             rho = DirectField(grid=grid) + dtol
#         else :
#             rho[:] = dtol
#         lattice = grid.lattice
#         metric = np.dot(lattice.T, lattice)
#         latp = np.zeros(3)
#         for i in range(3):
#             latp[i] = np.sqrt(metric[i, i])
#         gaps = latp / nr
#         for key in self._r :
#             r = self._r[key]
#             arho = self._arho[key]
#             if arho is None : continue
#             rcut = np.max(r)
#             rcut = min(rcut, 0.5 * np.min(latp))
#             rtol = r[0]
#             border = (rcut / gaps).astype(np.int32) + 1
#             ixyzA = np.mgrid[-border[0]:border[0]+1, -border[1]:border[1]+1, -border[2]:border[2]+1].reshape((3, -1))
#             prho = np.zeros((2 * border[0]+1, 2 * border[1]+1, 2 * border[2]+1))
#             tck = splrep(r, arho)
#             for i in range(ions.nat):
#                 if ions.symbols[i] != key:
#                     continue
#                 prho[:] = 0.0
#                 posi = ions.positions[i].reshape((1, 3))
#                 atomp = np.array(posi.to_crys()) * nr
#                 atomp = atomp.reshape((3, 1))
#                 ipoint = np.floor(atomp)
#                 px = atomp - ipoint
#                 l123A = np.mod(ipoint.astype(np.int32) - ixyzA, nr[:, None])
#
#                 positions = (ixyzA + px) * dnr
#                 positions = np.einsum("j...,kj->k...", positions, grid.lattice)
#                 dists = LA.norm(positions, axis = 0).reshape(prho.shape)
#                 index = np.logical_and(dists < rcut, dists > rtol)
#                 prho[index] = splev(dists[index], tck, der = 0)
#                 rho[l123A[0], l123A[1], l123A[2]] += prho.ravel()
#         nc = rho.integral()
#         sprint('Guess density : ', nc)
#         if ncharge is None :
#             ncharge = 0.0
#             for i in range(ions.nat) :
#                 ncharge += ions.charges[i]]
#         if ncharge > 1E-6 : rho[:] *= ncharge / nc
#         return rho
#
#     def get_3d_value_recipe(self, ions, grid, ncharge = None, rho = None, dtol=1E-30, direct=True, **kwargs):
#         return get_3d_value_recipe(self._r, self._arho, ions, grid, ncharge, rho, dtol, direct, **kwargs)
#
# def get_3d_value_recipe(r, arho, ions, grid, ncharge = None, rho = None, dtol=0.0, direct=True, pme=True, order=10, **kwargs):
#     """
#     """
#     if hasattr(grid, 'get_reciprocal'):
#         reciprocal_grid = grid.get_reciprocal()
#     else :
#         reciprocal_grid = grid
#         grid = grid.get_direct()
#
#     rho_g = Field(reciprocal_grid, direct = False)
#
#     radial = {}
#     vlines = {}
#     if pme :
#         Bspline = CBspline(ions=ions, grid=grid, order=order)
#         qa = np.empty(grid.nr)
#         for key in r:
#             r0 = r[key]
#             arho0 = arho[key]
#             if arho0 is None : continue
#             radial[key] = RadialGrid(r0, arho0, direct = False)
#             vlines[key] = radial[key].to_3d_grid(reciprocal_grid.q)
#             qa[:] = 0.0
#             for i in range(len(ions.positions)):
#                 if ions.symbols[i] == key:
#                     qa = Bspline.get_PME_Qarray(i, qa)
#             qarray = Field(grid=grid, data=qa, direct = True)
#             rho_g += vlines[key] * qarray.fft()
#         rho_g *= Bspline.Barray * grid.nnrR / grid.volume
#     else :
#         for key in r:
#             r0 = r[key]
#             arho0 = arho[key]
#             radial[key] = RadialGrid(r0, arho0, direct = False)
#             vlines[key] = radial[key].to_3d_grid(reciprocal_grid.q)
#             for i in range(ions.nat):
#                 if ions.symbols[i] == key:
#                     strf = ions.strf(reciprocal_grid, i)
#                     rho_g += vlines[key] * strf
#
#     if direct :
#         if rho is None :
#             rho = rho_g.ifft()
#         else :
#             rho[:] = rho_g.ifft()
#         rho[rho < dtol] = dtol
#         nc = rho.integral()
#         sprint('Guess density (Real): ', nc, comm=grid.mp.comm)
#         if ncharge is None :
#             ncharge = 0.0
#             for i in range(ions.nat) :
#                 ncharge += ions.charges[i]]
#         if ncharge > 1E-6 :
#             rho[:] *= ncharge / nc
#             sprint('Guess density (Scale): ', rho.integral(), comm=grid.mp.comm)
#     else :
#         if rho is None :
#             rho = rho_g
#         else :
#             rho[:] = rho_g
#     return rho
#
# def normalization_density(density, ncharge = None, grid = None, tol = 1E-300, method = 'shift'):
#     if grid is None :
#         grid = density.grid
#     # min_rho = grid.mp.amin(density)
#     # sprint('min0', min_rho, comm=grid.mp.comm)
#     if method == 'scale' :
#         if ncharge is not None :
#             ncharge = np.sum(density) * grid.dV
#         ncharge = grid.mp.asum(ncharge)
#         density[density < tol] = tol
#         nc = grid.mp.asum(np.sum(density) * grid.dV)
#         density *= ncharge / nc
#     elif method == 'shift' :
#         if ncharge is not None :
#             nc = grid.mp.asum(np.sum(density) * grid.dV)
#             density += (ncharge-nc)/grid.nnrR
#     else :
#         pass
#     if not hasattr(density, 'grid'):
#         density = Field(grid, data=density, direct=True)
#     # min_rho = grid.mp.amin(density)
#     # sprint('min1', min_rho, comm=grid.mp.comm)
#     return density
#
# def filter_density(grid, density, mu = 0.7, kt = 0.05):
#     grid._filter = 1.0
#     # Not use _filter
#     ncharge = np.sum(density)
#     if grid._filter is None :
#         recip_grid = density.grid.get_reciprocal()
#         q = recip_grid.q
#         mu *= np.max(q)
#         kt *= mu
#         grid._filter = fermi_dirac(q.real, mu = mu, kt = kt)
#     den_g = density.fft() * grid._filter
#     den = den_g.ifft(force_real=True)
#     #-----------------------------------------------------------------------
#     den = density
#     #-----------------------------------------------------------------------
#     den *= ncharge/np.sum(den)
#     den[den < 0] = 1E-300
#     return den
#
# def file2density(filename, density, grid = None):
#     if grid is None : grid = density.grid
#     ext = os.path.splitext(filename)[1].lower()
#     if ext == ".snpy":
#         density[:] = io.read_density(filename, grid=grid)
#     else :
#         fstr = f'!WARN : snpy format for density initialization will better, but this file is "{filename}".'
#         sprint(fstr, comm=grid.mp.comm, level=2)
#         if grid.mp.comm.rank == 0 :
#             density_gather = io.read_density(filename)
#         else :
#             density_gather = np.zeros(1)
#         grid.scatter(density_gather, out = density)

    def distance(self):
        self.i = 0
        while self.i < self.ions.nat:
            results = self.step()
            if results is not None:
                yield results
            self.i += 1

    def step(self):
        if self.ions.symbols[self.i] != self.key:
            return None
        self.prho[:] = 0.0
        posi = self.ions.get_scaled_positions()[self.i].reshape((1, 3))
        atomp = np.array(posi) * self.grid.nr
        atomp = atomp.reshape((3, 1))
        ipoint = np.floor(atomp + 1E-8)
        px = atomp - ipoint
        l123A = np.mod(ipoint.astype(np.int32) - self.ixyzA, self.grid.nr[:, None])
        positions = (self.ixyzA + px) * self.dnr
        positions = np.einsum("j...,kj->k...", positions, self.grid.lattice)
        dists = LA.norm(positions, axis=0).reshape(self.prho.shape)
        index = dists < self.rcut
        mask = self.grid.get_array_mask(l123A)
        return {
            "dists": dists,
            "index": index,
            "l123A": l123A,
            "mask": mask
        }

def gen_gaussian_density(ions, grid, options={}, density = None):
    """
    FWHM : 2*np.sqrt(2.0*np.log(2)) = 2.354820
    """
    fwhm = 2.354820
    nr = grid.nrR
    dnr = (1.0/nr).reshape((3, 1))
    gaps = grid.spacings
    if density is None : density = DirectField(grid=grid)
    ncharge = 0.0
    sigma_min = np.max(gaps) * 2 / fwhm
    scaled_postions=ions.get_scaled_positions()
    for key, option in options.items() :
        rcut = option.get('rcut', 5.0)
        sigma = option.get('sigma', 0.3)
        scale = option.get('scale', 0.0)
        if scale is None or abs(scale) < 1E-10 :
            continue
        sigma = max(sigma, sigma_min)
        border = (rcut / gaps).astype(np.int32) + 1
        border = np.minimum(border, nr//2)
        ixyzA = np.mgrid[-border[0]:border[0]+1, -border[1]:border[1]+1, -border[2]:border[2]+1].reshape((3, -1))
        prho = np.zeros((2 * border[0]+1, 2 * border[1]+1, 2 * border[2]+1))
        for i in range(ions.nat):
            if ions.symbols[i] != key:
                continue
            prho[:] = 0.0
            posi = scaled_postions[i].reshape((1, 3))
            atomp = posi * nr
            atomp = atomp.reshape((3, 1))
            ipoint = np.floor(atomp + 1E-8)
            # ipoint = np.floor(atomp)
            px = atomp - ipoint
            l123A = np.mod(ipoint.astype(np.int32) - ixyzA, nr[:, None])
            positions = (ixyzA + px) * dnr
            positions = np.einsum("j...,kj->k...", positions, grid.lattice)
            dists = LA.norm(positions, axis = 0).reshape(prho.shape)
            index = dists < rcut
            prho[index] = gaussian(dists[index], sigma) * scale
            # prho[index] = gaussian(dists[index], sigma, dim = 0) * scale
            mask = grid.get_array_mask(l123A)
            density[l123A[0][mask], l123A[1][mask], l123A[2][mask]] += prho.ravel()[mask]
            ncharge += scale

    return density, ncharge

def build_pseudo_density(pos, grid, scale = 0.0, sigma = 0.3, rcut = 5.0, density = None, add = True, deriv = 0):
    """
    FWHM : 2*np.sqrt(2.0*np.log(2)) = 2.354820
    """
    fwhm = 2.354820
    nr = grid.nrR
    dnr = (1.0/nr).reshape((3, 1))
    gaps = grid.spacings
    if density is None : density = DirectField(grid=grid)
    sigma_min = np.max(gaps) * 2 / fwhm
    #
    if scale is None : return density
    scale = np.asarray(scale)
    if np.all(np.abs(scale)) < 1E-10 : return density
    #
    sigma = max(sigma, sigma_min)
    border = (rcut / gaps).astype(np.int32) + 1
    border = np.minimum(border, nr//2)
    ixyzA = np.mgrid[-border[0]:border[0]+1, -border[1]:border[1]+1, -border[2]:border[2]+1].reshape((3, -1))
    prho = np.zeros((2 * border[0]+1, 2 * border[1]+1, 2 * border[2]+1))
    posi = grid.cell.scaled_positions(pos)
    atomp = posi * nr
    atomp = atomp.reshape((3, 1))
    ipoint = np.floor(atomp + 1E-8)
    # ipoint = np.floor(atomp)
    px = atomp - ipoint
    l123A = np.mod(ipoint.astype(np.int32) - ixyzA, nr[:, None])
    positions = (ixyzA + px) * dnr
    positions = np.einsum("j...,kj->k...", positions, grid.lattice)
    dists = LA.norm(positions, axis = 0).reshape(prho.shape)
    index = dists < rcut
    if scale.size > 1 :
        if deriv != 1 : raise AttributeError("'scale' with array only works for deriv==1")
        positions = positions.reshape((3, *prho.shape))
        for i, s in enumerate(scale):
            prho[index] += gaussian(dists[index], sigma) * positions[i][index]/sigma**2*scale[i]
    else :
        prho[index] = gaussian(dists[index], sigma) * scale
    mask = grid.get_array_mask(l123A)
    if add :
        density[l123A[0][mask], l123A[1][mask], l123A[2][mask]] += prho.ravel()[mask]
    else :
        density[l123A[0][mask], l123A[1][mask], l123A[2][mask]] = prho.ravel()[mask]
    return density
