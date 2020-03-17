import numpy as np
from scipy.linalg import eigh
from dftpy.field import DirectField
from dftpy.functionals import FunctionalClass, TotalEnergyAndPotential

class Casida(object):

    def __init__(self, rho0, KE, XC):
        if not isinstance(rho0, DirectField):
            raise TypeError("rho0 must be an rank-1 or -2 DFTpy DirectField.")
        if rho0.rank == 2:
            self.rho0 = rho0
            self.polarized = True
        elif rho0.rank == 1:
            self.rho0 = DirectField(rho0.grid, rank=2, griddata_3d=np.stack([rho0/2, rho0/2], axis=0))
            self.polarized = False
        else:
            raise AttributeError("rho0 must be an rank-1 or -2 DFTpy DirectField.")
        self.N = rho0.integral()
        self.grid = rho0.grid
        self.functional = TotalEnergyAndPotential(KineticEnergyFunctional = KE, XCFunctional = XC)
        self.fkxc = self.functional(self.rho0, calcType=['V2']).v2rho2

    def calc_k(self, psi_i, psi_j, vh):
        if not self.polarized:
            fkxc = DirectField(self.grid, rank=1, griddata_3d=(self.fkxc[0]+self.fkxc[1])/2.0)
            return (psi_i*(vh+fkxc*psi_j)).integral()
        else:
            raise Exception('Spin polarized Casida is not implemented')

    def calc_k_tri(self, psi_i, psi_j):
        if not self.polarized:
            fkxc = DirectField(self.grid, rank=1, griddata_3d=(self.fkxc[0]-self.fkxc[1])/2.0)
            return (psi_i*fkxc*psi_j).integral()
        else:
            raise Exception('Spin polarized Casida is not implemented')
    
    def calc_mu(self, psi_1, psi_2, direc=0):
        x = psi_1.grid.r[direc]
        mu = (psi_1*x*psi_2).integral()
        return mu

    def build_matrix(self, num_psi, eigs, psi_list, calc_triplet=False):
        hartree = FunctionalClass(type='HARTREE')
        self.c = np.empty([num_psi-1, num_psi-1], dtype = np.float64)
        if calc_triplet:
            self.c_tri = np.empty([num_psi-1, num_psi-1], dtype = np.float64)
        omega = eigs[1:]-eigs[0]
        for j in range(1, num_psi):
            psi_j = psi_list[0] * psi_list[j]
            vh = hartree(psi_j, calcType = ['V']).potential
            for i in range(j, num_psi):
                psi_i = psi_list[0] * psi_list[i]
                self.c[i-1,j-1] = self.calc_k(psi_i, psi_j, vh) * self.N * 2.0 * np.sqrt(omega[i-1]*omega[j-1])
                if calc_triplet:
                    self.c_tri[i-1,j-1] = self.calc_k_tri(psi_i, psi_j) * self.N * 2.0 * np.sqrt(omega[i-1]*omega[j-1])
                if not i == j:
                    self.c[j-1, i-1] = self.c[i-1, j-1]
                    if calc_triplet:
                        self.c_tri[j-1, i-1] = self.c_tri[i-1, j-1]

        self.c +=  np.identity(num_psi-1) * (omega * omega)
        if calc_triplet:
            self.c_tri +=  np.identity(num_psi-1) * (omega * omega)
        self.sqrtomega = np.sqrt(omega)
        #print(self.sqrtomega)
        #print(self.c)
        #print(self.c_tri)

        self.x = np.empty(num_psi-1, dtype = np.float64)
        self.y = np.empty(num_psi-1, dtype = np.float64)
        self.z = np.empty(num_psi-1, dtype = np.float64)
        for i in range(1, num_psi):
            self.x[i-1] = self.calc_mu(psi_list[0], psi_list[i])
            self.y[i-1] = self.calc_mu(psi_list[0], psi_list[i], direc=1)
            self.z[i-1] = self.calc_mu(psi_list[0], psi_list[i], direc=2)

    def __call__(self, calc_triplet=False):
        if not hasattr(self, 'c'):
            raise Exception("Matrix is not built yet. Run build_matrix first.")
        if calc_triplet and not hasattr(self, 'c_tri'):
            raise Exception("Matrix for triplet is not built yet. Run build_matrix with calc_triplet=True first.")
        num_modes = np.shape(self.c)[0]

        omega2, z_list = eigh(self.c)
        omega = np.sqrt(omega2)
        omega = np.real(omega)

        f = np.empty(num_modes, dtype = np.float64)
        for i in range(num_modes):
            tmp = np.sum(self.x * self.sqrtomega * z_list[:,i])
            f[i] = tmp * tmp
            tmp = np.sum(self.y * self.sqrtomega * z_list[:,i])
            f[i] += tmp * tmp
            tmp = np.sum(self.z * self.sqrtomega * z_list[:,i])
            f[i] += tmp * tmp

            f[i] = f[i] * 2.0 / 3.0

        if calc_triplet:
            omega2, z_list = eigh(self.c_tri)
            omega_tri = np.sqrt(omega2)
            omega_tri = np.real(omega_tri)

            f_tri = np.empty(num_modes, dtype = np.float64)
            for i in range(num_modes):
                tmp = np.sum(self.x * self.sqrtomega * z_list[:,i])
                f_tri[i] = tmp * tmp
                tmp = np.sum(self.y * self.sqrtomega * z_list[:,i])
                f_tri[i] += tmp * tmp
                tmp = np.sum(self.z * self.sqrtomega * z_list[:,i])
                f_tri[i] += tmp * tmp

                f_tri[i] = f_tri[i] * 2.0 / 3.0

            omega = np.real(np.concatenate((omega, omega_tri)))
            f = np.concatenate((f, f_tri))

        return omega, f

