#!/usr/bin/env python3
import os
import unittest
import numpy as np

from dftpy.functionals import FunctionalClass
from dftpy.constants import LEN_CONV
from dftpy.formats.qepp import PP
from dftpy.ewald import ewald
from dftpy.local_pseudopotential import NuclearElectronForce, NuclearElectronForcePME
from dftpy.local_pseudopotential import NuclearElectronStress, NuclearElectronStressPME

class Test(unittest.TestCase):
    def test_ie(self):
        dftpy_data_path = os.environ.get('DFTPY_DATA_PATH')
        print()
        print("*"*50)
        print("Testing loading pseudopotentials")
        mol = PP(filepp=dftpy_data_path+"/Al_fde_rho.pp").read()
        optional_kwargs = {}
        optional_kwargs["PP_list"] = {'Al': dftpy_data_path+"/Al_lda.oe01.recpot"}
        optional_kwargs["ions"]    = mol.ions 
        IONS = FunctionalClass(type='IONS', optional_kwargs=optional_kwargs)
        func  = IONS.ComputeEnergyPotential(rho=mol.field)
        a = func.potential
        optional_kwargs = {}
        optional_kwargs["PP_list"] = {'Al': dftpy_data_path+"/Al_lda.oe01.recpot"}

        rho = mol.field
        IE_Energy = func.energy
        IE_Force = NuclearElectronForce(mol.ions, rho, PP_file=optional_kwargs["PP_list"])
        IE_Stress = NuclearElectronStress(mol.ions, rho, PP_file=optional_kwargs["PP_list"])

        optional_kwargs = {}
        optional_kwargs["PP_list"] = {'Al': dftpy_data_path+"/Al_lda.oe01.recpot"}
        optional_kwargs["ions"]    = mol.ions 
        mol.ions.usePME = True
        IONS = FunctionalClass(type='IONS', optional_kwargs=optional_kwargs)
        func  = IONS.ComputeEnergyPotential(rho=mol.field)
        IE_Energy_PME = func.energy
        IE_Force_PME = NuclearElectronForcePME(mol.ions, rho, PP_file=optional_kwargs["PP_list"])
        IE_Stress_PME = NuclearElectronStressPME(mol.ions, rho, PP_file=optional_kwargs["PP_list"])

        print('IE energy', IE_Energy, IE_Energy_PME)
        self.assertTrue(np.isclose(IE_Energy, IE_Energy_PME, atol = 1.E-4))
        print('IE forces', IE_Force, IE_Force_PME)
        self.assertTrue(np.allclose(IE_Force, IE_Force_PME, atol = 1.E-4))
        print('IE stress', IE_Stress, IE_Stress_PME)
        self.assertTrue(np.allclose(IE_Stress, IE_Stress_PME, atol = 1.E-4))
        
    def test_ewald_PME(self):
        dftpy_data_path = os.environ.get('DFTPY_DATA_PATH')
        print()
        print("*"*50)
        print("Testing particle mesh Ewald method")
        mol = PP(filepp=dftpy_data_path+"/Al_fde_rho.pp").read()
        Ewald_ = ewald(rho = mol.field, ions = mol.ions, verbose = False)
        Ewald_PME = ewald(rho = mol.field, ions = mol.ions, verbose = False, PME = True)

        print('Ewald energy', Ewald_.energy, Ewald_PME.energy)
        self.assertTrue(np.allclose(Ewald_.energy, Ewald_PME.energy, atol = 1.E-5))
        print('Ewald forces', Ewald_.forces, Ewald_PME.forces)
        self.assertTrue(np.allclose(Ewald_.forces, Ewald_PME.forces, atol = 1.E-5))
        print('Ewald stress', Ewald_.stress, Ewald_PME.stress)
        self.assertTrue(np.allclose(Ewald_.stress, Ewald_PME.stress, atol = 1.E-5))

if __name__ == "__main__":
    unittest.main()
