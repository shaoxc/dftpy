from dftpy.functional.abstract_functional import AbstractFunctional
from dftpy.functional.functional_output import FunctionalOutput


class TotalFunctional(AbstractFunctional):
    """
     Object handling energy evaluation for the
     purposes of optimizing the electron density

     Attributes
     ----------

     KineticEnergyFunctional, XCFunctional, PSEUDO, HARTREE: Functional
         Instances of functional class needed for the computation
         of the chemical potential, total potential and total energy.

     Example
     -------

     KE = Functional(type='KEDF',name='TF')
     XC = Functional(type='XC',name='LDA')
     PSEUDO = Functional(type='PSEUDO', kwargs)
     HARTREE = Functional(type='HARTREE')

     EnergyEvaluator = TotalEnergyAndPotential(
         KineticEnergyFunctional = KE,
         XCFunctional = XC,
         PSEUDO = PSEUDO,
         HARTREE = HARTREE
     )

     or given a dict
     funcdict = {
         "KineticEnergyFunctional": KE,
         "XCFunctional": XC,
         "PSEUDO": PSEUDO,
         "HARTREE": HARTREE
     }
     EnergyEvaluator = TotalEnergyAndPotential(**funcdict)

     [the energy:]
     E = EnergyEvaluator.Energy(rho,ions)

     [total energy and potential:]
     out = EnergyEvaluator.compute(rho)

     [time for optimization of density:]
     in_for_scipy_minimize = EnergyEvaluator(phi)
    """

    def __init__(self, **kwargs):

        self.funcDict = {}
        self.funcDict.update(kwargs)
        # remove useless key
        for key, evalfunctional in list(self.funcDict.items()):
            if evalfunctional is None:
                del self.funcDict[key]

        self.UpdateNameType()

    def __repr__(self):
        return self.funcDict.__repr__()
    
    def __getitem__(self, key):
        return self.funcDict[key]
    
    def __setitem__(self, key, value):
        self.funcDict.update({key: value})

    def UpdateNameType(self):
        self.name = ""
        self.type = ""
        for key, evalfunctional in self.funcDict.items():
            if not isinstance(evalfunctional, AbstractFunctional):
                raise TypeError("{} must be AbstractFunctional".format(key))
            setattr(self, key, evalfunctional)
            self.name += getattr(evalfunctional, 'name') + " "
            self.type += getattr(evalfunctional, 'type') + " "

    def UpdateFunctional(self, keysToRemove=[], newFuncDict={}):
        for key in keysToRemove:
            del self.funcDict[key]

        self.funcDict.update(newFuncDict)
        self.UpdateNameType()

    def Subset(self, keys):
        subdict = dict((key, self.funcDict[key]) for key in keys)
        return TotalFunctional(**subdict)

    def compute(self, rho, calcType={"E", "V"}, *args, **kwargs):
        Obj = None
        for key, evalfunctional in self.funcDict.items():
            if Obj is None:
                Obj = evalfunctional(rho, calcType=calcType, *args, **kwargs)
            else:
                Obj += evalfunctional(rho, calcType=calcType, *args, **kwargs)
            # sss = evalfunctional(rho, ["E","V"])
            # sss.energy = rho.mp.vsum(sss.energy)
            # print('key', key, sss.energy)
        # print('-' * 80)
        if Obj is None:
            Obj = FunctionalOutput(name='NONE')
        if 'E' in calcType:
            Obj.energy = rho.mp.vsum(Obj.energy)
        return Obj

    def Energy(self, rho, ions, usePME=False):
        from dftpy.ewald import ewald

        ewald_ = ewald(rho=rho, ions=ions, PME=usePME)
        total_e = self.compute(rho, calcType={"E"})
        ewald_energy = rho.mp.vsum(ewald_.energy)
        # print('ewald', ewald_energy)
        return ewald_energy + total_e.energy
