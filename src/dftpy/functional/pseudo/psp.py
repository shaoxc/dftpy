import numpy as np

from dftpy.functional.pseudo.abstract_pseudo import BasePseudo

"""
Ref :
    https://docs.abinit.org/developers/psp8_info/
"""

class PSP(BasePseudo):
    def __init__(self, fname, direct = True, **kwargs):
        super().__init__(fname, direct = direct, **kwargs)

    def read(self, fname):
        # Only support psp8 format
        with open(fname, "r") as fr:
            lines = []
            for i, line in enumerate(fr):
                if i > 5 :
                    line = line.replace('D', 'E')
                lines.append(line)
        info = {}

        # line 2 :atomic number, pseudoion charge, date
        values = lines[1].split()
        atomicnum = int(float(values[0]))
        zval = float(values[1])
        # line 3 :pspcod,pspxc,lmax,lloc,mmax,r2well
        values = lines[2].split()
        if int(values[0]) != 8:
            raise AttributeError("Only support psp8 format pseudopotential with psp")
        info['info'] = lines[:6]
        info['atomicnum'] = atomicnum
        info['zval'] = zval
        info['pspcod'] = 8
        info['pspxc'] = int(values[1])
        info['lmax'] = int(values[2])
        info['lloc'] = int(values[3])
        info['mmax'] = int(values[4])
        info['r2well'] = int(values[5])
        # line 4 : rchrg fchrg qchrg
        values = lines[3].split()
        info['rchrg'] = float(values[0])
        info['fchrg'] = float(values[1])
        info['qchrg'] = float(values[2])
        #
        mmax = info['mmax']
        lloc = info['lloc']
        fchrg = info['fchrg']

        ibegin = 7
        iend = ibegin + mmax
        # line = " ".join([line for line in lines[ibegin:iend]])
        # data = np.fromstring(line, dtype=float, sep=" ")
        # data = np.array(line.split()).astype(np.float64) / HARTREE2EV / BOHR2ANG ** 3
        data = [line.split()[1:3] for line in lines[ibegin:iend]]
        data = np.asarray(data, dtype = np.float64)
        ibegin = 6+ (mmax + 1) * lloc + mmax
        iend = ibegin + mmax

        self.r = data[:, 0]
        self.v = data[:, 1]
        self.info = info
        self._zval = self.info['zval']

        if fchrg > 0.0 :
            core_density = [line.split()[1:3] for line in lines[ibegin:iend]]
            self._core_density_grid = core_density[:, 0]
            core_density = np.asarray(core_density, dtype = np.float64)
            core_density[:, 1] /= (4.0 * np.pi)
            self._core_density = core_density
