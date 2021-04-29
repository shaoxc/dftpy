import numpy as np


class PSP:
    def __init__(self, fname):
        self.fname = fname
        self.read(fname)

    def read(self, fname):
        # Only support psp8 format
        # HARTREE2EV = ENERGY_CONV["Hartree"]["eV"]
        # BOHR2ANG = LEN_CONV["Bohr"]["Angstrom"]
        with open(fname, "r") as outfil:
            lines = outfil.readlines()
        info = {}

        # line 2 :atomic number, pseudoion charge, date
        values = lines[1].split()
        atomicnum = int(float(values[0]))
        Zval = float(values[1])
        # line 3 :pspcod,pspxc,lmax,lloc,mmax,r2well
        values = lines[2].split()
        if int(values[0]) != 8:
            raise AttributeError("Only support psp8 format pseudopotential with psp")
        info['info'] = lines[:6]
        info['atomicnum'] = atomicnum
        info['zval'] = Zval
        info['pspcod'] = 8
        info['pspxc'] = int(values[1])
        info['lmax'] = int(values[2])
        info['lloc'] = int(values[3])
        info['r2well'] = int(values[5])
        # pspxc = int(value[1])
        mmax = int(values[4])

        ibegin = 7
        iend = ibegin + mmax
        # line = " ".join([line for line in lines[ibegin:iend]])
        # data = np.fromstring(line, dtype=float, sep=" ")
        # data = np.array(line.split()).astype(np.float) / HARTREE2EV / BOHR2ANG ** 3
        data = [line.split()[1:3] for line in lines[ibegin:iend]]
        data = np.asarray(data, dtype=float)

        self.r = data[:, 0]
        self.v = data[:, 1]
        self.info = info

    @property
    def zval(self):
        return self.info['zval']
