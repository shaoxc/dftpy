import numpy as np

from dftpy.constants import LEN_CONV, ENERGY_CONV


class USP:
    def __init__(self, fname):
        self.fname = fname
        self.read()

    def read(self, fname):
        """
        !!! NOT FULL TEST !!!
        Reads CASTEP-like usp PP file
        """
        if fname.lower().endswith(("usp", "uspcc")):
            ext = "usp"
        elif fname.lower().endswith('uspso'):
            ext = "uspso"
        else:
            raise AttributeError("Pseudopotential not supported : '{}'".format(fname))

        HARTREE2EV = ENERGY_CONV["Hartree"]["eV"]
        BOHR2ANG = LEN_CONV["Bohr"]["Angstrom"]
        with open(fname, "r") as outfil:
            lines = outfil.readlines()

        ibegin = 0
        for i in range(0, len(lines)):
            line = lines[i]
            if ext == 'usp':
                if "END COMMENT" in line:
                    ibegin = i + 4
                elif ibegin > 1 and (line.strip() == "1000" or len(line.strip()) == 1) and i - ibegin > 4:
                    iend = i
                    break
            elif ext == 'uspso':
                if "END COMMENT" in line:
                    ibegin = i + 5
                elif ibegin > 1 and (line.strip() == "1000" or len(line.strip()) == 5) and i - ibegin > 4:
                    iend = i
                    break

        line = " ".join([line.strip() for line in lines[ibegin:iend]])
        info = {}

        Zval = np.float(lines[ibegin - 2].strip())
        info['zval'] = Zval

        if "1000" in lines[iend] or len(lines[iend].strip()) == 1 or len(lines[iend].strip()) == 5:
            pass
        else:
            raise AttributeError("Error : Check the PP file : {}".format(fname))
        gmax = np.float(lines[ibegin - 1].split()[0]) * BOHR2ANG

        # v = np.array(line.split()).astype(np.float) / (HARTREE2EV*BOHR2ANG ** 3 * 4.0 * np.pi)
        self.v_g = np.array(line.split()).astype(np.float) / (HARTREE2EV * BOHR2ANG ** 3)
        self.r_g = np.linspace(0, gmax, num=len(self.v_g))
        self.v_g[1:] -= Zval * 4.0 * np.pi / self.r_g[1:] ** 2
        # -----------------------------------------------------------------------
        nlcc = int(lines[ibegin - 1].split()[1])
        if nlcc == 2 and ext == 'usp':
            # num_projectors
            for i in range(iend, len(lines)):
                l = lines[i].split()
                if len(l) == 2 and all([item.isdigit() for item in l]):
                    ibegin = i + 1
                    ngrid = int(l[1])
                    break
            core_grid = []
            for i in range(ibegin, len(lines)):
                l = list(map(float, lines[i].split()))
                core_grid.extend(l)
                if len(core_grid) >= ngrid:
                    core_grid = core_grid[:ngrid]
                    break
            info['core_grid'] = np.asarray(core_grid) * BOHR2ANG
            line = " ".join([line.strip() for line in lines[ibegin:]])
            data = np.array(line.split()).astype(np.float)
            info['core_value'] = data[-ngrid:]
        # -----------------------------------------------------------------------
        self.info = info

    @property
    def zval(self):
        return self.info['zval']
