import os
### Import fft library
try:
    import pyfftw
    FFTLIB = "pyfftw"
except Exception:
    FFTLIB = "numpy"

# FFTLIB = "numpy"
FFTLIB = os.environ.get('DFTPY_FFTLIB', FFTLIB)
# print('Use "%s" for Fourier Transform' % (FFTLIB))
SAVEFFT = os.environ.get('DFTPY_SAVEFFT', False)

LEN_UNITS = ["Bohr", "Angstrom", "nm", "m"]

LEN_CONV = {}
LEN_CONV["Bohr"] = {"Bohr": 1.0, "Angstrom": 0.5291772106712, "nm": 0.05291772106712, "m": 5.291772106712e-11}
LEN_CONV["Angstrom"] = {"Bohr": 1.8897261254535427, "Angstrom": 1.0, "nm": 1.0e-1, "m": 1.0e-10}
LEN_CONV["nm"] = {"Bohr": 18.897261254535427, "Angstrom": 10.0, "nm": 1.0, "m": 1.0e-9}
LEN_CONV["m"] = {"Bohr": 1.8897261254535427e10, "Angstrom": 1.0e10, "nm": 1.0e9, "m": 1.0}

ENERGY_CONV = {}
ENERGY_CONV["eV"] = {"eV": 1.0, "Hartree": 0.03674932598150397}
ENERGY_CONV["Hartree"] = {"eV": 27.2113834279111, "Hartree": 1.0}

FORCE_CONV = {}
FORCE_CONV["Ha/Bohr"] = {"Ha/Bohr": 1.0, "eV/A": ENERGY_CONV["Hartree"]["eV"] / LEN_CONV["Bohr"]["Angstrom"]}

STRESS_CONV = {}
STRESS_CONV["Ha/Bohr3"] = {
    "Ha/Bohr3": 1.0,
    "GPa": 29421.02648438959,
    "eV/A3": ENERGY_CONV["Hartree"]["eV"] / LEN_CONV["Bohr"]["Angstrom"] ** 3,
}

units_warning = "All the quantities in atomic units"

ZERO = 1E-30
