import os
import sys
from ase import units as Units
try:
    import pyfftw
    FFTLIB = "pyfftw"
except Exception:
    FFTLIB = "numpy"

# _Grav = Units._Grav
# _Nav = Units._Nav
# _amu = Units._amu
# _auf = Units._auf
# _aup = Units._aup
# _aut = Units._aut
# _auv = Units._auv
# _c = Units._c
# _e = Units._e
# _eps0 = Units._eps0
# _hbar = Units._hbar
# _hplanck = Units._hplanck
# _k = Units._k
# _me = Units._me
# _mp = Units._mp
# _mu0 = Units._mu0
# alpha = Units.alpha
# eV = Units.eV
# fs = Units.fs
# invcm = Units.invcm
# kB = Units.kB
# kJ = Units.kJ
# kcal = Units.kcal
# kg = Units.kg
# m = Units.m
# mol = Units.mol
# nm = Units.nm
# s = Units.s
# second = Units.second
# A = Units.A
# AUT = Units.AUT
# Ang = Units.Ang
# Angstrom = Units.Angstrom
# Bohr = Units.Bohr
# C = Units.C
# Debye = Units.Debye
# GPa = Units.GPa
# Ha = Units.Ha
# Hartree = Units.Hartree
# J = Units.J
# Pascal = Units.Pascal
# bar = Units.bar
# Ry = Units.Ry
# Rydberg = Units.Rydberg

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
#https://en.wikipedia.org/wiki/Hartree_atomic_units
TIME_CONV = {}
TIME_CONV["au"] = {
        's' : 2.4188843265857e-17,
        'fs' : 2.4188843265857e-2,
}

SPEED_OF_LIGHT = 137.035999084
C_TF = 2.87123400018819181594
TKF0 = 6.18733545256027186194

CBRT_TWO = 1.25992104989487316477

ZERO = 1E-30
# set to 0 if smaller than ZERO

environ = {} # You can change it anytime you want
environ['STDOUT'] = sys.stdout # file descriptor of sprint
try:
    environ['LOGLEVEL'] = int(os.environ.get('DFTPY_LOGLEVEL', 2)) # The level of sprint
except Exception :
    environ['LOGLEVEL'] = 2 # The level of sprint
"""
    0 : all
    1 : debug
    2 : info
    3 : warning
    4 : error
"""
environ['FFTLIB'] = os.environ.get('DFTPY_FFTLIB', FFTLIB)
# DFTpy old units
# Units.Bohr = 0.5291772106712
# Units.Ha = 27.2113834279111
