# DFTpy: Density Functional Theory in Python

`DFTpy` is an orbital-free Density Functional Theory code based on a plane-wave expansion of the electron density developed by [PRG](https://sites.rutgers.edu/prg/) at [Rutgers University-Newark](http://sasn.rutgers.edu). The code is based on [`pbcpy`](https://gitlab.com/ales.genova/pbcpy).
`DFTpy` is a serial code (some open-mp can be leveraged from the under-the-hood Intel mkl) capable of handling million-atom systems [reference to appear shortly]().

### Main features
 - `DFTpy` is fast, leveraging most known high-efficiency algorithms (such as PME).
 - `DFTpy` carries an implementation of atomistic real-time time-dependent orbital-free DFT [reference to appear]() also known as hydrodynamic DFT.
 - The Python code base is developer-friendly providing a low barrier to entry for new developers.

### Manual
[`DFTpy` manual](http://dftpy.rutgers.edu)

### Main classes of DFTpy
 - `FunctionalClass`
    1. GGA XC and KEDF handled locally as well as by `LibXC` (`pylibxc`)
    2. nonlocal KEDF (WT, SM, WGC, MGP, LMGP)
    3. `HARTREE` energy and `PSEUDO` electron-ion local pseudopotentials are also encoded in `FunctionalClass` instances
 - `TotalEnergyAndPotential`
    1. Main energy evaluator for `DFTpy`
    2. Input: `XC`, `KEDF`, `IONS` and `HARTREE` `FunctionalClass` instances
    3. Output: total energy and total potential
- `OptimizationClass`
    1. Optimizes the electron density given a `TotalEnergyAndPotential` evaluator
    2. Uses `scipy.minimize` as the under-the-hood minimizer as well as in-house algorithms
- `config`, a dictionary (and associated methods) handling I/O
- `DFTpyCalculator`, an API to [ASE](https://wiki.fysik.dtu.dk/ase/index.html) for geometry relaxations and molecular dynamics

### Contacts
 - [Xuecheng Shao](https://sites.rutgers.edu/prg/people/xuecheng-shao/)
 - [Kaili Jiang](https://sites.rutgers.edu/prg/people/kaili-jiang/)
 - [Michele Pavanello](https://sasn.rutgers.edu/about-us/faculty-staff/michele-pavanello)
 - [Alessandro Genova](mailto:ales.genova@gmail.com)
 

## PbcPy

[`pbcpy`](https://gitlab.com/ales.genova/pbcpy) is a Python3 package providing some useful abstractions to deal with
molecules and materials under periodic boundary conditions (PBC).

`pbcpy` has been developed at [PRG](https://sites.rutgers.edu/prg/) by:
- Alessandro Genova [@AlesGeno](https://twitter.com/AlesGeno) previously at PRG now [@Kitware](https://twitter.com/Kitware)

with contributions from:
- Tommaso Pavanello
- Michele Pavanello


