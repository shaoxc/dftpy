# DFTpy: Density Functional Theory in Python

## DFTpy

[![pipeline status](https://gitlab.com/pavanello-research-group/dftpy/badges/master/pipeline.svg)](https://gitlab.com/pavanello-research-group/dftpy/pipelines)
[![coverage report](https://gitlab.com/pavanello-research-group/dftpy/badges/master/coverage.svg)](https://gitlab.com/pavanello-research-group/dftpy/pipelines)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

`DFTpy` is an orbital-free Density Functional Theory code based on a plane-wave 
expansion of the electron density developed by [PRG](https://sites.rutgers.edu/prg/) 
at Rutgers University-Newark [@MikPavanello](https://twitter.com/MikPavanello)
. The code bases itself on `pbcpy` by Alessandro Genova (see below) [@AlesGeno](https://twitter.com/AlesGeno) previously at PRG now  [@Kitware](https://twitter.com/Kitware)

### Manual
[`DFTpy` manual](http://dftpy.rutgers.edu)


### Main classes of DFTpy
 - `FunctionalClass`
    1. GGA XC and KEDF handled locally as well as by `LibXC` (`pylibxc`)
    2. nonlocal KEDF (WT, SM, WGC, MGP, LMGP)
    3. `HARTREE` energy and `PSEUDO` electron-ion local pseudopotentials are also encoded in `FunctionalClass` instances
 - `TotalEnergyAndPotential`
    1. `DFTpy` energy evaluator.
    2. Input: `XC`, `KEDF`, `IONS` and `HARTREE` `FunctionalClass` instances
    3. Output: total energy and total potential
- `OptimizationClass`
    1. optimizes the electron density given a `TotalEnergyAndPotential` evaluator
    2. uses `scipy.minimize` as the under-the-hood minimizer as well as in-house algorithms
- `config` is a dictionary handling I/O
- API to [ASE](https://wiki.fysik.dtu.dk/ase/index.html) for geometry relaxations and molecular dynamics

### Contacts
 - [Xuecheng Shao](https://sites.rutgers.edu/prg/people/xuecheng-shao/)
 - [Michele Pavanello](https://sasn.rutgers.edu/about-us/faculty-staff/michele-pavanello)
 - [Alessandro Genova](mailto:ales.genova@gmail.com)
 

## PbcPy

[`pbcpy`](https://gitlab.com/ales.genova/pbcpy) is a Python3 package providing some useful abstractions to deal with
molecules and materials under periodic boundary conditions (PBC).

`pbcpy` has been developed at [PRG](https://sites.rutgers.edu/prg/) by:
- **Alessandro Genova**

with contributions from:
- Tommaso Pavanello
- Michele Pavanello


