# DFTpy

`DFTpy` is an Density Functional Theory code based on a plane-wave 
expansion of the electron density. The code bases itself on `pbcpy`.
 - `DFTpy` at [PRG](http://michelepavanello.com/) at Rutgers University-Newark @MikPavanello
 - `pbcpy` by Alessandro Genova @AlesGeno previously at PRG now @Kitware


## PbcPy

[![PyPI version](https://img.shields.io/pypi/v/pbcpy.svg)](https://pypi.python.org/pypi/pbcpy/)
[![PyPI status](https://img.shields.io/pypi/status/pbcpy.svg)](https://pypi.python.org/pypi/pbcpy/)
[![pipeline status](https://gitlab.com/ales.genova/pbcpy/badges/master/pipeline.svg)](https://gitlab.com/ales.genova/pbcpy/pipelines)
[![coverage report](https://gitlab.com/ales.genova/pbcpy/badges/master/coverage.svg)](https://gitlab.com/ales.genova/pbcpy/pipelines)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

`pbcpy` is a Python3 package providing some useful abstractions to deal with
molecules and materials under periodic boundary conditions (PBC).

`pbcpy` has been developed at [PRG](http://michelepavanello.com/) by:
- **Alessandro Genova**

with contributions from:
- Tommaso Pavanello
- Michele Pavanello

## Main classes of DFTpy
 - `FunctionalClass`
    1. GGA XC and KEDF handled by `LibXC` (`pylibxc`)
    2. nonlocal KEDF will be available soon
    3. `HARTREE` and electron-ion pseudopotentials (local) are also encoded in `FunctionalClass` instances
 - `TotalEnergyAndPotential`
    1. total energy evaluator
    2. uses `XC`, `KEDF`, `IONS` and `HARTREE` `FunctionalClass` instances
    3. `.Energy(rho,ions)` evaluates the energy
- `OptimizationClass`
    1. optimizes the electron density given a `TotalEnergyAndPotential`
    2. uses `scipy.minimize` as the under-the-hood minimizer

