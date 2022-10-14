===============
Stable releases
===============


Version 2.0
===========

Release date: `October 2022 <https://gitlab.com/pavanello-research-group/dftpy/-/tags/dftpy-2.0>`_

This version supports:
 - The Ions object is inherit from `ASE.atoms <https://wiki.fysik.dtu.dk/ase/ase/atoms.html>`_
 - Parallel execution with mpi4py_
 - Supported non-standard GGAs, such as LKT_ and PG_
 - Supported spin polarized runs
 - New td-OF-DFT code for real-time propagation and Casida
 - Nonadiabatic Pauli potential with the JP_ functional
 - Extensive upgrade of documentation and docstrings
 - Close to universal I/O capability and compatibility with ASE
 - Support for `finite temperature OF-DFT <https://arxiv.org/abs/2206.03754>`_


Version 1.0
===========

Release date: `January 2020 <https://gitlab.com/pavanello-research-group/dftpy/-/tags/dftpy-1.0>`_

This version supports:
 - OFDFT electron density optimization
 - Latest nonlocal KEDFs
 - Ab initio dynamics
 - Hydrodynamic DFT (TD-OFDFT)
 - I/O for plotting and to several file formats
 - Several examples (also on Jupyter notebooks)

.. _mpi4py: https://mpi4py.readthedocs.io/en/stable/
.. _LKT: https://journals.aps.org/prb/abstract/10.1103/PhysRevB.98.041111 
.. _JP: https://journals.aps.org/prb/abstract/10.1103/PhysRevB.104.235110
.. _PG: https://pubs.acs.org/doi/full/10.1021/acs.jpclett.8b01926

