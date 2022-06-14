.. _ofdft:

How does DFTpy work?
====================

Orbital-Free DFT
----------------

In OF-DFT, the ground state electron density, :math:`n_0(\mathbf{r})`, is obtained from the direct minimization of the ground state energy density functional, :math:`E[n]`. Namely,

.. math::
   n_0 = \text{argmin}_n \left[ E[n] - \mu \left( \int n(\mathbf{r}) d\mathbf{r} - N \right) \right]

where :math:`N` is the number of valence electrons in the system. Both :math:`n_0(\mathbf{r})` and  :math:`\mu` are determined during the minimization. The ground state energy is, :math:`E_0 = E[n_0]`.

In practice, the above minimization can only be carried out if the ground state energy functional is known as a pure functional of the density. The energy functional is a sum of several terms: 

.. math::
   E[n]=T_s[n]+E_H[n]+E_{xc}[n]+\int v_{ext}(\mathbf{r}) n(\mathbf{r}) d\mathbf{r}

where

    * :math:`T_s[n]`: noninteracting kinetic energy or KEDF. 
    * :math:`E_{xc}[n]`: exchange-correlation energy or EXC. 
    * :math:`E_{H}[n]=\frac{1}{2}\int \frac{n(\mathbf{r})n(\mathbf{r}^\prime)}{|\mathbf{r}-\mathbf{r}^\prime|}d\mathbf{r} d\mathbf{r}^\prime`: Hartree energy.
    * :math:`v_{ext}(\mathbf{r})`: the external potential (typically the electron-ion interaction).


.. note:: In DFTpy, :math:`T_s[n]` and :math:`E_{xc}[n]` are pure functionals of the density. Check out the tutorials for a list of available KEDF_ and EXC_ functionals, 



.. note::
   DFTpy solves the ground state problem with the so-called `direct energy minimization`. Other (faster) methods are available, such as OESCF_, which is implemented in eDFTpy_. OESCF_ may be implemented in DFTpy upon request.


Time-Dependent Orbital-Free DFT
-------------------------------

DFTpy can also describe systems out of equilibrium by propagating them in `real time`_ as well as in `frequency space`_ finding the roots of the frequency dependent polarizability (Casida). Because of the OF-DFT foundation, DFTpy implements the so-called time-dependent orbital-free DFT (td-OF-DFT) approach whereby a single Bosonic wavefunction, :math:`\Psi(\mathbf{r},t)` is propagated with a time-dependent KS-like Hamiltonian. Namely,

.. math::
   \hat{H}(t)  \Psi(\mathbf{r},t) = i \frac{d}{dt}\Psi(\mathbf{r},t)

where 

.. math:: 
   \hat{H}(t) = -\frac{1}{2} \nabla^2 + v_B(\mathbf{r},t).

The Bosonic KS-like potential is given by two major contributions

.. math::
   v_B(\mathbf{r},t) = v_s(\mathbf{r},t) + v_P(\mathbf{r},t)

where :math:`v_s(\mathbf{r},t)=v_{ext}(\mathbf{r},t)+v_H[n(t)](\mathbf{r},t)+v_{xc}[n(t)](\mathbf{r},t)` where the adiabatic approximation has been invoked. The `Pauli` potential is given by adiabatic and nonadiabatic contributions, 

.. math:: 
   v_P(\mathbf{r},t)=v_P^{ad}(\mathbf{r},t)+v_P^{nad}(\mathbf{r},t).


.. note::
   The adiabatic Pauli potential can be specified according to any given KEDF_ available in DFTpy.

The nonadiabatic contribution is usually neglected in the litarature. In DFTpy the JP_ functional is available,

.. math::
   v_P^{nad}(\mathbf{r},t) = -\frac{\pi^3}{12}\left(\frac{6}{k_F^2(\mathbf{r},t)}\mathcal{F}^{-1}\left\{i\mathbf{q}\cdot\mathbf{j}(\mathbf{q},t)\frac{1}{q}\right\}+\frac{1}{k_F^4(\mathbf{r},t)}\mathcal{F}^{-1}\left\{i\mathbf{q}\cdot\mathbf{j}(\mathbf{q},t)q\right\}\right)

where :math:`\mathbf{j}` and :math:`\mathbf{q}` are the electronic current density and the reciprocal space vector, respectively. The current density is determined by the standard equation :math:`\mathbf{j}(\mathbf{r})=\frac{1}{2i}\left[\Psi^*(\mathbf{r})\nabla\Psi(\mathbf{r})-\Psi(\mathbf{r})\nabla\Psi^*(\mathbf{r})\right]`.  :math:`\mathcal{F}` stands for Fourier transform and :math:`k_F(\mathbf{r},t)=[3\pi^2 n(\mathbf{r},t)]^{1/3}` is the Fermi wavevector function of the local electron density.


.. warning::
   The JP potential is numerically challenging. Refer to the original JP_ publication for details. 



.. note::
   Optical spectra and nonlinear electronic processes can be modelled by DFTpy. See tutorials for additional information. Ehrenfest dynamics is not yet available.


Short note on the implementation
--------------------------------

In DFTpy, the electron density is represented in a discrete set of points given by a Cartesian `grid` and contained in a simulation `cell` that is specified by 3 `lattice vectors`. The number of grid points and the cell size are regulated by the user. The Cartesian grid allows for an efficient parallelization of data and work (we use `mpi4py`), and for the exploitation of Fast Fourier Transforms for solving convolution integrals (such as the one needed to compute :math:`E_H[n]`). Either `NumPy.fft` or `PyFFT` are used depending on user input.


References
----------
* `DFTpy release paper (ground state and td-OF-DFT) <https://wires.onlinelibrary.wiley.com/doi/abs/10.1002/wcms.1482>`_
* `DFTpy td-OF-DFT (Casida) <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.103.245102>`_
* `DFTpy td-OF-DFT (JP nonadiabatic Pauli potential) <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.104.235110>`_
* `OESCF solver for OF-DFT <https://pubs.acs.org/doi/abs/10.1021/acs.jpclett.1c00716>`_


.. _KEDF: tutorials/config.html#kedf
.. _EXC: tutorials/config.html#exc
.. _JP: https://journals.aps.org/prb/abstract/10.1103/PhysRevB.104.235110
.. _`real time`: https://wires.onlinelibrary.wiley.com/doi/abs/10.1002/wcms.1482 
.. _`frequency space`: https://journals.aps.org/prb/abstract/10.1103/PhysRevB.103.245102
.. _OESCF: https://pubs.acs.org/doi/abs/10.1021/acs.jpclett.1c00716
.. _eDFTpy: http://edftpy.rutgers.edu
