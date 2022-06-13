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


Short note on the implementation
--------------------------------

In DFTpy, the electron density is represented in a discrete set of points given by a Cartesian `grid` and contained in a simulation `cell` that is specified by 3 `lattice vectors`. The number of grid points and the cell size are regulated by the user. The Cartesian grid allows for an efficient parallelization of data and work (we use `mpi4py`), and for the exploitation of Fast Fourier Transforms for solving convolution integrals (such as the one needed to compute :math:`E_H[n]`). Either `NumPy.fft` or `PyFFT` are used depending on user input.


.. _KEDF: tutorials/config.html#kedf
.. _EXC: tutorials/config.html#exc
