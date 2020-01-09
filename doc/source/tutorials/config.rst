.. _config:

====================
Script mode of DFTpy
====================

DFTpy is a set of python modules. However, it can be executed in the 'ol way by using the `dftpy` script which is generated at installation time. Here's a quick guide to the script's configuration dictionary, or `config`. 


.. list-table::

     * - `JOB`_
       - `PATH`_
       - `MATH`_
       - `PP`_
     * - `CELL`_
       - `GRID`_
       - `DENSITY`_
       - `EXC`_
     * - `KEDF`_
       - `OUTPUT`_
       - `OPT`_
       - `PROPAGATOR`_
     * - `TD`_
       -
       -
       -

.. warning:: 
    `PP`_ and `CELL`_ are mandatory inputs (i.e., no defaults are avaliable for them).

.. note::
    Defaults work well for most arguments.

    When *Options* is empty, it can accept any `float` or `integer`.


JOB
----------

    Control of the running job.

.. option:: task

    The task to be performed.
        *Options* : Optdensity, Calculation

        *Default* : Optdensity

.. option:: calctype

    The property to be calculated.
        *Options* : Energy, Potential, Both, Force, Stress

        *Default* : Energy

PATH
----------

    Specify the path of needed files.

.. option:: pppath

    The path of pseudopotential.
        *Options* :

        *Default* : None

.. option:: cellpath

    The path of input structure.
        *Options* :

        *Default* : None


MATH
----------

    Some methods and techniques that make `DFTpy` really fast.

.. option:: linearii

    Linear-scaling method to deal with Ion-Ion interactions (PME).
        *Options* : True, False

        *Default* : True

.. option:: linearie

    Linear-scaling method to deal with Ion-Electron interactions (PME).
        *Options* : True

        *Default* : True

.. option:: twostep

    A two-step method for performing density optimizations. '`True`' is equivalent to :option:`multistep` = 2.
        *Options* : True, False

        *Default* : False

.. option:: multistep

    A multi-step method for performing density optimizations.
        *Options* : 1,2,...

        *Default* : 1

.. option:: reuse

    Except in the first step, the initial density is given by the optimized density of the previous step.
        *Options* : True, False

        *Default* : True


PP
----------

    Pseudopotential file of each atom type.

        *e.g.*

        - *Al* = Al_lda.oe01.recpot
        - *Mg* = Mg_lda.oe01.recpot



CELL
----------

    Information about the input structure.

.. option:: cellfile

    The file of input structure.
        *Options* :

        *Default* : POSCAR

.. option:: elename

    The name of atom.
        *Options* :

        *Default* : Al

.. option:: zval

    The charge of atomic species.
        *Options* :

        *Default* : None

.. option:: format

    The format of structure file.
        *Options* : pp, vasp, xsf,...

        *Default* : None


GRID
----------

     Control the grid.

.. option:: ecut

    The kinetic energy cutoff (eV).
        *Options* :

        *Default* : 600

.. option:: spacing

    The spacing (or gap) separating nearest real space grid points. If set this, :option:`ecut` is disabled.
        *Options* :

        *Default* : None

.. option:: gfull

    Determines oif the number of grid points in the reciprocal and real space grids are equal. If '`False`' only use half grid, which will be faster. 

        *Options* : True, False

        *Default* : False

.. warning:: Be careful: '`gfull=True`' implies that the dftpy.field used is real in real space.

.. option:: nr

    The number of grid points in the direction of the three lattice vectors.
        *Options* :

        *Default* : None

        *e.g.*

            *nr* = 32 32 32


DENSITY
----------

    Control the charge density.

.. option:: densityini

    The initial density is given by homogeneous electron gas (HEG) or read from :option:`densityfile`. If set `Read`, must given the :option:`densityfile`.
        *Options* : HEG, Read

        *Default* : HEG

.. option:: densityfile

    The charge density for initial density, only works when if :option:`densityini` set `Read`.
        *Options* :

        *Default* : None

.. option:: densityoutput

    The output file of final density. The default is not output the density.
        *Options* :

        *Default* : None


EXC
----------

    Control the exchange-correlation functional.

.. option:: xc

    The kind of exchange-correlation functional. If not `LDA`, must have pylibxc_ installed.
        *Options* : LDA, PBE,...

        *Default* : LDA

.. warning:: Stress is not implemented for non-LDA xc functionals.

.. option:: x_str

    The type of exchange functional.
        *Options* :

        *Default* : lda_x

.. option:: c_str

    The type of correlation functional.
        *Options* :

        *Default* : lda_c_pz


KEDF
----------

    Control the kinetic energy density functional (KEDF). 
    `DFTpy` features most KEDFs, from GGAs to nonlocal to nonlocal with density dependent kernel.

.. option:: kedf

    The type of KEDF.
        *Options* : TF, vW, x_TF_y_vW, WT, MGP,...

        *Default* : WT

.. option:: x

    The ratio of TF KEDF.
        *Options* :

        *Default* : 1.0

.. option:: y

    The ratio of vW KEDF.
        *Options* :

        *Default* : 1.0

.. option:: alpha

    The alpha parameter typical in  nonlocal KEDF :math:`\rho^{\alpha}`.
        *Options* :

        *Default* : 5.0/6.0

.. option:: beta

    The beta parameter typical in  nonlocal KEDF :math:`\rho^{\beta}`.
        *Options* :

        *Default* : 5.0/6.0

.. option:: sigma

    A parameter used to smooth with a Gaussian convolution FFTs of problematic functions (e.g., invfft of :math:`{G^2\rho(G)}` ). 
        *Options* :

        *Default* : None

.. option:: nsp

    The number of :math:`{k_{f}}` points for splining `LWT` like nonlocal KEDFs. There are three options to achieve the same goal, the priority is :option:`nsp` -> :option:`delta` -> :option:`ratio`. Default is using :option:`ratio`.
        *Options* :

        *Default* : None

.. option:: delta

    The gap of :math:`{k_{f}}` for spline in `LWT` KEDF. There are three options to do same thing, the priority is :option:`nsp` -> :option:`delta` -> :option:`ratio`. Default is using :option:`ratio`.
        *Options* :

        *Default* : None

.. option:: ratio

    The ratio of :math:`{k_{f}}` for spline in `LWT` KEDF. There are three options to do same thing, the priority is :option:`nsp` -> :option:`delta` -> :option:`ratio`. Default is using :option:`ratio`.
        *Options* :

        *Default* : 1.2

.. option:: interp

    The interpolation method for `LWT` KEDF's kernel from the kernel table.
        *Options* : linear, newton, hermite

        *Default* : hermite

.. option:: kerneltype

    The kernel for `LWT` KEDF.
        *Options* : WT, MGP

        *Default* : WT

.. option:: symmetrization

    The symmetrization way for `MGP` KEDF. See `paper <https://aip.scitation.org/doi/abs/10.1063/1.5023926>`_.
        *Options* : None, Arithmetic, Geometric

        *Default* : None

.. option:: lumpfactor

    The kinetic electron for `LWT` KEDF.
        *Options* :

        *Default* : None

.. option:: neta

    The max number of discrete :math:`\eta` for `LWT` KEDF.
        *Options* :

        *Default* : 50000

.. option:: etamax

    The max value of \eta for kernel in `LWT` KEDF.
        *Options* :

        *Default* : 50.0

.. option:: order

    The order for the interpolation of the kernel in `LWT` KEDF. '0' means using the value of nearest-neighbor point is used.
        *Options* :

        *Default* : 3

.. option:: maxpoints

    The max number of integration points for the evaluation of the `MGP` kernel.
        *Options* :

        *Default* : 1000

.. option:: kdd

    The kernel density denpendent for `LWT` KEDF:
        + 1 : The origin `LWT` KEDF.
        + 2 : Considers the :math:`\rho^{\beta}(r')\omega(\rho(r),r-r')` term in the potential.
        + 3 : Also considers the derivative of kernel which is neglected in LWT. See `paper <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.100.041105>`_.

        *Options* : 1,2,3

        *Default* : 3

.. option:: rho0

    The 'average' density used for the definition of the Fermi momentum. Default is None, which means it calculated based on the total charge and system volume.
        *Options* :

        *Default* : None


OUTPUT
----------

    Control the output.

.. option:: time

    Output the time information of all parts.
        *Options* : True, False

        *Default* : True

.. option:: stress

    Output the stress information of all terms.
        *Options* :

        *Default* : True


OPT
----------

    Control the charge density optimization.

.. option:: method

    The density optimization method.
        *Options* : TN, LBFGS, CG-HS, CG-DY, CG-CD, CG-LS, CG-FR, CG-PR

        *Default* : CG-HS

.. option:: algorithm

    The direct minimization method : Energy (EMM) or Residual (RMM).
        *Options* : EMM, RMM

        *Default* : EMM

.. option:: vector

    The scheme to deal with search direction.
        *Options* :  Orthogonalization, Scaling

        *Default* : Orthogonalization

.. option:: c1

    The wolfe parameters `c1`
        *Options* :

        *Default* : 1e-4

.. option:: c2

    The wolfe parameters `c2`
        *Options* :

        *Default* : 2e-1

.. option:: maxls

    The max steps for line search.
        *Options* :

        *Default* : 10

.. option:: econv

    The energy convergence for last three steps (a.u./atom).
        *Options* :

        *Default* : 1e-6

.. option:: maxfun

    The max steps for function calls. For `TN` density optimization method its the max steps for searching direction.
        *Options* :

        *Default* : 50

.. option:: maxiter

    The max steps for optimization
        *Options* :

        *Default* : 100

.. option:: xtol

    Relative tolerance for an acceptable step.
        *Options* :

        *Default* : 1e-12

.. option:: h0

    The initial approximation for the inverse Hessian needed by `LBFGS`.
        *Options* :

        *Default* : 1.0


PROPAGATOR
----------

    Control of the propagator.
    `DFTpy` has an implementation of hydrodynamic TDDFT. This is essentially TDDFT with one orbital only, defined as :math:`{\psi(r,t)=\sqrt{\rho(r,t)}e^{iS(r,t)}}`, and determined by the following time-dependent Schroedinger equation,

.. math:: {-\frac{1}{2} \nabla^2 \psi(r,t) + v_s(r,t) \psi(r,t) = i\frac{d}{dt}\psi(r,t)}.

where :math:`{v_s = v_{xc} + v_H + v_{T_s} - v_{vW} + v_{dyn}}`, See `paper <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.111.175002>`_.


.. option:: type
    
    The type of propagator.
        *Options* : crank-nicolson, taylor, rk4 (experimental)

        *Default* : crank-nicolson

.. option:: int_t

    The time step in atomic units.
        *Options* :

        *Default* : 1.0e-3

.. option:: order

    The order used for the Taylor expansion propagator.
        *Options* :

        *Default* : 20

.. option:: linearsolver

    The linear solver used for the Crank-Nicolson expansion propagator (from `SciPy`).
        *Options* : bicg, bicgstab, cg, cgs, gmres, lgmres, minres, qmr

        *Default* : bicgstab

.. option:: tol

    The tolerance for the linear solver used for the Crank-Nicolson expansion propagator.
        *Options* :

        *Default* : 1e-10

.. option:: maxiter

    The max amount of iteration steps for the linear solver used for the Crank-Nicolson expansion propagator.
        *Options* :

        *Default* : 100


TD
--

    Control the TDDFT parameters that lie outside the propagator class.

.. option:: outfile

    The prefix of the output files.
        *Options* :

        *Default* : td_out

.. option:: tmax

    The total amount of time in atomic units.
        *Options* :

        *Default* : 1.0

.. option:: order

    The max amount of order of the prediction correction steps.
        *Options* :

        *Default* : 1

.. option:: direc

    The direction of the initial kick.
        *Options* : x, y, z

        *Default* : x

.. option:: strength

    The strength of the initial kick in atomic units.
        *Options* :

        *Default* : 1.0e-3


.. _pylibxc: https://tddft.org/programs/libxc/

