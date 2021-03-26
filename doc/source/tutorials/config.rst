
.. _config:

====================
Script mode of DFTpy
====================

DFTpy is a set of python modules. However, it can be executed in the old way by using the `dftpy` script which is generated at installation time. Here's a quick guide to the script's configuration dictionary, or `config`.


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
       - `CASIDA`_
       - `INVERSION`_
       -

.. warning::
    `PP`_ is a mandatory input (i.e., no default is avaliable for it).

.. note::
    Defaults work well for most arguments.

    When *Options* is empty, it can accept any value.

.. _pylibxc: https://tddft.org/programs/libxc/

JOB
-----------------

	Control of the running job.


.. list-table::

		* - :ref:`task<JOB-task>`
		  - :ref:`calctype<JOB-calctype>`


.. _JOB-task:

**task**
	The task to be performed.

		*Options* : Optdensity, Calculation, Propagate, Casida, Diagonize, Inversion

		*Default* : Optdensity


.. _JOB-calctype:

**calctype**
	The property to be calculated.

		*Options* : Energy, Potential, Both, Force, Stress

		*Default* : Energy


PATH
-----------------

	Specify the path of needed files.


.. list-table::

		* - :ref:`pppath<PATH-pppath>`
		  - :ref:`cellpath<PATH-cellpath>`


.. _PATH-pppath:

**pppath**
	The path of pseudopotential.

		*Options* : 

		*Default* : ./


.. _PATH-cellpath:

**cellpath**
	The path of input structure.

		*Options* : 

		*Default* : ./


MATH
-----------------

	Some methods and techniques that make DFTpy really fast.


.. list-table::

		* - :ref:`linearii<MATH-linearii>`
		  - :ref:`linearie<MATH-linearie>`
		  - :ref:`twostep<MATH-twostep>`
		  - :ref:`multistep<MATH-multistep>`
		* - :ref:`reuse<MATH-reuse>`
		  -
		  -
		  -


.. _MATH-linearii:

**linearii**
	Linear-scaling method to deal with Ion-Ion interactions (PME).

		*Options* : True, False

		*Default* : True


.. _MATH-linearie:

**linearie**
	Linear-scaling method to deal with Ion-Electron interactions (PME).

		*Options* : True, False

		*Default* : True


.. _MATH-twostep:

**twostep**
	A two-step method for performing density optimizations. '`True`' is equivalent to :ref:`multistep<MATH-multistep>` = 2.

		*Options* : True, False

		*Default* : False


.. _MATH-multistep:

**multistep**
	A multi-step method for performing density optimizations.

		*Options* : 1,2,...

		*Default* : 1


.. _MATH-reuse:

**reuse**
	Except in the first step, the initial density is given by the optimized density of the previous step.

		*Options* : True, False

		*Default* : True


PP
-----------------

	Control of the running job.

	*e.g.* : 

		Al = Al_lda.oe01.recpot



CELL
-----------------

	Information about the input structure.


.. list-table::

		* - :ref:`cellfile<CELL-cellfile>`
		  - :ref:`elename<CELL-elename>`
		  - :ref:`zval<CELL-zval>`
		  - :ref:`format<CELL-format>`


.. _CELL-cellfile:

**cellfile**
	The file of input structure. 

		*Options* : 

		*Default* : POSCAR


.. _CELL-elename:

**elename**
	The name of atom.

		*Options* : 

		*Default* : Al


.. _CELL-zval:

**zval**
	The charge of atomic species.

		*Options* : 

		*Default* : None


.. _CELL-format:

**format**
	The format of structure file.

		*Options* : pp, vasp, xsf, snpy, ...

		*Default* : None

.. note::
 Only `snpy` format support parallel read and write

GRID
-----------------

	Control the grid.


.. list-table::

		* - :ref:`ecut<GRID-ecut>`
		  - :ref:`spacing<GRID-spacing>`
		  - :ref:`gfull<GRID-gfull>`
		  - :ref:`nr<GRID-nr>`
		* - :ref:`maxprime<GRID-maxprime>`
		  - :ref:`scale<GRID-scale>`
		  - :ref:`cplx<GRID-cplx>`
		  -


.. _GRID-ecut:

**ecut**
	The kinetic energy cutoff (eV).

		*Options* : 

		*Default* : 600


.. _GRID-spacing:

**spacing**
	The spacing (or gap) separating nearest real space grid points. If set this, :ref:`ecut<Grid-ecut>` is disabled.

		*Options* : 

		*Default* : None


.. _GRID-gfull:

**gfull**
	Determines oif the number of grid points in the reciprocal and real space grids are equal. If '`False`' only use half grid, which will be faster.

		*Options* : True, False

		*Default* : False

.. note::
 `gfull=False`' implies that the the number of points of reciprocal space is only half of real space.

.. _GRID-nr:

**nr**
	The number of grid points in the direction of the three lattice vectors.

		*Options* : 

		*Default* : None

		*e.g.* : 

			nr = 32 32 32

.. _GRID-maxprime:

**maxprime**
	The max prime of guess best number of grid points for FFT

		*Options* : 3, 5, 7, 11, 13, 17,..., 97

		*Default* : 13


.. _GRID-scale:

**scale**
	The minimum scale for guess the best number of grid points

		*Options* : 

		*Default* : 0.99


.. _GRID-cplx:

**cplx**
	The type of real space value

		*Options* : True, False

		*Default* : False


DENSITY
-----------------

	Control the charge density.


.. list-table::

		* - :ref:`nspin<DENSITY-nspin>`
		  - :ref:`magmom<DENSITY-magmom>`
		  - :ref:`densityini<DENSITY-densityini>`
		  - :ref:`densityfile<DENSITY-densityfile>`
		* - :ref:`densityoutput<DENSITY-densityoutput>`
		  -
		  -
		  -


.. _DENSITY-nspin:

**nspin**
	non/spin-polarized calculation

		*Options* : 1, 2

		*Default* : 1


.. _DENSITY-magmom:

**magmom**
	Total electronic magnetization.

		*Options* : 

		*Default* : 0


.. _DENSITY-densityini:

**densityini**
	The initial density is given by homogeneous electron gas (HEG) or read from :ref:`densityfile<DENSITY-densityfile>`. If set `Read`, must given the :ref:`densityfile<DENSITY-densityfile>`.

		*Options* : HEG, Read

		*Default* : HEG


.. _DENSITY-densityfile:

**densityfile**
	The charge density for initial density, only works when if :ref:`densityini<DENSITY-densityini>` set `Read`.

		*Options* : 

		*Default* : None


.. _DENSITY-densityoutput:

**densityoutput**
	The output file of final density. The default is not output the density.

		*Options* : 

		*Default* : None


EXC
-----------------

	Control the exchange-correlation functional.


.. list-table::

		* - :ref:`xc<EXC-xc>`
		  - :ref:`x_str<EXC-x_str>`
		  - :ref:`c_str<EXC-c_str>`


.. _EXC-xc:

**xc**
	The kind of exchange-correlation functional. If not `LDA`, must have pylibxc_ installed.

		*Options* : LDA, PBE

		*Default* : LDA


.. _EXC-x_str:

**x_str**
	The type of exchange functional.

		*Options* : 

		*Default* : lda_x


.. _EXC-c_str:

**c_str**
	The type of correlation functional.

		*Options* : 

		*Default* : lda_c_pz


KEDF
-----------------

	Control the kinetic energy density functional (KEDF). DFTpy features most KEDFs, from GGAs to nonlocal to nonlocal with density dependent kernel.


.. list-table::

		* - :ref:`kedf<KEDF-kedf>`
		  - :ref:`x<KEDF-x>`
		  - :ref:`y<KEDF-y>`
		  - :ref:`alpha<KEDF-alpha>`
		* - :ref:`beta<KEDF-beta>`
		  - :ref:`sigma<KEDF-sigma>`
		  - :ref:`nsp<KEDF-nsp>`
		  - :ref:`interp<KEDF-interp>`
		* - :ref:`kerneltype<KEDF-kerneltype>`
		  - :ref:`symmetrization<KEDF-symmetrization>`
		  - :ref:`lumpfactor<KEDF-lumpfactor>`
		  - :ref:`neta<KEDF-neta>`
		* - :ref:`etamax<KEDF-etamax>`
		  - :ref:`order<KEDF-order>`
		  - :ref:`ratio<KEDF-ratio>`
		  - :ref:`maxpoints<KEDF-maxpoints>`
		* - :ref:`delta<KEDF-delta>`
		  - :ref:`kdd<KEDF-kdd>`
		  - :ref:`rho0<KEDF-rho0>`
		  - :ref:`k_str<KEDF-k_str>`
		* - :ref:`params<KEDF-params>`
		  - :ref:`kfmin<KEDF-kfmin>`
		  - :ref:`kfmax<KEDF-kfmax>`
		  - :ref:`rhomax<KEDF-rhomax>`
		* - :ref:`ldw<KEDF-ldw>`
		  -
		  -
		  -


.. _KEDF-kedf:

**kedf**
	The type of KEDF.

		*Options* : TF, vW, x_TF_y_vW, WT, SM, FP, MGP, MGPA, MGPG, LMGP, LMGPA, LMGPG

		*Default* : WT


.. _KEDF-x:

**x**
	The ratio of TF KEDF.

		*Options* : 

		*Default* : 1


.. _KEDF-y:

**y**
	The ratio of vW KEDF.

		*Options* : 

		*Default* : 1


.. _KEDF-alpha:

**alpha**
	The alpha parameter typical in  nonlocal KEDF :math:`\rho^{\alpha}`.

		*Options* : 

		*Default* : 0.8333333333333333


.. _KEDF-beta:

**beta**
	The beta parameter typical in  nonlocal KEDF :math:`\rho^{\beta}`.

		*Options* : 

		*Default* : 0.8333333333333333


.. _KEDF-sigma:

**sigma**
	A parameter used to smooth with a Gaussian convolution FFTs of problematic functions (e.g., invfft of :math:`{G^2\rho(G)}` ).

		*Options* : 

		*Default* : None


.. _KEDF-nsp:

**nsp**
	The number of :math:`{k_{f}}` points for splining `LWT` like nonlocal KEDFs. There are three options to achieve the same goal, the priority is :ref:`nsp<KEDF-nsp>` -> :ref:`delta<KEDF-delta>` -> :ref:`ratio<KEDF-ratio>`. Default is using :ref:`ratio<KEDF-ratio>`.

		*Options* : 

		*Default* : None


.. _KEDF-interp:

**interp**
	The interpolation method for `LWT` KEDF's kernel from the kernel table.

		*Options* : 

		*Default* : hermite


.. _KEDF-kerneltype:

**kerneltype**
	The kernel for `LWT` KEDF.

		*Options* : linear, newton, hermite

		*Default* : WT


.. _KEDF-symmetrization:

**symmetrization**
	The symmetrization way for `MGP` KEDF. See `paper <https://aip.scitation.org/doi/abs/10.1063/1.5023926>`_.

		*Options* : None, Arithmetic, Geometric

		*Default* : None


.. _KEDF-lumpfactor:

**lumpfactor**
	The kinetic electron for `LWT` KEDF.

		*Options* : 

		*Default* : None


.. _KEDF-neta:

**neta**
	The max number of discrete :math:`\eta` for `LWT` KEDF.

		*Options* : 

		*Default* : 50000


.. _KEDF-etamax:

**etamax**
	The max value of \eta for kernel in `LWT` KEDF.

		*Options* : 

		*Default* : 50


.. _KEDF-order:

**order**
	The order for the interpolation of the kernel in `LWT` KEDF. '0' means using the value of nearest-neighbor point is used.

		*Options* : 1, 2, 3, 4, 5

		*Default* : 3


.. _KEDF-ratio:

**ratio**
	The ratio of :math:`{k_{f}}` for spline in `LWT` KEDF. There are three options to do same thing, the priority is :ref:`nsp<KEDF-nsp>` -> :ref:`delta<KEDF-delta>` -> :ref:`ratio<KEDF-ratio>`. Default is using :ref:`ratio<KEDF-ratio>`.

		*Options* : 

		*Default* : 1.2


.. _KEDF-maxpoints:

**maxpoints**
	The max number of integration points for the evaluation of the `MGP` kernel.

		*Options* : 

		*Default* : 1000


.. _KEDF-delta:

**delta**
	The gap of spline

		*Options* : 

		*Default* : None


.. _KEDF-kdd:

**kdd**
	The kernel density denpendent for `LWT` KEDF: 
		+ 1 : The origin `LWT` KEDF. 
		+ 2 : Considers the :math:`\rho^{\beta}(r')\omega(\rho(r),r-r')` term in the potential.
		+ 3 : Also considers the derivative of kernel which is neglected in LWT. See `paper <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.100.041105>`_.  

		*Options* : 1, 2, 3

		*Default* : 3


.. _KEDF-rho0:

**rho0**
	The 'average' density used for the definition of the Fermi momentum. Default is None, which means it calculated based on the total charge and system volume.

		*Options* : 

		*Default* : None


.. _KEDF-k_str:

**k_str**
	Functional type for GGA/LIBXC_KEDF

		*Options* : 

		*Default* : gga_k_revapbe

.. warning::
 The functional type of GGA and LIBXC_KEDF are different.

.. _KEDF-params:

**params**
	Parameters for GGA KEDF functionals

		*Options* : 

		*Default* : None


.. _KEDF-kfmin:

**kfmin**
	Lower limit of kf

		*Options* : 

		*Default* : None


.. _KEDF-kfmax:

**kfmax**
	Upper limit of kf

		*Options* : 

		*Default* : None


.. _KEDF-rhomax:

**rhomax**
	Maximum/cutoff density

		*Options* : 

		*Default* : None


.. _KEDF-ldw:

**ldw**
	local density weight

		*Options* : 

		*Default* : None


OUTPUT
-----------------

	Control the output.


.. list-table::

		* - :ref:`time<OUTPUT-time>`
		  - :ref:`stress<OUTPUT-stress>`


.. _OUTPUT-time:

**time**
	Output the time information of all parts.

		*Options* : True, False

		*Default* : True


.. _OUTPUT-stress:

**stress**
	Output the stress information of all terms.

		*Options* : True, False

		*Default* : True


OPT
-----------------

	Control the charge density optimization.


.. list-table::

		* - :ref:`method<OPT-method>`
		  - :ref:`algorithm<OPT-algorithm>`
		  - :ref:`vector<OPT-vector>`
		  - :ref:`c1<OPT-c1>`
		* - :ref:`c2<OPT-c2>`
		  - :ref:`maxls<OPT-maxls>`
		  - :ref:`econv<OPT-econv>`
		  - :ref:`maxfun<OPT-maxfun>`
		* - :ref:`maxiter<OPT-maxiter>`
		  - :ref:`xtol<OPT-xtol>`
		  - :ref:`h0<OPT-h0>`
		  -


.. _OPT-method:

**method**
	The density optimization method.

		*Options* : TN, LBFGS, CG-HS, CG-DY, CG-CD, CG-LS, CG-FR, CG-PR

		*Default* : CG-HS


.. _OPT-algorithm:

**algorithm**
	The direct minimization method : Energy (EMM) or Residual (RMM).

		*Options* : EMM, RMM

		*Default* : EMM


.. _OPT-vector:

**vector**
	The scheme to deal with search direction.

		*Options* : Orthogonalization, Scaling

		*Default* : Orthogonalization


.. _OPT-c1:

**c1**
	The wolfe parameters `c1`

		*Options* : 

		*Default* : 0.0001


.. _OPT-c2:

**c2**
	The wolfe parameters `c2`

		*Options* : 

		*Default* : 0.2


.. _OPT-maxls:

**maxls**
	The max steps for line search. 

		*Options* : 

		*Default* : 10


.. _OPT-econv:

**econv**
	The energy convergence for last three steps (a.u./atom).

		*Options* : 

		*Default* : 1e-06


.. _OPT-maxfun:

**maxfun**
	The max steps for function calls. For `TN` density optimization method its the max steps for searching direction.

		*Options* : 

		*Default* : 50


.. _OPT-maxiter:

**maxiter**
	The max steps for optimization

		*Options* : 

		*Default* : 100


.. _OPT-xtol:

**xtol**
	Relative tolerance for an acceptable step.

		*Options* : 

		*Default* : 1e-12


.. _OPT-h0:

**h0**
	The initial approximation for the inverse Hessian needed by `LBFGS`.

		*Options* : 

		*Default* : 1


PROPAGATOR
-----------------

	Control of the propagator. `DFTpy` has an implementation of hydrodynamic TDDFT. This is essentially TDDFT with one orbital only, defined as :math:`{\psi(r,t)=\sqrt{\rho(r,t)}e^{iS(r,t)}}`, and determined by the following time-dependent Schroedinger equation, 

		 :math:`{-\frac{1}{2} \nabla^2 \psi(r,t) + v_s(r,t) \psi(r,t) = i\frac{d}{dt}\psi(r,t)}`,

	where :math:`{v_s = v_{xc} + v_H + v_{T_s} - v_{vW} + v_{dyn}}`, See `paper <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.111.175002>`_.


.. list-table::

		* - :ref:`type<PROPAGATOR-type>`
		  - :ref:`order<PROPAGATOR-order>`
		  - :ref:`linearsolver<PROPAGATOR-linearsolver>`
		  - :ref:`tol<PROPAGATOR-tol>`
		* - :ref:`maxiter<PROPAGATOR-maxiter>`
		  - :ref:`atol<PROPAGATOR-atol>`
		  -
		  -


.. _PROPAGATOR-type:

**type**
	The type of propagator.

		*Options* : crank-nicolson, taylor, rk4 (experimental)

		*Default* : crank-nicolson


.. _PROPAGATOR-order:

**order**
	The order used for the Taylor expansion propagator.

		*Options* : 

		*Default* : 20


.. _PROPAGATOR-linearsolver:

**linearsolver**
	The linear solver used for the Crank-Nicolson propagator. The solvers with a name end with `-scipy` are from the `SciPy` package and should be used in serial calculations only. 

		*Options* : bicg, bicgstab, cg, bicg-scipy, bicgstab-scipy, cg-scipy, cgs-scipy, gmres-scipy, lgmres-scipy, minres-scipy, qmr-scipy

		*Default* : cg


.. _PROPAGATOR-tol:

**tol**
	The relative tolerance for the linear solver used for the Crank-Nicolson propagator.

		*Options* : 

		*Default* : 1e-10


.. _PROPAGATOR-maxiter:

**maxiter**
	The max amount of iteration steps for the linear solver used for the Crank-Nicolson propagator.

		*Options* : 

		*Default* : 100


.. _PROPAGATOR-atol:

**atol**
	The absolute tolerance for the linear solver used for the Crank-Nicolson propagator.

		*Options* : 

		*Default* : None


TD
-----------------

	Control the TDDFT parameters that lie outside the propagator class.


.. list-table::

		* - :ref:`outfile<TD-outfile>`
		  - :ref:`timestep<TD-timestep>`
		  - :ref:`tmax<TD-tmax>`
		  - :ref:`max_pc<TD-max_pc>`
		* - :ref:`tol_pc<TD-tol_pc>`
		  - :ref:`atol_pc<TD-atol_pc>`
		  - :ref:`direc<TD-direc>`
		  - :ref:`strength<TD-strength>`
		* - :ref:`dynamic_potential<TD-dynamic_potential>`
		  - :ref:`max_runtime<TD-max_runtime>`
		  - :ref:`restart<TD-restart>`
		  -


.. _TD-outfile:

**outfile**
	The prefix of the output files.

		*Options* : 

		*Default* : td_out


.. _TD-timestep:

**timestep**
	The time step in atomic units.

		*Options* : 

		*Default* : 0.001


.. _TD-tmax:

**tmax**
	The total amount of time in atomic units.

		*Options* : 

		*Default* : 1


.. _TD-max_pc:

**max_pc**
	The max amount of the predictor-corrector steps.

		*Options* : 

		*Default* : 1


.. _TD-tol_pc:

**tol_pc**
	The relative tolerance for the predictor-corrector.

		*Options* : 

		*Default* : 1e-08


.. _TD-atol_pc:

**atol_pc**
	The absolute tolerance for the predictor-corrector.

		*Options* : 

		*Default* : 1e-10


.. _TD-direc:

**direc**
	The direction of the initial kick.

		*Options* : 

		*Default* : 0


.. _TD-strength:

**strength**
	The strength of the initial kick in atomic units.

		*Options* : 

		*Default* : 0.001


.. _TD-dynamic_potential:

**dynamic_potential**
	Include dynamic potential. (See Eq. (15) of `paper <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.98.144302>`_.

		*Options* : True, False

		*Default* : False


.. _TD-max_runtime:

**max_runtime**
	Max amount of running time in seconds before the program saves the intermediate result and quitting.

		*Options* : 

		*Default* : 0


.. _TD-restart:

**restart**
	Restart the propagation from a previously saved intermediate result.

		*Options* : True, False

		*Default* : False


CASIDA
-----------------

	Control of the CASIDA.


.. list-table::

		* - :ref:`numeig<CASIDA-numeig>`
		  - :ref:`diagonize<CASIDA-diagonize>`
		  - :ref:`tda<CASIDA-tda>`


.. _CASIDA-numeig:

**numeig**
	Number of eigenstates used in constructing casida matrix.

		*Options* : 

		*Default* : None


.. _CASIDA-diagonize:

**diagonize**
	If true, diagonize the Hamiltonian before construct the Casida matrix. If false, read the eigenstates from a saved file.

		*Options* : True, False

		*Default* : True


.. _CASIDA-tda:

**tda**
	Use Tamm-Dancoff approximation.

		*Options* : True, False

		*Default* : False


INVERSION
-----------------

	Control of the INVERSION.


.. list-table::

		* - :ref:`rho_in<INVERSION-rho_in>`
		  - :ref:`v_out<INVERSION-v_out>`


.. _INVERSION-rho_in:

**rho_in**
	Input file for the density.

		*Options* : 

		*Default* : None


.. _INVERSION-v_out:

**v_out**
	Output file for the potential.

		*Options* : 

		*Default* : None

