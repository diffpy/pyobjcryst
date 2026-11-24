=============
Release notes
=============

.. current developments

2026.2.0
=====

**Added:**

* Add explicitly constructable reflection profiles
* Expose objcryst's anisotropic Pseudo-Voigt profile
* Add PowderPatternDiffraction.SetProfile
* Expose reflection profile ``ReflectionProfilePseudoVoigt`` and ``ReflectionProfilePseudoVoigtAnisotropic`` as constructible Python classes.
* Add ``IsAnisotropic()`` method to ``ReflectionProfile``.
* Expose the time-of-flight reflection profile ReflectionProfileDoubleExponentialPseudoVoigt.
* Add ``lsq.GetVarianceCovarianceMap()`` to get the variance-covariance matrix as a dict keyed by ``(param_name, param_name)`` tuples.
* Expose reflection profile ``ReflectionProfilePseudoVoigtTCH`` with TCH instead of explicit Eta mixing.
* Expose the March-Dollase preferred-orientation correction TextureMarchDollase (AddPhase, SetPhasePar, DeletePhase and getters) and PowderPatternDiffraction.GetTextureMarchDollase. SetPhasePar and DeletePhase rely on the upstream ObjCryst++ fix (vincefn/objcryst#105) which recalculates the powder pattern when the texture changes.
* Add SCons ``arch`` option to set architecture during compilation. By default, no arch is specified.
* Add Rwp threshold ``minRwpVar`` for ``lsq.Refine()`` and ``lsq.SafeRefine()``.
* Add method ``GetRwHistory`` on LSQ object to get history of Rw values through one run of ``lsq.Refine()``.
* Expose ``PowderPatternDiffraction.X2XCorrPhase()`` for per-phase flat-detector displacement correction.
* Support for Python 3.15.

**Changed:**

* PowderPatternDiffraction.SetProfile gives a ReflectionProfileDoubleExponentialPseudoVoigt the unit cell of the component's crystal.
* Removed scatterers, scattering powers and molecule sub-objects are now deleted instead of leaked, unless still referenced from Python.
* Build the PyPI wheels with ``cibuildwheel`` (manylinux, macOS x86_64/arm64 and Windows AMD64), configured in ``[tool.cibuildwheel]`` in ``pyproject.toml``.
* The package version is now derived from the git tag via ``setuptools-scm``.
* Make ``LSQ.PrepareRefParList()`` silent by default and add a ``verbose`` argument to control warnings about duplicate parameter names.
* Boost headers are now taken from the PyPI ``libboost-headers`` package when building with pip, falling back to system/conda Boost headers (or ``-DBOOST_INCLUDE_DIR=<path>``).

**Fixed:**

* ``LSQ.PrepareRefParList()`` is silent by default again in the nanobind
bindings, as documented. Pass ``verbose=True`` to show the warnings
about duplicate parameter names.
* Fix crashes (use-after-free, double free) and stale wrappers when ObjCryst++ deletes objects still referenced from Python (#121, #132).
* Return numpy arrays from ``UnitCell.GetLatticePar()``,
``UnitCell.GetOrthonormalCoords()``, ``Crystal.GetMinDistanceTable()``
and the ``ScatteringPower`` methods ``GetScatteringFactor()``,
``GetTemperatureFactor()``, ``GetResonantScattFactReal()`` and
``GetResonantScattFactImag()``, which raised a ``TypeError`` in the
nanobind bindings.
* PyPI wheels are now self-contained and no longer need ``libObjCryst`` or ``libboost_python`` from a conda environment; previously ``import pyobjcryst`` failed in a plain virtual environment.
* Declare ``packaging`` as a runtime dependency (used by ``pyobjcryst.powderpattern``).
* Fix memory leak in makeNdArray
* Fix ``EraseAllParamSet`` on compiled ``RefinableObj`` (from ``LSQ.GetCompiledRefinedObj()``).
* Fixed powder pattern plot(hkl=True) when used before the figure has been displayed at least once.
* Update ObjCryst++ to vincefn/objcryst master (ba6ca89): fix TextureEllipsoid, invalidate March-Dollase cache on phase change, recalculate powder pattern on Radiation change, update Le-Bail extracted intensities when the number of reflections changes.
* Preserve scattering power lifetimes in nanobind Atom construction,
copying, and initialization to prevent use-after-free crashes.

**Removed:**

* ``libboost-python`` and ``libobjcryst`` from the conda requirements, as they are no longer needed with the nanobind build.


2026.1.0
=====

**Added:**

* ``quick_index()`` now accepts a ``length_min`` parameter (default: 2.5 Å)
to control the minimum unit cell length explored during indexing.
* Exposed `ReflectionProfile` methods (`GetProfile`, `GetFullProfileWidth`, `XMLOutput`, `XMLInput`) via Python bindings. Added unit tests.
* The binding to `ReflectionProfile.GetProfile(x, xcenter, h, k, l)` accepts python sequences / `numpy` arrays for the `x` argument, thanks to the helper function `assignCrystVector`.
* Expose PowderPattern.GetPowderPatternObsSigma and PowderPattern.SetPowderPatternObsSigma from objcryst
* Add ``AGENTS.md`` documenting contribution guidelines for AI coding agents
(dependency chain, commit conventions, news-item policy, human-review
requirement).
* Add support for Python 3.14.
* Exposed ``Radiation.GetLinearPolarRate()``, ``Radiation.SetLinearPolarRate()``,
``Radiation.GetClockWavelength()``, and ``Radiation.GetClockRadiation()`` via
Python bindings. Closes `#38 <https://github.com/diffpy/pyobjcryst/issues/38>`_.
``Radiation.GetClockLinearPolarRate()`` is also bound and will be available once
`diffpy/libobjcryst` picks up `vincefn/objcryst PR #79
<https://github.com/vincefn/objcryst/pull/79>`_, which adds the dedicated
polarisation clock and fixes the polarisation-correction recalculation trigger.

**Changed:**

* ``quick_index()`` default minimum unit cell length lowered from 3 Å to 2.5 Å,
allowing correct indexing of compact structures such as α-Fe, B, or ZRNCl
(`issue #47 <https://github.com/diffpy/pyobjcryst/issues/47>`_).

**Deprecated:**


**Fixed:**

* Exposed the correctly spelled `ORTHORHOMBIC` while preserving `ORTHOROMBIC` as temporary backend compatible alias
* Building with `pip install .` now uses `sysconfig` to locate `ObjCryst++`` libraries outside conda environments.
* Missing definition of `ScatteringData` in `powderpatterndiffraction_ext.ccp`
* Fixed `PowderPattern.AddPowderPatternDiffraction()` so a no-reflections failure does not leave a failed diffraction component attached to the powder pattern.
* `quick_fit_profile(displ_transl=True)` now correctly enables refinement of `2ThetaDispl` and `2ThetaTransp` by unfixing both parameters.

**Removed:**

* Remove support for Python 3.11.

**Security:**



2025.1.0
=====

**Fixed:**

* Code brought up to scikit-package level 5 standards with automated release workflows

**Removed:**

* Removed fallback version handling in `src/pyobjcryst/version.py`.


Version 2024.2.2
=====

**Fixed:**

* Correct powder pattern plotting with a non-empty name

Version 2024.2.1
=====

**Changed:**

* PowderPattern:
  * Fix reusing a matplotlib figure when plotting
  * Add ``figure`` property

Version 2024.2
=====

**Changed:**

* **DiffractionDataSingleCrystal**: add ``SetHklIobs``, ``SetIobs``, ``SetSigma``, ``GetSigma``, ``GetChi2``, ``FitScaleFactorForRw`` and ``FitScaleFactorForR`` (`issue #42 <https://github.com/diffpy/pyobjcryst/issues/42>`_)
* Add a single crystal data notebook example
* Online documentation notebooks now include the plots `<https://pyobjcryst.readthedocs.io/en/latest/examples>`_

**Fixed:**

* From libobjcryst: update the ScatteringComponentList when a Scatterer is removed from a Crystal (`issue #41 <https://github.com/diffpy/pyobjcryst/issues/41>`_)

Version 2024.1
=====

**Changed:**

* Add python access to MolZAtom, for ``Molecule.AsZMatrix()``

Version 2.2.6
=====

**Changed:**

* Support for Windows and Python>=3.8
* Added a zoom limit for 3D crystal views

**Fixed:**

* Correct error preventing pyobjcryst import for Windows and Python>=3.8 (`issue #33 <https://github.com/diffpy/pyobjcryst/issues/33>`_)
* Fix for matplotlib >=3.7.0 when removing hkl labels

Version 2.2.5
=====

**Changed:**

* Raise an exception if ``alpha``, ``beta`` or ``gamma`` are not within ``]0;pi[`` when changing lattice angles
* Add ``UnitCell.ChangeSpaceGroup()``

**Fixed:**

* Avoid duplication of plots when using ipympl (aka ``%matplotlib widget``)
* Correct powder pattern tests to avoid warnings

**Deprecated:**

* ``loadCrystal`` – use ``create_crystal_from_cif()`` instead

Version 2.2.4
=====

**Changed:**

* The list of HKL reflections will now be automatically re-generated for a ``PowderPatternDiffraction`` when the Crystal's spacegroup changes, or the lattice parameters are modified by more than 0.5%

**Fixes:**

* Fixed the powder pattern indexing test

Version 2.2.3
=====

**Added:**

* Support for Windows install (works with Python 3.7, and with PyPy 3.8 and 3.9)
* Native support for Apple arm64 (M1, M2) processors
* Fourier maps calculation
* Add ``gDiffractionDataSingleCrystalRegistry`` to globals

Version 2.2.2
=====

**Changed:**

* Add correct wrapping for C++-instantiated objects available through global registries, e.g. when loading an XML file. The objects are decorated with the python functions when accessed through the global registries ``GetObj()``
* Moved global object registries to ``pyobjcryst.globals``
* Update documentation

**Fixed:**

* Fix access to ``PRISM_TETRAGONAL_DICAP``, ``PRISM_TRIGONAL``, ``ICOSAHEDRON`` and ``TRIANGLE_PLANE``
* Fix powder pattern plot issues (NaN and update of hkl text with recent matplotlib versions)

Version 2.2.1 -- 2021-11-28
=====

* Add quantitative phase analysis with ``PowderPattern.qpa()``, including an example notebook using the QPA Round-Robin data
* Correct import of ``urllib.request.urllopen()`` when loading CIF or z-matrix files from HTTP URLs
* Fix blank line javascript output when updating the Crystal 3D view
* Add ``RefinableObj.xml()`` to directly get the XMLOutput as a string
* Add example notebooks to the sphinx-generated html documentation
* Fix issue when using ``Crystal.XMLInput()`` for a non-empty structure. Existing scattering power will be reused when possible, and otherwise not deleted anymore (which could lead to crashes)

Version 2.2.0 -- 2021-06-08
=====

* Add access to ``Radiation`` class & functions to change RadiationType, wavelength in ``PowderPattern`` and ``ScatteringData`` (and hence ``DiffractionDataSingleCrystal``) classes
* Fix the custodian_ward when creating a ``PowderPatternDiffraction``: ``PowderPatternDiffraction`` must persist while ``PowderPattern`` exists, and Crystal must persist while ``PowderPatternDiffraction`` exists
* Add 3D Crystal viewer ``pyobjcryst.crystal.Crystal.widget_3d``

Version 2.1.0 -- 2019-03-11
=====

**Added:**

* Support for Python 3.7
* Validation of compiler options from ``python-config``
* Make scons scripts compatible with Python 3 and Python 2
* Support ``np.array`` arguments for ``SetPowderPatternX``, ``SetPowderPatternObs``
* Declare compatible version requirements for client Anaconda packages
* Facility for silencing spurious console output from libobjcryst

**Changed:**

* Build Anaconda package with Anaconda C++ compiler
* Update to libobjcryst 2017.2.x

**Deprecated:**

* Variable ``__gitsha__`` in the ``version`` module, renamed to ``__git_commit__``

**Removed:**

* Support for Python 3.4

**Fixed:**

* Ambiguous use of boost::python classes and functions
* Name suffix resolution of ``boost_python`` shared library
* ``SetPowderPatternX`` crash for zero-length argument
* Incorrectly doubled return value from ``GetInversionCenter``
