**Added:**

* <news item>

**Changed:**

* Build the PyPI wheels with ``cibuildwheel`` (manylinux, macOS x86_64/arm64 and Windows AMD64), configured in ``[tool.cibuildwheel]`` in ``pyproject.toml``.
* The package version is now derived from the git tag via ``setuptools-scm``.

**Deprecated:**

* <news item>

**Removed:**

* <news item>

**Fixed:**

* PyPI wheels are now self-contained and no longer need ``libObjCryst`` or ``libboost_python`` from a conda environment; previously ``import pyobjcryst`` failed in a plain virtual environment.
* Declare ``packaging`` as a runtime dependency (used by ``pyobjcryst.powderpattern``).

**Security:**

* <news item>
