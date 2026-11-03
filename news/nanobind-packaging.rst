**Added:**

* Support for Python 3.15.

**Changed:**

* Boost headers are now taken from the PyPI ``libboost-headers`` package when building with pip, falling back to system/conda Boost headers (or ``-DBOOST_INCLUDE_DIR=<path>``).
* Set package version to ``2026.2.0rc0``.

**Deprecated:**

* <news item>

**Removed:**

* ``libboost-python`` and ``libobjcryst`` from the conda requirements, as they are no longer needed with the nanobind build.

**Fixed:**

* <news item>

**Security:**

* <news item>
