**Added:**

* Support for Python 3.15.

**Changed:**

* Boost headers are now taken from the PyPI ``libboost-headers`` package when building with pip, falling back to system/conda Boost headers (or ``-DBOOST_INCLUDE_DIR=<path>``).

**Deprecated:**

* <news item>

**Removed:**

* ``libboost-python`` and ``libobjcryst`` from the conda requirements, as they are no longer needed with the nanobind build.

**Fixed:**

* Update ObjCryst++ to vincefn/objcryst master (ba6ca89): fix TextureEllipsoid, invalidate March-Dollase cache on phase change, recalculate powder pattern on Radiation change, update Le-Bail extracted intensities when the number of reflections changes.

**Security:**

* <news item>
