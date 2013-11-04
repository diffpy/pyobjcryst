#!/usr/bin/env python

from setuptools import setup, find_packages
from setuptools import Extension
import glob

# Include directories
from numpy.distutils.misc_util import get_numpy_include_dirs
include_dirs = get_numpy_include_dirs()


# Figure out which boost library to use. This doesn't appear to consult
# LD_LIBRARY_PATH.
def get_boost_libraries():
    """Check for installed boost_python shared library.

    Returns list of required boost_python shared libraries that are installed
    on the system. If required libraries are not found, an Exception will be
    thrown.
    """
    baselib = "boost_python"
    boostlibtags = ['', '-mt']
    from ctypes.util import find_library
    for tag in boostlibtags:
        lib = baselib + tag
        found = find_library(lib)
        if found: break

    # Raise Exception if we don't find anything
    if not found:
        raise Exception("Cannot find shared boost_library library")

    libs = [lib]
    return libs


# Define the extension
libraries = ["ObjCryst"]
libraries += get_boost_libraries()
module = Extension('pyobjcryst._pyobjcryst',
        glob.glob("extensions/*.cpp"),
        include_dirs = include_dirs,
        libraries = libraries
        )

# define distribution
dist =  setup(
        name = "pyobjcryst",
        version = "1.0b1",
        author = "Christopher L. Farrow",
        author_email = "clf2121@columbia.edu",
        description = "Bindings of ObjCryst++ into python",
        license = "BSD",
        url = "http://www.diffpy.org/",

        # What we're installing
        packages = ['pyobjcryst'],
        ext_modules = [module],
        zip_safe = False,
)

# End of file
