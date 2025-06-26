#!/usr/bin/env python

# Installation script for pyobjcryst
"""pyobjcryst - Python bindings to ObjCryst++ Object-Oriented Crystallographic
Library

Packages:   pyobjcryst
"""

import glob
import os
import platform
import re
import sys
from os.path import join as pjoin

import numpy as np
from setuptools import Extension, setup

# Use this version when git data are not available as in a git zip archive.
# Update when tagging a new release.
FALLBACK_VERSION = "2024.2.1"

# define extension arguments here
ext_kws = {
    "libraries": ["ObjCryst"],
    "extra_compile_args": [
        "-std=c++11",
        "-DBOOST_ERROR_CODE_HEADER_ONLY",
        "-DREAL=double",
    ],
    "extra_link_args": [],
    "include_dirs": [np.get_include()],
    "library_dirs": [],
}
if platform.system() == "Windows":
    ext_kws["extra_compile_args"] = [
        "-DBOOST_ERROR_CODE_HEADER_ONLY",
        "-DREAL=double",
    ]
    if "CONDA_PREFIX" in os.environ:
        ext_kws["include_dirs"] += [
            pjoin(os.environ["CONDA_PREFIX"], "include"),
            pjoin(os.environ["CONDA_PREFIX"], "Library", "include"),
        ]
        ext_kws["library_dirs"] += [
            pjoin(os.environ["CONDA_PREFIX"], "Library", "lib"),
            pjoin(os.environ["CONDA_PREFIX"], "Library", "bin"),
            pjoin(os.environ["CONDA_PREFIX"], "libs"),
        ]
        ext_kws["libraries"] = ["libObjCryst"]
elif platform.system() == "Darwin":
    ext_kws["extra_compile_args"] += ["-fno-strict-aliasing"]

# determine if we run with Python 3.
PY3 = sys.version_info[0] == 3


# Figure out the tagged name of boost_python library.
def get_boost_libraries():
    """Check for installed boost_python shared library.

    Returns list of required boost_python shared libraries that are
    installed on the system. If required libraries are not found, an
    Exception will be thrown.
    """
    baselib = "boost_python"
    major, minor = (str(x) for x in sys.version_info[:2])
    pytags = [major + minor, major, ""]
    mttags = ["", "-mt"]
    boostlibtags = [(pt + mt) for mt in mttags for pt in pytags] + [""]
    from ctypes.util import find_library

    for tag in boostlibtags:
        lib = baselib + tag
        found = find_library(lib)
        if found:
            break

    # Show warning when library was not detected.
    if not found:
        import platform
        import warnings

        ldevname = "LIBRARY_PATH"
        if platform.system() == "Darwin":
            ldevname = "DYLD_FALLBACK_LIBRARY_PATH"
        wmsg = (
            "Cannot detect name suffix for the %r library.  "
            "Consider setting %s."
        ) % (baselib, ldevname)
        warnings.warn(wmsg)

    libs = [lib]
    return libs


def create_extensions():
    "Initialize Extension objects for the setup function."
    blibs = [n for n in get_boost_libraries() if not n in ext_kws["libraries"]]
    ext_kws["libraries"] += blibs
    ext = Extension(
        "pyobjcryst._pyobjcryst", glob.glob("src/extensions/*.cpp"), **ext_kws
    )
    return [ext]


# versioncfgfile holds version data for git commit hash and date.
# It must reside in the same directory as version.py.
MYDIR = os.path.dirname(os.path.abspath(__file__))
versioncfgfile = os.path.join(MYDIR, "src/pyobjcryst/version.cfg")
gitarchivecfgfile = os.path.join(MYDIR, ".gitarchive.cfg")


def gitinfo():
    from subprocess import PIPE, Popen

    kw = dict(stdout=PIPE, cwd=MYDIR, universal_newlines=True)
    proc = Popen(["git", "describe", "--match=v[[:digit:]]*", "--tags"], **kw)
    desc = proc.stdout.read()
    proc = Popen(["git", "log", "-1", "--format=%H %ct %ci"], **kw)
    glog = proc.stdout.read()
    rv = {}
    rv["version"] = ".post".join(desc.strip().split("-")[:2]).lstrip("v")
    rv["commit"], rv["timestamp"], rv["date"] = glog.strip().split(None, 2)
    return rv


def getversioncfg():
    if PY3:
        from configparser import RawConfigParser
    else:
        from ConfigParser import RawConfigParser
    vd0 = dict(version=FALLBACK_VERSION, commit="", date="", timestamp=0)
    # first fetch data from gitarchivecfgfile, ignore if it is unexpanded
    g = vd0.copy()
    cp0 = RawConfigParser(vd0)
    cp0.read(gitarchivecfgfile)
    if len(cp0.get("DEFAULT", "commit")) > 20:
        g = cp0.defaults()
        mx = re.search(r"\btag: v(\d[^,]*)", g.pop("refnames"))
        if mx:
            g["version"] = mx.group(1)
    # then try to obtain version data from git.
    gitdir = os.path.join(MYDIR, ".git")
    if os.path.exists(gitdir) or "GIT_DIR" in os.environ:
        try:
            g = gitinfo()
        except OSError:
            pass
    # finally, check and update the active version file
    cp = RawConfigParser()
    cp.read(versioncfgfile)
    d = cp.defaults()
    rewrite = not d or (
        g["commit"]
        and (
            g["version"] != d.get("version") or g["commit"] != d.get("commit")
        )
    )
    if rewrite:
        cp.set("DEFAULT", "version", g["version"])
        cp.set("DEFAULT", "commit", g["commit"])
        cp.set("DEFAULT", "date", g["date"])
        cp.set("DEFAULT", "timestamp", g["timestamp"])
        with open(versioncfgfile, "w") as fp:
            cp.write(fp)
    return cp


versiondata = getversioncfg()

with open(os.path.join(MYDIR, "README.rst")) as fp:
    long_description = fp.read()

# define distribution
setup_args = dict(
    name="pyobjcryst",
    version=versiondata.get("DEFAULT", "version"),
    author="Simon J.L. Billinge",
    author_email="sb2896@columbia.edu",
    maintainer="Vincent-Favre-Nicolin",
    maintainer_email="favre@esrf.fr",
    description="Python bindings to the ObjCryst++ library.",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    license="BSD-style license",
    url="https://github.com/diffpy/pyobjcryst",
    # Required python packages
    install_requires=["numpy", "packaging"],
    extras_require={
        "gui": [
            "ipywidgets",
            "jupyter",
            "matplotlib",
            "ipympl",
            "py3dmol>=2.0.1",
        ],
        "doc": [
            "sphinx",
            "m2r2",
            "sphinx_py3doc_enhanced_theme",
            "nbsphinx",
            "nbsphinx-link",
        ],
    },
    # What we're installing
    packages=["pyobjcryst", "pyobjcryst.tests"],
    package_dir={"": "src"},
    test_suite="pyobjcryst.tests",
    include_package_data=True,
    zip_safe=False,
    keywords="objcryst atom structure crystallography powder diffraction",
    classifiers=[
        # List of possible values at
        # http://pypi.python.org/pypi?:action=list_classifiers
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries",
    ],
)

if __name__ == "__main__":
    setup_args["ext_modules"] = create_extensions()
    setup(**setup_args)
