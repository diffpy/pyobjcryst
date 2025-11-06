#!/usr/bin/env python

# Installation script for pyobjcryst
"""pyobjcryst - Python bindings to ObjCryst++ Object-Oriented Crystallographic
Library

Packages:   pyobjcryst
"""

import glob
import os
import sys
import sysconfig
from ctypes.util import find_library
from pathlib import Path

import numpy as np
from setuptools import Extension, setup

# Helper functions -----------------------------------------------------------


def get_boost_libraries():
    # the names we'll search for
    major, minor = sys.version_info.major, sys.version_info.minor
    candidates = [
        f"boost_python{major}{minor}",
        f"boost_python{major}",
        "boost_python",
    ]

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        libdir = os.path.join(conda_prefix, "lib")
        for name in candidates:
            so = f"lib{name}.so"
            if os.path.isfile(os.path.join(libdir, so)):
                # return the plain "boost_python311" etc (no "lib" prefix or ".so")
                return [name]

    # fallback to ldconfig
    for name in candidates:
        found = find_library(name)
        if found:
            # find_library may return "libboost_python3.so.1.74.0" etc
            # strip off lib*.so.* if you like, or just return name
            return [name]

    raise RuntimeError("Cannot find a suitable Boost.Python library.")


def get_env_config():
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        if os.name == "nt":
            inc = Path(conda_prefix) / "Library" / "include"
            lib = Path(conda_prefix) / "Library" / "lib"
        else:
            inc = Path(conda_prefix) / "include"
            lib = Path(conda_prefix) / "lib"
        return {"include_dirs": [str(inc)], "library_dirs": [str(lib)]}

    # no conda env: fallback to system/venv Python include/lib dirs
    py_inc = sysconfig.get_paths().get("include")
    libdir = sysconfig.get_config_var("LIBDIR") or "/usr/lib"
    return {"include_dirs": [p for p in [py_inc] if p], "library_dirs": [libdir]}


def create_extensions():
    include_dirs = get_env_config().get("include_dirs") + [np.get_include()]
    library_dirs = get_env_config().get("library_dirs")

    if os.name == "nt":
        objcryst_lib = "libObjCryst"
    else:
        objcryst_lib = "ObjCryst"

    libraries = [objcryst_lib] + get_boost_libraries()
    extra_objects = []
    extra_compile_args = []
    extra_link_args = []

    define_macros = []

    if os.name == "nt":
        extra_compile_args = [
            "-DBOOST_ERROR_CODE_HEADER_ONLY",
            "-DREAL=double",
        ]
    else:
        extra_compile_args = [
            "-std=c++11",
            "-DBOOST_ERROR_CODE_HEADER_ONLY",
            "-DREAL=double",
            "-fno-strict-aliasing",
        ]

    ext_kws = {
        "include_dirs": include_dirs,
        "libraries": libraries,
        "library_dirs": library_dirs,
        "define_macros": define_macros,
        "extra_compile_args": extra_compile_args,
        "extra_link_args": extra_link_args,
        "extra_objects": extra_objects,
    }
    ext = Extension(
        "pyobjcryst._pyobjcryst", glob.glob("src/extensions/*.cpp"), **ext_kws
    )
    return [ext]


def ext_modules():
    if set(sys.argv) & {"build_ext", "bdist_wheel", "install"}:
        return create_extensions()
    return []


if __name__ == "__main__":
    setup(ext_modules=ext_modules())
