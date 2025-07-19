#!/usr/bin/env python

# Installation script for pyobjcryst
"""pyobjcryst - Python bindings to ObjCryst++ Object-Oriented Crystallographic
Library

Packages:   pyobjcryst
"""

import glob
import os
import sys
from ctypes.util import find_library
from pathlib import Path

import numpy as np
from setuptools import Extension, setup

# Helper functions -----------------------------------------------------------


def get_boost_libraries():
    base_lib = "boost_python"
    major, minor = str(sys.version_info[0]), str(sys.version_info[1])
    tags = [f"{major}{minor}", major, ""]
    mttags = ["", "-mt"]
    candidates = [base_lib + tag for tag in tags for mt in mttags] + [base_lib]
    for lib in candidates:
        if find_library(lib):
            return [lib]
    raise RuntimeError("Cannot find a suitable Boost.Python library.")


def get_env_config():
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        raise EnvironmentError(
            "CONDA_PREFIX environment variable is not set. "
            "Please activate your conda environment before running setup.py."
        )
    if os.name == "nt":
        inc = Path(conda_prefix) / "Library" / "include"
        lib = Path(conda_prefix) / "Library" / "lib"
    else:
        inc = Path(conda_prefix) / "include"
        lib = Path(conda_prefix) / "lib"

    return {"include_dirs": [str(inc)], "library_dirs": [str(lib)]}


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


setup_args = dict(
    ext_modules=[],
)

if __name__ == "__main__":
    setup_args["ext_modules"] = create_extensions()
    setup(**setup_args)
