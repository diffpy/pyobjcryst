#!/usr/bin/env python

# Installation script for pyobjcryst
"""pyobjcryst - Python bindings to ObjCryst++ Object-Oriented Crystallographic
Library

Packages:   pyobjcryst
"""

import glob
import os
from pathlib import Path
import numpy as np

from setuptools import Extension, setup

# Helper functions -----------------------------------------------------------

def check_boost_libraries(lib_dir):
    pattern = "libboost_python*.*" if os.name != "nt" else "boost_python*.lib"
    found = list(lib_dir.glob(pattern))
    if not found:
        raise EnvironmentError(
            f"No boost_python libraries found in conda environment at {lib_dir}. "
            "Please install libboost_python in your conda environment."
        )

    # convert into linker names
    lib = []
    for libpath in found:
        name = libpath.stem
        if name.startswith("lib"):
            name = name[3:]
        lib.append(name)
    return lib

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

    libraries = ["ObjCryst"] + check_boost_libraries(Path(library_dirs[0]))
    extra_objects = []
    extra_compile_args = []
    extra_link_args = []

    define_macros = []

    if os.name == "nt":
        extra_compile_args = ['-DBOOST_ERROR_CODE_HEADER_ONLY', '-DREAL=double']
    else:
        extra_compile_args = ['-std=c++11', '-DBOOST_ERROR_CODE_HEADER_ONLY', '-DREAL=double', '-fno-strict-aliasing']

    ext_kws = {
        "include_dirs": include_dirs,
        "libraries": libraries,
        "library_dirs": library_dirs,
        "define_macros": define_macros,
        "extra_compile_args": extra_compile_args,
        "extra_link_args": extra_link_args,
        "extra_objects": extra_objects,
    }
    ext = Extension('pyobjcryst._pyobjcryst', glob.glob("src/extensions/*.cpp"), **ext_kws)
    return [ext]


setup_args = dict(
    ext_modules=[],
)

if __name__ == "__main__":
    setup_args["ext_modules"] = create_extensions()
    setup(**setup_args)
