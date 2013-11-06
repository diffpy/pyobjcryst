#!/usr/bin/env python

from setuptools import setup
from setuptools import Extension
import glob

# versioncfgfile holds version data for git commit hash and date.
# It must reside in the same directory as version.py.
versioncfgfile = 'pyobjcryst/version.cfg'

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


def gitinfo():
    from subprocess import Popen, PIPE
    proc = Popen(['git', 'describe'], stdout=PIPE)
    desc = proc.stdout.read()
    proc = Popen(['git', 'log', '-1', '--format=%H %ai'], stdout=PIPE)
    glog = proc.stdout.read()
    rv = {}
    rv['version'] = '-'.join(desc.strip().split('-')[:2])
    rv['commit'], rv['date'] = glog.strip().split(None, 1)
    return rv


def getversioncfg():
    import os
    from ConfigParser import SafeConfigParser
    cp = SafeConfigParser()
    cp.read(versioncfgfile)
    if not os.path.isdir('.git'):  return cp
    d = cp.defaults()
    g = gitinfo()
    if g['commit'] != d.get('commit'):
        cp.set('DEFAULT', 'version', g['version'])
        cp.set('DEFAULT', 'commit', g['commit'])
        cp.set('DEFAULT', 'date', g['date'])
        cp.write(open(versioncfgfile, 'w'))
    return cp

cp = getversioncfg()

# define distribution
dist =  setup(
        name = "pyobjcryst",
        version = cp.get('DEFAULT', 'version'),
        author = "Christopher L. Farrow",
        author_email = "clf2121@columbia.edu",
        description = "Bindings of ObjCryst++ into python",
        license = "BSD",
        url = "http://www.diffpy.org/",

        # What we're installing
        packages = ['pyobjcryst', 'pyobjcryst.tests'],
        test_suite = 'pyobjcryst.tests',
        include_package_data = True,
        ext_modules = [module],
        zip_safe = False,
)

# End of file
