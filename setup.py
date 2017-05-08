#!/usr/bin/env python

# Installation script for pyobjcryst

"""pyobjcryst - Python bindings to ObjCryst++ Object-Oriented Crystallographic
Library

Packages:   pyobjcryst
"""

import os
import re
import sys
import glob
from setuptools import setup
from setuptools import Extension
from numpy.distutils.misc_util import get_numpy_include_dirs

# Use this version when git data are not available as in a git zip archive.
# Update when tagging a new release.
FALLBACK_VERSION = '2.0.2.post0'

# define extension arguments here
ext_kws = {
        'libraries' : ['ObjCryst'],
        'extra_compile_args' : [],
        'extra_link_args' : [],
        'include_dirs' : get_numpy_include_dirs(),
}

# determine if we run with Python 3.
PY3 = (sys.version_info[0] == 3)

# Figure out the tagged name of boost_python library.
def get_boost_libraries():
    """Check for installed boost_python shared library.

    Returns list of required boost_python shared libraries that are installed
    on the system. If required libraries are not found, an Exception will be
    thrown.
    """
    baselib = "boost_python3" if PY3 else "boost_python"
    boostlibtags = ['', '-mt'] + ['']
    from ctypes.util import find_library
    for tag in boostlibtags:
        lib = baselib + tag
        found = find_library(lib)
        if found: break

    # Show warning when library was not detected.
    if not found:
        import platform
        import warnings
        ldevname = 'LIBRARY_PATH'
        if platform.system() == 'Darwin':
            ldevname = 'DYLD_FALLBACK_LIBRARY_PATH'
        wmsg = ("Cannot detect name suffix for the %r library.  "
                "Consider setting %s.") % (baselib, ldevname)
        warnings.warn(wmsg)

    libs = [lib]
    return libs


def create_extensions():
    "Initialize Extension objects for the setup function."
    blibs = [n for n in get_boost_libraries()
            if not n in ext_kws['libraries']]
    ext_kws['libraries'] += blibs
    ext = Extension('pyobjcryst._pyobjcryst',
                    glob.glob("src/extensions/*.cpp"),
                    **ext_kws)
    return [ext]


# versioncfgfile holds version data for git commit hash and date.
# It must reside in the same directory as version.py.
MYDIR = os.path.dirname(os.path.abspath(__file__))
versioncfgfile = os.path.join(MYDIR, 'src/pyobjcryst/version.cfg')
gitarchivecfgfile = versioncfgfile.replace('version.cfg', 'gitarchive.cfg')


def gitinfo():
    from subprocess import Popen, PIPE
    kw = dict(stdout=PIPE, cwd=MYDIR, universal_newlines=True)
    proc = Popen(['git', 'describe', '--match=v[[:digit:]]*'], **kw)
    desc = proc.stdout.read()
    proc = Popen(['git', 'log', '-1', '--format=%H %at %ai'], **kw)
    glog = proc.stdout.read()
    rv = {}
    rv['version'] = '.post'.join(desc.strip().split('-')[:2]).lstrip('v')
    rv['commit'], rv['timestamp'], rv['date'] = glog.strip().split(None, 2)
    return rv


def getversioncfg():
    if PY3:
        from configparser import RawConfigParser
    else:
        from ConfigParser import RawConfigParser
    vd0 = dict(version=FALLBACK_VERSION, commit='', date='', timestamp=0)
    # first fetch data from gitarchivecfgfile, ignore if it is unexpanded
    g = vd0.copy()
    cp0 = RawConfigParser(vd0)
    cp0.read(gitarchivecfgfile)
    if '$Format:' not in cp0.get('DEFAULT', 'commit'):
        g = cp0.defaults()
        mx = re.search(r'\btag: v(\d[^,]*)', g.pop('refnames'))
        if mx:
            g['version'] = mx.group(1)
    # then try to obtain version data from git.
    gitdir = os.path.join(MYDIR, '.git')
    if os.path.exists(gitdir) or 'GIT_DIR' in os.environ:
        try:
            g = gitinfo()
        except OSError:
            pass
    # finally, check and update the active version file
    cp = RawConfigParser()
    cp.read(versioncfgfile)
    d = cp.defaults()
    rewrite = not d or (g['commit'] and (
        g['version'] != d.get('version') or g['commit'] != d.get('commit')))
    if rewrite:
        cp.set('DEFAULT', 'version', g['version'])
        cp.set('DEFAULT', 'commit', g['commit'])
        cp.set('DEFAULT', 'date', g['date'])
        cp.set('DEFAULT', 'timestamp', g['timestamp'])
        with open(versioncfgfile, 'w') as fp:
            cp.write(fp)
    return cp

versiondata = getversioncfg()

# define distribution
setup_args = dict(
    name = "pyobjcryst",
    version = versiondata.get('DEFAULT', 'version'),
    author = "Simon J.L. Billinge",
    author_email = "sb2896@columbia.edu",
    maintainer = 'Pavol Juhas',
    maintainer_email = 'pavol.juhas@gmail.com',
    description = "Python bindings to the ObjCryst++ library.",
    license = "BSD-style license",
    url = "https://github.com/diffpy/pyobjcryst",

    # What we're installing
    packages = ['pyobjcryst', 'pyobjcryst.tests'],
    package_dir = {'' : 'src'},
    test_suite = 'pyobjcryst.tests',
    include_package_data = True,
    zip_safe = False,

    keywords = "objcryst atom structure crystallography",
    classifiers = [
        # List of possible values at
        # http://pypi.python.org/pypi?:action=list_classifiers
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Programming Language :: C++',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Software Development :: Libraries',
    ],
)

if __name__ == '__main__':
    setup_args['ext_modules'] = create_extensions()
    setup(**setup_args)

# End of file
