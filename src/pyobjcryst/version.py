#!/usr/bin/env python
##############################################################################
#
# pyobjcryst        by DANSE Diffraction group
#                   Simon J. L. Billinge
#                   (c) 2009 The Trustees of Columbia University
#                   in the City of New York.  All rights reserved.
#
# File coded by:    Chris Farrow
#
# See AUTHORS.txt for a list of people who contributed.
# See LICENSE_DANSE.txt for license information.
#
##############################################################################

"""
Definition of __version__, __date__, __timestamp__, __git_commit__,
libobjcryst_version_info.

Notes
-----
Variable `__gitsha__` is deprecated as of version 2.1.
Use `__git_commit__` instead.
"""

__all__ = ['__date__', '__git_commit__', '__timestamp__', '__version__',
           'libobjcryst_version_info']

import os.path

from pkg_resources import resource_filename


# obtain version information from the version.cfg file
cp = dict(version='', date='', commit='', timestamp='0')
fcfg = resource_filename(__name__, 'version.cfg')
if not os.path.isfile(fcfg):    # pragma: no cover
    from warnings import warn
    warn('Package metadata not found, execute "./setup.py egg_info".')
    fcfg = os.devnull
with open(fcfg) as fp:
    kwords = [[w.strip() for w in line.split(' = ', 1)]
              for line in fp if line[:1].isalpha() and ' = ' in line]
assert all(w[0] in cp for w in kwords), "received unrecognized keyword"
cp.update(kwords)

__version__ = cp['version']
__date__ = cp['date']
__git_commit__ = cp['commit']
__timestamp__ = int(cp['timestamp'])

# TODO remove deprecated __gitsha__ in version 2.2.
__gitsha__ = __git_commit__

del cp, fcfg, fp, kwords

# version information on the active libObjCryst library ----------------------

from collections import namedtuple
from pyobjcryst._pyobjcryst import _get_libobjcryst_version_info_dict

libobjcryst_version_info = namedtuple('libobjcryst_version_info',
        "major minor micro patch version_number version date git_commit")
vd = _get_libobjcryst_version_info_dict()
libobjcryst_version_info = libobjcryst_version_info(
        version = vd['version_str'],
        version_number = vd['version'],
        major = vd['major'],
        minor = vd['minor'],
        micro = vd['micro'],
        patch = vd['patch'],
        date = vd['date'],
        git_commit = vd['git_commit'])
del vd
