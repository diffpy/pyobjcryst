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


"""Definition of __version__, __date__, __gitsha__, libobjcryst_version_info.
"""

from pkg_resources import resource_filename

import sys
if sys.version_info[0] >= 3:
    from configparser import RawConfigParser
else:
    from ConfigParser import RawConfigParser
del sys

# obtain version information from the version.cfg file
cp = RawConfigParser(dict(version='', date='', commit='', timestamp=0))
if not cp.read(resource_filename(__name__, 'version.cfg')):
    from warnings import warn
    warn('Package metadata not found, execute "./setup.py egg_info".')

__version__ = cp.get('DEFAULT', 'version')
__date__ = cp.get('DEFAULT', 'date')
__gitsha__ = cp.get('DEFAULT', 'commit')
__timestamp__ = cp.getint('DEFAULT', 'timestamp')

del cp

# version information on the active libObjCryst library ----------------------

from collections import namedtuple
from pyobjcryst._pyobjcryst import _get_libobjcryst_version_info_dict

libobjcryst_version_info = namedtuple('libobjcryst_version_info',
        "major minor micro patch version_number version date git_sha")
vd = _get_libobjcryst_version_info_dict()
libobjcryst_version_info = libobjcryst_version_info(
        version = vd['version_str'],
        version_number = vd['version'],
        major = vd['major'],
        minor = vd['minor'],
        micro = vd['micro'],
        patch = vd['patch'],
        date = vd['date'],
        git_sha = vd['git_sha'])
del vd

# End of file
