#!/usr/bin/env python
##############################################################################
#
# PyObjCryst        by DANSE Diffraction group
#                   Simon J. L. Billinge
#                   (c) 2009 The Trustees of Columbia University
#                   in the City of New York.  All rights reserved.
#
# File coded by:    Chris Farrow
#
# See AUTHORS.txt for a list of people who contributed.
# See LICENSE.txt for license information.
#
##############################################################################

"""Definition of __version__, __date__, __gitsha__ for pyobjcryst.
"""

from pkg_resources import resource_stream
from ConfigParser import SafeConfigParser

# obtain version information from the version.cfg file
cp = SafeConfigParser()
cp.readfp(resource_stream(__name__, 'version.cfg'))

__version__ = cp.get('DEFAULT', 'version')
__date__ = cp.get('DEFAULT', 'date')
__gitsha__ = cp.get('DEFAULT', 'commit')

del cp

# End of file
