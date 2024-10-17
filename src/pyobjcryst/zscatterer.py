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

"""Python wrapping of Zscatterer.

See the online ObjCryst++ documentation (https://objcryst.readthedocs.io).

Changes from ObjCryst::ZAtom
- XMLOutput and Input are not wrapped.

Changes from ObjCryst++
- XMLOutput and Input are not wrapped.
"""

__all__ = ["ZScatterer", "ZAtom", "ZPolyhedron",
           "RegularPolyhedraType", "GlobalScatteringPower"]

from urllib.request import urlopen

from pyobjcryst._pyobjcryst import ZScatterer as ZScatterer_orig
from pyobjcryst._pyobjcryst import ZAtom
from pyobjcryst._pyobjcryst import ZPolyhedron
from pyobjcryst._pyobjcryst import RegularPolyhedraType
from pyobjcryst._pyobjcryst import GlobalScatteringPower


class ZScatterer(ZScatterer_orig):

    def ImportFenskeHallZMatrix(self, src, named=False):
        """
        Import atoms from a Fenske-Hall z-matrix
        :param src: either a python filed (opened in 'rb' mode), or
            a filename, or an url ("http://...") to a text file with the z-matrix
        :param named: if True, allows to read a named Z-matrix - the formatting
            is similar to a Fenske-Hall z-matrix but only relies on spaces between the
            different fields instead of a strict number of characters.
        """
        if isinstance(src, str):
            if len(src) > 4:
                if src[:4].lower() == 'http':
                    return super().ImportFenskeHallZMatrix(urlopen(src), named)
            with open(src, 'rb') as fhz:  # Make sure file object is closed
                super().ImportFenskeHallZMatrix(fhz, named)
        else:
            super().ImportFenskeHallZMatrix(src, named)
