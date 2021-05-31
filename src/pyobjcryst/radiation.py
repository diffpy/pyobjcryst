#!/usr/bin/env python
##############################################################################
#
# pyobjcryst
#
# File coded by:    Vincent Favre-Nicolin
#
# See AUTHORS.txt for a list of people who contributed.
# See LICENSE.txt for license information.
#
##############################################################################

"""Python wrapping of Radiation from ScatteringData.h

See the online ObjCryst++ documentation (http://vincefn.net/ObjCryst/).

Changes from ObjCryst::Radiation::
        In development !
"""

__all__ = ["Radiation", "RadiationType", "WavelengthType"]

from pyobjcryst._pyobjcryst import Radiation, RadiationType, WavelengthType

