#!/usr/bin/env python
##############################################################################
#
# PyObjCryst        by DANSE Diffraction group
#                   Simon J. L. Billinge
#                   (c) 2009 Trustees of the Columbia University
#                   in the City of New York.  All rights reserved.
#
# File coded by:    Chris Farrow
#
# See AUTHORS.txt for a list of people who contributed.
# See LICENSE.txt for license information.
#
##############################################################################

"""Python wrapping of ScatteringPower.h

See the online ObjCryst++ documentation (http://vincefn.net/ObjCryst/).

Changes from ObjCryst::ScatteringComponent
- Added attributes X, Y, Z, Occupancy to conform to MolAtom.

Changes from ObjCryst::ScatteringComponentList
- Wrapped as a to-python converter only (no constructor)

"""

from _pyobjcryst import ScatteringPower
from _pyobjcryst import ScatteringComponent
from _pyobjcryst import ScatteringPowerAtom
from _pyobjcryst import ScatteringComponentList
