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
"""Python wrapping of ScatteringPower.h.

See the online ObjCryst++ documentation (https://objcryst.readthedocs.io).

Changes from ObjCryst::ScatteringComponent
- Added attributes X, Y, Z, Occupancy to conform to MolAtom.

Changes from ObjCryst::ScatteringComponentList
- Wrapped as a to-python converter only (no constructor)
"""

__all__ = [
    "ScatteringPower",
    "ScatteringComponent",
    "ScatteringPowerAtom",
    "ScatteringComponentList",
    "gScatteringPowerRegistry",
]

from pyobjcryst._pyobjcryst import (
    ScatteringComponent,
    ScatteringComponentList,
    ScatteringPower,
    ScatteringPowerAtom,
    gScatteringPowerRegistry,
)
