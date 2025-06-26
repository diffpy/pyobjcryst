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
"""Python wrapping of Polyhedron.h.

See the online ObjCryst++ documentation (
https://objcryst.readthedocs.io).
"""

__all__ = [
    "MakeTetrahedron",
    "MakeOctahedron",
    "MakeSquarePlane",
    "MakeCube",
    "MakeAntiPrismTetragonal",
    "MakePrismTrigonal",
    "MakeIcosahedron",
    "MakeTriangle",
]

from pyobjcryst._pyobjcryst import (
    MakeAntiPrismTetragonal,
    MakeCube,
    MakeIcosahedron,
    MakeOctahedron,
    MakePrismTrigonal,
    MakeSquarePlane,
    MakeTetrahedron,
    MakeTriangle,
)
