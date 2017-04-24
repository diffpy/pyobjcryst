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

"""Python wrapping of Polyhedron.h

See the online ObjCryst++ documentation (http://vincefn.net/ObjCryst/).
"""

__all__ = ["MakeTetrahedron", "MakeOctahedron", "MakeSquarePlane",
           "MakeCube", "MakeAntiPrismTetragonal", "MakePrismTrigonal",
           "MakeIcosahedron", "MakeTriangle"]

from pyobjcryst._pyobjcryst import MakeTetrahedron
from pyobjcryst._pyobjcryst import MakeOctahedron
from pyobjcryst._pyobjcryst import MakeSquarePlane
from pyobjcryst._pyobjcryst import MakeCube
from pyobjcryst._pyobjcryst import MakeAntiPrismTetragonal
from pyobjcryst._pyobjcryst import MakePrismTrigonal
from pyobjcryst._pyobjcryst import MakeIcosahedron
from pyobjcryst._pyobjcryst import MakeTriangle
