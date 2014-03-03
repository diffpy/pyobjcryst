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

"""Python wrapping of Polyhedron.h

See the online ObjCryst++ documentation (http://vincefn.net/ObjCryst/).

"""

from _pyobjcryst import MakeTetrahedron
from _pyobjcryst import MakeOctahedron
from _pyobjcryst import MakeSquarePlane
from _pyobjcryst import MakeCube
from _pyobjcryst import MakeAntiPrismTetragonal
from _pyobjcryst import MakePrismTrigonal
from _pyobjcryst import MakeIcosahedron
from _pyobjcryst import MakeTriangle
