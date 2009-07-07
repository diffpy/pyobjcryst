#!/usr/bin/env python
########################################################################
#
# pyobjcryst        by DANSE Diffraction group
#                   Simon J. L. Billinge
#                   (c) 2009 trustees of the Michigan State University
#                   All rights reserved.
#
# File coded by:    Chris Farrow
#
# See AUTHORS.txt for a list of people who contributed.
# See LICENSE.txt for license information.
#
########################################################################

"""Python wrapping of Molecule.h

See the online ObjCryst++ documentation (http://vincefn.net/ObjCryst/).

Classes:
    Molecule
    MolAtom
    MolBond
    MolBondAngle
    MolDihedralAngle
    Quaternion
    RigidGroup
    StretchMode
    StretchModeBondLength
    StretchModeBondAngle
    StretchModeTorsion
    StretchModeTwist

Methods:
    GetBondLength
    GetBondAngle
    GetDihedralAngle
    MakeTetrahedron
    MakeOctahedron
    MakeSquarePlane
    MakeCube
    MakeAntiPrismTetragonal
    MakePrismTrigonal
    MakeIcosahedron
    MakeTriangle

"""

# TODO - MolRing

from _pyobjcryst import Molecule
from _pyobjcryst import GetBondLength
from _pyobjcryst import GetBondAngle
from _pyobjcryst import GetDihedralAngle
from _pyobjcryst import MolAtom
from _pyobjcryst import MolBond
from _pyobjcryst import MolBondAngle
from _pyobjcryst import MolDihedralAngle
from _pyobjcryst import Quaternion
from _pyobjcryst import RigidGroup
from _pyobjcryst import StretchMode
from _pyobjcryst import StretchModeBondLength
from _pyobjcryst import StretchModeBondAngle
from _pyobjcryst import StretchModeTorsion
from _pyobjcryst import StretchModeTwist
from _pyobjcryst import MakeTetrahedron
from _pyobjcryst import MakeOctahedron
from _pyobjcryst import MakeSquarePlane
from _pyobjcryst import MakeCube
from _pyobjcryst import MakeAntiPrismTetragonal
from _pyobjcryst import MakePrismTrigonal
from _pyobjcryst import MakeIcosahedron
from _pyobjcryst import MakeTriangle
        
__id__ = "$Id$"
