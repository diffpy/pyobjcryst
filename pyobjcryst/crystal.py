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

"""Python wrapping of Crystal.h.

See the online ObjCryst++ documentation (http://vincefn.net/ObjCryst/).

Classes:
    Crystal
    BumpMergePar

"""

from _pyobjcryst import Crystal
from _pyobjcryst import BumpMergePar
from _pyobjcryst import CreateCrystalFromCIF

__id__ = "$Id$"
