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

"""Python wrapping of Crystal.h.

See the online ObjCryst++ documentation (http://vincefn.net/ObjCryst/).

Changes from ObjCryst::Crystal

- CIFOutput accepts a python file-like object
- CalcDynPopCorr is not enabled, as the API states that this is for internal
  use only.

Other Changes

- CreateCrystalFromCIF is placed here instead of in a seperate CIF module. This
  method accepts a python file rather than a CIF object.
"""

from _pyobjcryst import Crystal
from _pyobjcryst import BumpMergePar
from _pyobjcryst import CreateCrystalFromCIF
