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

"""Python wrapping of Scatterer.h

See the online ObjCryst++ documentation (http://vincefn.net/ObjCryst/).

Changes from ObjCryst::Scatterer

- C++ methods that can return const or non-const objects return non-const
  objects in python.
- Operator string() is not exposed.
- Internal use only methods have not been exposed.
- InitRefParList is not exposed, as it is not used inside of Scatterer.
- GetClockScattCompList is exposed using a workaround, because it is not
  implemented in the library.
- Methods related to visualization are not exposed.
"""

from _pyobjcryst import Scatterer
