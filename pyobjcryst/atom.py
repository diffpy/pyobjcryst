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

"""Python wrapping of Atom.h

See the online ObjCryst++ documentation (http://vincefn.net/ObjCryst/).

Changes from ObjCryst::Atom

- The default constructor has been disabled. When not followed-up by Init, it
  will cause segmentation faults, even if it is printed.
"""

from _pyobjcryst import Atom
