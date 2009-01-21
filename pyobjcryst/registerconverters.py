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

"""\
Empty module that will register the c++ to Python converters upon import.
This must be imported along with any other pyobjcryst module.
"""

__all__ = []

import sys
# Only import once or suffer RuntimeWarning-barf all over the screen
if not "registerconverters" in sys.modules:
    from _registerconverters import *
