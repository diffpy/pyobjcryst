#!/usr/bin/env python
##############################################################################
#
# PyObjCryst        Complex Modeling Initiative
#                   Pavol Juhas
#                   (c) 2013 Brookhaven National Laboratory,
#                   Upton, New York.  All rights reserved.
#
# File coded by:    Pavol Juhas
#
# See AUTHORS.txt for a list of people who contributed.
# See LICENSE.txt for license information.
#
##############################################################################

"""Convenience module for executing all unit tests with

python -m pyobjcryst.tests.run
"""

if __name__ == '__main__':
    import sys
    from pyobjcryst.tests import test
    # produce zero exit code for a successful test
    sys.exit(not test().wasSuccessful())

# End of file
