#!/usr/bin/env python
##############################################################################
#
# pyobjcryst        Complex Modeling Initiative
#                   (c) 2013 Brookhaven Science Associates,
#                   Brookhaven National Laboratory.
#                   All rights reserved.
#
# File coded by:    Pavol Juhas
#
# See AUTHORS.txt for a list of people who contributed.
# See LICENSE.txt for license information.
#
##############################################################################

"""Unit tests for pyobjcryst.
"""

import unittest


def testsuite(pattern=''):
    '''Create a unit tests suite for the pyobjcryst package.

    Parameters
    ----------
    pattern : str, optional
        Regular expression pattern for selecting test cases.
        Select all tests when empty.

    Returns
    -------
    suite : `unittest.TestSuite`
        The TestSuite object containing the matching tests.
    '''
    import re
    from itertools import chain
    from pkg_resources import resource_filename
    loader = unittest.defaultTestLoader
    thisdir = resource_filename(__name__, '')
    suite_all = loader.discover(thisdir)
    # shortcut when pattern is not specified
    if not pattern:
        return suite_all
    # here we need to filter the suite by pattern
    suite = unittest.TestSuite()
    rx = re.compile(pattern)
    tcases = chain.from_iterable(chain.from_iterable(suite_all))
    for tc in tcases:
        tcwords = tc.id().rsplit('.', 2)
        shortname = '.'.join(tcwords[-2:])
        if rx.search(shortname):
            suite.addTest(tc)
    return suite


def test():
    '''Execute all unit tests for the pyobjcryst package.

    Returns
    -------
    result : `unittest.TestResult`
    '''
    suite = testsuite()
    runner = unittest.TextTestRunner()
    result = runner.run(suite)
    return result


# End of file
