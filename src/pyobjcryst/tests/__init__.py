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
        Select all tests when empty.  Ignore the pattern when
        any of unit test modules fails to import.

    Returns
    -------
    suite : `unittest.TestSuite`
        The TestSuite object containing the matching tests.
    '''
    import re
    from os.path import dirname
    from itertools import chain
    from pkg_resources import resource_filename
    loader = unittest.defaultTestLoader
    thisdir = resource_filename(__name__, '')
    depth = __name__.count('.') + 1
    topdir = thisdir
    for i in range(depth):
        topdir = dirname(topdir)
    suite_all = loader.discover(thisdir, top_level_dir=topdir)
    # always filter the suite by pattern to test-cover the selection code.
    suite = unittest.TestSuite()
    rx = re.compile(pattern)
    tsuites = list(chain.from_iterable(suite_all))
    tsok = all(isinstance(ts, unittest.TestSuite) for ts in tsuites)
    if not tsok:    # pragma: no cover
        return suite_all
    tcases = chain.from_iterable(tsuites)
    for tc in tcases:
        tcwords = tc.id().split('.')
        shortname = '.'.join(tcwords[-3:])
        if rx.search(shortname):
            suite.addTest(tc)
    # verify all tests are found for an empty pattern.
    assert pattern or suite_all.countTestCases() == suite.countTestCases()
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
