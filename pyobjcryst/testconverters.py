#!/usr/bin/env python
"""Test the converters.

This verifies results from tests built into the _registerconverters module.
"""

if __name__ == "__main__":

    import _registerconverters
    import numpy

    tv = numpy.array(range(3), dtype=float)
    tm = numpy.array(range(6), dtype=float).reshape(3,2)

    # Check to see if the above arrays are equal to the test arrays
    assert( not (tv -_registerconverters.getTestVector()).any() )
    assert( not (tm -_registerconverters.getTestMatrix()).any() )

    print "Tests passed!"
