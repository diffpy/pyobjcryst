#!/usr/bin/env python
##############################################################################
#
# globals
#
# File coded by:    Vincent Favre-Nicolin
#
# See AUTHORS.txt for a list of people who contributed.
# See LICENSE.txt for license information.
#
##############################################################################
"""Global objects are exposed here.

These are the main objects registries, which are tweaked to wrap pure
C++ objects with the python methods.
"""

__all__ = [
    "gCrystalRegistry",
    "gPowderPatternRegistry",
    "gRefinableObjRegistry",
    "gScattererRegistry",
    "gOptimizationObjRegistry",
    "gTopRefinableObjRegistry",
    "gDiffractionDataSingleCrystalRegistry",
]

from pyobjcryst._pyobjcryst import (
    gCrystalRegistry,
    gDiffractionDataSingleCrystalRegistry,
    gOptimizationObjRegistry,
    gPowderPatternRegistry,
    gRefinableObjRegistry,
    gScattererRegistry,
    gTopRefinableObjRegistry,
)

from .globaloptim import wrap_boost_optimizationobjregistry
from .refinableobj import wrap_boost_refinableobjregistry

# Wrap registries with python methods
wrap_boost_refinableobjregistry(gCrystalRegistry)
wrap_boost_refinableobjregistry(gPowderPatternRegistry)
wrap_boost_refinableobjregistry(gRefinableObjRegistry)
wrap_boost_refinableobjregistry(gTopRefinableObjRegistry)
wrap_boost_optimizationobjregistry(gOptimizationObjRegistry)
