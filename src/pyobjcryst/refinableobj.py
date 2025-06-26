#!/usr/bin/env python
##############################################################################
#
# pyobjcryst        by DANSE Diffraction group
#                   Simon J. L. Billinge
#                   (c) 2009 The Trustees of Columbia University
#                   in the City of New York.  All rights reserved.
#
# File coded by:    Chris Farrow
#
# See AUTHORS.txt for a list of people who contributed.
# See LICENSE_DANSE.txt for license information.
#
##############################################################################
"""Python wrapping of RefinableObj.h.

See the online ObjCryst++ documentation (https://objcryst.readthedocs.io).

Changes from ObjCryst::RefinableObj

- XMLOutput and XMLInput accept python file-like objects.
- GetPar that takes a const double* is not exposed, as it is designed for
  internal use.
- GetParamSet returns a copy of the internal data so that no indirect
  manipulation can take place from python.
- SetDeleteRefParInDestructor(false) is called in the constructors of the
  python class and the parameter accessors.
- SetDeleteRefParInDestructor is not exposed.
- RemovePar is overloaded to return None.

Changes from ObjCryst::RefinablePar

- The constructor has been changed to accept a double, rather than a pointer to
  a double.
- The copy and default constructors and Init are not wrapped in order to avoid
  memory corruption. Since boost cannot implicitly handle double* object, a
  wrapper class had to be created. However, this wrapper class cannot be used
  to convert RefinablePar objected created in c++.  Thus,
  ObjCryst::RefinablePar objects created in c++ are passed into python as
  instances of _RefinablePar, which is a python wrapper around
  ObjCryst::RefinablePar. The RefinablePar python class is a wrapper around
  the C++ class PyRefinablePar, which manages its own double*.  These python
  classes are interchangable once instantiated, so users should not notice.
- XML input/output are not exposed.

Changes from ObjCryst::RefinableObjClock

- operator= is wrapped as the SetEqual method
  a.SetEqual(b) -> a = b

Changes from ObjCryst::ObjRegistry

- DeleteAll not wrapped

Changes from ObjCryst::Restraint

- The default and copy constructors are not wrapped, nor is Init.
- GetType returns a non-const reference to the RefParType.  This should be a
  no-no, but RefParType has no mutating methods, so this should no lead to
  trouble.
- XML input/output are not exposed.
"""

__all__ = [
    "RefinableObjClock",
    "RefinableObj",
    "RefObjOpt",
    "RefinableObjRegistry",
    "RefParType",
    "RefParDerivStepModel",
    "RefinablePar",
    "Restraint",
    "ScattererRegistry",
    "ScatteringPowerRegistry",
    "ZAtomRegistry",
    "refpartype_crystal",
    "refpartype_objcryst",
    "refpartype_scatt",
    "refpartype_scatt_transl",
    "refpartype_scatt_transl_x",
    "refpartype_scatt_transl_y",
    "refpartype_scatt_transl_z",
    "refpartype_scatt_orient",
    "refpartype_scatt_conform",
    "refpartype_scatt_conform_bondlength",
    "refpartype_scatt_conform_bondangle",
    "refpartype_scatt_conform_dihedangle",
    "refpartype_scatt_conform_x",
    "refpartype_scatt_conform_y",
    "refpartype_scatt_conform_z",
    "refpartype_scatt_occup",
    "refpartype_scattdata",
    "refpartype_scattdata_background",
    "refpartype_scattdata_scale",
    "refpartype_scattdata_profile",
    "refpartype_scattdata_profile_type",
    "refpartype_scattdata_profile_width",
    "refpartype_scattdata_profile_asym",
    "refpartype_scattdata_corr",
    "refpartype_scattdata_corr_pos",
    "refpartype_scattdata_radiation",
    "refpartype_scattdata_radiation_wavelength",
    "refpartype_scattpow",
    "refpartype_scattpow_temperature",
    "refpartype_unitcell",
    "refpartype_unitcell_length",
    "refpartype_unitcell_angle",
]

from types import MethodType

from pyobjcryst._pyobjcryst import (
    RefinableObj,
    RefinableObjClock,
    RefinableObjRegistry,
    RefinablePar,
    RefObjOpt,
    RefParDerivStepModel,
    RefParType,
    Restraint,
    ScattererRegistry,
    ScatteringPowerRegistry,
    ZAtomRegistry,
    refpartype_crystal,
    refpartype_objcryst,
    refpartype_scatt,
    refpartype_scatt_conform,
    refpartype_scatt_conform_bondangle,
    refpartype_scatt_conform_bondlength,
    refpartype_scatt_conform_dihedangle,
    refpartype_scatt_conform_x,
    refpartype_scatt_conform_y,
    refpartype_scatt_conform_z,
    refpartype_scatt_occup,
    refpartype_scatt_orient,
    refpartype_scatt_transl,
    refpartype_scatt_transl_x,
    refpartype_scatt_transl_y,
    refpartype_scatt_transl_z,
    refpartype_scattdata,
    refpartype_scattdata_background,
    refpartype_scattdata_corr,
    refpartype_scattdata_corr_pos,
    refpartype_scattdata_profile,
    refpartype_scattdata_profile_asym,
    refpartype_scattdata_profile_type,
    refpartype_scattdata_profile_width,
    refpartype_scattdata_radiation,
    refpartype_scattdata_radiation_wavelength,
    refpartype_scattdata_scale,
    refpartype_scattpow,
    refpartype_scattpow_temperature,
    refpartype_unitcell,
    refpartype_unitcell_angle,
    refpartype_unitcell_length,
)


class ObjRegistryWrapper(RefinableObjRegistry):
    """Wrapper class with a GetObj() method which can correctly wrap C++
    objects with the python methods.

    This is only needed when the objects have been created from C++,
    e.g. when loading an XML file.
    """

    def GetObj(self, i):
        o = self._GetObj(i)
        if o.GetClassName() == "Crystal":
            from .crystal import wrap_boost_crystal

            wrap_boost_crystal(o)
        elif o.GetClassName() == "PowderPattern":
            from .powderpattern import wrap_boost_powderpattern

            wrap_boost_powderpattern(o)
        elif o.GetClassName() == "MonteCarloObj":
            from .globaloptim import wrap_boost_montecarlo

            wrap_boost_montecarlo(o)
        return o


def wrap_boost_refinableobjregistry(o):
    """This function is used to wrap a C++ Object by adding the python methods
    to it.

    :param c: the C++ created object to which the python function must
        be added.
    """
    # TODO: moving the original function is not very pretty. Is there a better way ?
    if "_GetObj" not in dir(o):
        o._GetObj = o.GetObj
        o.GetObj = MethodType(ObjRegistryWrapper.GetObj, o)
