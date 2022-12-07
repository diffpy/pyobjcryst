/*****************************************************************************
*
* pyobjcryst        by DANSE Diffraction group
*                   Simon J. L. Billinge
*                   (c) 2009 The Trustees of Columbia University
*                   in the City of New York.  All rights reserved.
*
* File coded by:    Chris Farrow
*
* See AUTHORS.txt for a list of people who contributed.
* See LICENSE_DANSE.txt for license information.
*
******************************************************************************
*
* boost::python bindings to ObjCryst::UnitCell.
*
*****************************************************************************/

#include <boost/python/class.hpp>
#include <boost/python/def.hpp>
#include <boost/python/tuple.hpp>
#include <boost/python/copy_const_reference.hpp>

#include <string>

#include <ObjCryst/ObjCryst/UnitCell.h>
#include <ObjCryst/CrystVector/CrystVector.h>

#include "helpers.hpp"

using namespace boost::python;
using namespace ObjCryst;
namespace bp = boost::python;

namespace {

bp::tuple FractionalToOrthonormalCoords(const UnitCell& uc,
        double x, double y, double z)
{
    uc.FractionalToOrthonormalCoords(x,y,z);
    return bp::make_tuple(x,y,z);
}

bp::tuple OrthonormalToFractionalCoords(const UnitCell& uc,
        double x, double y, double z)
{
    uc.OrthonormalToFractionalCoords(x,y,z);
    return bp::make_tuple(x,y,z);
}

bp::tuple MillerToOrthonormalCoords(const UnitCell& uc,
        double x, double y, double z)
{
    uc.MillerToOrthonormalCoords(x,y,z);
    return bp::make_tuple(x,y,z);
}

bp::tuple OrthonormalToMillerCoords(const UnitCell& uc,
        double x, double y, double z)
{
    uc.OrthonormalToMillerCoords(x,y,z);
    return bp::make_tuple(x,y,z);
}

// Setter for the lattice parameters.

void _seta(UnitCell& u, double val)
{
    u.GetPar("a").SetValue(val);
}

double _geta(UnitCell& u)
{
    return u.GetLatticePar(0);
}

void _setb(UnitCell& u, double val)
{
    u.GetPar("b").SetValue(val);
}

double _getb(UnitCell& u)
{
    return u.GetLatticePar(1);
}

void _setc(UnitCell& u, double val)
{
    u.GetPar("c").SetValue(val);
}

double _getc(UnitCell& u)
{
    return u.GetLatticePar(2);
}

void _setalpha(UnitCell& u, double val)
{
    if((val<=0)||(val>=M_PI)) throw ObjCrystException("alpha must be within ]0;pi[");
    RefinablePar &p = u.GetPar("alpha");
    if(p.IsUsed()) p.SetValue(val);
    // Throwing an exception here would be risky - a warning would be more adequate
    // else throw ObjCrystException("alpha is fixed and cannot be changed");
}

double _getalpha(UnitCell& u)
{
    return u.GetLatticePar(3);
}

void _setbeta(UnitCell& u, double val)
{
    if((val<=0)||(val>=M_PI)) throw ObjCrystException("beta must be within ]0;pi[");
    RefinablePar &p = u.GetPar("beta");
    if(p.IsUsed()) p.SetValue(val);
    // Throwing an exception here would be risky - a warning would be more adequate
    // else throw ObjCrystException("beta is fixed and cannot be changed");
}

double _getbeta(UnitCell& u)
{
    return u.GetLatticePar(4);
}

void _setgamma(UnitCell& u, double val)
{
    if((val<=0)||(val>=M_PI)) throw ObjCrystException("gamma must be within ]0;pi[");
    RefinablePar &p = u.GetPar("gamma");
    if(p.IsUsed()) p.SetValue(val);
    // Throwing an exception here would be risky - a warning would be more adequate
    // else throw ObjCrystException("gamma is fixed and cannot be changed");
}

double _getgamma(UnitCell& u)
{
    return u.GetLatticePar(5);
}

void SafeChangeSpaceGroup(UnitCell& u, const std::string& sgid)
{
    MuteObjCrystUserInfo muzzle;
    // this may throw invalid_argument which is translated to ValueError
    u.ChangeSpaceGroup(sgid);
}


}

void wrap_unitcell()
{
    scope().attr("refpartype_unitcell") = object(ptr(gpRefParTypeUnitCell));
    scope().attr("refpartype_unitcell_length") = object(ptr(gpRefParTypeUnitCellLength));
    scope().attr("refpartype_unitcell_angle") = object(ptr(gpRefParTypeUnitCellAngle));

    class_<UnitCell, bases<RefinableObj> >
        ("UnitCell")
        // Constructors
        .def(init<const double, const double, const double, const std::string&>())
        .def(init<const double, const double, const double,
            const double, const double, const double,
            const std::string&>())
        .def(init<const UnitCell&>())
        .def("GetLatticePar",
            (CrystVector<double> (UnitCell::*)() const) &UnitCell::GetLatticePar)
        .def("GetLatticePar",
            (double (UnitCell::*)(const int) const) &UnitCell::GetLatticePar)
        .def("GetClockLatticePar", &UnitCell::GetClockLatticePar,
                return_value_policy<copy_const_reference>())
        .def("GetBMatrix", &UnitCell::GetBMatrix,
                return_value_policy<copy_const_reference>())
        .def("GetOrthMatrix", &UnitCell::GetOrthMatrix,
                return_value_policy<copy_const_reference>())
        .def("GetClockMetricMatrix", &UnitCell::GetClockMetricMatrix,
                return_value_policy<copy_const_reference>())
        .def("GetOrthonormalCoords", &UnitCell::GetOrthonormalCoords)
        // Modified to return a tuple
        .def("OrthonormalToFractionalCoords",
                &OrthonormalToFractionalCoords)
        // Modified to return a tuple
        .def("FractionalToOrthonormalCoords",
                &FractionalToOrthonormalCoords)
        // Modified to return a tuple
        .def("MillerToOrthonormalCoords",
                &MillerToOrthonormalCoords)
        // Modified to return a tuple
        .def("OrthonormalToMillerCoords",
                &OrthonormalToMillerCoords)
        .def("GetSpaceGroup", (SpaceGroup& (UnitCell::*)()) &UnitCell::GetSpaceGroup,
                return_internal_reference<>())
        .def("ChangeSpaceGroup", &SafeChangeSpaceGroup)
        .def("GetVolume", &UnitCell::GetVolume)
        .def("__str__", &__str__<UnitCell>)
        // python-only
        .add_property("a", &_geta, &_seta)
        .add_property("b", &_getb, &_setb)
        .add_property("c", &_getc, &_setc)
        .add_property("alpha", &_getalpha, &_setalpha)
        .add_property("beta", &_getbeta, &_setbeta)
        .add_property("gamma", &_getgamma, &_setgamma)
        ;
}
