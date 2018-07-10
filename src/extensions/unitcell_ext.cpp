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
    u.GetPar("alpha").SetValue(val);
}

double _getalpha(UnitCell& u)
{
    return u.GetLatticePar(3);
}

void _setbeta(UnitCell& u, double val)
{
    u.GetPar("beta").SetValue(val);
}

double _getbeta(UnitCell& u)
{
    return u.GetLatticePar(4);
}

void _setgamma(UnitCell& u, double val)
{
    u.GetPar("gamma").SetValue(val);
}

double _getgamma(UnitCell& u)
{
    return u.GetLatticePar(5);
}


}

void wrap_unitcell()
{

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
