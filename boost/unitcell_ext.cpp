/*****************************************************************************
*
* pyobjcryst        by DANSE Diffraction group
*                   Simon J. L. Billinge
*                   (c) 2009 Trustees of the Columbia University
*                   in the City of New York.  All rights reserved.
*
* File coded by:    Chris Farrow
*
* See AUTHORS.txt for a list of people who contributed.
* See LICENSE.txt for license information.
*
******************************************************************************
*
* boost::python bindings to ObjCryst::UnitCell.
*
* $Id$
*
*****************************************************************************/

#include "ObjCryst/UnitCell.h"
#include "CrystVector/CrystVector.h"

#include <boost/utility.hpp>
#include <boost/python.hpp>
#include <boost/python/class.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>

#include <string>
#include <iostream>

#include "helpers.hpp"

using namespace boost::python;
using namespace ObjCryst;

namespace {

tuple FractionalToOrthonormalCoords(const UnitCell &uc, 
        float x, float y, float z)
{
    uc.FractionalToOrthonormalCoords(x,y,z);
    return make_tuple(x,y,z);
}

tuple OrthonormalToFractionalCoords(const UnitCell &uc, 
        float x, float y, float z)
{
    uc.OrthonormalToFractionalCoords(x,y,z);
    return make_tuple(x,y,z);
}

tuple MillerToOrthonormalCoords(const UnitCell &uc, 
        float x, float y, float z)
{
    uc.MillerToOrthonormalCoords(x,y,z);
    return make_tuple(x,y,z);
}

tuple OrthonormalToMillerCoords(const UnitCell &uc, 
        float x, float y, float z)
{
    uc.OrthonormalToMillerCoords(x,y,z);
    return make_tuple(x,y,z);
}

// Setter for the lattice parameters.

void _seta(UnitCell& u, float val)
{
    u.GetPar("a").SetValue(val);
}

float _geta(UnitCell& u)
{
    return u.GetLatticePar(0);
}

void _setb(UnitCell& u, float val)
{
    u.GetPar("b").SetValue(val);
}

float _getb(UnitCell& u)
{
    return u.GetLatticePar(1);
}

void _setc(UnitCell& u, float val)
{
    u.GetPar("c").SetValue(val);
}

float _getc(UnitCell& u)
{
    return u.GetLatticePar(2);
}

void _setalpha(UnitCell& u, float val)
{
    u.GetPar("alpha").SetValue(val);
}

float _getalpha(UnitCell& u)
{
    return u.GetLatticePar(3);
}

void _setbeta(UnitCell& u, float val)
{
    u.GetPar("beta").SetValue(val);
}

float _getbeta(UnitCell& u)
{
    return u.GetLatticePar(4);
}

void _setgamma(UnitCell& u, float val)
{
    u.GetPar("gamma").SetValue(val);
}

float _getgamma(UnitCell& u)
{
    return u.GetLatticePar(5);
}


}

BOOST_PYTHON_MODULE(_unitcell)
{

    class_<UnitCell, bases<RefinableObj> > 
        ("UnitCell", init<>())
        // Constructors
        .def(init<const float, const float, const float, const std::string&>())
        .def(init<const float, const float, const float, 
            const float, const float, const float,
            const std::string&>())
        .def(init<const UnitCell&>())
        .def("GetLatticePar", 
            (CrystVector<float> (UnitCell::*)() const) &UnitCell::GetLatticePar)
        .def("GetLatticePar", 
            (float (UnitCell::*)(const int) const) &UnitCell::GetLatticePar)
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
