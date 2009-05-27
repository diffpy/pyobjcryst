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
#include <boost/python/make_constructor.hpp>

#include <string>
#include <iostream>

#include "helpers.hpp"

using namespace boost::python;
using namespace ObjCryst;

namespace {

// Factories that set SetDeleteRefParInDestructor(0)

UnitCell* UnitCellDefault()
{
    UnitCell* u = new UnitCell();
    u->SetDeleteRefParInDestructor(0);
    return u;
}

UnitCell* UnitCell4(const float a, const float b, const float c,
        const std::string& SpaceGroupId)
{
    UnitCell* u = new UnitCell(a, b, c, SpaceGroupId);
    u->SetDeleteRefParInDestructor(0);
    return u;
}

UnitCell* UnitCell7(const float a, const float b, const float c,
        const float alpha, const float beta, const float gamma,
        const std::string& SpaceGroupId)
{
    UnitCell* u = new UnitCell(a, b, c, alpha, beta, gamma, SpaceGroupId);
    u->SetDeleteRefParInDestructor(0);
    return u;
}

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

}

BOOST_PYTHON_MODULE(_unitcell)
{

    class_<UnitCell, bases<RefinableObj> > 
        ("UnitCell", init<const UnitCell &>())
        // Constructors
        .def("__init__", make_constructor(UnitCellDefault))
        .def("__init__", make_constructor(UnitCell4))
        .def("__init__", make_constructor(UnitCell7))
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
        ;
}
