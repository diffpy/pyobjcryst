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
* boost::python bindings to ObjCryst::ScatteringPowerSphere.
*
* $Id$
*
*****************************************************************************/

#include <boost/utility.hpp>
#include <boost/python.hpp>
#include <boost/python/class.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>

#include <string>

#include "ObjCryst/ScatteringPower.h"
#include "ObjCryst/ScatteringPowerSphere.h"
#include "RefinableObj/RefinableObj.h"
#include "CrystVector/CrystVector.h"

using namespace boost::python;
using namespace ObjCryst;


namespace {

// Factories that set SetDeleteRefParInDestructor(0)
ScatteringPowerSphere* ScatteringPowerSphereDefault()
{
    ScatteringPowerSphere* sp = new ScatteringPowerSphere();
    sp->SetDeleteRefParInDestructor(0);
    return sp;
}

ScatteringPowerSphere* ScatteringPowerSphere3(const std::string& name,
        const float radius, const float bIso)
{
    ScatteringPowerSphere* sp = new ScatteringPowerSphere(name, radius, bIso);
    sp->SetDeleteRefParInDestructor(0);
    return sp;
}

}

BOOST_PYTHON_MODULE(_scatteringpowersphere)
{

    class_<ScatteringPowerSphere, bases<ScatteringPower> > 
        ("ScatteringPowerSphere")
        .def("__init__", make_constructor(ScatteringPowerSphereDefault))
        .def("__init__", make_constructor(ScatteringPowerSphere3))
        .def(init<const std::string&, const float, optional<const float> >())
        .def("Init", &ScatteringPowerSphere::Init,
                (boost::python::arg("name"),
                boost::python::arg("radius"),
                boost::python::arg("biso")=1.0
                ))
        .def("GetRadius", &ScatteringPowerSphere::GetRadius)
        ;
}
