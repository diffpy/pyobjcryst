/*****************************************************************************
*
* PyObjCryst        by DANSE Diffraction group
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
#include <boost/python/def.hpp>

#include <string>

#include "ObjCryst/ScatteringPower.h"
#include "ObjCryst/ScatteringPowerSphere.h"
#include "RefinableObj/RefinableObj.h"
#include "CrystVector/CrystVector.h"

using namespace boost::python;
using namespace ObjCryst;


void wrap_scatteringpowersphere()
{

    class_<ScatteringPowerSphere, bases<ScatteringPower> > 
        ("ScatteringPowerSphere", init<>())
        .def(init<const std::string&, const double, optional<const double> >())
        .def("Init", &ScatteringPowerSphere::Init,
                (boost::python::arg("name"),
                boost::python::arg("radius"),
                boost::python::arg("biso")=1.0
                ))
        .def("GetRadius", &ScatteringPowerSphere::GetRadius)
        ;
}
