/*****************************************************************************
*
* PyObjCryst        by DANSE Diffraction group
*                   Simon J. L. Billinge
*                   (c) 2009 The Trustees of Columbia University
*                   in the City of New York.  All rights reserved.
*
* File coded by:    Chris Farrow
*
* See AUTHORS.txt for a list of people who contributed.
* See LICENSE.txt for license information.
*
******************************************************************************
*
* boost::python bindings to ObjCryst::GlobalScatteringPower from
* ObjCryst/ObjCryst/ZScatterer.h
*
*****************************************************************************/

#include <boost/python.hpp>
#include <boost/utility.hpp>
#include <boost/python/class.hpp>
#include <boost/python/def.hpp>

#include <string>
#include <iostream>

#include <ObjCryst/ObjCryst/ScatteringPower.h>
#include <ObjCryst/ObjCryst/ZScatterer.h>
#include <ObjCryst/RefinableObj/RefinableObj.h>
#include <ObjCryst/CrystVector/CrystVector.h>

using namespace boost::python;
using namespace ObjCryst;

void wrap_globalscatteringpower()
{

    class_<GlobalScatteringPower, bases<ScatteringPower> > ("GlobalScatteringPower",
        init<>())
        .def(init<const ZScatterer &>())
        .def(init<const GlobalScatteringPower&>())
        .def("Init", &GlobalScatteringPower::Init)
        .def("GetRadius", &GlobalScatteringPower::GetRadius)
        ;
}
