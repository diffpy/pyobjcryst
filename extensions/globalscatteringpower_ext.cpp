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
* boost::python bindings to ObjCryst::GlobalScatteringPower from
* ObjCryst/ObjCryst/ZScatterer.h
*
*****************************************************************************/

#include <boost/python/class.hpp>

#include <ObjCryst/ObjCryst/ScatteringPower.h>
#include <ObjCryst/ObjCryst/ZScatterer.h>

using namespace boost::python;
using namespace ObjCryst;

void wrap_globalscatteringpower()
{

    typedef void (GlobalScatteringPower::*GSPInitType)(const ZScatterer&);
    GSPInitType theinit = &GlobalScatteringPower::Init;

    class_<GlobalScatteringPower, bases<ScatteringPower> >("GlobalScatteringPower")
        .def(init<const ZScatterer &>())
        .def(init<const GlobalScatteringPower&>())
        .def("Init", theinit)
        .def("GetRadius", &GlobalScatteringPower::GetRadius)
        ;
}
