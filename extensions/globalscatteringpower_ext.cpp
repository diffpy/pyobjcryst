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
* boost::python bindings to ObjCryst::GlobalScatteringPower from
* ObjCryst/ZScatterer.h
*
* $Id$
*
*****************************************************************************/

#include <boost/utility.hpp>
#include <boost/python.hpp>
#include <boost/python/class.hpp>
#include <boost/python/def.hpp>

#include <string>
#include <iostream>

#include "ObjCryst/ScatteringPower.h"
#include "ObjCryst/ZScatterer.h"
#include "RefinableObj/RefinableObj.h"
#include "CrystVector/CrystVector.h"

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
