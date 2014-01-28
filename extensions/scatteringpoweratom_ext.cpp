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
* boost::python bindings to ObjCryst::ScatteringPowerAtom.
*
* $Id$
*
*****************************************************************************/

#include <boost/python.hpp>
#include <boost/utility.hpp>
#include <boost/python/class.hpp>
#include <boost/python/def.hpp>
#include <boost/python/args.hpp>

#include <string>
#include <iostream>

#include <ObjCryst/ObjCryst/ScatteringPower.h>
#include <ObjCryst/RefinableObj/RefinableObj.h>
#include <ObjCryst/CrystVector/CrystVector.h>

namespace bp = boost::python;
using namespace boost::python;
using namespace ObjCryst;


void wrap_scatteringpoweratom()
{

    class_<ScatteringPowerAtom, bases<ScatteringPower> > ("ScatteringPowerAtom",
            init<const ScatteringPowerAtom&>())
        .def(
            init<const std::string&, const std::string&, optional<const double> >
            ((bp::arg("name"), bp::arg("symbol"), bp::arg("bIso")=1.0)))
        .def("Init", &ScatteringPowerAtom::Init,
                (bp::arg("name"),
                bp::arg("symbol"),
                bp::arg("biso")=1.0
                ))
        .def("SetSymbol", &ScatteringPowerAtom::SetSymbol)
        .def("GetElementName", &ScatteringPowerAtom::GetElementName)
        .def("GetAtomicNumber", &ScatteringPowerAtom::GetAtomicNumber)
        ;
}
