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
* boost::python bindings to ObjCryst::ScatteringPowerAtom.
*
*****************************************************************************/

#include <boost/python/class.hpp>
#include <boost/python/args.hpp>

#include <string>

#include <ObjCryst/ObjCryst/ScatteringPower.h>

namespace bp = boost::python;
using namespace boost::python;
using namespace ObjCryst;


void wrap_scatteringpoweratom()
{

    typedef void (ScatteringPowerAtom::*SPAInitType)(
            const string&, const string&, const double);
    SPAInitType theinit = &ScatteringPowerAtom::Init;

    class_<ScatteringPowerAtom, bases<ScatteringPower> > ("ScatteringPowerAtom",
            init<const ScatteringPowerAtom&>())
        .def(
            init<const std::string&, const std::string&, bp::optional<const double> >
            ((bp::arg("name"), bp::arg("symbol"), bp::arg("bIso")=1.0)))
        .def("Init", theinit,
                (bp::arg("name"),
                bp::arg("symbol"),
                bp::arg("biso")=1.0
                ))
        .def("SetSymbol", &ScatteringPowerAtom::SetSymbol)
        .def("GetElementName", &ScatteringPowerAtom::GetElementName)
        .def("GetAtomicNumber", &ScatteringPowerAtom::GetAtomicNumber)
        ;
}
