/*****************************************************************************
*
* pyobjcryst
*
* File coded by:    Vincent Favre-Nicolin
*
* See AUTHORS.txt for a list of people who contributed.
* See LICENSE.txt for license information.
*
******************************************************************************
*
* boost::python bindings to ObjCryst::PowderPatternBackground.
*
* Changes from ObjCryst::PowderPatternBackground
*
* Other Changes
*
*****************************************************************************/

#include <boost/python/class.hpp>
#include <boost/python/args.hpp>
#include <boost/python/copy_const_reference.hpp>

#undef B0

#include <ObjCryst/ObjCryst/PowderPattern.h>

#include "helpers.hpp"

namespace bp = boost::python;
using namespace ObjCryst;

namespace {

void _SetInterpPoints(PowderPatternBackground& b,
        bp::object tth, bp::object backgd)
{
    CrystVector_REAL cvtth, cvbackg;
    assignCrystVector(cvtth, tth);
    assignCrystVector(cvbackg, backgd);
    b.SetInterpPoints(cvtth, cvbackg);
}

}   // namespace


void wrap_powderpatternbackground()
{
    using namespace boost::python;
    class_<PowderPatternBackground, bases<PowderPatternComponent>, boost::noncopyable>(
            "PowderPatternBackground", no_init)
        //.def("SetParentPowderPattern", &PowderPatternBackground::SetParentPowderPattern)
        .def("GetPowderPatternCalc",
                &PowderPatternBackground::GetPowderPatternCalc,
                return_value_policy<copy_const_reference>())
        .def("ImportUserBackground",
                &PowderPatternBackground::ImportUserBackground,
                bp::arg("filename"))
        .def("SetInterpPoints",
                _SetInterpPoints,
                (bp::arg("tth"), bp::arg("backgd")))
        .def("OptimizeBayesianBackground",
                &PowderPatternBackground::OptimizeBayesianBackground)
        .def("FixParametersBeyondMaxresolution",
                &PowderPatternBackground::FixParametersBeyondMaxresolution,
                bp::arg("obj"))
        ;
}
