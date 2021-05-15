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
#include <boost/python/scope.hpp>

#undef B0

#include <ObjCryst/ObjCryst/PowderPattern.h>

#include "helpers.hpp"

namespace bp = boost::python;
using namespace ObjCryst;
using namespace boost::python;

namespace {

void _SetInterpPoints(PowderPatternBackground& b,
        bp::object tth, bp::object backgd)
{
    CrystVector_REAL cvtth, cvbackg;
    assignCrystVector(cvtth, tth);
    assignCrystVector(cvbackg, backgd);
    b.SetInterpPoints(cvtth, cvbackg);
}

void _OptimizeBayesianBackground(PowderPatternBackground *pbackgd, const bool verbose=false)
{
      CaptureStdOut gag;
      if(verbose) gag.release();
      pbackgd->OptimizeBayesianBackground();
}

const CrystVector_REAL& _GetInterpPointsX(PowderPatternBackground *pbackgd)
{
  return *(pbackgd->GetInterpPoints().first);
}

const CrystVector_REAL& _GetInterpPointsY(PowderPatternBackground *pbackgd)
{
  return *(pbackgd->GetInterpPoints().second);
}


}   // namespace


void wrap_powderpatternbackground()
{
    scope().attr("refpartype_scattdata_background") = object(ptr(gpRefParTypeScattDataBackground));

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
        .def("GetInterpPointsX",
                &_GetInterpPointsX,
                return_value_policy<copy_const_reference>())
        .def("GetInterpPointsY",
                &_GetInterpPointsY,
                return_value_policy<copy_const_reference>())
        .def("OptimizeBayesianBackground",
                &_OptimizeBayesianBackground, (bp::arg("verbose")=false))
        .def("FixParametersBeyondMaxresolution",
                &PowderPatternBackground::FixParametersBeyondMaxresolution,
                bp::arg("obj"))
        ;
}
