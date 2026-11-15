/*
 * pyobjcryst nanobind port — PowderPatternBackground bindings
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#undef B0
#include <ObjCryst/ObjCryst/PowderPattern.h>

#include "helpers_nb.hpp"

namespace nb = nanobind;
using namespace ObjCryst;

namespace {

void _SetInterpPoints(PowderPatternBackground& b, nb::object tth, nb::object backgd)
{
    CrystVector_REAL cvtth, cvbackg;
    assignCrystVector(cvtth, tth);
    assignCrystVector(cvbackg, backgd);
    b.SetInterpPoints(cvtth, cvbackg);
}

void _OptimizeBayesianBackground(PowderPatternBackground& b, const bool verbose = false)
{
    CaptureStdOut gag;
    if (verbose) gag.release();
    b.OptimizeBayesianBackground();
}

} // namespace

void wrap_powderpatternbackground(nb::module_& m)
{
    m.attr("refpartype_scattdata_background") = gpRefParTypeScattDataBackground;

    nb::class_<PowderPatternBackground, PowderPatternComponent>(m, "PowderPatternBackground")
        .def("GetPowderPatternCalc",
             [](PowderPatternBackground& b){ return crystvec_to_array(b.GetPowderPatternCalc()); })
        .def("ImportUserBackground", &PowderPatternBackground::ImportUserBackground,
             nb::arg("filename"))
        .def("SetInterpPoints", &_SetInterpPoints, nb::arg("tth"), nb::arg("backgd"))
        .def("GetInterpPointsX",
             [](PowderPatternBackground& b){ return crystvec_to_array(*b.GetInterpPoints().first); })
        .def("GetInterpPointsY",
             [](PowderPatternBackground& b){ return crystvec_to_array(*b.GetInterpPoints().second); })
        .def("OptimizeBayesianBackground", &_OptimizeBayesianBackground,
             nb::arg("verbose") = false)
        .def("FixParametersBeyondMaxresolution",
             &PowderPatternBackground::FixParametersBeyondMaxresolution, nb::arg("obj"))
        ;
}
