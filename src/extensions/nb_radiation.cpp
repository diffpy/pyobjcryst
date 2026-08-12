/*
 * pyobjcryst nanobind port — Radiation bindings
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <ObjCryst/ObjCryst/ScatteringData.h>

namespace nb = nanobind;
using namespace ObjCryst;

namespace {
double _GetWavelength(Radiation& r) { return r.GetWavelength()(0); }
} // namespace

void wrap_radiation(nb::module_& m)
{
    nb::class_<Radiation, RefinableObj>(m, "Radiation")
        .def(nb::init<>())
        .def("SetRadiationType",      &Radiation::SetRadiationType)
        .def("GetRadiationType",      &Radiation::GetRadiationType)
        .def("SetWavelengthType",     &Radiation::SetWavelengthType)
        .def("GetWavelengthType",     &Radiation::GetWavelengthType)
        .def("GetWavelength",         &_GetWavelength)
        .def("SetWavelength",
             nb::overload_cast<const REAL>(&Radiation::SetWavelength))
        .def("SetWavelength",
             nb::overload_cast<const std::string&, const REAL>(&Radiation::SetWavelength),
             nb::arg("XRayTubeElementName"), nb::arg("alpha2Alpha2ratio") = 0.5)
        .def("GetXRayTubeDeltaLambda",     &Radiation::GetXRayTubeDeltaLambda)
        .def("GetXRayTubeAlpha2Alpha1Ratio",&Radiation::GetXRayTubeAlpha2Alpha1Ratio)
        // Ported forward from upstream main (closes diffpy/pyobjcryst#38):
        // LinearPolarRate accessors and clock accessors, added after this
        // branch's original fork. No LIBOBJCRYST_VERSION guard needed here
        // (unlike main's version) since we vendor the ObjCryst++ source
        // directly rather than depending on an externally-installed version.
        .def("GetLinearPolarRate",    &Radiation::GetLinearPolarRate)
        .def("SetLinearPolarRate",    &Radiation::SetLinearPolarRate)
        .def("GetClockWavelength",
             &Radiation::GetClockWavelength, nb::rv_policy::reference_internal)
        .def("GetClockRadiation",
             &Radiation::GetClockRadiation, nb::rv_policy::reference_internal)
        .def("GetClockLinearPolarRate",
             &Radiation::GetClockLinearPolarRate, nb::rv_policy::reference_internal)
        ;
}
