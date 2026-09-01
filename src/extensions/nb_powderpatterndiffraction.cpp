/*
 * pyobjcryst nanobind port — PowderPatternDiffraction bindings
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <ObjCryst/ObjCryst/General.h>

#undef B0
#include <ObjCryst/ObjCryst/PowderPattern.h>
#include <ObjCryst/ObjCryst/ScatteringData.h>
#include <ObjCryst/ObjCryst/ReflectionProfile.h>
#include <ObjCryst/RefinableObj/RefinableObj.h>

#include "helpers_nb.hpp"

namespace nb = nanobind;
using namespace ObjCryst;

void wrap_powderpatterndiffraction(nb::module_& m)
{
    nb::enum_<ReflectionProfileType>(m, "ReflectionProfileType")
        .value("PROFILE_GAUSSIAN",    PROFILE_GAUSSIAN)
        .value("PROFILE_LORENTZIAN",  PROFILE_LORENTZIAN)
        .value("PROFILE_PSEUDO_VOIGT",PROFILE_PSEUDO_VOIGT)
        .value("PROFILE_PSEUDO_VOIGT_FINGER_COX_JEPHCOAT",
               PROFILE_PSEUDO_VOIGT_FINGER_COX_JEPHCOAT)
        .value("PROFILE_PEARSON_VII", PROFILE_PEARSON_VII)
        .export_values()
        ;

    // PowderPatternDiffraction : virtual public PowderPatternComponent,
    //                            public ScatteringData
    // (see PowderPattern.h). The link to PowderPatternComponent is VIRTUAL
    // -- declaring it as the nanobind base here (as an earlier version of
    // this file did) computes a wrong pointer offset (§1 in
    // nanobind_migration_notes.md), which is exactly the bug class that
    // caused the Crystal.GetScatteringPower() segfault elsewhere in this
    // migration. The link to ScatteringData is a plain (non-virtual) single
    // base, so it is the one declared here; ScatteringData is already
    // registered standalone in nb_scatteringdata.cpp and brings GetCrystal(),
    // GetNbRefl(), and friends along automatically. PowderPatternComponent's
    // GetParentPowderPattern() and the RefinableObj methods reachable only
    // via the (now-undeclared) virtual base are rebound explicitly below,
    // mirroring nb_powderpatterncomponent.cpp.
    nb::class_<PowderPatternDiffraction, ScatteringData>(
            m, "PowderPatternDiffraction")
        .def("GetParentPowderPattern",
             [](PowderPatternDiffraction& d) -> PowderPattern& {
                 return static_cast<PowderPatternComponent&>(d).GetParentPowderPattern(); },
             nb::rv_policy::reference_internal)
        .def("GetName", [](PowderPatternDiffraction& d) -> std::string {
                            return static_cast<RefinableObj&>(d).GetName(); })
        .def("SetName", [](PowderPatternDiffraction& d, const std::string& n) {
                            static_cast<RefinableObj&>(d).SetName(n); })
        .def("GetClassName", [](PowderPatternDiffraction& d) -> std::string {
                            return static_cast<RefinableObj&>(d).GetClassName(); })
        .def("FixAllPar",   [](PowderPatternDiffraction& d) { static_cast<RefinableObj&>(d).FixAllPar(); })
        .def("UnFixAllPar", [](PowderPatternDiffraction& d) { static_cast<RefinableObj&>(d).UnFixAllPar(); })
        .def("GetPowderPatternCalc",
             [](PowderPatternDiffraction& d){ return crystvec_to_array(d.GetPowderPatternCalc()); })
        .def("SetReflectionProfilePar",
             &PowderPatternDiffraction::SetReflectionProfilePar,
             nb::arg("type") = PROFILE_PSEUDO_VOIGT,
             nb::arg("fwhmCagliotiW") = 1e-6,
             nb::arg("fwhmCagliotiU") = 0,
             nb::arg("fwhmCagliotiV") = 0,
             nb::arg("eta0") = 0.5,
             nb::arg("eta1") = 0)
        .def("GetProfile",
             nb::overload_cast<>(&PowderPatternDiffraction::GetProfile),
             nb::rv_policy::reference_internal)
        .def("SetProfile",
             [](PowderPatternDiffraction& d, ReflectionProfile& p) {
                 d.SetProfile(p.CreateCopy());
             },
             nb::arg("profile"),
             "Install an independent copy of the given profile.")
        .def("SetExtractionMode",
             &PowderPatternDiffraction::SetExtractionMode,
             nb::arg("extract") = true, nb::arg("init") = false)
        .def("GetExtractionMode",
             &PowderPatternDiffraction::GetExtractionMode)
        .def("ExtractLeBail",
             &PowderPatternDiffraction::ExtractLeBail,
             nb::arg("nbcycle") = 1)
        .def("SetCrystal",
             &PowderPatternDiffraction::SetCrystal,
             nb::arg("crystal"), nb::keep_alive<1,2>())
        .def("GetNbReflBelowMaxSinThetaOvLambda",
             &PowderPatternDiffraction::GetNbReflBelowMaxSinThetaOvLambda)
        .def("GetFhklObsSq",
             [](PowderPatternDiffraction& d){ return crystvec_to_array(d.GetFhklObsSq()); })
        .def("X2XCorrPhase", &PowderPatternDiffraction::X2XCorrPhase,
             "Apply the flat-detector displacement correction for this phase's offset.")
        ;
}
