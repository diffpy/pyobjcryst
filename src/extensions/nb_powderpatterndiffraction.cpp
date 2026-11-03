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
#include <ObjCryst/ObjCryst/ScatteringCorr.h>
#include <ObjCryst/RefinableObj/RefinableObj.h>

#include "helpers_nb.hpp"

namespace nb = nanobind;
using namespace ObjCryst;

namespace {

void SetProfileCopy(PowderPatternDiffraction& pdiff,
                    const ReflectionProfile& profile)
{
    ReflectionProfile* copy = profile.CreateCopy();
    // The TOF profile needs a unit cell to compute d_hkl, as in
    // PowderPatternDiffraction::SetCrystal and XMLInput.
    auto* tofprofile =
        dynamic_cast<ReflectionProfileDoubleExponentialPseudoVoigt*>(copy);
    if (tofprofile != nullptr && pdiff.HasCrystal())
        tofprofile->SetUnitCell(pdiff.GetCrystal());
    pdiff.SetProfile(copy);
}

// PowderPatternDiffraction has no public accessor for its texture
// correction, but registers it as a sub-object.
TextureMarchDollase& GetTextureMarchDollase(PowderPatternDiffraction& pdiff)
{
    ObjRegistry<RefinableObj>& subobjs = pdiff.GetSubObjRegistry();
    for (long i = 0; i < subobjs.GetNb(); ++i)
    {
        auto* texture = dynamic_cast<TextureMarchDollase*>(&subobjs.GetObj(i));
        if (texture != nullptr) return *texture;
    }
    throw ObjCrystException("PowderPatternDiffraction has no "
                            "TextureMarchDollase sub-object. Please report "
                            "this as a pyobjcryst bug.");
}

} // namespace

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
              &SetProfileCopy,
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
        .def("GetTextureMarchDollase", &GetTextureMarchDollase,
             nb::rv_policy::reference_internal,
             "Return the March-Dollase preferred-orientation correction.")
        ;

    // TextureMarchDollase : public ScatteringCorr, public RefinableObj --
    // RefinableObj is a second base at a non-zero offset, which nanobind's
    // base casting cannot handle (same failure mode as the virtual bases in
    // nanobind_migration_notes.md). Register it standalone and forward the
    // RefinableObj surface through T instead.
    auto texture = nb::class_<TextureMarchDollase>(m, "TextureMarchDollase");
    bind_refinableobj_forwarding(texture);
    texture
        .def("AddPhase", &TextureMarchDollase::AddPhase,
             nb::arg("fraction"), nb::arg("coeffMarch"),
             nb::arg("h"), nb::arg("k"), nb::arg("l"))
        .def("SetPhasePar", &TextureMarchDollase::SetPhasePar,
             nb::arg("i"), nb::arg("fraction"), nb::arg("coeffMarch"),
             nb::arg("h"), nb::arg("k"), nb::arg("l"))
        .def("DeletePhase", &TextureMarchDollase::DeletePhase, nb::arg("i"))
        .def("GetNbPhase", &TextureMarchDollase::GetNbPhase)
        .def("GetFraction", &TextureMarchDollase::GetFraction, nb::arg("i"))
        .def("GetMarchCoeff", &TextureMarchDollase::GetMarchCoeff, nb::arg("i"))
        .def("GetPhaseH", &TextureMarchDollase::GetPhaseH, nb::arg("i"))
        .def("GetPhaseK", &TextureMarchDollase::GetPhaseK, nb::arg("i"))
        .def("GetPhaseL", &TextureMarchDollase::GetPhaseL, nb::arg("i"))
        ;
}
