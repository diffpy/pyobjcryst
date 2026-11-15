/*
 * pyobjcryst nanobind port — ScatteringData bindings
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/map.h>

#include <ObjCryst/ObjCryst/ScatteringData.h>
#include <ObjCryst/CrystVector/CrystVector.h>

#include "helpers_nb.hpp"

namespace nb = nanobind;
using namespace ObjCryst;

namespace {

double _GetWavelength(ScatteringData& s)
{
    return s.GetWavelength()(0);
}

nb::dict _GetScatteringFactor(ScatteringData& data)
{
    const std::map<const ScatteringPower*, CrystVector_REAL>& vsf
        = data.GetScatteringFactor();
    nb::dict d;
    for (auto& kv : vsf) {
        nb::object key = nb::cast(kv.first, nb::rv_policy::reference);
        d[key] = crystvec_to_array(kv.second);
    }
    return d;
}

} // namespace

void wrap_scatteringdata(nb::module_& m)
{
    m.attr("refpartype_scattdata")                   = gpRefParTypeScattData;
    m.attr("refpartype_scattdata_scale")             = gpRefParTypeScattDataScale;
    m.attr("refpartype_scattdata_profile")           = gpRefParTypeScattDataProfile;
    m.attr("refpartype_scattdata_profile_type")      = gpRefParTypeScattDataProfileType;
    m.attr("refpartype_scattdata_profile_width")     = gpRefParTypeScattDataProfileWidth;
    m.attr("refpartype_scattdata_profile_asym")      = gpRefParTypeScattDataProfileAsym;
    m.attr("refpartype_scattdata_corr")              = gpRefParTypeScattDataCorr;
    m.attr("refpartype_scattdata_corr_pos")          = gpRefParTypeScattDataCorrPos;
    m.attr("refpartype_scattdata_radiation")         = gpRefParTypeRadiation;
    m.attr("refpartype_scattdata_radiation_wavelength") = gpRefParTypeRadiationWavelength;

    nb::class_<ScatteringData> cls(m, "ScatteringData");
    cls
        .def("GenHKLFullSpace2",
             nb::overload_cast<const REAL, const bool>(&ScatteringData::GenHKLFullSpace2),
             nb::arg("maxsithsl"), nb::arg("unique") = false)
        .def("GenHKLFullSpace",
             nb::overload_cast<const REAL, const bool>(&ScatteringData::GenHKLFullSpace),
             nb::arg("maxtheta"), nb::arg("unique") = false)
        .def("SetCrystal", &ScatteringData::SetCrystal)
        .def("GetCrystal",
             nb::overload_cast<>(&ScatteringData::GetCrystal),
             nb::rv_policy::reference_internal)
        .def("HasCrystal", &ScatteringData::HasCrystal)
        .def("GetNbRefl",  &ScatteringData::GetNbRefl)
        .def("GetH",  [](ScatteringData& s){ return crystvec_to_array(s.GetH()); })
        .def("GetK",  [](ScatteringData& s){ return crystvec_to_array(s.GetK()); })
        .def("GetL",  [](ScatteringData& s){ return crystvec_to_array(s.GetL()); })
        .def("GetH2Pi", [](ScatteringData& s){ return crystvec_to_array(s.GetH2Pi()); })
        .def("GetK2Pi", [](ScatteringData& s){ return crystvec_to_array(s.GetK2Pi()); })
        .def("GetL2Pi", [](ScatteringData& s){ return crystvec_to_array(s.GetL2Pi()); })
        .def("GetReflX", [](ScatteringData& s){ return crystvec_to_array(s.GetReflX()); })
        .def("GetReflY", [](ScatteringData& s){ return crystvec_to_array(s.GetReflY()); })
        .def("GetReflZ", [](ScatteringData& s){ return crystvec_to_array(s.GetReflZ()); })
        .def("GetSinThetaOverLambda", [](ScatteringData& s){ return crystvec_to_array(s.GetSinThetaOverLambda()); })
        .def("GetTheta", [](ScatteringData& s){ return crystvec_to_array(s.GetTheta()); })
        .def("GetClockTheta", &ScatteringData::GetClockTheta, nb::rv_policy::reference_internal)
        .def("GetFhklCalcSq", [](ScatteringData& s){ return crystvec_to_array(s.GetFhklCalcSq()); })
        .def("GetFhklCalcReal",[](ScatteringData& s){ return crystvec_to_array(s.GetFhklCalcReal()); })
        .def("GetFhklCalcImag",[](ScatteringData& s){ return crystvec_to_array(s.GetFhklCalcImag()); })
        .def("GetFhklObsSq",   [](ScatteringData& s){ return crystvec_to_array(s.GetFhklObsSq()); })
        .def("GetRadiation",
             nb::overload_cast<>(&ScatteringData::GetRadiation, nb::const_),
             nb::rv_policy::reference_internal)
        .def("GetRadiationType", &ScatteringData::GetRadiationType)
        .def("GetWavelength",    &_GetWavelength)
        .def("SetIsIgnoringImagScattFact", &ScatteringData::SetIsIgnoringImagScattFact)
        .def("IsIgnoringImagScattFact",    &ScatteringData::IsIgnoringImagScattFact)
        .def("PrintFhklCalc",       [](const ScatteringData& s){ s.PrintFhklCalc(); })
        .def("PrintFhklCalcDetail", [](const ScatteringData& s){ s.PrintFhklCalcDetail(); })
        .def("SetMaxSinThetaOvLambda", &ScatteringData::SetMaxSinThetaOvLambda)
        .def("GetMaxSinThetaOvLambda", &ScatteringData::GetMaxSinThetaOvLambda)
        .def("GetNbReflBelowMaxSinThetaOvLambda", &ScatteringData::GetNbReflBelowMaxSinThetaOvLambda)
        .def("GetClockNbReflBelowMaxSinThetaOvLambda",
             &ScatteringData::GetClockNbReflBelowMaxSinThetaOvLambda,
             nb::rv_policy::reference_internal)
        .def("GetScatteringFactor", &_GetScatteringFactor)
        ;

    // Restore RefinableObj's method surface (Print, GetName, FixAllPar,
    // BeginOptimization, XMLOutput, AddPar, ...): ScatteringData inherits
    // RefinableObj virtually, so none of it was reachable before -- see
    // helpers_nb.hpp for why forwarding like this is safe. Covers
    // DiffractionDataSingleCrystal and PowderPatternDiffraction
    // automatically (both inherit ScatteringData non-virtually and already
    // have it correctly declared as their nanobind base).
    bind_refinableobj_forwarding(cls);
}
