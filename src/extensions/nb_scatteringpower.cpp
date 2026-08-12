/*
 * pyobjcryst nanobind port — ScatteringPower bindings
 */

#include <nanobind/nanobind.h>
#include <nanobind/trampoline.h>
#include <nanobind/stl/string.h>

#include <ObjCryst/ObjCryst/ScatteringPower.h>
#include <ObjCryst/RefinableObj/RefinableObj.h>
#include <ObjCryst/CrystVector/CrystVector.h>
#include "helpers_nb.hpp"

namespace nb = nanobind;
using namespace ObjCryst;

struct PyScatteringPower : ScatteringPower {
    NB_TRAMPOLINE(ScatteringPower, 13);

    CrystVector_REAL GetScatteringFactor(const ScatteringData& d, const int idx) const override {
        NB_OVERRIDE_PURE(GetScatteringFactor, d, idx);
    }
    REAL GetForwardScatteringFactor(const RadiationType t) const override {
        NB_OVERRIDE_PURE(GetForwardScatteringFactor, t);
    }
    CrystVector_REAL GetTemperatureFactor(const ScatteringData& d, const int idx) const override {
        NB_OVERRIDE_PURE(GetTemperatureFactor, d, idx);
    }
    CrystMatrix_REAL GetResonantScattFactReal(const ScatteringData& d, const int idx) const override {
        NB_OVERRIDE_PURE(GetResonantScattFactReal, d, idx);
    }
    CrystMatrix_REAL GetResonantScattFactImag(const ScatteringData& d, const int idx) const override {
        NB_OVERRIDE_PURE(GetResonantScattFactImag, d, idx);
    }
    REAL GetRadius() const override { NB_OVERRIDE_PURE(GetRadius); }
    // GetMaximumLikelihood*/SetMaximumLikelihood* are not virtual; not in trampoline
    bool IsScatteringFactorAnisotropic() const override {
        NB_OVERRIDE(IsScatteringFactorAnisotropic);
    }
    bool IsTemperatureFactorAnisotropic() const override {
        NB_OVERRIDE(IsTemperatureFactorAnisotropic);
    }
    bool IsResonantScatteringAnisotropic() const override {
        NB_OVERRIDE(IsResonantScatteringAnisotropic);
    }
    // GetSymbol: const string& return; override not supported, use base class
    void SetBiso(const REAL b) override { NB_OVERRIDE(SetBiso, b); }
    void SetBij(const size_t& i, const size_t& j, const REAL b) override {
        NB_OVERRIDE(SetBij, i, j, b);
    }
    REAL GetFormalCharge() const override { NB_OVERRIDE(GetFormalCharge); }
    void SetFormalCharge(const REAL c) override { NB_OVERRIDE(SetFormalCharge, c); }
protected:
    void InitRefParList() override {}
};

namespace {

template <size_t I, size_t J>
double _GetBij(ScatteringPower& sp) { return sp.GetBij(I, J); }

template <size_t I, size_t J>
void _SetBij(ScatteringPower& sp, const double bd) { sp.SetBij(I, J, static_cast<REAL>(bd)); }

nb::tuple _GetColourRGB(ScatteringPower& sp) {
    return nb::make_tuple(sp.GetColourRGB()[0], sp.GetColourRGB()[1], sp.GetColourRGB()[2]);
}

} // namespace

void wrap_scatteringpower(nb::module_& m)
{
    m.attr("refpartype_scattpow")             = gpRefParTypeScattPow;
    m.attr("refpartype_scattpow_temperature") = gpRefParTypeScattPowTemperature;
    m.attr("gScatteringPowerRegistry")        = &gScatteringPowerRegistry;

    nb::class_<ScatteringPower, PyScatteringPower> cls(m, "ScatteringPower");
    cls
        .def("GetScatteringFactor",  &ScatteringPower::GetScatteringFactor,
             nb::arg("data"), nb::arg("spgSymPosIndex") = -1)
        .def("GetForwardScatteringFactor", &ScatteringPower::GetForwardScatteringFactor)
        .def("GetTemperatureFactor", &ScatteringPower::GetTemperatureFactor,
             nb::arg("data"), nb::arg("spgSymPosIndex") = -1)
        .def("GetResonantScattFactReal", &ScatteringPower::GetResonantScattFactReal,
             nb::arg("data"), nb::arg("spgSymPosIndex") = -1)
        .def("GetResonantScattFactImag", &ScatteringPower::GetResonantScattFactImag,
             nb::arg("data"), nb::arg("spgSymPosIndex") = -1)
        .def("IsScatteringFactorAnisotropic",  &ScatteringPower::IsScatteringFactorAnisotropic)
        .def("IsTemperatureFactorAnisotropic", &ScatteringPower::IsTemperatureFactorAnisotropic)
        .def("IsResonantScatteringAnisotropic",&ScatteringPower::IsResonantScatteringAnisotropic)
        .def("GetSymbol", &ScatteringPower::GetSymbol)
        .def("GetBiso",
             nb::overload_cast<>(&ScatteringPower::GetBiso, nb::const_))
        .def("SetBiso",  &ScatteringPower::SetBiso)
        .def("GetBij",
             nb::overload_cast<const size_t&, const size_t&>(&ScatteringPower::GetBij, nb::const_))
        .def("SetBij",
             nb::overload_cast<const size_t&, const size_t&, const REAL>(&ScatteringPower::SetBij))
        .def("IsIsotropic",          &ScatteringPower::IsIsotropic)
        .def("GetDynPopCorrIndex",   &ScatteringPower::GetDynPopCorrIndex)
        .def("GetNbScatteringPower", &ScatteringPower::GetNbScatteringPower)
        .def("GetLastChangeClock",   &ScatteringPower::GetLastChangeClock,
             nb::rv_policy::reference_internal)
        .def("GetRadius",                           &ScatteringPower::GetRadius)
        .def("GetMaximumLikelihoodPositionError",   &ScatteringPower::GetMaximumLikelihoodPositionError)
        .def("SetMaximumLikelihoodPositionError",   &ScatteringPower::SetMaximumLikelihoodPositionError)
        .def("GetMaximumLikelihoodNbGhostAtom",     &ScatteringPower::GetMaximumLikelihoodNbGhostAtom)
        .def("SetMaximumLikelihoodNbGhostAtom",     &ScatteringPower::SetMaximumLikelihoodNbGhostAtom)
        .def("GetMaximumLikelihoodParClock", &ScatteringPower::GetMaximumLikelihoodParClock,
             nb::rv_policy::reference_internal)
        .def("GetFormalCharge",  &ScatteringPower::GetFormalCharge)
        .def("SetFormalCharge",  &ScatteringPower::SetFormalCharge)
        .def("GetColourRGB",     &_GetColourRGB)
        .def("GetColour",        &_GetColourRGB)
        .def("SetColour",
             nb::overload_cast<const float, const float, const float>(&ScatteringPower::SetColour),
             nb::arg("r"), nb::arg("g"), nb::arg("b"))
        .def_prop_rw("Biso",
             nb::overload_cast<>(&ScatteringPower::GetBiso, nb::const_),
             &ScatteringPower::SetBiso)
        .def_prop_rw("B11", &_GetBij<1,1>, &_SetBij<1,1>)
        .def_prop_rw("B22", &_GetBij<2,2>, &_SetBij<2,2>)
        .def_prop_rw("B33", &_GetBij<3,3>, &_SetBij<3,3>)
        .def_prop_rw("B12", &_GetBij<1,2>, &_SetBij<1,2>)
        .def_prop_rw("B13", &_GetBij<1,3>, &_SetBij<1,3>)
        .def_prop_rw("B23", &_GetBij<2,3>, &_SetBij<2,3>)
        // RefinableObj methods (virtual base — call via ScatteringPower* to avoid vbase offset issue)
        .def("GetName",       [](ScatteringPower& sp) -> std::string { return sp.GetName(); })
        .def("SetName",       [](ScatteringPower& sp, const std::string& n) { sp.SetName(n); })
        .def("GetClassName",  [](ScatteringPower& sp) -> std::string { return sp.GetClassName(); })
        .def("Print",         [](ScatteringPower& sp) { sp.Print(); })
        .def("GetNbPar",      [](ScatteringPower& sp) { return sp.GetNbPar(); })
        .def("GetPar",        [](ScatteringPower& sp, const std::string& name) -> RefinablePar& {
                                 return sp.GetPar(name); }, nb::rv_policy::reference_internal)
        .def("GetPar",        [](ScatteringPower& sp, long i) -> RefinablePar& {
                                 return sp.GetPar(i); }, nb::rv_policy::reference_internal)
        .def("GetClockMaster",[](ScatteringPower& sp) -> const RefinableObjClock& {
                                 return sp.GetClockMaster(); }, nb::rv_policy::reference_internal)
        .def("__str__",       [](ScatteringPower& sp) { return sp.GetName(); })
        ;

    // Restore the rest of RefinableObj's method surface (a handful of
    // methods -- GetName/SetName/GetClassName/Print/GetNbPar/GetPar/
    // GetClockMaster -- were already covered one-off above; everything
    // else -- FixAllPar, XMLOutput, BeginOptimization, GetLogLikelihood,
    // AddPar, CreateParamSet, GetOption, UpdateDisplay, ... -- was not).
    // See helpers_nb.hpp for why this is safe despite ScatteringPower's
    // virtual inheritance of RefinableObj. Covers ScatteringPowerSphere
    // automatically (non-virtual inheritance, already correctly declared as
    // its nanobind base); ScatteringPowerAtom and GlobalScatteringPower
    // additionally inherit ScatteringPower virtually and need this called
    // on them directly too (see nb_scatteringpoweratom.cpp,
    // nb_globalscatteringpower.cpp).
    bind_refinableobj_forwarding(cls);
}
