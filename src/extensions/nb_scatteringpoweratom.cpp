/*
 * pyobjcryst nanobind port — ScatteringPowerAtom bindings
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <ObjCryst/ObjCryst/ScatteringPower.h>
#include <ObjCryst/RefinableObj/RefinableObj.h>

#include "helpers_nb.hpp"

namespace nb = nanobind;
using namespace ObjCryst;

namespace {

// Helper: call ScatteringPower methods via the virtual base safely
// Use static_cast to get ScatteringPower* first, THEN access mName via
// RefinableObj methods (which are virtual and will work through the vtable)

template<size_t I, size_t J>
double _GetBijSPA(ScatteringPowerAtom& sp) { return sp.GetBij(I, J); }

template<size_t I, size_t J>
void _SetBijSPA(ScatteringPowerAtom& sp, double b) { sp.SetBij(I, J, static_cast<REAL>(b)); }

} // namespace

void wrap_scatteringpoweratom(nb::module_& m)
{
    // NOTE: intentionally registered WITHOUT ScatteringPower as a declared
    // nanobind base. Unlike ScatteringPowerSphere (plain `public
    // ScatteringPower`), ScatteringPowerAtom uses VIRTUAL inheritance
    // (`class ScatteringPowerAtom : virtual public ScatteringPower`, see
    // ScatteringPower.h). nanobind's automatic base<->derived pointer-offset
    // machinery (nb::class_<Derived, Base>) assumes a fixed, statically
    // computable offset, which virtual inheritance does not provide — this
    // is exactly the "Pattern 1/2: leaf class, no declared virtual base"
    // case documented in nanobind_migration_notes.md. All ScatteringPower
    // methods needed on ScatteringPowerAtom are therefore explicitly
    // rebound below (not inherited via nanobind), matching the working
    // Atom/Scatterer pattern elsewhere in this codebase. Do not add
    // ScatteringPower here without re-verifying against that virtual base.
    nb::class_<ScatteringPowerAtom> cls(m, "ScatteringPowerAtom");
    cls
        .def(nb::init<const ScatteringPowerAtom&>())
        .def(nb::init<const std::string&, const std::string&, const REAL>(),
             nb::arg("name"), nb::arg("symbol"), nb::arg("bIso") = 1.0)
        .def("Init",
             nb::overload_cast<const std::string&, const std::string&, const REAL>(
                 &ScatteringPowerAtom::Init),
             nb::arg("name"), nb::arg("symbol"), nb::arg("biso") = 1.0)
        .def("SetSymbol",       &ScatteringPowerAtom::SetSymbol)
        .def("GetElementName",  &ScatteringPowerAtom::GetElementName)
        .def("GetAtomicNumber", &ScatteringPowerAtom::GetAtomicNumber)
        .def("GetAtomicWeight", &ScatteringPowerAtom::GetAtomicWeight)
        // ScatteringPower methods (would need virtual cast, use direct call)
        .def("GetSymbol",   [](ScatteringPowerAtom& sp) -> std::string { return sp.GetSymbol(); })
        .def("GetBiso",     [](ScatteringPowerAtom& sp) { return (double)sp.GetBiso(); })
        .def("SetBiso",     [](ScatteringPowerAtom& sp, double b) { sp.SetBiso(static_cast<REAL>(b)); })
        .def("GetRadius",   [](ScatteringPowerAtom& sp) { return (double)sp.GetRadius(); })
        .def("GetForwardScatteringFactor", [](ScatteringPowerAtom& sp, RadiationType t) {
                                return (double)sp.GetForwardScatteringFactor(t); })
        .def("GetMaximumLikelihoodPositionError",
             [](ScatteringPowerAtom& sp) { return (double)sp.GetMaximumLikelihoodPositionError(); })
        .def("SetMaximumLikelihoodPositionError",
             [](ScatteringPowerAtom& sp, double e) { sp.SetMaximumLikelihoodPositionError(static_cast<REAL>(e)); })
        .def("GetMaximumLikelihoodNbGhostAtom",
             [](ScatteringPowerAtom& sp) { return (double)sp.GetMaximumLikelihoodNbGhostAtom(); })
        .def("SetMaximumLikelihoodNbGhostAtom",
             [](ScatteringPowerAtom& sp, double n) { sp.SetMaximumLikelihoodNbGhostAtom(static_cast<REAL>(n)); })
        .def("GetFormalCharge", [](ScatteringPowerAtom& sp) { return (double)sp.GetFormalCharge(); })
        .def("SetFormalCharge", [](ScatteringPowerAtom& sp, double c) { sp.SetFormalCharge(static_cast<REAL>(c)); })
        // RefinableObj methods (virtual base — access via ScatteringPowerAtom directly)
        .def("GetName",     [](ScatteringPowerAtom& sp) -> std::string { return std::string(sp.GetName()); })
        .def("SetName",     [](ScatteringPowerAtom& sp, const std::string& n) { static_cast<RefinableObj&>(sp).SetName(n); })
        .def("GetClassName",[](ScatteringPowerAtom& sp) -> std::string { return std::string(sp.GetClassName()); })
        .def("Print",       [](ScatteringPowerAtom& sp) { sp.Print(); })
        .def("GetNbPar",    [](ScatteringPowerAtom& sp) { return sp.GetNbPar(); })
        .def("GetPar",      [](ScatteringPowerAtom& sp, const std::string& n) -> RefinablePar& {
                                return sp.GetPar(n); }, nb::rv_policy::reference_internal)
        .def("GetPar",      [](ScatteringPowerAtom& sp, long i) -> RefinablePar& {
                                return sp.GetPar(i); }, nb::rv_policy::reference_internal)
        .def("__str__",     [](ScatteringPowerAtom& sp) -> std::string { return std::string(sp.GetName()); })
        .def_prop_rw("Biso",
             [](ScatteringPowerAtom& sp) { return (double)sp.GetBiso(); },
             [](ScatteringPowerAtom& sp, double b) { sp.SetBiso(static_cast<REAL>(b)); })
        .def_prop_rw("B11", &_GetBijSPA<1,1>, &_SetBijSPA<1,1>)
        .def_prop_rw("B22", &_GetBijSPA<2,2>, &_SetBijSPA<2,2>)
        .def_prop_rw("B33", &_GetBijSPA<3,3>, &_SetBijSPA<3,3>)
        .def_prop_rw("B12", &_GetBijSPA<1,2>, &_SetBijSPA<1,2>)
        .def_prop_rw("B13", &_GetBijSPA<1,3>, &_SetBijSPA<1,3>)
        .def_prop_rw("B23", &_GetBijSPA<2,3>, &_SetBijSPA<2,3>)
        // Additional ScatteringPower methods needed by tests
        .def("IsIsotropic",     [](ScatteringPowerAtom& sp) { return sp.IsIsotropic(); })
        .def("IsScatteringFactorAnisotropic", [](ScatteringPowerAtom& sp) { return sp.IsScatteringFactorAnisotropic(); })
        .def("IsTemperatureFactorAnisotropic", [](ScatteringPowerAtom& sp) { return sp.IsTemperatureFactorAnisotropic(); })
        ;

    // ScatteringPowerAtom inherits ScatteringPower *virtually* too (see the
    // NOTE above), so -- unlike ScatteringPowerSphere -- it can't ride on
    // ScatteringPower's nanobind base for either RefinableObj's or
    // ScatteringPower's own method surface; both need to be forwarded
    // directly. Most of ScatteringPower's own methods were already covered
    // one-off above (some already went through this file specifically
    // because they need REAL<->double conversions ScatteringPower's own
    // binding doesn't need); bind_scatteringpower_forwarding fills in the
    // rest (GetScatteringFactor, GetTemperatureFactor,
    // GetResonantScattFactReal/Imag, GetDynPopCorrIndex,
    // GetNbScatteringPower, GetLastChangeClock, GetMaximumLikelihoodParClock,
    // GetColour(RGB)/SetColour, generic GetBij/SetBij, ...); any name
    // already bound above simply keeps winning (first registered wins).
    bind_scatteringpower_forwarding(cls);
    bind_refinableobj_forwarding(cls);
}
