/*
 * pyobjcryst nanobind port — ScatteringComponent bindings
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <ObjCryst/ObjCryst/ScatteringPower.h>

#include "helpers_nb.hpp"

namespace nb = nanobind;
using namespace ObjCryst;

namespace {

// Return the scattering power as the correct concrete type pointer
nb::object getScatteringPowerPython(ScatteringComponent& s, nb::handle parent)
{
    const ScatteringPower* sp = s.mpScattPow;
    if (!sp) return nb::none();
    // Try ScatteringPowerAtom first
    const ScatteringPowerAtom* spa = dynamic_cast<const ScatteringPowerAtom*>(sp);
    if (spa) {
        return nb::cast(const_cast<ScatteringPowerAtom*>(spa),
                        nb::rv_policy::reference_internal, parent);
    }
    // Fall back to ScatteringPower
    return nb::cast(const_cast<ScatteringPower*>(sp),
                    nb::rv_policy::reference_internal, parent);
}

} // namespace

void wrap_scatteringcomponent(nb::module_& m)
{
    nb::class_<ScatteringComponent>(m, "ScatteringComponent")
        .def(nb::init<>())
        .def("Print",         &ScatteringComponent::Print)
        .def_rw("mX",         &ScatteringComponent::mX)
        .def_rw("X",          &ScatteringComponent::mX)
        .def_rw("mY",         &ScatteringComponent::mY)
        .def_rw("Y",          &ScatteringComponent::mY)
        .def_rw("mZ",         &ScatteringComponent::mZ)
        .def_rw("Z",          &ScatteringComponent::mZ)
        .def_rw("mOccupancy", &ScatteringComponent::mOccupancy)
        .def_rw("Occupancy",  &ScatteringComponent::mOccupancy)
        .def_ro("mDynPopCorr",&ScatteringComponent::mDynPopCorr)
        .def_prop_ro("mpScattPow",
            [](nb::handle self) -> nb::object {
                ScatteringComponent& s = nb::cast<ScatteringComponent&>(self);
                return getScatteringPowerPython(s, self);
            })
        .def("__str__", [](const ScatteringComponent& s){ return obj_str(s); })
        ;
}
