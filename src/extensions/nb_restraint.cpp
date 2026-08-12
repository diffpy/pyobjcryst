/*
 * pyobjcryst nanobind port — Restraint and RefObjOpt bindings
 */

#include <nanobind/nanobind.h>
#include <nanobind/trampoline.h>
#include <nanobind/stl/string.h>

#include <ObjCryst/RefinableObj/RefinableObj.h>

namespace nb = nanobind;
using namespace ObjCryst;

// ---------------------------------------------------------------------------
// Trampoline for Restraint (virtual methods)
// ---------------------------------------------------------------------------
struct PyRestraint : Restraint {
    NB_TRAMPOLINE(Restraint, 3);

    const RefParType* GetType() const override {
        NB_OVERRIDE(GetType);
    }
    void SetType(const RefParType* type) override {
        NB_OVERRIDE(SetType, type);
    }
    REAL GetLogLikelihood() const override {
        NB_OVERRIDE(GetLogLikelihood);
    }
};

void wrap_restraint(nb::module_& m)
{
    nb::class_<Restraint, PyRestraint>(m, "Restraint")
        .def(nb::init<>())
        .def(nb::init<const RefParType*>(), nb::arg("type"),
             nb::keep_alive<1, 2>())
        .def("GetType", &Restraint::GetType, nb::rv_policy::reference_internal)
        .def("SetType", &Restraint::SetType, nb::arg("type"),
             nb::keep_alive<1, 2>())
        .def("GetLogLikelihood", &Restraint::GetLogLikelihood)
        ;
}
