/*
 * pyobjcryst nanobind port — Quaternion bindings
 */

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>

#undef B0
#include <ObjCryst/ObjCryst/General.h>
#include <ObjCryst/ObjCryst/Molecule.h>

namespace nb = nanobind;
using namespace ObjCryst;

namespace {

nb::tuple _RotateVector(const Quaternion& q, double v1, double v2, double v3)
{
    REAL rv1 = v1, rv2 = v2, rv3 = v3;
    q.RotateVector(rv1, rv2, rv3);
    return nb::make_tuple((double)rv1, (double)rv2, (double)rv3);
}

} // namespace

void wrap_quaternion(nb::module_& m)
{
    nb::class_<Quaternion>(m, "Quaternion")
        .def(nb::init<const REAL, const REAL, const REAL, const REAL, bool>(),
             nb::arg("q0"), nb::arg("q1"), nb::arg("q2"), nb::arg("q3"),
             nb::arg("unit") = true)
        .def("GetConjugate",   &Quaternion::GetConjugate)
        .def("RotateVector",   &_RotateVector,
             nb::arg("v1"), nb::arg("v2"), nb::arg("v3"))
        .def("Normalize",      &Quaternion::Normalize)
        .def("GetNorm",        &Quaternion::GetNorm)
        .def_static("RotationQuaternion", &Quaternion::RotationQuaternion,
             nb::arg("ang"), nb::arg("v1"), nb::arg("v2"), nb::arg("v3"))
        .def_prop_rw("Q0",
            [](const Quaternion& q){ return (double)q.Q0(); },
            [](Quaternion& q, double v){ q.Q0() = static_cast<REAL>(v); })
        .def_prop_rw("Q1",
            [](const Quaternion& q){ return (double)q.Q1(); },
            [](Quaternion& q, double v){ q.Q1() = static_cast<REAL>(v); })
        .def_prop_rw("Q2",
            [](const Quaternion& q){ return (double)q.Q2(); },
            [](Quaternion& q, double v){ q.Q2() = static_cast<REAL>(v); })
        .def_prop_rw("Q3",
            [](const Quaternion& q){ return (double)q.Q3(); },
            [](Quaternion& q, double v){ q.Q3() = static_cast<REAL>(v); })
        .def("__mul__",  [](const Quaternion& a, const Quaternion& b){ return a * b; })
        .def("__imul__", [](Quaternion& a, const Quaternion& b) -> Quaternion& { a *= b; return a; })
        ;
}
