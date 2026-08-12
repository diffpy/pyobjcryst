/*
 * pyobjcryst nanobind port — Atom bindings
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <ObjCryst/ObjCryst/Atom.h>
#include <ObjCryst/ObjCryst/ScatteringPower.h>
#include <ObjCryst/RefinableObj/RefinableObj.h>

namespace nb = nanobind;
using namespace ObjCryst;

namespace {

// Wrap a (possibly null) ScatteringPower* for return to Python. Its dynamic
// type is almost always ScatteringPowerAtom, which inherits VIRTUALLY from
// ScatteringPower and is therefore intentionally registered in nanobind
// WITHOUT a declared base (see nb_scatteringpoweratom.cpp) -- nanobind's
// automatic base<->derived pointer adjustment assumes a fixed, statically
// computable offset, which virtual inheritance does not provide. Detect the
// concrete type with a real C++ dynamic_cast (correct for virtual
// inheritance) before handing nanobind an already-correctly-typed pointer.
// See the identical fix in nb_crystal.cpp::wrapScatteringPowerPython for the
// segfault this avoids.
nb::object getScatteringPowerPython(nb::handle self)
{
    const Atom& a = nb::cast<const Atom&>(self);
    if (a.IsDummy()) return nb::none();
    const ScatteringPower& sp = a.GetScatteringPower();
    if (auto* spa = dynamic_cast<const ScatteringPowerAtom*>(&sp))
        return nb::cast(const_cast<ScatteringPowerAtom*>(spa), nb::rv_policy::reference_internal, self);
    return nb::cast(const_cast<ScatteringPower*>(&sp), nb::rv_policy::reference_internal, self);
}

// Helper to get ScatteringPower* from various Python types (ScatteringPowerAtom, etc.)
// Since these classes may use virtual inheritance, we extract the pointer via the concrete type
const ScatteringPower* getScatteringPowerPtr(nb::object obj)
{
    // Handle None -> nullptr (dummy atom)
    if (obj.is_none()) return nullptr;
    // Try ScatteringPowerAtom first (most common)
    try {
        return static_cast<const ScatteringPower*>(&nb::cast<ScatteringPowerAtom&>(obj));
    } catch (...) {}
    // Try ScatteringPower directly
    return &nb::cast<const ScatteringPower&>(obj);
}

} // namespace

void wrap_atom(nb::module_& m)
{
    nb::class_<Atom, Scatterer>(m, "Atom", nb::is_final())
        .def("__init__",
             [](Atom* self, const Atom& old) {
                 new (self) Atom(old);
             }, nb::arg("old"))
        .def("__init__",
             [](Atom* self, REAL x, REAL y, REAL z,
                const std::string& name, nb::object pow, REAL popu) {
                 const ScatteringPower* sp = getScatteringPowerPtr(pow);
                 new (self) Atom(x, y, z, name, sp, popu);
             },
             nb::arg("x"), nb::arg("y"), nb::arg("z"), nb::arg("name"),
             nb::arg("pow").none(), nb::arg("popu") = (REAL)1.0)
        .def("Init",
             [](Atom& self, REAL x, REAL y, REAL z,
                const std::string& name, nb::object pow, REAL popu) {
                 const ScatteringPower* sp = getScatteringPowerPtr(pow);
                 self.Init(x, y, z, name, sp, popu);
             },
             nb::arg("x"), nb::arg("y"), nb::arg("z"), nb::arg("name"),
             nb::arg("pow").none(), nb::arg("popu") = (REAL)1.0)
        .def("GetMass",   &Atom::GetMass)
        .def("GetRadius", &Atom::GetRadius)
        .def("IsDummy",   &Atom::IsDummy)
        .def("GetScatteringPower", &getScatteringPowerPython)
        // GetName and other RefinableObj methods via Scatterer
        .def("GetName",     [](Atom& a) -> std::string { return std::string(a.GetName()); })
        .def("SetName",     [](Atom& a, const std::string& n) {
                                static_cast<RefinableObj&>(a).SetName(n); })
        ;
}
