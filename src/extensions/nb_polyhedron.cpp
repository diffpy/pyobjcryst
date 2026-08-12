#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <ObjCryst/ObjCryst/Polyhedron.h>
#include <ObjCryst/ObjCryst/ScatteringPower.h>
#include "helpers_nb.hpp"

namespace nb = nanobind;
using namespace ObjCryst;

namespace {

// All Make*() factory functions below take `const ScatteringPower*`
// arguments, but the values passed from Python are almost always
// ScatteringPowerAtom instances. ScatteringPowerAtom inherits VIRTUALLY from
// ScatteringPower (see ScatteringPower.h) and is therefore intentionally
// registered in nanobind WITHOUT a declared base (see
// nb_scatteringpoweratom.cpp) -- nanobind's automatic base<->derived pointer
// adjustment assumes a fixed, statically computable offset, which virtual
// inheritance does not provide. Left as a plain `const ScatteringPower*`
// parameter, nanobind refuses to accept a Python ScatteringPowerAtom
// instance there at all (TypeError). Resolve it manually: try the concrete
// registered type first (exact match, no offset needed), falling back to
// ScatteringPower directly. Mirrors getScatteringPowerPtr in nb_atom.cpp /
// extractScatteringPowerArg in nb_molecule.cpp.
const ScatteringPower* extractScatteringPowerArg(nb::object obj)
{
    if (obj.is_none()) return nullptr;
    try {
        return static_cast<const ScatteringPower*>(&nb::cast<ScatteringPowerAtom&>(obj));
    } catch (...) {}
    return &nb::cast<const ScatteringPower&>(obj);
}

Molecule* _MakeTetrahedron(Crystal& cryst, const std::string& name,
                            nb::object centralAtom, nb::object peripheralAtom, const REAL dist)
{
    return MakeTetrahedron(cryst, name, extractScatteringPowerArg(centralAtom),
                            extractScatteringPowerArg(peripheralAtom), dist);
}
Molecule* _MakeOctahedron(Crystal& cryst, const std::string& name,
                          nb::object centralAtom, nb::object peripheralAtom, const REAL dist)
{
    return MakeOctahedron(cryst, name, extractScatteringPowerArg(centralAtom),
                           extractScatteringPowerArg(peripheralAtom), dist);
}
Molecule* _MakeSquarePlane(Crystal& cryst, const std::string& name,
                           nb::object centralAtom, nb::object peripheralAtom, const REAL dist)
{
    return MakeSquarePlane(cryst, name, extractScatteringPowerArg(centralAtom),
                            extractScatteringPowerArg(peripheralAtom), dist);
}
Molecule* _MakeCube(Crystal& cryst, const std::string& name,
                    nb::object centralAtom, nb::object peripheralAtom, const REAL dist)
{
    return MakeCube(cryst, name, extractScatteringPowerArg(centralAtom),
                     extractScatteringPowerArg(peripheralAtom), dist);
}
Molecule* _MakeAntiPrismTetragonal(Crystal& cryst, const std::string& name,
                                   nb::object centralAtom, nb::object peripheralAtom, const REAL dist)
{
    return MakeAntiPrismTetragonal(cryst, name, extractScatteringPowerArg(centralAtom),
                                    extractScatteringPowerArg(peripheralAtom), dist);
}
Molecule* _MakePrismTrigonal(Crystal& cryst, const std::string& name,
                             nb::object centralAtom, nb::object peripheralAtom, const REAL dist)
{
    return MakePrismTrigonal(cryst, name, extractScatteringPowerArg(centralAtom),
                              extractScatteringPowerArg(peripheralAtom), dist);
}
Molecule* _MakeIcosahedron(Crystal& cryst, const std::string& name,
                          nb::object centralAtom, nb::object peripheralAtom, const REAL dist)
{
    return MakeIcosahedron(cryst, name, extractScatteringPowerArg(centralAtom),
                            extractScatteringPowerArg(peripheralAtom), dist);
}
Molecule* _MakeTriangle(Crystal& cryst, const std::string& name,
                        nb::object centralAtom, nb::object peripheralAtom, const REAL dist)
{
    return MakeTriangle(cryst, name, extractScatteringPowerArg(centralAtom),
                         extractScatteringPowerArg(peripheralAtom), dist);
}

} // namespace

void wrap_polyhedron(nb::module_& m)
{
    m.def("MakeTetrahedron",    &_MakeTetrahedron,
          nb::arg("cryst"), nb::arg("name"), nb::arg("centralAtom").none(),
          nb::arg("peripheralAtom").none(), nb::arg("dist"),
          nb::rv_policy::take_ownership);

    m.def("MakeOctahedron",     &_MakeOctahedron,
          nb::arg("cryst"), nb::arg("name"), nb::arg("centralAtom").none(),
          nb::arg("peripheralAtom").none(), nb::arg("dist"),
          nb::rv_policy::take_ownership);

    m.def("MakeSquarePlane",    &_MakeSquarePlane,
          nb::arg("cryst"), nb::arg("name"), nb::arg("centralAtom").none(),
          nb::arg("peripheralAtom").none(), nb::arg("dist"),
          nb::rv_policy::take_ownership);

    m.def("MakeCube",           &_MakeCube,
          nb::arg("cryst"), nb::arg("name"), nb::arg("centralAtom").none(),
          nb::arg("peripheralAtom").none(), nb::arg("dist"),
          nb::rv_policy::take_ownership);

    m.def("MakeAntiPrismTetragonal", &_MakeAntiPrismTetragonal,
          nb::arg("cryst"), nb::arg("name"), nb::arg("centralAtom").none(),
          nb::arg("peripheralAtom").none(), nb::arg("dist"),
          nb::rv_policy::take_ownership);

    m.def("MakePrismTrigonal",  &_MakePrismTrigonal,
          nb::arg("cryst"), nb::arg("name"), nb::arg("centralAtom").none(),
          nb::arg("peripheralAtom").none(), nb::arg("dist"),
          nb::rv_policy::take_ownership);

    m.def("MakeIcosahedron",    &_MakeIcosahedron,
          nb::arg("cryst"), nb::arg("name"), nb::arg("centralAtom").none(),
          nb::arg("peripheralAtom").none(), nb::arg("dist"),
          nb::rv_policy::take_ownership);

    m.def("MakeTriangle",       &_MakeTriangle,
          nb::arg("cryst"), nb::arg("name"), nb::arg("centralAtom").none(),
          nb::arg("peripheralAtom").none(), nb::arg("dist"),
          nb::rv_policy::take_ownership);
}
