#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <sstream>
#include <ObjCryst/ObjCryst/Molecule.h>
#include <ObjCryst/ObjCryst/ScatteringPower.h>
#include "helpers_nb.hpp"

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
    const MolAtom& a = nb::cast<const MolAtom&>(self);
    if (a.IsDummy()) return nb::none();
    const ScatteringPower& sp = a.GetScatteringPower();
    if (auto* spa = dynamic_cast<const ScatteringPowerAtom*>(&sp))
        return nb::cast(const_cast<ScatteringPowerAtom*>(spa), nb::rv_policy::reference_internal, self);
    return nb::cast(const_cast<ScatteringPower*>(&sp), nb::rv_policy::reference_internal, self);
}
std::string __str__(MolAtom& a)
{
    std::ostringstream s;
    s << a.GetName() << " " << a.GetX() << " " << a.GetY() << " " << a.GetZ();
    return s.str();
}
} // namespace

void wrap_molatom(nb::module_& m)
{
    nb::class_<MolAtom>(m, "MolAtom")
        .def(nb::init<const MolAtom&>())
        .def("GetName", nb::overload_cast<>(&MolAtom::GetName, nb::const_))
        .def("SetName", &MolAtom::SetName)
        .def("GetMolecule", nb::overload_cast<>(&MolAtom::GetMolecule),
             nb::rv_policy::reference_internal)
        .def("GetX", &MolAtom::GetX)
        .def("GetY", &MolAtom::GetY)
        .def("GetZ", &MolAtom::GetZ)
        .def("GetOccupancy", &MolAtom::GetOccupancy)
        .def("SetX", &MolAtom::SetX)
        .def("SetY", &MolAtom::SetY)
        .def("SetZ", &MolAtom::SetZ)
        .def("SetOccupancy", &MolAtom::SetOccupancy)
        .def("IsDummy", &MolAtom::IsDummy)
        .def("GetScatteringPower", &getScatteringPowerPython)
        .def("SetIsInRing", &MolAtom::SetIsInRing)
        .def("IsInRing", &MolAtom::IsInRing)
        .def_prop_rw("X", &MolAtom::GetX, &MolAtom::SetX)
        .def_prop_rw("Y", &MolAtom::GetY, &MolAtom::SetY)
        .def_prop_rw("Z", &MolAtom::GetZ, &MolAtom::SetZ)
        .def_prop_rw("Occupancy", &MolAtom::GetOccupancy, &MolAtom::SetOccupancy)
        .def_prop_rw("x", &MolAtom::GetX, &MolAtom::SetX)
        .def_prop_rw("y", &MolAtom::GetY, &MolAtom::SetY)
        .def_prop_rw("z", &MolAtom::GetZ, &MolAtom::SetZ)
        .def_prop_rw("occ", &MolAtom::GetOccupancy, &MolAtom::SetOccupancy)
        .def("__str__", &__str__)
        .def("int_ptr", &MolAtom::int_ptr)
        ;
}
