#include <nanobind/nanobind.h>
#include <nanobind/make_iterator.h>
#include <set>
#include <ObjCryst/ObjCryst/Molecule.h>
#include "helpers_nb.hpp"

namespace nb = nanobind;
using namespace ObjCryst;

// RigidGroup is a typedef for std::set<MolAtom*>. The legacy Boost.Python
// binding registered RigidGroup with `bases<MolAtomSet>` so that it
// inherited the full Python-set-like interface (add/discard/remove/update/
// clear/__contains__/__getitem__/__len__) from a generic MolAtomSet wrapper
// -- see the module docstring in pyobjcryst/molecule.py: "RigidGroup is
// wrapped to have python-set methods rather than stl::set methods." nanobind
// has no equivalent to bases<> mixing in a second wrapped C++ base that
// isn't in RigidGroup's real inheritance chain, so that interface has to be
// bound directly here instead. (__iter__ is a new addition -- Boost's
// MolAtomSet never defined it either, relying on the legacy __getitem__
// fallback iteration protocol -- but nanobind classes don't get that
// fallback, so without an explicit __iter__ a RigidGroup wouldn't even
// support `for atom in group:`.)
namespace {

void _addRG(RigidGroup& rg, MolAtom& a)
{
    rg.insert(&a);
}

void _updateRG(RigidGroup& rg, nb::object other)
{
    std::set<MolAtom*> s = pyIterableToSet<MolAtom>(other);
    rg.insert(s.begin(), s.end());
}

bool _containsRG(const RigidGroup& rg, MolAtom& a)
{
    return rg.find(&a) != rg.end();
}

MolAtom& _getItemRG(const RigidGroup& rg, size_t i)
{
    if (i >= rg.size())
        throw nb::index_error("index out of range");
    RigidGroup::const_iterator p = rg.begin();
    std::advance(p, i);
    return **p;
}

void _discardRG(RigidGroup& rg, MolAtom& a)
{
    rg.erase(&a);
}

void _removeRG(RigidGroup& rg, MolAtom& a)
{
    auto it = rg.find(&a);
    if (it == rg.end())
        throw nb::key_error("KeyError");
    rg.erase(it);
}

} // namespace

void wrap_rigidgroup(nb::module_& m)
{
    nb::class_<RigidGroup>(m, "RigidGroup")
        .def(nb::init<>())
        .def(nb::init<const RigidGroup&>())
        .def("GetName", &RigidGroup::GetName)
        .def("int_ptr", &RigidGroup::int_ptr)
        // Python-set-like interface (faithful to the legacy MolAtomSet
        // mixin -- see the file-level comment above).
        .def("add", &_addRG, nb::keep_alive<1,2>())
        .def("clear", [](RigidGroup& rg){ rg.clear(); })
        .def("discard", &_discardRG)
        .def("remove", &_removeRG)
        .def("update", &_updateRG, nb::keep_alive<1,2>())
        .def("__contains__", &_containsRG)
        .def("__getitem__", &_getItemRG, nb::rv_policy::reference_internal)
        .def("__len__", [](const RigidGroup& rg){ return rg.size(); })
        .def("__iter__", [](RigidGroup& rg){
            return nb::make_iterator(nb::find(nb::type<RigidGroup>()),
                                      "rigidgroup_iter", rg.begin(), rg.end());
        }, nb::keep_alive<0,1>())
        ;
}
