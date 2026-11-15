#include <nanobind/nanobind.h>
#include <nanobind/make_iterator.h>
#include <ObjCryst/ObjCryst/Molecule.h>
#include "helpers_nb.hpp"

namespace nb = nanobind;
using namespace ObjCryst;

namespace {
MolAtom& _GetAtom(MolBondAngle& mb, size_t i)
{
    switch(i) {
        case 0: return mb.GetAtom1();
        case 1: return mb.GetAtom2();
        case 2: return mb.GetAtom3();
        default: throw nb::index_error("Index out of range");
    }
}
} // namespace

void wrap_molbondangle(nb::module_& m)
{
    nb::class_<MolBondAngle, Restraint>(m, "MolBondAngle")
        .def("GetMolecule", nb::overload_cast<>(&MolBondAngle::GetMolecule),
             nb::rv_policy::reference_internal)
        .def("GetName", &MolBondAngle::GetName)
        .def("GetLogLikelihood", nb::overload_cast<>(&MolBondAngle::GetLogLikelihood, nb::const_))
        .def("GetLogLikelihood",
             nb::overload_cast<const bool, const bool>(&MolBondAngle::GetLogLikelihood, nb::const_))
        .def("GetAngle",      &MolBondAngle::GetAngle)
        .def("GetAngle0",     &MolBondAngle::GetAngle0)
        .def("GetAngleDelta", &MolBondAngle::GetAngleDelta)
        .def("GetAngleSigma", &MolBondAngle::GetAngleSigma)
        .def("SetAngle0",     &MolBondAngle::SetAngle0)
        .def("SetAngleDelta", &MolBondAngle::SetAngleDelta)
        .def("SetAngleSigma", &MolBondAngle::SetAngleSigma)
        .def("GetAtom1", nb::overload_cast<>(&MolBondAngle::GetAtom1),
             nb::rv_policy::reference_internal)
        .def("GetAtom2", nb::overload_cast<>(&MolBondAngle::GetAtom2),
             nb::rv_policy::reference_internal)
        .def("GetAtom3", nb::overload_cast<>(&MolBondAngle::GetAtom3),
             nb::rv_policy::reference_internal)
        .def("SetAtom1", &MolBondAngle::SetAtom1, nb::keep_alive<1,2>())
        .def("SetAtom2", &MolBondAngle::SetAtom2, nb::keep_alive<1,2>())
        .def("SetAtom3", &MolBondAngle::SetAtom3, nb::keep_alive<1,2>())
        .def_prop_ro("Angle", &MolBondAngle::GetAngle)
        .def_prop_rw("Angle0", &MolBondAngle::GetAngle0, &MolBondAngle::SetAngle0)
        .def_prop_rw("AngleDelta", &MolBondAngle::GetAngleDelta, &MolBondAngle::SetAngleDelta)
        .def_prop_rw("AngleSigma", &MolBondAngle::GetAngleSigma, &MolBondAngle::SetAngleSigma)
        .def("__getitem__", &_GetAtom, nb::rv_policy::reference_internal)
        // Iterate over the atoms involved (present in the legacy
        // Boost.Python binding via MolBondAngle::begin()/end()).
        .def("__iter__", [](MolBondAngle& mb){
            return nb::make_iterator(nb::find(nb::type<MolBondAngle>()),
                                      "molbondangle_atom_iter", mb.begin(), mb.end());
        }, nb::keep_alive<0,1>())
        .def("int_ptr", &MolBondAngle::int_ptr)
        ;
}
