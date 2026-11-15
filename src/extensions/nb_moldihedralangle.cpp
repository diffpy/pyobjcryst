#include <nanobind/nanobind.h>
#include <nanobind/make_iterator.h>
#include <ObjCryst/ObjCryst/Molecule.h>
#include "helpers_nb.hpp"

namespace nb = nanobind;
using namespace ObjCryst;

namespace {
MolAtom& _GetAtom(MolDihedralAngle& mb, size_t i)
{
    switch(i) {
        case 0: return mb.GetAtom1();
        case 1: return mb.GetAtom2();
        case 2: return mb.GetAtom3();
        case 3: return mb.GetAtom4();
        default: throw nb::index_error("Index out of range");
    }
}
} // namespace

void wrap_moldihedralangle(nb::module_& m)
{
    nb::class_<MolDihedralAngle, Restraint>(m, "MolDihedralAngle")
        .def("GetMolecule", nb::overload_cast<>(&MolDihedralAngle::GetMolecule),
             nb::rv_policy::reference_internal)
        .def("GetName", &MolDihedralAngle::GetName)
        .def("GetLogLikelihood", nb::overload_cast<>(&MolDihedralAngle::GetLogLikelihood, nb::const_))
        .def("GetLogLikelihood",
             nb::overload_cast<const bool, const bool>(&MolDihedralAngle::GetLogLikelihood, nb::const_))
        .def("GetAngle",      &MolDihedralAngle::GetAngle)
        .def("GetAngle0",     &MolDihedralAngle::GetAngle0)
        .def("GetAngleDelta", &MolDihedralAngle::GetAngleDelta)
        .def("GetAngleSigma", &MolDihedralAngle::GetAngleSigma)
        .def("SetAngle0",     &MolDihedralAngle::SetAngle0)
        .def("SetAngleDelta", &MolDihedralAngle::SetAngleDelta)
        .def("SetAngleSigma", &MolDihedralAngle::SetAngleSigma)
        .def("GetAtom1", nb::overload_cast<>(&MolDihedralAngle::GetAtom1),
             nb::rv_policy::reference_internal)
        .def("GetAtom2", nb::overload_cast<>(&MolDihedralAngle::GetAtom2),
             nb::rv_policy::reference_internal)
        .def("GetAtom3", nb::overload_cast<>(&MolDihedralAngle::GetAtom3),
             nb::rv_policy::reference_internal)
        .def("GetAtom4", nb::overload_cast<>(&MolDihedralAngle::GetAtom4),
             nb::rv_policy::reference_internal)
        .def("SetAtom1", &MolDihedralAngle::SetAtom1, nb::keep_alive<1,2>())
        .def("SetAtom2", &MolDihedralAngle::SetAtom2, nb::keep_alive<1,2>())
        .def("SetAtom3", &MolDihedralAngle::SetAtom3, nb::keep_alive<1,2>())
        .def("SetAtom4", &MolDihedralAngle::SetAtom4, nb::keep_alive<1,2>())
        .def_prop_ro("Angle", &MolDihedralAngle::GetAngle)
        .def_prop_rw("Angle0", &MolDihedralAngle::GetAngle0, &MolDihedralAngle::SetAngle0)
        .def_prop_rw("AngleDelta", &MolDihedralAngle::GetAngleDelta, &MolDihedralAngle::SetAngleDelta)
        .def_prop_rw("AngleSigma", &MolDihedralAngle::GetAngleSigma, &MolDihedralAngle::SetAngleSigma)
        .def("__getitem__", &_GetAtom, nb::rv_policy::reference_internal)
        // Iterate over the atoms involved (present in the legacy
        // Boost.Python binding via MolDihedralAngle::begin()/end()).
        .def("__iter__", [](MolDihedralAngle& mb){
            return nb::make_iterator(nb::find(nb::type<MolDihedralAngle>()),
                                      "moldihedralangle_atom_iter", mb.begin(), mb.end());
        }, nb::keep_alive<0,1>())
        .def("int_ptr", &MolDihedralAngle::int_ptr)
        ;
}
