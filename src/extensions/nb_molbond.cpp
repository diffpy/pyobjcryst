#include <nanobind/nanobind.h>
#include <ObjCryst/ObjCryst/Molecule.h>
#include "helpers_nb.hpp"

namespace nb = nanobind;
using namespace ObjCryst;

namespace {
MolAtom& _GetAtom(MolBond& mb, size_t i)
{
    switch(i) {
        case 0: return mb.GetAtom1();
        case 1: return mb.GetAtom2();
        default: throw nb::index_error("Index out of range");
    }
}
} // namespace

void wrap_molbond(nb::module_& m)
{
    nb::class_<MolBond, Restraint>(m, "MolBond")
        .def("GetMolecule", nb::overload_cast<>(&MolBond::GetMolecule),
             nb::rv_policy::reference_internal)
        .def("GetLogLikelihood", nb::overload_cast<>(&MolBond::GetLogLikelihood, nb::const_))
        .def("GetLogLikelihood",
             nb::overload_cast<const bool, const bool>(&MolBond::GetLogLikelihood, nb::const_))
        .def("GetName", &MolBond::GetName)
        .def("GetAtom1", nb::overload_cast<>(&MolBond::GetAtom1),
             nb::rv_policy::reference_internal)
        .def("GetAtom2", nb::overload_cast<>(&MolBond::GetAtom2),
             nb::rv_policy::reference_internal)
        .def("SetAtom1", &MolBond::SetAtom1, nb::keep_alive<1,2>())
        .def("SetAtom2", &MolBond::SetAtom2, nb::keep_alive<1,2>())
        .def("GetLength", &MolBond::GetLength)
        .def("GetLength0", &MolBond::GetLength0)
        .def("GetLengthDelta", &MolBond::GetLengthDelta)
        .def("GetLengthSigma", &MolBond::GetLengthSigma)
        .def("GetBondOrder", &MolBond::GetBondOrder)
        .def("SetLength0", &MolBond::SetLength0)
        .def("SetLengthDelta", &MolBond::SetLengthDelta)
        .def("SetLengthSigma", &MolBond::SetLengthSigma)
        .def("SetBondOrder", &MolBond::SetBondOrder)
        .def("IsFreeTorsion", &MolBond::IsFreeTorsion)
        .def("SetFreeTorsion", &MolBond::SetFreeTorsion)
        .def_prop_ro("Length", &MolBond::GetLength)
        .def_prop_rw("Length0", &MolBond::GetLength0, &MolBond::SetLength0)
        .def_prop_rw("LengthDelta", &MolBond::GetLengthDelta, &MolBond::SetLengthDelta)
        .def_prop_rw("LengthSigma", &MolBond::GetLengthSigma, &MolBond::SetLengthSigma)
        .def_prop_rw("BondOrder", &MolBond::GetBondOrder, &MolBond::SetBondOrder)
        .def("__getitem__", &_GetAtom, nb::rv_policy::reference_internal)
        .def("int_ptr", &MolBond::int_ptr)
        ;
}
