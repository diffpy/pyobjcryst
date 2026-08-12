#include <nanobind/nanobind.h>
#include <nanobind/trampoline.h>
#include <set>
#include <ObjCryst/ObjCryst/Molecule.h>
#include "helpers_nb.hpp"

namespace nb = nanobind;
using namespace ObjCryst;

typedef std::set<MolAtom*> MolAtomSet;

struct PyStretchMode : StretchMode {
    NB_TRAMPOLINE(StretchMode, 3);
    void CalcDeriv(const bool derivllk=true) const override
    { NB_OVERRIDE_PURE(CalcDeriv, derivllk); }
    void Stretch(const REAL change, const bool keepCenter=true) override
    { NB_OVERRIDE_PURE(Stretch, change, keepCenter); }
    void RandomStretch(const REAL amplitude, const bool keepCenter=true) override
    { NB_OVERRIDE_PURE(RandomStretch, amplitude, keepCenter); }
};

namespace {

// StretchModeBondLength atom helpers
void _AddAtomSMBL(StretchModeBondLength& mode, MolAtom& a)
{ mode.mvTranslatedAtomList.insert(&a); }

void _AddAtomsSMBL(StretchModeBondLength& mode, nb::object& l)
{
    for (auto h : l) mode.mvTranslatedAtomList.insert(&nb::cast<MolAtom&>(h));
}

nb::list _GetAtomsSMBL(StretchModeBondLength& mode)
{
    return ptrcontainerToPyList<MolAtomSet>(mode.mvTranslatedAtomList);
}

MolAtom* _GetAtom0BL(StretchModeBondLength& m) { return m.mpAtom0; }
MolAtom* _GetAtom1BL(StretchModeBondLength& m) { return m.mpAtom1; }

// Angle/torsion mode helpers
template<class T>
void _AddAtom(T& mode, MolAtom& a) { mode.mvRotatedAtomList.insert(&a); }

template<class T>
void _AddAtoms(T& mode, nb::object& l)
{ for (auto h : l) mode.mvRotatedAtomList.insert(&nb::cast<MolAtom&>(h)); }

template<class T>
nb::list _GetAtoms(T& mode)
{ return ptrcontainerToPyList<MolAtomSet>(mode.mvRotatedAtomList); }

template<class T> MolAtom* _GetAtom0(T& m) { return m.mpAtom0; }
template<class T> MolAtom* _GetAtom1(T& m) { return m.mpAtom1; }
template<class T> MolAtom* _GetAtom2(T& m) { return m.mpAtom2; }

} // namespace

void wrap_stretchmode(nb::module_& m)
{
    nb::class_<StretchMode, PyStretchMode>(m, "StretchMode")
        .def("CalcDeriv", &StretchMode::CalcDeriv, nb::arg("derivllk") = true)
        .def("Stretch", &StretchMode::Stretch, nb::arg("amplitude"), nb::arg("keepCenter") = true)
        .def("RandomStretch", &StretchMode::RandomStretch, nb::arg("amplitude"), nb::arg("keepCenter") = true)
        ;

    nb::class_<StretchModeBondLength, StretchMode>(m, "StretchModeBondLength")
        .def(nb::init<MolAtom&, MolAtom&, MolBond*>(),
             nb::arg("at0"), nb::arg("at1"), nb::arg("pBond") = nullptr,
             nb::keep_alive<1,2>(), nb::keep_alive<1,3>())
        .def("AddAtom", &_AddAtomSMBL, nb::keep_alive<1,2>())
        .def("AddAtoms", &_AddAtomsSMBL, nb::keep_alive<1,2>())
        .def("GetAtoms", &_GetAtomsSMBL)
        .def_prop_ro("mpAtom0",
            &_GetAtom0BL, nb::rv_policy::reference_internal)
        .def_prop_ro("mpAtom1",
            &_GetAtom1BL, nb::rv_policy::reference_internal)
        ;

    nb::class_<StretchModeBondAngle, StretchMode>(m, "StretchModeBondAngle")
        .def(nb::init<MolAtom&, MolAtom&, MolAtom&, MolBondAngle*>(),
             nb::arg("at0"), nb::arg("at1"), nb::arg("at2"), nb::arg("pBondAngle") = nullptr,
             nb::keep_alive<1,2>(), nb::keep_alive<1,3>(), nb::keep_alive<1,4>())
        .def("AddAtom", &_AddAtom<StretchModeBondAngle>, nb::keep_alive<1,2>())
        .def("AddAtoms", &_AddAtoms<StretchModeBondAngle>, nb::keep_alive<1,2>())
        .def("GetAtoms", &_GetAtoms<StretchModeBondAngle>)
        .def_prop_ro("mpAtom0",
            [](StretchModeBondAngle& m){ return m.mpAtom0; }, nb::rv_policy::reference_internal)
        .def_prop_ro("mpAtom1",
            [](StretchModeBondAngle& m){ return m.mpAtom1; }, nb::rv_policy::reference_internal)
        .def_prop_ro("mpAtom2",
            [](StretchModeBondAngle& m){ return m.mpAtom2; }, nb::rv_policy::reference_internal)
        ;

    nb::class_<StretchModeTorsion, StretchMode>(m, "StretchModeTorsion")
        .def(nb::init<MolAtom&, MolAtom&, MolDihedralAngle*>(),
             nb::arg("at0"), nb::arg("at1"), nb::arg("pDihedralAngle") = nullptr,
             nb::keep_alive<1,2>(), nb::keep_alive<1,3>())
        .def("AddAtom", &_AddAtom<StretchModeTorsion>, nb::keep_alive<1,2>())
        .def("AddAtoms", &_AddAtoms<StretchModeTorsion>, nb::keep_alive<1,2>())
        .def("GetAtoms", &_GetAtoms<StretchModeTorsion>)
        .def_prop_ro("mpAtom1",
            [](StretchModeTorsion& m){ return m.mpAtom1; }, nb::rv_policy::reference_internal)
        .def_prop_ro("mpAtom2",
            [](StretchModeTorsion& m){ return m.mpAtom2; }, nb::rv_policy::reference_internal)
        ;

    nb::class_<StretchModeTwist, StretchMode>(m, "StretchModeTwist")
        .def(nb::init<MolAtom&, MolAtom&>(),
             nb::arg("at0"), nb::arg("at1"),
             nb::keep_alive<1,2>(), nb::keep_alive<1,3>())
        .def("AddAtom", &_AddAtom<StretchModeTwist>, nb::keep_alive<1,2>())
        .def("AddAtoms", &_AddAtoms<StretchModeTwist>, nb::keep_alive<1,2>())
        .def("GetAtoms", &_GetAtoms<StretchModeTwist>)
        .def_prop_ro("mpAtom1",
            [](StretchModeTwist& m){ return m.mpAtom1; }, nb::rv_policy::reference_internal)
        .def_prop_ro("mpAtom2",
            [](StretchModeTwist& m){ return m.mpAtom2; }, nb::rv_policy::reference_internal)
        ;
}
