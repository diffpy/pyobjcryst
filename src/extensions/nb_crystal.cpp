/*
 * pyobjcryst nanobind port — Crystal bindings
 */

#include <nanobind/nanobind.h>
#include <nanobind/trampoline.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/map.h>

#include <string>
#include <sstream>

#undef B0
#include <ObjCryst/ObjCryst/Crystal.h>
#include <ObjCryst/ObjCryst/Atom.h>
#include <ObjCryst/ObjCryst/Molecule.h>
#include <ObjCryst/ObjCryst/ScatteringPower.h>
#include <ObjCryst/ObjCryst/CIF.h>
#include <ObjCryst/ObjCryst/UnitCell.h>

#include "helpers_nb.hpp"

namespace nb = nanobind;
using namespace ObjCryst;

struct PyCrystal : Crystal {
    NB_TRAMPOLINE(Crystal, 1);
    void UpdateDisplay() const override { NB_OVERRIDE(UpdateDisplay); }
};

namespace {

// Crystal::GetScatteringPower() returns a ScatteringPower& whose dynamic
// type is almost always ScatteringPowerAtom. ScatteringPowerAtom inherits
// VIRTUALLY from ScatteringPower (see ScatteringPower.h), so nanobind's
// automatic base<->derived pointer-adjustment (used when it detects, via
// RTTI, that the returned object's most-derived registered type differs
// from the static return type) cannot compute a correct offset — it isn't
// a fixed, statically-known value for virtual bases. Left to its own
// devices, nanobind was producing a corrupted pointer for the
// ScatteringPowerAtom Python wrapper: reads/writes through it landed on
// unrelated memory, and calling any inherited method that touches the
// RefinableObjClock member (e.g. Biso's getter, which Clicks the clock)
// segfaulted walking a bogus mvParent tree (see
// tests/test_cif.py::test_TiO2_rutile_cif).
//
// The fix (same pattern already used in nb_scatteringcomponent.cpp's
// getScatteringPowerPython, and documented as "Pattern 4: Const Pointer
// Type Confusion" in nanobind_migration_notes.md): perform the downcast
// ourselves with a real C++ dynamic_cast (which correctly walks the
// object's actual vtable/RTTI at runtime, unlike nanobind's static offset
// table), and only then hand nanobind an already-correctly-typed pointer,
// so its automatic machinery has nothing left to compute.
nb::object wrapScatteringPowerPython(ScatteringPower& sp, nb::handle parent)
{
    if (auto* spa = dynamic_cast<ScatteringPowerAtom*>(&sp))
        return nb::cast(spa, nb::rv_policy::reference_internal, parent);
    return nb::cast(&sp, nb::rv_policy::reference_internal, parent);
}

} // namespace

namespace {

void _AddScatterer(Crystal& crystal, nb::handle obj)
{
    if (obj.is_none()) throw nb::value_error("Cannot add nonexistent Scatterer");
    Scatterer* scatt = nullptr;
    try {
        scatt = static_cast<Scatterer*>(&nb::cast<Atom&>(obj));
    } catch (...) {
        try {
            scatt = &nb::cast<Scatterer&>(obj);
        } catch (...) {
            throw nb::type_error("AddScatterer: argument must be a Scatterer instance");
        }
    }
    if (!scatt) throw nb::value_error("Cannot add nonexistent Scatterer");
    if (scatt->GetClassName() == "Atom") {
        Atom* pat = dynamic_cast<Atom*>(scatt);
        if (!pat->IsDummy()) {
            const ScatteringPower* psp = &pat->GetScatteringPower();
            if (crystal.GetScatteringPowerRegistry().Find(psp) < 0)
                throw ObjCrystException("The Atom's scattering power must be added to the Crystal first.");
        }
    } else if (scatt->GetClassName() == "Molecule") {
        Molecule* pm = dynamic_cast<Molecule*>(scatt);
        for (int i = 0; i < pm->GetNbComponent(); i++) {
            if (!pm->GetAtom(i).IsDummy()) {
                if (crystal.GetScatteringPowerRegistry().Find(&pm->GetAtom(i).GetScatteringPower()) < 0)
                    throw ObjCrystException("The Molecule scattering powers must be added to the Crystal first.");
            }
        }
    }
    crystal.AddScatterer(scatt);
}

void _RemoveScatterer(Crystal& crystal, nb::handle obj)
{
    if (obj.is_none()) throw nb::value_error("Cannot remove nonexistent Scatterer");
    Scatterer* scatt = nullptr;
    try {
        scatt = static_cast<Scatterer*>(&nb::cast<Atom&>(obj));
    } catch (...) {
        try {
            scatt = &nb::cast<Scatterer&>(obj);
        } catch (...) {
            throw nb::type_error("RemoveScatterer: argument must be a Scatterer instance");
        }
    }
    if (!scatt) throw nb::value_error("Cannot remove nonexistent Scatterer");
    crystal.RemoveScatterer(scatt, false);
}

Scatterer& _GetScattByIndex(Crystal& crystal, int idx)
{
    int i = check_index(idx, crystal.GetNbScatterer(), ALLOW_NEGATIVE);
    return crystal.GetScatt(i);
}

Scatterer& _GetScattByName(Crystal& crystal, const std::string& name)
{
    try {
        CaptureStdOut gag;
        return crystal.GetScatt(name);
    } catch (ObjCrystException&) {
        throw nb::value_error(("Invalid atom name: " + name).c_str());
    }
}

void _AddScatteringPower(Crystal& crystal, nb::handle obj)
{
    if (obj.is_none()) throw nb::value_error("Cannot add nonexistent ScatteringPower");
    // Accept ScatteringPowerAtom, ScatteringPowerSphere, or any ScatteringPower
    // Use explicit static_cast through the concrete type to handle virtual inheritance
    ScatteringPower* sp = nullptr;
    try { sp = static_cast<ScatteringPower*>(&nb::cast<ScatteringPowerAtom&>(obj)); goto found; }
    catch (...) {}
    try { sp = &nb::cast<ScatteringPower&>(obj); goto found; }
    catch (...) {}
    throw nb::type_error("AddScatteringPower: argument must be a ScatteringPower instance");
found:
    if (!sp) throw nb::value_error("Cannot add nonexistent ScatteringPower");
    crystal.AddScatteringPower(sp);
}

void _RemoveScatteringPower(Crystal& crystal, nb::handle obj)
{
    if (obj.is_none()) throw nb::value_error("Cannot remove nonexistent ScatteringPower");
    ScatteringPower* sp = nullptr;
    try { sp = static_cast<ScatteringPower*>(&nb::cast<ScatteringPowerAtom&>(obj)); goto found; }
    catch (...) {}
    try { sp = &nb::cast<ScatteringPower&>(obj); goto found; }
    catch (...) {}
    throw nb::type_error("RemoveScatteringPower: argument must be a ScatteringPower instance");
found:
    if (!sp) throw nb::value_error("Cannot remove nonexistent ScatteringPower");
    crystal.RemoveScatteringPower(sp, false);
}

nb::list _GetScatteringComponentList(Crystal& c)
{
    const ScatteringComponentList& scl = c.GetScatteringComponentList();
    nb::list l;
    for (int i = 0; i < scl.GetNbComponent(); ++i)
        l.append(nb::cast(scl(i)));
    return l;
}

std::string _CIF(const Crystal& c, double mindist)
{
    std::stringstream s;
    c.CIFOutput(s, mindist);
    return s.str();
}

void _CIFOutput(Crystal& c, nb::object output, double mindist)
{
    std::ostringstream os;
    c.CIFOutput(os, mindist);
    output.attr("write")(os.str());
}

void _ImportCrystalFromCIF(Crystal& cryst, nb::object input,
                           const bool oneScatteringPowerPerElement = false,
                           const bool connectAtoms = false)
{
    MuteObjCrystUserInfo muzzle;
    CaptureStdOut gag;
    std::string s = read_pyfile_to_string(input);
    std::istringstream in(s);
    ObjCryst::CIF cif(in);
    const bool verbose = false;
    const bool checkSymAsXYZ = true;
    ObjCryst::CreateCrystalFromCIF(cif, verbose, checkSymAsXYZ,
        oneScatteringPowerPerElement, connectAtoms, &cryst);
}

Crystal* _CreateCrystalFromCIF(nb::object input,
                                const bool oneScatteringPowerPerElement = false,
                                const bool connectAtoms = false)
{
    MuteObjCrystUserInfo muzzle;
    CaptureStdOut gag;
    std::string s = read_pyfile_to_string(input);
    std::istringstream in(s);
    ObjCryst::CIF cif(in);
    int idx0 = gCrystalRegistry.GetNb();
    ObjCryst::CreateCrystalFromCIF(cif, false, true,
        oneScatteringPowerPerElement, connectAtoms);
    gag.release();
    muzzle.release();
    if (gCrystalRegistry.GetNb() == idx0)
        throw ObjCrystException("Cannot create crystal from CIF");
    ObjCryst::Crystal* c = &gCrystalRegistry.GetObj(gCrystalRegistry.GetNb() - 1);
    c->SetDeleteSubObjInDestructor(false);
    c->SetDeleteRefParInDestructor(false);
    return c;
}

} // namespace

void wrap_crystal(nb::module_& m)
{
    m.attr("refpartype_crystal") = gpRefParTypeCrystal;
    m.attr("gCrystalRegistry")  = &gCrystalRegistry;

    nb::class_<Crystal, UnitCell, PyCrystal>(m, "Crystal")
        .def("__init__", [](Crystal* self) { new (self) Crystal(); })
        .def("__init__", [](Crystal* self, REAL a, REAL b, REAL c,
                            const std::string& sgId) {
                 new (self) Crystal(a, b, c, sgId);
                 self->SetDeleteSubObjInDestructor(false);
                 self->SetDeleteRefParInDestructor(false);
             },
             nb::arg("a"), nb::arg("b"), nb::arg("c"), nb::arg("SpaceGroupId"))
        .def("__init__", [](Crystal* self, REAL a, REAL b, REAL c,
                            REAL al, REAL be, REAL ga,
                            const std::string& sgId) {
                 new (self) Crystal(a, b, c, al, be, ga, sgId);
                 self->SetDeleteSubObjInDestructor(false);
                 self->SetDeleteRefParInDestructor(false);
             },
             nb::arg("a"), nb::arg("b"), nb::arg("c"),
             nb::arg("alpha"), nb::arg("beta"), nb::arg("gamma"),
             nb::arg("SpaceGroupId"))
        .def("AddScatterer",
             [](nb::handle crystal_h, nb::handle obj) {
                 Crystal& crystal = nb::cast<Crystal&>(crystal_h);
                 _AddScatterer(crystal, obj);
                 // Keep scatterer alive as long as crystal is alive (but not for None)
                 if (!obj.is_none()) {
                     nb::detail::keep_alive(crystal_h.ptr(), obj.ptr());
                 }
             }, nb::arg("arg").none())
        .def("RemoveScatterer", &_RemoveScatterer, nb::arg("arg").none())
        .def("GetNbScatterer",  &Crystal::GetNbScatterer)
        .def("GetScatt",
             [](Crystal& c, const std::string& name) -> Scatterer& {
                 return _GetScattByName(c, name);
             }, nb::rv_policy::reference_internal)
        .def("GetScatt",
             [](Crystal& c, int idx) -> Scatterer& {
                 return _GetScattByIndex(c, idx);
             }, nb::rv_policy::reference_internal)
        .def("GetScatterer",
             [](Crystal& c, const std::string& name) -> Scatterer& {
                 return _GetScattByName(c, name);
             }, nb::rv_policy::reference_internal)
        .def("GetScatterer",
             [](Crystal& c, int idx) -> Scatterer& {
                 return _GetScattByIndex(c, idx);
             }, nb::rv_policy::reference_internal)
        .def("GetScattererRegistry",
             nb::overload_cast<>(&Crystal::GetScattererRegistry),
             nb::rv_policy::reference_internal)
        .def("GetScatteringPowerRegistry",
             nb::overload_cast<>(&Crystal::GetScatteringPowerRegistry),
             nb::rv_policy::reference_internal)
        .def("AddScatteringPower",    &_AddScatteringPower,    nb::keep_alive<1,2>(), nb::arg("arg").none())
        .def("RemoveScatteringPower", &_RemoveScatteringPower, nb::arg("arg").none())
        .def("GetScatteringPower",
             [](nb::handle self, const std::string& name) -> nb::object {
                 Crystal& c = nb::cast<Crystal&>(self);
                 return wrapScatteringPowerPython(c.GetScatteringPower(name), self);
             })
        .def("GetMasterClockScatteringPower", &Crystal::GetMasterClockScatteringPower,
             nb::rv_policy::reference_internal)
        .def("GetScatteringComponentList", &_GetScatteringComponentList)
        .def("GetClockScattCompList", &Crystal::GetClockScattCompList,
             nb::rv_policy::reference_internal)
        .def("GetMinDistanceTable", &Crystal::GetMinDistanceTable,
             nb::arg("minDistance") = 1.0)
        .def("PrintMinDistanceTable",
             [](const Crystal& c, double d){ c.PrintMinDistanceTable(d); },
             nb::arg("minDistance") = 1.0)
        .def("ResetDynPopCorr",  &Crystal::ResetDynPopCorr)
        .def("SetUseDynPopCorr", &Crystal::SetUseDynPopCorr)
        .def("GetUseDynPopCorr", &Crystal::GetUseDynPopCorr)
        .def("GetBumpMergeCost", &Crystal::GetBumpMergeCost)
        .def("SetBumpMergeDistance",
             (void (Crystal::*)(const ScatteringPower&, const ScatteringPower&, const REAL))
                 &Crystal::SetBumpMergeDistance,
             nb::arg("scatt1"), nb::arg("scatt2"), nb::arg("dist") = 1.5)
        .def("SetBumpMergeDistance",
             (void (Crystal::*)(const ScatteringPower&, const ScatteringPower&,
                                const REAL, const bool))
                 &Crystal::SetBumpMergeDistance,
             nb::arg("scatt1"), nb::arg("scatt2"), nb::arg("dist"), nb::arg("allowMerge"))
        .def("RemoveBumpMergeDistance", &Crystal::RemoveBumpMergeDistance)
        .def("GetBumpMergeParList",
             nb::overload_cast<>(&Crystal::GetBumpMergeParList),
             nb::rv_policy::reference_internal)
        .def("GetClockScattererList", &Crystal::GetClockScattererList,
             nb::rv_policy::reference_internal)
        .def("CIFOutput", &_CIFOutput, nb::arg("file"), nb::arg("mindist") = 0)
        .def("CIF",       &_CIF,       nb::arg("mindist") = 0)
        .def("AddBondValenceRo",    &Crystal::AddBondValenceRo)
        .def("RemoveBondValenceRo", &Crystal::AddBondValenceRo)
        .def("GetBondValenceCost",  &Crystal::GetBondValenceCost)
        .def("GetBondValenceRoList",
             nb::overload_cast<>(&Crystal::GetBondValenceRoList),
             nb::rv_policy::reference_internal)
        .def("ConnectAtoms", &Crystal::ConnectAtoms,
             nb::arg("min_relat_dist") = 0.4, nb::arg("max_relat_dist") = 1.3,
             nb::arg("warnuser_fail") = false)
        .def("GetFormula", &Crystal::GetFormula)
        .def("GetWeight",  &Crystal::GetWeight)
        .def("ImportCrystalFromCIF", &_ImportCrystalFromCIF,
             nb::arg("input"),
             nb::arg("oneScatteringPowerPerElement") = false,
             nb::arg("connectAtoms") = false)
        .def("UpdateDisplay", &Crystal::UpdateDisplay)
        ;

    nb::class_<Crystal::BumpMergePar>(m, "BumpMergePar")
        .def_rw("mDist2",      &Crystal::BumpMergePar::mDist2)
        .def_rw("mCanOverlap", &Crystal::BumpMergePar::mCanOverlap)
        ;

    m.def("CreateCrystalFromCIF", &_CreateCrystalFromCIF,
          nb::arg("file"),
          nb::arg("oneScatteringPowerPerElement") = false,
          nb::arg("connectAtoms") = false,
          nb::rv_policy::take_ownership);
}
