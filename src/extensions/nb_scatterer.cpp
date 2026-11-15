/*
 * pyobjcryst nanobind port — Scatterer bindings
 */

#include <nanobind/nanobind.h>
#include <nanobind/trampoline.h>
#include <nanobind/stl/string.h>

#include <ObjCryst/RefinableObj/RefinableObj.h>
#include <ObjCryst/ObjCryst/Scatterer.h>
#include <ObjCryst/ObjCryst/Molecule.h>
#include <ObjCryst/ObjCryst/Atom.h>
#include <ObjCryst/ObjCryst/ZScatterer.h>

#include "helpers_nb.hpp"

namespace nb = nanobind;
using namespace ObjCryst;

struct PyScatterer : Scatterer {
    NB_TRAMPOLINE(Scatterer, 9);

    void SetX(const REAL x) override { NB_OVERRIDE(SetX, x); }
    void SetY(const REAL y) override { NB_OVERRIDE(SetY, y); }
    void SetZ(const REAL z) override { NB_OVERRIDE(SetZ, z); }
    void SetOccupancy(const REAL occ) override { NB_OVERRIDE(SetOccupancy, occ); }
    Scatterer* CreateCopy() const override { NB_OVERRIDE_PURE(CreateCopy); }
    int GetNbComponent() const override { NB_OVERRIDE_PURE(GetNbComponent); }
    const ScatteringComponentList& GetScatteringComponentList() const override {
        NB_OVERRIDE_PURE(GetScatteringComponentList);
    }
    std::string GetComponentName(const int i) const override {
        NB_OVERRIDE_PURE(GetComponentName, i);
    }
    void Print() const override { NB_OVERRIDE_PURE(Print); }
    std::ostream& POVRayDescription(std::ostream& os, const CrystalPOVRayOptions& opt) const override {
        return os;
    }
    // GetClockScattCompList: forwards a *protected* Scatterer member
    // (mClockScattCompList) not reachable from any free function -- the
    // Boost.Python binding used the same trick (subclassing Scatterer to
    // gain access), see scatterer_ext.cpp's ScattererWrap::_GetClockScattCompList.
    const RefinableObjClock& GetClockScattCompList() const {
        return mClockScattCompList;
    }
#ifdef OBJCRYST_GL
    void GLInitDisplayList(const bool noSymmetrics,
                           const REAL xMin, const REAL xMax,
                           const REAL yMin, const REAL yMax,
                           const REAL zMin, const REAL zMax,
                           const bool displayEnantiomer,
                           const bool displayNames,
                           const bool hideHydrogens,
                           const REAL fadeDistance,
                           const bool fullMoleculeInLimits) const override {}
#endif
protected:
    void InitRefParList() override {}
};

namespace {

nb::list _GetScatteringComponentList(Scatterer& s)
{
    const ScatteringComponentList& scl = s.GetScatteringComponentList();
    nb::list l;
    for (int i = 0; i < scl.GetNbComponent(); ++i)
        l.append(nb::cast(scl(i)));
    return l;
}

} // namespace

void wrap_scatterer(nb::module_& m)
{
    m.attr("refpartype_scatt")                       = gpRefParTypeScatt;
    m.attr("refpartype_scatt_transl")                = gpRefParTypeScattTransl;
    m.attr("refpartype_scatt_transl_x")              = gpRefParTypeScattTranslX;
    m.attr("refpartype_scatt_transl_y")              = gpRefParTypeScattTranslY;
    m.attr("refpartype_scatt_transl_z")              = gpRefParTypeScattTranslZ;
    m.attr("refpartype_scatt_orient")                = gpRefParTypeScattOrient;
    m.attr("refpartype_scatt_conform")               = gpRefParTypeScattConform;
    m.attr("refpartype_scatt_conform_bondlength")    = gpRefParTypeScattConformBondLength;
    m.attr("refpartype_scatt_conform_bondangle")     = gpRefParTypeScattConformBondAngle;
    m.attr("refpartype_scatt_conform_dihedangle")    = gpRefParTypeScattConformDihedAngle;
    m.attr("refpartype_scatt_conform_x")             = gpRefParTypeScattConformX;
    m.attr("refpartype_scatt_conform_y")             = gpRefParTypeScattConformY;
    m.attr("refpartype_scatt_conform_z")             = gpRefParTypeScattConformZ;
    m.attr("refpartype_scatt_occup")                 = gpRefParTypeScattOccup;
    m.attr("gScattererRegistry")                     = &gScattererRegistry;

    nb::class_<Scatterer, PyScatterer> cls(m, "Scatterer");
    cls
        .def("GetX",         &Scatterer::GetX)
        .def("GetY",         &Scatterer::GetY)
        .def("GetZ",         &Scatterer::GetZ)
        .def("GetOccupancy", &Scatterer::GetOccupancy)
        .def("SetX",         &Scatterer::SetX)
        .def("SetY",         &Scatterer::SetY)
        .def("SetZ",         &Scatterer::SetZ)
        .def("SetOccupancy", &Scatterer::SetOccupancy)
        .def("GetClockScatterer",
             nb::overload_cast<>(&Scatterer::GetClockScatterer),
             nb::rv_policy::reference_internal)
        .def("SetCrystal", &Scatterer::SetCrystal, nb::keep_alive<1,2>())
        .def("GetCrystal",
             nb::overload_cast<>(&Scatterer::GetCrystal),
             nb::rv_policy::reference_internal)
        .def("GetNbComponent",            &Scatterer::GetNbComponent)
        .def("GetComponentName",          &Scatterer::GetComponentName)
        .def("GetScatteringComponentList",&_GetScatteringComponentList)
        .def("Print",                     &Scatterer::Print)
        .def("__str__",   [](const Scatterer& s){ return obj_str(s); })
        .def_prop_rw("X",         &Scatterer::GetX,         &Scatterer::SetX)
        .def_prop_rw("Y",         &Scatterer::GetY,         &Scatterer::SetY)
        .def_prop_rw("Z",         &Scatterer::GetZ,         &Scatterer::SetZ)
        .def_prop_rw("Occupancy", &Scatterer::GetOccupancy, &Scatterer::SetOccupancy)
        // RefinableObj methods (virtual base — use lambdas to avoid vbase offset issue)
        .def("GetName",     [](Scatterer& s) -> std::string { return s.GetName(); })
        .def("SetName",     [](Scatterer& s, const std::string& n) { s.SetName(n); })
        .def("GetClassName",[](Scatterer& s) -> std::string { return s.GetClassName(); })
        .def("GetNbPar",    [](Scatterer& s) { return s.GetNbPar(); })
        .def("GetPar",      [](Scatterer& s, const std::string& name) -> RefinablePar& {
                                return s.GetPar(name); }, nb::rv_policy::reference_internal)
        .def("GetPar",      [](Scatterer& s, long i) -> RefinablePar& {
                                return s.GetPar(i); }, nb::rv_policy::reference_internal)
        // NB: parameter is Scatterer&, not PyScatterer& -- this must work on
        // Atom/Molecule/... instances too, which are plain Scatterer
        // subclasses (never actually constructed as a PyScatterer). Casting
        // through PyScatterer is safe here because PyScatterer adds no data
        // members of its own; it exists purely to obtain access to the
        // protected mClockScattCompList (same trick Boost.Python's
        // ScattererWrap used -- see scatterer_ext.cpp).
        .def("GetClockScattCompList",
             [](Scatterer& s) -> const RefinableObjClock& {
                 return static_cast<PyScatterer&>(s).GetClockScattCompList();
             },
             nb::rv_policy::reference_internal)
        ;

    // Restore the rest of RefinableObj's method surface (Print, GetName,
    // GetNbPar, and GetPar above were already covered one-off; everything
    // else -- FixAllPar, XMLOutput, BeginOptimization, GetLogLikelihood,
    // AddPar, CreateParamSet, GetOption, UpdateDisplay, ... -- was not).
    // See helpers_nb.hpp for why this is safe despite Scatterer's virtual
    // inheritance of RefinableObj. Covers Atom, Molecule, ZScatterer, and
    // ZPolyhedron automatically, since they inherit Scatterer non-virtually
    // and already have it correctly declared as their nanobind base.
    bind_refinableobj_forwarding(cls);
}
