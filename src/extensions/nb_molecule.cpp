#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/make_iterator.h>

#include <vector>
#include <string>
#include <sstream>
#include <set>
#include <map>

#undef B0
#include <ObjCryst/RefinableObj/RefinableObj.h>
#include <ObjCryst/ObjCryst/Molecule.h>
#include <ObjCryst/ObjCryst/ZScatterer.h>
#include <ObjCryst/ObjCryst/Crystal.h>
#include <ObjCryst/ObjCryst/ScatteringPower.h>

#include "helpers_nb.hpp"

namespace nb = nanobind;
using namespace ObjCryst;

namespace {

// ScatteringPowerAtom inherits VIRTUALLY from ScatteringPower (see
// ScatteringPower.h), so it is intentionally registered in nanobind WITHOUT
// a declared base (see nb_scatteringpoweratom.cpp) -- nanobind's automatic
// base<->derived pointer adjustment assumes a fixed, statically computable
// offset, which virtual inheritance does not provide. Left as a plain
// `const ScatteringPower* pow` parameter, nanobind refuses to accept a
// Python ScatteringPowerAtom instance there at all (TypeError), since it
// has no declared relationship to ScatteringPower. Resolve it manually:
// try the concrete registered type first (exact match, no offset needed),
// falling back to ScatteringPower directly. Mirrors getScatteringPowerPtr
// in nb_atom.cpp.
const ScatteringPower* extractScatteringPowerArg(nb::object obj)
{
    if (obj.is_none()) return nullptr;
    try {
        return static_cast<const ScatteringPower*>(&nb::cast<ScatteringPowerAtom&>(obj));
    } catch (...) {}
    return &nb::cast<const ScatteringPower&>(obj);
}

// Wrap a (possibly null) ScatteringPower* for return to Python. Its dynamic
// type is almost always ScatteringPowerAtom; detect that with a real C++
// dynamic_cast (correct for virtual inheritance, unlike nanobind's own
// offset-based polymorphic auto-detection -- see the identical fix and
// longer explanation in nb_crystal.cpp::wrapScatteringPowerPython) before
// handing nanobind an already-correctly-typed pointer.
nb::object wrapScatteringPowerReturn(const ScatteringPower* sp, nb::handle parent)
{
    if (!sp) return nb::none();
    if (auto* spa = dynamic_cast<const ScatteringPowerAtom*>(sp))
        return nb::cast(const_cast<ScatteringPowerAtom*>(spa), nb::rv_policy::reference_internal, parent);
    return nb::cast(const_cast<ScatteringPower*>(sp), nb::rv_policy::reference_internal, parent);
}

MolAtom& _AddAtom(Molecule& m, const double x, const double y, const double z,
                  nb::object pow, const std::string& name,
                  const bool updateDisplay=true)
{
    const ScatteringPower* sp = extractScatteringPowerArg(pow);
    m.AddAtom(x, y, z, sp, name, updateDisplay);
    m.SetDeleteSubObjInDestructor(false);
    return *m.GetAtomList().back();
}

MolBond& _AddBond(Molecule& m, MolAtom& atom1, MolAtom& atom2, const double length,
                  const double sigma, const double delta, const double bondOrder=1.,
                  const bool updateDisplay=true)
{
    m.AddBond(atom1, atom2, length, sigma, delta, bondOrder, updateDisplay);
    m.SetDeleteSubObjInDestructor(false);
    return *m.GetBondList().back();
}

MolBondAngle& _AddBondAngle(Molecule& m, MolAtom& atom1, MolAtom& atom2, MolAtom& atom3,
                             const double angle, const double sigma, const double delta,
                             const bool updateDisplay=true)
{
    m.AddBondAngle(atom1, atom2, atom3, angle, sigma, delta, updateDisplay);
    m.SetDeleteSubObjInDestructor(false);
    return *m.GetBondAngleList().back();
}

MolDihedralAngle& _AddDihedralAngle(Molecule& m, MolAtom& a1, MolAtom& a2,
                                     MolAtom& a3, MolAtom& a4,
                                     const double angle, const double sigma, const double delta,
                                     const bool updateDisplay=true)
{
    m.AddDihedralAngle(a1, a2, a3, a4, angle, sigma, delta, updateDisplay);
    m.SetDeleteSubObjInDestructor(false);
    return *m.GetDihedralAngleList().back();
}

MolAtom& _GetAtomIdx(Molecule& m, int idx)
{
    int i = check_index(idx, m.GetNbComponent(), ALLOW_NEGATIVE);
    return m.GetAtom(i);
}

MolAtom* _FindAtom(Molecule& m, const std::string& name)
{
    auto ii = m.FindAtom(name);
    return (ii != m.mvpAtom.rend()) ? *ii : nullptr;
}

MolAtom* _GetAtomByName(Molecule& m, const std::string& name)
{
    MolAtom* rv = _FindAtom(m, name);
    if (!rv) throw nb::value_error(("Invalid atom name: " + name).c_str());
    return rv;
}

MolBond& _GetBondIdx(Molecule& m, int idx)
{
    auto& v = m.GetBondList();
    if (idx < 0) idx += (int)v.size();
    if (v.empty() || idx < 0 || idx >= (int)v.size())
        throw nb::index_error("Index out of range");
    return *v[idx];
}

MolBondAngle& _GetBondAngleIdx(Molecule& m, int idx)
{
    auto& v = m.GetBondAngleList();
    if (idx < 0) idx += (int)v.size();
    if (v.empty() || idx < 0 || idx >= (int)v.size())
        throw nb::index_error("Index out of range");
    return *v[idx];
}

MolDihedralAngle& _GetDihedralAngleIdx(Molecule& m, int idx)
{
    auto& v = m.GetDihedralAngleList();
    if (idx < 0) idx += (int)v.size();
    if (v.empty() || idx < 0 || idx >= (int)v.size())
        throw nb::index_error("Index out of range");
    return *v[idx];
}

RigidGroup& _AddRigidGroup(Molecule& m, const RigidGroup& r, const bool ud=true)
{
    m.AddRigidGroup(r, ud);
    return *m.GetRigidGroupList().back();
}

RigidGroup& _AddRigidGroupIterable(Molecule& m, nb::object& l, const bool ud=true)
{
    RigidGroup rg;
    for (auto h : l) rg.insert(&nb::cast<MolAtom&>(h));
    m.AddRigidGroup(rg, ud);
    return *m.GetRigidGroupList().back();
}

nb::object _FindBond(const Molecule& m, const MolAtom& a1, const MolAtom& a2)
{
    auto it = m.FindBond(a1, a2);
    if (it == m.GetBondList().end()) return nb::none();
    return nb::cast(*it, nb::rv_policy::reference);
}

nb::object _FindBondAngle(const Molecule& m, const MolAtom& a1, const MolAtom& a2, const MolAtom& a3)
{
    auto it = m.FindBondAngle(a1, a2, a3);
    if (it == m.GetBondAngleList().end()) return nb::none();
    return nb::cast(*it, nb::rv_policy::reference);
}

nb::object _FindDihedralAngle(const Molecule& m, const MolAtom& a1, const MolAtom& a2,
                               const MolAtom& a3, const MolAtom& a4)
{
    auto it = m.FindDihedralAngle(a1, a2, a3, a4);
    if (it == m.GetDihedralAngleList().end()) return nb::none();
    return nb::cast(*it, nb::rv_policy::reference);
}

nb::list _GetAtomList(const Molecule& m)
{ return ptrcontainerToPyList<const std::vector<MolAtom*>>(m.GetAtomList()); }

nb::list _GetBondList(const Molecule& m)
{ return ptrcontainerToPyList<const std::vector<MolBond*>>(m.GetBondList()); }

nb::list _GetBondAngleList(const Molecule& m)
{ return ptrcontainerToPyList<const std::vector<MolBondAngle*>>(m.GetBondAngleList()); }

nb::list _GetDihedralAngleList(const Molecule& m)
{ return ptrcontainerToPyList<const std::vector<MolDihedralAngle*>>(m.GetDihedralAngleList()); }

nb::list _GetRigidGroupList(const Molecule& m)
{ return ptrcontainerToPyList<const std::vector<RigidGroup*>>(m.GetRigidGroupList()); }

nb::list _GetStretchModeBondLengthList(const Molecule& m)
{ return containerToPyList<const std::list<StretchModeBondLength>>(m.GetStretchModeBondLengthList()); }

nb::list _GetStretchModeBondAngleList(const Molecule& m)
{ return containerToPyList<const std::list<StretchModeBondAngle>>(m.GetStretchModeBondAngleList()); }

nb::list _GetStretchModeTorsionList(const Molecule& m)
{ return containerToPyList<const std::list<StretchModeTorsion>>(m.GetStretchModeTorsionList()); }

void _RotateAtomGroup(Molecule& m, const MolAtom& at1, const MolAtom& at2,
                      const nb::object& atoms, const double angle, const bool keepCenter=true)
{
    auto catoms = pyIterableToSet<MolAtom>(atoms);
    m.RotateAtomGroup(at1, at2, catoms, angle, keepCenter);
}

void _RotateAtomGroupVec(Molecule& m, const MolAtom& at1, const REAL vx,
                          const double vy, const double vz, const nb::object& atoms,
                          const double angle, const bool keepCenter=true)
{
    auto catoms = pyIterableToSet<MolAtom>(atoms);
    m.RotateAtomGroup(at1, vx, vy, vz, catoms, angle, keepCenter);
}

// Accepts two 3-tuples (a point and a direction vector) instead of two
// MolAtom instances, mirroring the legacy Boost.Python _RotateAtomGroup2Vec
// in src/extensions/molecule_ext.cpp: creates a temporary dummy atom at v1,
// rotates around the axis defined by (temp atom, direction v2), then removes
// the temporary atom again.
void _RotateAtomGroup2Vec(Molecule& m, nb::object v1, nb::object v2,
                          const nb::object& atoms, const double angle,
                          const bool keepCenter=true)
{
    double x = nb::cast<double>(v1[0]);
    double y = nb::cast<double>(v1[1]);
    double z = nb::cast<double>(v1[2]);
    MolAtom& a = _AddAtom(m, x, y, z, nb::none(), "_rag2vectemp", false);
    x = nb::cast<double>(v2[0]);
    y = nb::cast<double>(v2[1]);
    z = nb::cast<double>(v2[2]);
    _RotateAtomGroupVec(m, a, x, y, z, atoms, angle, keepCenter);
    m.RemoveAtom(a, true);
}

void _TranslateAtomGroup(Molecule& m, const nb::object& atoms, const double dx,
                          const double dy, const double dz, const bool keepCenter=true)
{
    auto catoms = pyIterableToSet<MolAtom>(atoms);
    m.TranslateAtomGroup(catoms, dx, dy, dz, keepCenter);
}

nb::dict _GetConnectivityTable(Molecule& m)
{
    const auto& ct = m.GetConnectivityTable();
    nb::dict d;
    for (const auto& kv : ct) {
        nb::object key = nb::cast(kv.first, nb::rv_policy::reference);
        d[key] = ptrcontainerToPyList<const std::set<MolAtom*>>(kv.second);
    }
    return d;
}

nb::list _AsZMatrix(const Molecule& m, const bool keeporder)
{
    return containerToPyList<const std::vector<MolZAtom>>(m.AsZMatrix(keeporder));
}

std::string quatparname(const Molecule& m, int idx)
{
    using namespace std;
    static bool didseparator = false;
    static bool prefixmolname = false;
    static string separator;
    if (!didseparator) {
        map<string,int> qnames;
        for (long i = 0; i < m.GetNbPar(); ++i) {
            const string& pname = m.GetPar(i).GetName();
            size_t n = pname.size();
            if (n < 2) continue;
            if (pname[n-2] != 'Q') continue;
            if (pname.find_last_of("0123", n-1) == string::npos) continue;
            qnames[pname.substr(0, n-2)] += 1;
        }
        const string& mname = m.GetName();
        for (auto& kv : qnames) {
            if (kv.second == 4) {
                const string& qnm = kv.first;
                prefixmolname = (qnm.size() >= mname.size() &&
                                 qnm.substr(0, mname.size()) == mname);
                size_t p0 = prefixmolname ? mname.size() : 0;
                separator = qnm.substr(p0);
                didseparator = true;
            }
        }
    }
    std::ostringstream rv;
    rv << (prefixmolname ? m.GetName() : "") << separator << 'Q' << idx;
    return rv.str();
}

void _setQ0(Molecule& m, double v) { m.GetPar(quatparname(m,0)).SetValue(v); }
double _getQ0(Molecule& m) { return m.GetPar(quatparname(m,0)).GetValue(); }
void _setQ1(Molecule& m, double v) { m.GetPar(quatparname(m,1)).SetValue(v); }
double _getQ1(Molecule& m) { return m.GetPar(quatparname(m,1)).GetValue(); }
void _setQ2(Molecule& m, double v) { m.GetPar(quatparname(m,2)).SetValue(v); }
double _getQ2(Molecule& m) { return m.GetPar(quatparname(m,2)).GetValue(); }
void _setQ3(Molecule& m, double v) { m.GetPar(quatparname(m,3)).SetValue(v); }
double _getQ3(Molecule& m) { return m.GetPar(quatparname(m,3)).GetValue(); }

nb::object molzatom_spow(nb::handle self)
{
    const MolZAtom& a = nb::cast<const MolZAtom&>(self);
    return wrapScatteringPowerReturn(a.mpPow ? a.mpPow : nullptr, self);
}

// AddPar on Molecule: Molecule does not nanobind-inherit RefinableObj (it is
// only ever exposed via Scatterer, and reaching RefinableObj itself would hit
// the same virtual-inheritance offset problem documented throughout this
// migration), so reach the base via static_cast, exactly mirroring the
// already-working _AddPar/_AddParObj pattern in nb_refinableobj.cpp.
void _AddParMol(Molecule& m, RefinablePar* p)
{
    static_cast<RefinableObj&>(m).AddPar(p);
    static_cast<RefinableObj&>(m).SetDeleteRefParInDestructor(0);
}
void _AddParObjMol(Molecule& m, nb::object o, const bool copyParam = false)
{
    static_cast<RefinableObj&>(m).AddPar(extractRefinableObjArg(o), copyParam);
    static_cast<RefinableObj&>(m).SetDeleteRefParInDestructor(0);
}

// __getitem__ slice support, mirroring Python list semantics. Registered as
// a separate overload from the plain-int _GetAtomIdx (below) so nanobind's
// own overload resolution distinguishes `int` from `slice` arguments,
// rather than us hand-rolling that dispatch. self is taken as nb::handle so
// each returned MolAtom* can be explicitly tied (via rv_policy and parent)
// to the Molecule's lifetime, matching the pattern used everywhere else in
// this file for reference_internal returns.
nb::list _GetItemSlice(nb::handle self, const nb::slice& sl)
{
    Molecule& m = nb::cast<Molecule&>(self);
    auto [start, stop, step, slice_length] = sl.compute(m.GetAtomList().size());
    nb::list rv;
    for (size_t i = 0; i < slice_length; ++i) {
        rv.append(nb::cast(&_GetAtomIdx(m, (int)start), nb::rv_policy::reference_internal, self));
        start += step;
    }
    return rv;
}

} // namespace

void wrap_molecule(nb::module_& m)
{
    // nb::dynamic_attr() is required because the legacy Python-layer helper
    // ImportFenskeHallZMatrix() (see pyobjcryst/molecule.py) sets an ad-hoc
    // `m._crystal = cryst` attribute on the Molecule instance to keep the
    // Crystal alive (a pre-existing hack from the Boost.Python era, kept
    // as-is rather than restructured here).
    nb::class_<Molecule, Scatterer>(m, "Molecule", nb::is_final(), nb::dynamic_attr())
        .def(nb::init<const Molecule&>(), nb::arg("oldMolecule"))
        .def(nb::init<Crystal&, const std::string&>(),
             nb::arg("cryst"), nb::arg("name") = "")
        .def("GetFormula", &Molecule::GetFormula)
        .def("AddAtom", &_AddAtom,
             nb::arg("x"), nb::arg("y"), nb::arg("z"), nb::arg("pPow").none(),
             nb::arg("name"), nb::arg("updateDisplay") = true,
             nb::keep_alive<1,5>(), nb::rv_policy::reference_internal)
        .def("RemoveAtom", [](Molecule& mol, MolAtom& a){ mol.RemoveAtom(a, false); })
        .def("RemoveAtom", [](Molecule& mol, int idx){ mol.RemoveAtom(_GetAtomIdx(mol, idx), false); })
        .def("AddBond", &_AddBond,
             nb::arg("atom1"), nb::arg("atom2"), nb::arg("length"),
             nb::arg("sigma"), nb::arg("delta"), nb::arg("bondOrder") = 1.,
             nb::arg("updateDisplay") = true,
             // No nb::keep_alive<1,2/3>() here (Molecule keeping atom1/atom2
             // alive): both atoms already arrive with their OWN reverse-
             // direction link back to this Molecule (established when they
             // were obtained via GetAtom()/AddAtom(), which use
             // rv_policy::reference_internal). Adding the forward direction
             // here too closes a Molecule<->MolAtom reference cycle that
             // nanobind objects don't participate in Python's cyclic GC for
             // by default, so it never gets collected (a real, reproduced
             // leak -- see §10 in nanobind_migration_notes.md). The forward
             // direction is not actually needed: MolAtom has no standalone
             // Python-owned existence (see nb_molatom.cpp) independent of
             // the Molecule that returned it, so nothing here can dangle
             // once the redundant direction is removed.
             nb::rv_policy::reference_internal)
        .def("RemoveBond", [](Molecule& mol, const MolBond& b){ mol.RemoveBond(b, false); })
        .def("RemoveBond", [](Molecule& mol, int idx){ mol.RemoveBond(_GetBondIdx(mol, idx), false); })
        .def("GetBond", &_GetBondIdx, nb::rv_policy::reference_internal)
        .def("FindBond", &_FindBond)
        .def("AddBondAngle", &_AddBondAngle,
             nb::arg("atom1"), nb::arg("atom2"), nb::arg("atom3"),
             nb::arg("angle"), nb::arg("sigma"), nb::arg("delta"),
             nb::arg("updateDisplay") = true,
             // See the comment on AddBond above -- same reasoning applies to
             // atom1/atom2/atom3 here (§10).
             nb::rv_policy::reference_internal)
        .def("RemoveBondAngle", [](Molecule& mol, MolBondAngle& ba){ mol.RemoveBondAngle(ba, false); })
        .def("RemoveBondAngle", [](Molecule& mol, int idx){ mol.RemoveBondAngle(_GetBondAngleIdx(mol, idx), false); })
        .def("GetBondAngle", &_GetBondAngleIdx, nb::rv_policy::reference_internal)
        .def("FindBondAngle", &_FindBondAngle)
        .def("AddDihedralAngle", &_AddDihedralAngle,
             nb::arg("atom1"), nb::arg("atom2"), nb::arg("atom3"), nb::arg("atom4"),
             nb::arg("angle"), nb::arg("sigma"), nb::arg("delta"),
             nb::arg("updateDisplay") = true,
             // See the comment on AddBond above -- same reasoning applies to
             // atom1..atom4 here (§10).
             nb::rv_policy::reference_internal)
        .def("RemoveDihedralAngle", [](Molecule& mol, MolDihedralAngle& da){ mol.RemoveDihedralAngle(da, false); })
        .def("RemoveDihedralAngle", [](Molecule& mol, int idx){ mol.RemoveDihedralAngle(_GetDihedralAngleIdx(mol, idx), false); })
        .def("GetDihedralAngle", &_GetDihedralAngleIdx, nb::rv_policy::reference_internal)
        .def("FindDihedralAngle", &_FindDihedralAngle)
        .def("AddRigidGroup",
             [](Molecule& mol, const RigidGroup& group, const bool updateDisplay) -> RigidGroup& { return _AddRigidGroup(mol, group, updateDisplay); },
             nb::arg("group"), nb::arg("updateDisplay") = true,
             nb::rv_policy::reference_internal)
        .def("AddRigidGroup",
             [](Molecule& mol, nb::object& group, const bool updateDisplay) -> RigidGroup& { return _AddRigidGroupIterable(mol, group, updateDisplay); },
             nb::arg("group"), nb::arg("updateDisplay") = true,
             nb::rv_policy::reference_internal)
        .def("RemoveRigidGroup", [](Molecule& mol, RigidGroup& rg, const bool ud){ mol.RemoveRigidGroup(rg, ud, false); },
             nb::arg("group"), nb::arg("updateDisplay") = true)
        .def("GetAtom", [](Molecule& mol, int idx) -> MolAtom& { return _GetAtomIdx(mol, idx); }, nb::rv_policy::reference_internal)
        .def("GetAtom", [](Molecule& mol, const std::string& name) -> MolAtom& { return *_GetAtomByName(mol, name); }, nb::rv_policy::reference_internal)
        .def("FindAtom", &_FindAtom, nb::rv_policy::reference_internal)
        .def("OptimizeConformation", &Molecule::OptimizeConformation,
             nb::arg("nbTrial") = 10000, nb::arg("stopCost") = 0.)
        .def("OptimizeConformationSteepestDescent",
             &Molecule::OptimizeConformationSteepestDescent,
             nb::arg("maxStep") = 0.1, nb::arg("nbSteps") = 1)
        .def("GetNbAtoms", [](Molecule& mol){ return mol.GetAtomList().size(); })
        .def("GetNbBonds", [](Molecule& mol){ return mol.GetBondList().size(); })
        .def("GetNbBondAngles", [](Molecule& mol){ return mol.GetBondAngleList().size(); })
        .def("GetNbDihedralAngles", [](Molecule& mol){ return mol.GetDihedralAngleList().size(); })
        .def("GetNbRigidGroups", [](Molecule& mol){ return mol.GetRigidGroupList().size(); })
        .def("GetAtomList", &_GetAtomList)
        .def("GetBondList", &_GetBondList)
        .def("GetBondAngleList", &_GetBondAngleList)
        .def("GetDihedralAngleList", &_GetDihedralAngleList)
        .def("GetRigidGroupList", &_GetRigidGroupList)
        // Named iterators, present in the legacy Boost.Python binding
        // alongside the default atom __iter__ (see below). IterAtom is
        // functionally redundant with __iter__ (same begin/end pair) but
        // kept for interface fidelity; IterBond/IterBondAngle/
        // IterDihedralAngle have no other way to iterate those lists.
        .def("IterAtom", [](Molecule& mol){
            auto& v = mol.GetAtomList();
            return nb::make_iterator(nb::find(nb::type<Molecule>()), "atom_iter", v.begin(), v.end());
        }, nb::keep_alive<0,1>())
        .def("IterBond", [](Molecule& mol){
            auto& v = mol.GetBondList();
            return nb::make_iterator(nb::find(nb::type<Molecule>()), "bond_iter", v.begin(), v.end());
        }, nb::keep_alive<0,1>())
        .def("IterBondAngle", [](Molecule& mol){
            auto& v = mol.GetBondAngleList();
            return nb::make_iterator(nb::find(nb::type<Molecule>()), "bondangle_iter", v.begin(), v.end());
        }, nb::keep_alive<0,1>())
        .def("IterDihedralAngle", [](Molecule& mol){
            auto& v = mol.GetDihedralAngleList();
            return nb::make_iterator(nb::find(nb::type<Molecule>()), "dihedralangle_iter", v.begin(), v.end());
        }, nb::keep_alive<0,1>())
        .def("GetStretchModeBondLengthList", &_GetStretchModeBondLengthList)
        .def("GetStretchModeBondAngleList", &_GetStretchModeBondAngleList)
        .def("GetStretchModeTorsionList", &_GetStretchModeTorsionList)
        .def("RotateAtomGroup",
             [](Molecule& mol, const MolAtom& at1, const MolAtom& at2, const nb::object& atoms, const double angle, const bool keepCenter){
                 _RotateAtomGroup(mol, at1, at2, atoms, angle, keepCenter);
             },
             nb::arg("at1"), nb::arg("at2"), nb::arg("atoms"), nb::arg("angle"), nb::arg("keepCenter") = true)
        .def("RotateAtomGroup",
             [](Molecule& mol, const MolAtom& at1, const double vx, const double vy, const double vz, const nb::object& atoms, const double angle, const bool keepCenter){
                 _RotateAtomGroupVec(mol, at1, vx, vy, vz, atoms, angle, keepCenter);
             },
             nb::arg("at1"), nb::arg("vx"), nb::arg("vy"), nb::arg("vz"), nb::arg("atoms"), nb::arg("angle"), nb::arg("keepCenter") = true)
        .def("RotateAtomGroup", &_RotateAtomGroup2Vec,
             nb::arg("v1"), nb::arg("v2"), nb::arg("atoms"), nb::arg("angle"), nb::arg("keepCenter") = true)
        .def("TranslateAtomGroup", &_TranslateAtomGroup,
             nb::arg("atoms"), nb::arg("dx"), nb::arg("dy"), nb::arg("dz"), nb::arg("keepCenter") = true)
        .def("GetConnectivityTable", &_GetConnectivityTable)
        .def("GetBondListClock", nb::overload_cast<>(&Molecule::GetBondListClock),
             nb::rv_policy::reference_internal)
        .def("GetAtomPositionClock", nb::overload_cast<>(&Molecule::GetAtomPositionClock),
             nb::rv_policy::reference_internal)
        .def("GetRigidGroupClock", nb::overload_cast<>(&Molecule::GetRigidGroupClock),
             nb::rv_policy::reference_internal)
        .def("RigidifyWithDihedralAngles", &Molecule::RigidifyWithDihedralAngles)
        .def("BondLengthRandomChange", &Molecule::BondLengthRandomChange,
             nb::arg("mode"), nb::arg("amplitude"), nb::arg("respectRestraint") = true)
        .def("BondAngleRandomChange", &Molecule::BondAngleRandomChange,
             nb::arg("mode"), nb::arg("amplitude"), nb::arg("respectRestraint") = true)
        .def("DihedralAngleRandomChange", &Molecule::DihedralAngleRandomChange,
             nb::arg("mode"), nb::arg("amplitude"), nb::arg("respectRestraint") = true)
        .def("GetCenterAtom", &Molecule::GetCenterAtom,
             nb::rv_policy::reference_internal)
        .def("SetCenterAtom", &Molecule::SetCenterAtom, nb::keep_alive<1,2>())
        .def("AsZMatrix", &_AsZMatrix, nb::arg("keeporder"))
        .def("BuildRingList", &Molecule::BuildRingList)
        .def("BuildConnectivityTable", &Molecule::BuildConnectivityTable)
        .def("BuildRotorGroup", &Molecule::BuildRotorGroup)
        .def("TuneGlobalOptimRotationAmplitude", &Molecule::TuneGlobalOptimRotationAmplitude)
        .def("BuildFlipGroup", &Molecule::BuildFlipGroup)
        .def("BuildStretchModeBondLength", &Molecule::BuildStretchModeBondLength)
        .def("BuildStretchModeBondAngle", &Molecule::BuildStretchModeBondAngle)
        .def("BuildStretchModeTorsion", &Molecule::BuildStretchModeTorsion)
        .def("BuildStretchModeTwist", &Molecule::BuildStretchModeTwist)
        .def("BuildStretchModeGroups", &Molecule::BuildStretchModeGroups)
        .def("UpdateScattCompList", &Molecule::UpdateScattCompList)
        .def("InitOptions", &Molecule::InitOptions)
        .def("__getitem__", &_GetItemSlice)
        .def("__getitem__", &_GetAtomIdx, nb::rv_policy::reference_internal)
        .def("__len__", [](Molecule& mol){ return mol.GetAtomList().size(); })
        .def("GetLogLikelihood", [](Molecule& mol) -> double {
                                    return (double)static_cast<RefinableObj&>(mol).GetLogLikelihood(); })
        .def("xml", [](const Molecule& mol) -> std::string {
                        std::ostringstream s;
                        static_cast<const RefinableObj&>(mol).XMLOutput(s, 0);
                        return s.str();
                    })
        .def("AddPar", &_AddParMol, nb::arg("par"), nb::keep_alive<1,2>())
        .def("AddPar", &_AddParObjMol, nb::arg("newRefParList"), nb::arg("copyParam") = false,
             nb::keep_alive<1,2>())
        .def("__iter__", [](Molecule& mol){
            auto& v = mol.GetAtomList();
            return nb::make_iterator(nb::find(nb::type<Molecule>()), "atom_iter", v.begin(), v.end());
        }, nb::keep_alive<0,1>())
        .def_prop_rw("Q0", &_getQ0, &_setQ0)
        .def_prop_rw("Q1", &_getQ1, &_setQ1)
        .def_prop_rw("Q2", &_getQ2, &_setQ2)
        .def_prop_rw("Q3", &_getQ3, &_setQ3)
        ;

    m.def("GetBondLength", &GetBondLength);
    m.def("GetBondAngle", &GetBondAngle);
    m.def("GetDihedralAngle", &GetDihedralAngle);

    m.def("ZScatterer2Molecule",
          (Molecule* (*)(ZScatterer*)) &ZScatterer2Molecule,
          nb::arg("zscatt"), nb::rv_policy::take_ownership);

    nb::class_<MolZAtom>(m, "MolZAtom")
        .def(nb::init<const MolZAtom&>())
        .def("GetScatteringPower", &molzatom_spow)
        .def_ro("bond_atom", &MolZAtom::mBondAtom)
        .def_ro("bond_angle_atom", &MolZAtom::mBondAngleAtom)
        .def_ro("dihdral_angle_atom", &MolZAtom::mDihedralAtom)
        .def_ro("bond_length", &MolZAtom::mBondLength)
        .def_ro("bond_angle", &MolZAtom::mBondAngle)
        .def_ro("dihdral_angle", &MolZAtom::mDihedralAngle)
        ;
}
