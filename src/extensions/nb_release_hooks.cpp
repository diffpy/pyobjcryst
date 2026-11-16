/*
 * pyobjcryst nanobind port -- ObjCryst release hooks
 *
 * ObjCryst deletes the sub-objects it owns (reflection profiles, refinable
 * parameters, scatterers, molecule atoms, ...) through ObjCryst::ReleaseObj().
 * Python may still hold wrappers for these objects. Without the hooks below
 * such wrappers dangle, and nanobind may even return a stale wrapper (with
 * the wrong type) for a new object allocated at the same address.
 *
 * The hooks look for live Python wrappers of the released object:
 * - no wrapper: return false, ObjCryst deletes the object;
 * - a Python-owned wrapper exists: return true, that wrapper already owns the
 *   object and deletes it when it is garbage-collected;
 * - only non-owning wrappers exist: return true and attach a deleter to one
 *   of them, so the object is deleted when that wrapper dies. The object
 *   memory stays allocated as long as any wrapper is alive, so no new object
 *   can be created at the same address.
 *
 * RefinablePar is special: its value pointer refers to storage owned by its
 * former parent, which may already be freed or reallocated. Wrappers of a
 * released parameter are therefore invalidated (any further use raises a
 * TypeError), and the parameter is only kept allocated until they die.
 */

#include <nanobind/nanobind.h>

#include <type_traits>
#include <typeinfo>
#include <unordered_set>
#include <vector>

#undef B0
#include <ObjCryst/RefinableObj/RefinableObj.h>
#include <ObjCryst/ObjCryst/Atom.h>
#include <ObjCryst/ObjCryst/Molecule.h>
#include <ObjCryst/ObjCryst/PowderPattern.h>
#include <ObjCryst/ObjCryst/ScatteringCorr.h>
#include <ObjCryst/ObjCryst/ScatteringData.h>
#include <ObjCryst/ObjCryst/ScatteringPower.h>
#include <ObjCryst/ObjCryst/ZScatterer.h>

#include "nb_release_hooks.hpp"

namespace nb = nanobind;
using namespace ObjCryst;

namespace {

struct Probe
{
    const std::type_info* type;
    const void* addr;
};

int restraint_atoms_alive = 0;

// Objects currently kept alive by a deleter attached to a wrapper. Only
// accessed with the GIL held.
std::unordered_set<const void*>& adopted()
{
    static auto* s = new std::unordered_set<const void*>();
    return *s;
}

// Look up an existing wrapper registered at `addr` whose type is compatible
// with `type` or with the dynamic type `dyn`. Never returns a new wrapper.
void find_wrapper(const Probe& probe, const std::type_info* dyn,
                  std::vector<nb::object>& found)
{
    if (probe.addr == nullptr) return;
    bool is_new = false;
    PyObject* o = NB_CALL(nb_type_put)(
        NB_CTX, probe.type, dyn, const_cast<void*>(probe.addr),
        nb::rv_policy::none, nullptr, &is_new);
    if (o == nullptr)
    {
        PyErr_Clear();
        return;
    }
    nb::object w = nb::steal(o);
    if (w.is_none()) return;
    if (is_new)
    {
        // nanobind creates a fresh wrapper when an unrelated instance is
        // registered at the same address. Discard it without running any
        // C++ destructor.
        nb::inst_set_state(w, true, false);
        return;
    }
    for (const nb::object& f : found)
        if (f.is(w)) return;
    found.push_back(std::move(w));
}

std::vector<nb::object> find_wrappers(const std::vector<Probe>& probes,
                                      const std::type_info* dyn)
{
    std::vector<nb::object> found;
    for (const Probe& p : probes) find_wrapper(p, dyn, found);
    return found;
}

template <class Target, class T>
void add_probe(std::vector<Probe>& probes, T* p)
{
    if (Target* t = dynamic_cast<Target*>(p))
        probes.push_back({&typeid(Target), static_cast<const void*>(t)});
}

template <class T> const std::type_info* dynamic_type(T* p)
{
    if constexpr (std::is_polymorphic_v<T>) return &typeid(*p);
    else return &typeid(T);
}

template <class T> void delete_adopted(void* p) noexcept
{
    adopted().erase(p);
    try { delete static_cast<T*>(p); }
    catch (...) {}
}

// Hand ownership of `p` to its Python wrappers. `invalidate` marks all
// non-owning wrappers as unusable.
template <class T>
nb::handle adopt(T* p, std::vector<nb::object>& wrappers, bool invalidate)
{
    nb::handle owner;
    for (const nb::object& w : wrappers)
        if (nb::inst_state(w).second)
        {
            owner = w;
            break;
        }
    // A Python-owned wrapper already deletes the object when it dies. Its
    // object is still valid, even for a RefinablePar (which then uses its
    // own storage).
    const bool python_owned = owner.is_valid();
    if (!python_owned)
    {
        owner = wrappers.front();
        adopted().insert(p);
        nb::keep_alive_cb(owner, p, &delete_adopted<T>);
    }
    for (const nb::object& w : wrappers)
    {
        if (invalidate && !python_owned) nb::inst_set_state(w, false, false);
        if (!w.is(owner)) nb::keep_alive_obj(w, owner);
    }
    return owner;
}

nb::object wrap_scattering_power(const ScatteringPower& sp)
{
    ScatteringPower* psp = const_cast<ScatteringPower*>(&sp);
    if (auto* spa = dynamic_cast<ScatteringPowerAtom*>(psp))
        return nb::cast(spa, nb::rv_policy::reference);
    if (auto* gsp = dynamic_cast<GlobalScatteringPower*>(psp))
        return nb::cast(gsp, nb::rv_policy::reference);
    return nb::cast(psp, nb::rv_policy::reference);
}

// ~Atom deregisters from its scattering power, which its former Crystal may
// delete later. Keep the scattering power alive as long as the atom: if the
// Crystal releases it, the hook then finds this wrapper and adopts it too.
void pin_dependencies(RefinableObj* p, nb::handle owner)
{
    if (auto* atom = dynamic_cast<Atom*>(p))
        if (!atom->IsDummy())
            nb::keep_alive_obj(owner, wrap_scattering_power(atom->GetScatteringPower()));
}

void no_pin(void*, nb::handle) {}

void pin_atom(const MolAtom& atom, nb::handle owner)
{
    nb::object w = nb::cast(const_cast<MolAtom*>(&atom), nb::rv_policy::reference);
    nb::keep_alive_obj(owner, w);
}

// A restraint or rigid group kept alive by Python must not outlive its atoms.
// The atoms are only accessed in a KeepRestraintAtomsAlive scope: e.g.
// ~Molecule releases atoms before bonds.
void pin_atoms(MolBond* b, nb::handle owner)
{
    if (restraint_atoms_alive == 0) return;
    pin_atom(b->GetAtom1(), owner);
    pin_atom(b->GetAtom2(), owner);
}

void pin_atoms(MolBondAngle* b, nb::handle owner)
{
    if (restraint_atoms_alive == 0) return;
    pin_atom(b->GetAtom1(), owner);
    pin_atom(b->GetAtom2(), owner);
    pin_atom(b->GetAtom3(), owner);
}

void pin_atoms(MolDihedralAngle* b, nb::handle owner)
{
    if (restraint_atoms_alive == 0) return;
    pin_atom(b->GetAtom1(), owner);
    pin_atom(b->GetAtom2(), owner);
    pin_atom(b->GetAtom3(), owner);
    pin_atom(b->GetAtom4(), owner);
}

void pin_atoms(RigidGroup* g, nb::handle owner)
{
    if (restraint_atoms_alive == 0) return;
    for (MolAtom* a : *g) pin_atom(*a, owner);
}

template <class T, class Pin>
bool release_hook(T* p, const std::vector<Probe>& probes, bool invalidate,
                  Pin pin) noexcept
{
    if (p == nullptr || !Py_IsInitialized() || !nb::is_alive()) return false;
    PyGILState_STATE gstate = PyGILState_Ensure();
    bool keep = false;
    {
        nb::error_scope scope;
        try
        {
            if (adopted().count(p) != 0)
                keep = true;
            else
            {
                std::vector<nb::object> wrappers =
                    find_wrappers(probes, dynamic_type(p));
                if (!wrappers.empty())
                {
                    // Even if something fails below, a wrapper still refers
                    // to the object: leaking it is the safe choice.
                    keep = true;
                    nb::handle owner = adopt(p, wrappers, invalidate);
                    pin(p, owner);
                }
            }
        }
        catch (...)
        {
        }
        PyErr_Clear();
    }
    PyGILState_Release(gstate);
    return keep;
}

bool release_refinableobj(RefinableObj* p) noexcept
{
    std::vector<Probe> probes;
    try
    {
        probes.push_back({&typeid(RefinableObj), p});
        add_probe<Scatterer>(probes, p);
        add_probe<ScatteringPower>(probes, p);
        add_probe<ScatteringPowerAtom>(probes, p);
        add_probe<GlobalScatteringPower>(probes, p);
        add_probe<ScatteringData>(probes, p);
        add_probe<PowderPatternComponent>(probes, p);
        add_probe<PowderPatternDiffraction>(probes, p);
        add_probe<PowderPatternBackground>(probes, p);
        add_probe<TextureMarchDollase>(probes, p);
        probes.push_back({&typeid(*p), dynamic_cast<const void*>(p)});
    }
    catch (...)
    {
        return false;
    }
    return release_hook(p, probes, false, &pin_dependencies);
}

bool release_refinablepar(RefinablePar* p) noexcept
{
    std::vector<Probe> probes;
    try
    {
        probes.push_back({&typeid(RefinablePar), p});
        probes.push_back({&typeid(Restraint), static_cast<Restraint*>(p)});
    }
    catch (...)
    {
        return false;
    }
    return release_hook(p, probes, true, &no_pin);
}

template <class T> bool release_restraint(T* p) noexcept
{
    std::vector<Probe> probes;
    try
    {
        probes.push_back({&typeid(T), p});
        probes.push_back({&typeid(Restraint), static_cast<Restraint*>(p)});
    }
    catch (...)
    {
        return false;
    }
    return release_hook(p, probes, false, [](T* r, nb::handle o) { pin_atoms(r, o); });
}

template <class T> bool release_plain(T* p) noexcept
{
    std::vector<Probe> probes;
    try
    {
        probes.push_back({&typeid(T), p});
    }
    catch (...)
    {
        return false;
    }
    return release_hook(p, probes, false, &no_pin);
}

bool release_rigidgroup(RigidGroup* p) noexcept
{
    std::vector<Probe> probes;
    try
    {
        probes.push_back({&typeid(RigidGroup), p});
    }
    catch (...)
    {
        return false;
    }
    return release_hook(p, probes, false, [](RigidGroup* r, nb::handle o) { pin_atoms(r, o); });
}

} // namespace

KeepRestraintAtomsAlive::KeepRestraintAtomsAlive() { ++restraint_atoms_alive; }
KeepRestraintAtomsAlive::~KeepRestraintAtomsAlive() { --restraint_atoms_alive; }

void install_release_hooks()
{
    SetReleaseObjHook(RefinableObjReleaseHook(&release_refinableobj));
    SetReleaseObjHook(RefinableParReleaseHook(&release_refinablepar));
    SetReleaseObjHook(MolAtomReleaseHook(&release_plain<MolAtom>));
    SetReleaseObjHook(MolBondReleaseHook(&release_restraint<MolBond>));
    SetReleaseObjHook(MolBondAngleReleaseHook(&release_restraint<MolBondAngle>));
    SetReleaseObjHook(MolDihedralAngleReleaseHook(&release_restraint<MolDihedralAngle>));
    SetReleaseObjHook(RigidGroupReleaseHook(&release_rigidgroup));
}
