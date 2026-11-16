/*
 * pyobjcryst nanobind port -- ObjCryst release hooks
 */

#ifndef PYOBJCRYST_NB_RELEASE_HOOKS_HPP
#define PYOBJCRYST_NB_RELEASE_HOOKS_HPP

// Install the ObjCryst::ReleaseObj hooks (called once at module import).
void install_release_hooks();

// While an instance of this guard exists, molecule restraints and rigid
// groups kept alive by a Python wrapper also keep their atoms alive. Only use
// it when all atoms of the released restraints are known to be valid (e.g.
// not from ~Molecule, which releases atoms before bonds).
struct KeepRestraintAtomsAlive
{
    KeepRestraintAtomsAlive();
    ~KeepRestraintAtomsAlive();
    KeepRestraintAtomsAlive(const KeepRestraintAtomsAlive&) = delete;
    KeepRestraintAtomsAlive& operator=(const KeepRestraintAtomsAlive&) = delete;
};

#endif
