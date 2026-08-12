# nanobind Migration Notes for pyobjcryst

## Status (2026-08-12)

**The migration is functionally complete for the audited API surface, with known, accepted
gaps listed below — it is not "fully done" in the sense of zero open items.**

What's done and verified:
- The C++ binding layer is fully ported from Boost.Python + SCons to nanobind + CMake +
  scikit-build-core. `src/pyobjcryst/*.py` (the pure-Python layer) and `tests/*.py` are untouched —
  nothing there is Boost.Python-specific.
- All 45 `nb_*.cpp` binding files (`src/extensions/`) have real implementations covering the same
  class surface as the old `*_ext.cpp` files, built against `extern/objcryst`
  (`vincefn/objcryst@bfdcfda`, vendored as a submodule).
- A systematic audit (diffing every Python-exposed name in the old Boost.Python bindings against
  the nanobind ones) found and fixed every discovered regression, most recently a significant one:
  `Scatterer`, `ScatteringPower`, `ScatteringData`, and `PowderPatternComponent` (and everything
  under them) had lost most of `RefinableObj`'s method surface because of a virtual-inheritance
  limitation in nanobind — see [Virtual Inheritance](#virtual-inheritance) below. That fix is
  covered by real, committed tests.
- **161/161 tests pass** (`pytest tests/ -q`).

Known, accepted gaps — real, not oversights, but still open:
- **Python-side method override does not work across the `RefinableObj` virtual-inheritance
  seam.** A Python subclass of `Atom`, `Molecule`, `ScatteringPowerAtom`,
  `DiffractionDataSingleCrystal`, `PowderPatternBackground`, etc. that overrides an inherited
  `RefinableObj` method (e.g. `UpdateDisplay`) will **not** have that override seen by
  C++-internal virtual dispatch — only plain forwarding calls were restored, not overridability.
  This was an explicit scope decision (general override support was confirmed not needed), not an
  oversight, but it is a real regression vs. Boost.Python's `bases<RefinableObj>`, which handled
  this transparently. Override support that *does* still work: `RefinableObj`'s own ~15 virtuals
  when subclassing `RefinableObj` directly; `Scatterer`/`ScatteringPower`/`Restraint`'s own
  (non-inherited) virtuals; and `UpdateDisplay` specifically on `Crystal`, `PowderPattern`, and
  `MonteCarloObj` (all three reach `RefinableObj` through a non-virtual chain, so no seam exists
  there). See [Virtual Inheritance](#virtual-inheritance) for why this is a hard nanobind
  limitation, not a bug.
- **Reference-counting leak**: `nanobind: leaked N instances!` warnings print at interpreter exit.
  The `Molecule`↔`MolAtom` cycle is fixed. `Crystal`↔`Scatterer`/`Molecule`/`ScatteringPower` cycles
  remain (56 leaked instances in the current full-suite run) — not safe to fix the same way, see
  [keep_alive Reference Cycles](#keep_alive-reference-cycles). Not blocking functionality.
- **CI has not been exercised against this build.** `.github/workflows/` is inherited from `main`
  unmodified (it calls a shared reusable workflow); it has never actually run against the
  nanobind/CMake build because nothing on this branch has been pushed yet.
- **Cross-platform**: everything above has been built and tested on Linux only. macOS/Windows are
  unverified.
- **Performance**: not benchmarked against Boost.Python.
- **Versioning**: `pyproject.toml` uses a static placeholder version (`2024.2.1.dev0`) instead of
  `main`'s dynamic tag-based versioning (`setuptools-git-versioning`) — scikit-build-core needs a
  different mechanism for that; flagged as a `TODO` in `pyproject.toml` itself, not yet resolved.

None of the above blocks day-to-day use of the library from Python — every method that was
reachable under Boost.Python is reachable under nanobind (modulo the override caveat above, which
only matters if you're writing a Python subclass that overrides a virtual method).

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Building & Testing](#building--testing)
4. [Virtual Inheritance](#virtual-inheritance)
5. [Other Recurring Gotchas](#other-recurring-gotchas)
6. [Binding Patterns Quick Reference](#binding-patterns-quick-reference)
7. [Architecture Decisions](#architecture-decisions)
8. [Pitfalls Checklist](#pitfalls-checklist)

---

## Overview

**Why nanobind instead of Boost.Python:** much less code per binding (~20-40 lines vs. 70+),
faster compilation, smaller binaries, and native `scikit-build-core`/`pyproject.toml` integration
instead of SCons. The trade-off, discovered during this migration, is that nanobind's automatic
base↔derived pointer adjustment does not support C++ virtual inheritance at all — which ObjCryst++
uses extensively (`Scatterer`, `ScatteringPower`, `ScatteringData`, `PowderPatternComponent` all
inherit `RefinableObj` virtually) — so a large fraction of the actual migration work was working
around that one limitation. [Virtual Inheritance](#virtual-inheritance) covers this in full.

---

## Architecture

### File structure

```
pyobjcryst/
├── extern/
│   ├── objcryst/     # ObjCryst++ C++ core (git submodule, vincefn/objcryst@bfdcfda)
│   ├── newmat/       # matrix library (vendored)
│   └── cctbx/        # crystallography utilities (vendored, hand-pared subset)
├── src/
│   ├── pyobjcryst/   # Python package layer — unchanged, not Boost.Python-specific
│   └── extensions/   # nanobind C++ bindings: nb_module.cpp, helpers_nb.hpp, 45 nb_*.cpp files
├── CMakeLists.txt
├── pyproject.toml    # scikit-build-core backend
└── tests/
```

### Build pipeline

```
pyproject.toml (scikit-build-core)
    -> CMakeLists.txt
         -> extern/newmat/  -> static libNewmat
         -> extern/cctbx/   -> static libCCTBX
         -> extern/objcryst/-> static libObjCryst  (REAL=double forwarded via target_compile_definitions)
         -> src/extensions/ -> nanobind extension _pyobjcryst (links all static libs)
    -> wheel: pyobjcryst/*.py + _pyobjcryst.so
```

Python-facing API is unchanged: `from pyobjcryst import Crystal, SpaceGroup, Atom, ...` still
works, re-exported from `pyobjcryst._pyobjcryst`.

---

## Building & Testing

### Environment

```bash
# Miniforge/mamba, created OUTSIDE the git checkout. Tested with python=3.14.
mamba create -n pyobjcryst-nb python=3.14 numpy nanobind scikit-build-core \
    ninja cmake libboost-devel libboost-headers libboost
conda activate pyobjcryst-nb
git submodule update --init   # fetches extern/objcryst
```

**Compiler note:** if the host's glibc is older than conda-forge's bundled GCC assumes
(`GLIBC_ABI_GNU2_TLS` error at import time — conda-forge GCC 14+ defaults to the `gnu2` TLS
dialect), build with the system compiler instead, keeping the rest of the toolchain from conda:

```bash
CC=/usr/bin/gcc CXX=/usr/bin/g++ pip install -e . --no-build-isolation -Cbuild-dir=build
```

### Build & test

```bash
pip install -e . --no-build-isolation           # in-place / development
pip wheel . -w dist/                             # production wheel

pytest tests/ -q                                 # full suite: 161 passed
```

A bare `cmake --build build` refreshes the compiled extension but **not** the editable install —
after any C++ change, re-run the `pip install -e .` command above (with `-Cbuild-dir=build` to
reuse the existing build directory rather than reconfiguring from scratch).

`nanobind: leaked N instances!` / `leaked keep_alive records!` / `leaked types!` warnings print
after every test run — pre-existing, tracked in [Status](#status-2026-08-12) and
[keep_alive Reference Cycles](#keep_alive-reference-cycles), not currently blocking anything.

### Build troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `nanobind not found` | not installed | `conda install nanobind` |
| `ObjCryst submodule empty` | not initialized | `git submodule update --init --recursive` |
| `version.h: No such file` | CMake config not run | `pip install -e . --no-build-isolation` |
| Segfault / bus error in a test | pointer-offset bug (virtual inheritance) | check whether the crashing class's C++ header declares `virtual` inheritance anywhere in its ancestry; see [Virtual Inheritance](#virtual-inheritance) |
| `RuntimeError: std::bad_cast`, no useful location | usually `nb::cast<std::string>` on a `bytes` object | check every `.attr("read")()` call site — see [Python file-like input](#python-file-like-input-str-vs-bytes) |
| Binary needs a newer glibc than the host ships (`GLIBC_ABI_GNU2_TLS`) | conda-forge GCC's `gnu2` TLS dialect | build with the system compiler: `CC=/usr/bin/gcc CXX=/usr/bin/g++ pip install -e . --no-build-isolation` |
| gdb: `(no debugging symbols found)` | scikit-build-core strips symbols on install | `pip install -e . --no-build-isolation -Ccmake.define.CMAKE_BUILD_TYPE=RelWithDebInfo`, then copy the unstripped `.so` over the installed one |
| Values off in the last few digits | `REAL` (`ObjCryst/ObjCryst/General.h`) defaults to `float`; must be forced to `double` | confirm `CMakeLists.txt` sets `target_compile_definitions(ObjCryst PUBLIC REAL=double ...)`, and that any helper uses the `CrystVector_REAL`/`CrystMatrix_REAL` macros rather than a hardcoded `<float>`/`<double>` |

Useful gdb recipe for a segfault/bad_cast inside the extension:
```bash
gdb -q --args python -m pytest tests/test_whatever.py -k test_name
(gdb) catch throw   # for std::bad_cast / C++ exceptions
(gdb) run
(gdb) bt
```

---

## Virtual Inheritance

**This is the single largest source of bugs in this migration — read this section before binding
or extending any class near `RefinableObj`, `Scatterer`, `ScatteringPower`, `ScatteringData`, or
`PowderPatternComponent`.**

### The core problem

ObjCryst++ uses *virtual* inheritance in several places: `Scatterer : virtual public
RefinableObj`, `ScatteringPower : virtual public RefinableObj`, `ScatteringData : virtual public
RefinableObj`, `PowderPatternComponent : virtual public RefinableObj`, and a second hop for
`ScatteringPowerAtom : virtual public ScatteringPower` and `GlobalScatteringPower : virtual public
ScatteringPower`.

nanobind's automatic base↔derived pointer adjustment (used by `nb::class_<Derived, Base>(...)`)
assumes a **fixed, statically computable** offset between `Derived*` and `Base*`. That's only
valid for *non-virtual* inheritance. Declaring the nanobind base across a virtual link does **not**
raise an error — it silently computes a *wrong* offset, which can look fine for fields at offset 0
and then segfault or return garbage the moment a method touching state further into the object is
called.

**Rule of thumb: if class `D` inherits `virtual`ly from `B` anywhere in its hierarchy, never write
`nb::class_<D, B>(...)`.** Register `D` standalone and handle every base↔derived crossing by hand.
Grep the whole `extern/objcryst/ObjCryst` tree for `": *virtual public"` before starting new
binding work — it's cheap and saves a debugging session per class. Plain (non-virtual) inheritance
is fine and should still use nanobind's automatic base declaration (e.g. `ScatteringPowerSphere :
public ScatteringPower`, no `virtual`, is safely `nb::class_<ScatteringPowerSphere,
ScatteringPower>`).

### Four ways this bites, and the fix for each

**1. Return direction** — a function returns a base-typed reference/pointer
(`ScatteringPower&`) whose dynamic type is a virtually-derived leaf (`ScatteringPowerAtom`). Fix:
`dynamic_cast` to the concrete type before handing it to nanobind, so it wraps an already-exact-type
pointer (offset-0-safe):

```cpp
nb::object wrapScatteringPowerReturn(const ScatteringPower* sp, nb::handle parent) {
    if (!sp) return nb::none();
    if (auto* spa = dynamic_cast<const ScatteringPowerAtom*>(sp))
        return nb::cast(const_cast<ScatteringPowerAtom*>(spa), nb::rv_policy::reference_internal, parent);
    return nb::cast(const_cast<ScatteringPower*>(sp), nb::rv_policy::reference_internal, parent);
}
```
Applies even when the returned type is an *intermediate* base, not the ultimate leaf (e.g.
`PowderPattern::GetPowderPatternComponent(i)` returns `PowderPatternComponent&`, whose dynamic type
is `PowderPatternDiffraction`) — every function returning a shared ancestor type anywhere in a
virtual hierarchy needs this, not just the one function returning the immediate leaf type.

**2. Argument direction** — a virtually-derived leaf is registered *without* a declared nanobind
base (per the rule above), so nanobind refuses it wherever a C++ function's parameter type is the
*base* (`void AddAtom(..., const ScatteringPower* pow, ...)` rejects a Python `ScatteringPowerAtom`
with `TypeError: incompatible function arguments`). Fix: accept `nb::object` and resolve manually,
trying the concrete leaf type first:

```cpp
const ScatteringPower* extractScatteringPowerArg(nb::object obj) {
    if (obj.is_none()) return nullptr;
    try { return static_cast<const ScatteringPower*>(&nb::cast<ScatteringPowerAtom&>(obj)); } catch (...) {}
    return &nb::cast<const ScatteringPower&>(obj);
}
```

**3. Generic "any `RefinableObj`" argument functions** — the same argument-direction problem, but
for functions generic over *any* `RefinableObj` (`MonteCarloObj::AddRefinableObj`,
`LSQNumObj::SetRefinedObj`, `RefinableObj::AddPar`): nanobind has no conversion path from a Python
`DiffractionDataSingleCrystal`/`Atom`/`Molecule`/etc. to `RefinableObj&`, since none of those
concrete classes declare that base. Fix: a shared helper (`extractRefinableObjArg`, in
`helpers_nb.hpp`/`nb_refinableobj.cpp`) trying the safe cast first, then falling back through every
known virtually-linked type with `static_cast` (always well-defined at compile time — the problem
is only ever nanobind's own *runtime* offset table, never a real `static_cast`):

```cpp
RefinableObj& extractRefinableObjArg(nb::object obj) {
    try { return nb::cast<RefinableObj&>(obj); } catch (...) {}
    try { return static_cast<RefinableObj&>(nb::cast<ScatteringData&>(obj)); } catch (...) {}
    try { return static_cast<RefinableObj&>(nb::cast<PowderPatternComponent&>(obj)); } catch (...) {}
    try { return static_cast<RefinableObj&>(nb::cast<Scatterer&>(obj)); } catch (...) {}
    try { return static_cast<RefinableObj&>(nb::cast<ScatteringPower&>(obj)); } catch (...) {}
    try { return static_cast<RefinableObj&>(nb::cast<ScatteringPowerAtom&>(obj)); } catch (...) {}
    return nb::cast<RefinableObj&>(obj);  // nothing matched -- raise a clear TypeError
}
```

**4. Under-declaring a *safe* base** — the mirror-image mistake: forgetting to declare a nanobind
base across a link that's perfectly safe (plain, non-virtual). Symptom looks identical
(`TypeError: incompatible function arguments`) but the cause is a *missing* declaration, not a
dangerous one. Fix: register the base type too (it needs no bindings of its own, just an internal
name), then declare the relationship normally:
```cpp
nb::class_<PeakList>(m, "_PeakListBase");          // internal-only
nb::class_<PeakListNB, PeakList>(m, "PeakList");   // now declared
```

### Restoring `RefinableObj`'s method surface on virtually-inherited classes

Because `Scatterer`, `ScatteringPower`, `ScatteringData`, and `PowderPatternComponent` can never
nanobind-declare `RefinableObj` as their base (per the rule above), none of their ~44 inherited
methods (`Print`, `XMLOutput`/`XMLInput`, `BeginOptimization`/`EndOptimization`,
`GetLogLikelihood`, `UpdateDisplay`, `AddPar`/`RemovePar`, `CreateParamSet`, `GetOption`, ...) are
reachable from Python by default — under Boost.Python, `bases<RefinableObj>` handled this
transparently via real casts, so this is a genuine regression, not a pre-existing limitation.

Fix: two reusable template helpers in `helpers_nb.hpp`, `bind_refinableobj_forwarding<T>()` and
`bind_scatteringpower_forwarding<T>()`, each binding one method-set's full surface via plain
forwarding `.def()` calls (`&T::Method`, or a lambda for anything needing a helper/numpy
conversion). This is safe despite the virtual base because `T` is fully known at compile time at
the call site, so the *compiler* resolves the pointer adjustment — exactly the mechanism
Boost.Python relied on, nothing to do with nanobind's own (virtual-base-unsafe) casting. Called
once each in `nb_scatterer.cpp`, `nb_scatteringpower.cpp`, `nb_scatteringdata.cpp`, and
`nb_powderpatterncomponent.cpp` (cascades automatically to every non-virtually-inherited child —
`Atom`, `Molecule`, `ZScatterer`, `PowderPatternBackground`, `PowderPatternDiffraction`, etc. — via
ordinary Python MRO); `ScatteringPowerAtom` and `GlobalScatteringPower` need both helpers called
directly on them, since they have a *second* virtual-inheritance hop of their own.

**This restores callability, not overridability** — see [Status](#status-2026-08-12) for why that
distinction matters and is a deliberate, accepted scope limit (nanobind's `NB_TRAMPOLINE` macro
requires its argument to be a *direct* base, a hard C++ rule that blocks chained/templated
trampolines the way pybind11 supports them — confirmed by compiler error in a standalone repro,
not a nanobind design choice that could be worked around).

Two real bugs turned up while adding test coverage for this fix, both worth remembering as general
lessons:
- A binding that reaches a protected member via a trampoline/alias helper class (here,
  `PyScatterer::GetClockScattCompList()`, added to reach `Scatterer`'s protected
  `mClockScattCompList`) must take the **real base type** (`Scatterer&`) as its bound parameter,
  not the trampoline/alias type (`PyScatterer&`) — a plain `Atom`/`Molecule` instance is never
  actually constructed as the trampoline, so binding against the alias type raises `TypeError` for
  every ordinary instance. Cast through the alias *inside* the lambda instead:
  ```cpp
  .def("GetClockScattCompList",
       [](Scatterer& s) -> const RefinableObjClock& {
           return static_cast<PyScatterer&>(s).GetClockScattCompList();  // safe: no added data members
       }, nb::rv_policy::reference_internal)
  ```
- `bind_scatteringpower_forwarding<T>()` forwarded `GetBij`/`SetBij` but missed the `B11`/`B12`/
  `B13`/`B22`/`B23`/`B33`/`Biso` *properties* that `nb_scatteringpower.cpp` defines separately via
  file-local templates — a reminder that a forwarding helper needs to be checked against the full
  `dir()` of the canonical class, not just its method list, when a class also exposes properties.

---

## Other Recurring Gotchas

### `None` argument handling

nanobind rejects `None` by default for any typed `nb::arg`. Allow it explicitly:
```cpp
.def("AddScatterer", &Crystal::AddScatterer, nb::arg("pScatt").none(), nb::arg("ownsScatt") = true)
```
`.none()` is required even for generic `nb::object` parameters — it isn't automatic just because
the C++ parameter type is generic.

**Limit**: `.none()` only bridges `None -> nullptr` for `nb::object` parameters and pointers to a
`nb::class_`-registered type. It does **not** work for a pointer to a plain arithmetic type (e.g.
`REAL* derivpar` in `RecUnitCell::hkl2d`) — nanobind's arithmetic `type_caster<double>` rejects
`None` outright before `.none()` on the outer `nb::arg` ever gets a chance to matter. Fix: wrap in
a helper taking `nb::object` and bridge manually:
```cpp
float _hkl2d(const RecUnitCell& r, float h, float k, float l, nb::object derivpar, unsigned int derivhkl) {
    if (derivpar.is_none()) return r.hkl2d(h, k, l, nullptr, derivhkl);
    REAL d = 0;
    return r.hkl2d(h, k, l, &d, derivhkl);  // Python float is immutable -- there's no way to write
}                                             // the derivative back to the caller regardless of binding tech
```

### Custom container types (`CrystVector`/`CrystMatrix`)

nanobind has no built-in support for ObjCryst++'s custom matrix library. Convert explicitly via a
heap-allocated copy wrapped in an `nb::ndarray` capsule (see `crystvec_to_array`/
`crystmat_to_array` in `helpers_nb.hpp`). Always use the `CrystVector_REAL`/`CrystMatrix_REAL`
macros, never a hardcoded `<float>`/`<double>`, so a future `REAL` macro change doesn't silently
produce wrong-precision code.

### Lambda return type silently deduced by value

A `.def(...)` that wraps a reference-returning helper in a lambda **without an explicit trailing
return type** deduces `auto`, which strips the reference and returns *by value* — silently
*copying* the object, even with `nb::rv_policy::reference_internal` set (that policy only controls
how nanobind wraps whatever the lambda actually returns; it can't recover a reference from an
already-copied value). This caused a real bug: `Molecule.AddRigidGroup`'s lambda copied the
`RigidGroup` it had just added, so `Molecule.RemoveRigidGroup` (which finds the group to remove by
pointer identity) silently became a no-op with no error anywhere.

Fix: always give the lambda an explicit trailing return type when it wraps a reference-returning
helper: `[](...) -> RigidGroup& { return _AddRigidGroupIterable(...); }`. Rule of thumb: any
`.def()` combining a lambda with `nb::rv_policy::reference`/`reference_internal` needs a `-> T&`/
`-> T*` on the lambda. Binding a bare function pointer directly has no such problem — nothing is
deduced.

### `keep_alive` reference cycles

`nanobind: leaked N instances!` warnings are a **real leak**, not a shutdown-ordering artifact —
confirmed by a plain, non-pytest script that creates a `Molecule`, calls `AddBondAngle` once, drops
all references, and calls `gc.collect()` twice: still leaks 6 instances at exit. Root cause:
nanobind types don't participate in Python's cyclic GC by default (no `tp_traverse`/`tp_clear`),
so a bidirectional `keep_alive` between two live objects — parent keeps child alive (a `.def()`
declares `keep_alive<1,N>()`) *and* child keeps parent alive (a `.def()` returns via
`rv_policy::reference_internal`, nanobind's shorthand for `keep_alive<0,1>()`) — forms a cycle
nothing can ever collect.

Fixed for `Molecule`↔`MolAtom`: removed the redundant Molecule-keeps-atom direction from
`AddBond`/`AddBondAngle`/`AddDihedralAngle` (safe because `MolAtom` has no standalone Python-owned
existence — its only constructor is a copy constructor, so it's always reached via
`GetAtom()`/`AddAtom()`, which already protects it via the *other* direction).

**Not safe to fix the same way**, and still open: `Crystal`↔`Scatterer`/`Molecule`/
`ScatteringPower` (and possibly `RefinableObj`↔`RefinablePar`). Unlike `MolAtom`, these classes
*do* have real standalone Python-owned constructors (`sp = ScatteringPowerAtom("C", "C")` with no
parent) — the forward direction is protecting a real gap (between construction and being added to
a `Crystal`, nothing else keeps the child's wrapper alive), so removing it would trade a leak for a
use-after-free. The real fix is nanobind's `tp_traverse`/`tp_clear` mechanism, which likely requires
replacing `keep_alive<>` with explicit `nb::object` members on every class in each cycle — a
substantial redesign, not attempted. When auditing a new class, grep for `keep_alive<1,` and check
whether the same child type also has an accessor using `rv_policy::reference_internal` back onto
the parent — that combination is the signature to look for.

### Python file-like input (`str` vs `bytes`)

Any binding calling `.attr("read")()` on a caller-supplied Python object and casting straight to
`std::string` breaks the moment the file was opened in binary mode (`.read()` returns `bytes`,
`nb::cast<std::string>` throws `std::bad_cast`). Use a shared helper:
```cpp
inline std::string read_pyfile_to_string(nb::object input) {
    nb::object data = input.attr("read")();
    if (nb::isinstance<nb::bytes>(data)) {
        nb::bytes b = nb::cast<nb::bytes>(data);
        return std::string(b.c_str(), b.size());
    }
    return nb::cast<std::string>(data);
}
```

### `bases<>` mixing in an unrelated class has no nanobind equivalent

Boost.Python's `bases<>` can mix a wrapped class's Python methods into another wrapped class
regardless of the real C++ inheritance graph — used to give `RigidGroup` (a real
`std::set<MolAtom*>` subclass) a Python-set interface (`add`/`discard`/`remove`/`update`/
`__contains__`/`__getitem__`) via an unrelated wrapper class, `MolAtomSet`. nanobind's
`nb::class_<T, Base>` requires `Base` to actually be `T`'s C++ base (or `static_cast`-reachable),
so this doesn't carry over — the interface has to be rebound directly on the real class instead
(done for `RigidGroup` in `nb_rigidgroup.cpp`, including a new `__iter__` nanobind classes don't
get for free the way legacy `__getitem__`-fallback iteration did). If auditing for other silent
interface losses like this, look for any Boost.Python `bases<X, Y>` where `Y` isn't actually `X`'s
C++ base.

### A vendored C++ core bug, not a binding bug

`MonteCarloObj`'s optimizer defaults to auto-saving progress, formatting the current
log-likelihood into a filename with `sprintf(buf[30], "...%f", ...)`. For a bad starting
configuration the log-likelihood can be enormous, overflowing the 30-byte buffer; glibc's
fortified `sprintf` aborts the process rather than corrupting the stack — surfacing as an
uncatchable `Fatal Python error: Aborted`, not a Python exception. Fixed in the vendored
`extern/objcryst/ObjCryst/RefinableObj/GlobalOptimObj.cpp` itself (five sites): widened buffers,
switched to `snprintf`, and switched `%f` -> `%g` (bounded length regardless of magnitude). Worth
remembering if `extern/objcryst` is ever re-synced from upstream — that sync would silently drop
this fix unless it's re-applied or merged upstream.

---

## Binding Patterns Quick Reference

**Simple class, no virtuals:**
```cpp
nb::class_<MyClass>(m, "MyClass")
    .def(nb::init<>())
    .def("method", &MyClass::method)
    .def_prop_rw("prop", &MyClass::GetProp, &MyClass::SetProp)
    ;
```

**Class with overridable virtuals (trampoline) — only where `Base` is a *direct*, non-virtual
base of the trampoline, per [Virtual Inheritance](#virtual-inheritance):**
```cpp
struct PyMyClass : MyClass {
    NB_TRAMPOLINE(MyClass, 2);  // 2 virtual methods
    int VirtualMethod1() override { NB_OVERRIDE(int, VirtualMethod1); }
    void VirtualMethod2(int x) override { NB_OVERRIDE(void, VirtualMethod2, x); }
};
nb::class_<MyClass, PyMyClass>(m, "MyClass").def(nb::init<>());
```

**String representation** (many ObjCryst objects have a useful `operator<<`):
```cpp
.def("__str__", [](const T& obj){ return obj_str(obj); })   // obj_str() in helpers_nb.hpp, mutes debug spam
.def("__repr__", [](const T& obj){ return obj_str(obj); })
```

**Method overload resolution** — nanobind can't distinguish C++ overloads at binding time; use a
lambda or a pointer-to-member-function cast:
```cpp
.def("SetX", nb::overload_cast<const REAL>(&T::SetX))
```

See [Virtual Inheritance](#virtual-inheritance) for the standalone-registration pattern for
virtually-inherited leaves, and [Other Recurring Gotchas](#other-recurring-gotchas) for `None`
handling, container conversion, lambda return types, `keep_alive`, and file-like input.

---

## Architecture Decisions

| Decision | Rationale |
|---|---|
| Single `_pyobjcryst` extension | simpler than per-class extensions; nanobind modules can be large |
| Static linking (newmat, cctbx, ObjCryst) | avoids runtime library conflicts on Windows/macOS wheels |
| scikit-build-core + CMake | modern standard for C++ Python extensions, replaces SCons |
| ObjCryst++ as a submodule | tracks upstream, simpler than a fork |
| Vendored newmat/cctbx (not conda-forge `cctbx-base`) | see below |

**Vendored `cctbx` vs. conda-forge's `cctbx-base`** (investigated, not adopted): `cctbx-base`
does contain the right modules, compiled and working, but isn't structured for third-party C++
use — no stable `include/` directory (real headers live inside a Python-version-specific
`site-packages` path), no CMake/pkg-config export, a transitive dependency on Boost headers not
otherwise needed, and a ~160 MB/123-package install to obtain four submodules the current vendored
copy already provides in 2.8 MB with zero extra runtime dependencies. Not worth the trade today. If
revisited, the more promising angle is asking upstream cctbx for a minimal
`sgtbx`+`uctbx`+`eltbx`+`miller`-only build target with its own clean `include/` output — not
depending on `cctbx-base` as it currently ships.

---

## Pitfalls Checklist

**Before writing/extending a binding for class `X`:**
- [ ] Grep `X`'s header and its whole ancestry (not just the direct parent) for `: *virtual
      public` — if found, do **not** declare that base in `nb::class_<X, Base>`; use the manual
      `dynamic_cast`/`nb::cast<Leaf&>` patterns instead

**While writing:**
- [ ] Every optional pointer/object argument needs `.none()` on its `nb::arg(...)`, including plain
      `nb::object` params — except pointers to a plain arithmetic type, where `.none()` has no
      effect; wrap those in an `nb::object`-taking helper instead
- [ ] Every function returning a base-typed reference/pointer whose dynamic type is commonly a
      virtually-derived leaf needs the `dynamic_cast`-and-recast treatment
- [ ] Every function generically *accepting* a base type as an argument needs its argument routed
      through a helper like `extractRefinableObjArg()`
- [ ] Any `.def(...)` combining a lambda with `nb::rv_policy::reference`/`reference_internal` needs
      an explicit `-> T&`/`-> T*` trailing return type on the lambda
- [ ] Any binding calling `.attr("read")()` on a caller-supplied object should go through
      `read_pyfile_to_string()`, not a raw `nb::cast<std::string>`
- [ ] Any container type (`CrystVector`, `CrystMatrix`) crossing the boundary should use the
      `CrystVector_REAL`/`CrystMatrix_REAL` macros, never a hardcoded `<float>`/`<double>`
- [ ] A binding reaching a protected/private member via a trampoline/alias helper class must take
      the *real* base type as its bound parameter, not the trampoline/alias type — otherwise it
      raises `TypeError` on every instance not literally constructed as that alias

**After writing, before declaring done:**
- [ ] Rebuilt and re-ran the specific test file, then the full suite, and confirmed no new failures
- [ ] If the class exposes properties as well as methods, checked the forwarding/coverage against
      the full `dir()` of the canonical class, not just its method list
- [ ] If a genuinely new bug class was found, added it to this document with a concrete repro —
      the pattern-matching value compounds; most bugs in this migration turned out to be new
      instances of an already-documented class, not new classes outright
