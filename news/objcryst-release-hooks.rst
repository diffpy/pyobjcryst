**Added:**

* <news item>

**Changed:**

* ``Crystal.RemoveScatterer``, ``Crystal.RemoveScatteringPower`` (when the scattering power is no longer used) and the ``Molecule.Remove*`` methods now delete the removed object unless a Python wrapper still references it, instead of leaking it.
* Update the vendored ObjCryst++ to the ``ReleaseObj()`` hook API (vincefn/objcryst#106).

**Deprecated:**

* <news item>

**Removed:**

* <news item>

**Fixed:**

* Fix dangling Python wrappers, use-after-free and double free when ObjCryst++ deletes sub-objects that are still referenced from Python (#121, #132). Removed or replaced objects (reflection profiles, scatterers, scattering powers, molecule atoms and restraints) now stay valid while a Python wrapper references them, and ``GetProfile`` no longer returns a stale wrapper of the wrong type after ``SetProfile``. Wrappers of deleted ``RefinablePar`` objects (e.g. after ``PowderPatternBackground.SetInterpPoints`` or ``TextureMarchDollase.DeletePhase``) are invalidated and raise ``TypeError`` instead of accessing freed memory.
* Fix the double free when ``Crystal.ConnectAtoms`` merges atoms owned by Python.

**Security:**

* <news item>
