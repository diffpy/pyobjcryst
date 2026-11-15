#!/usr/bin/env python
"""Lifetime tests for Atom and its ScatteringPower (see issue #123).

An Atom only stores a raw pointer to its ScatteringPower and calls
``DeRegisterClient`` on it in its destructor, so the Python Atom must keep
the Python ScatteringPower alive.
"""

import gc
import os
import subprocess
import sys
import textwrap

import pytest

from pyobjcryst.atom import Atom
from pyobjcryst.scatteringpower import ScatteringPowerAtom


def _refcount(obj):
    gc.collect()
    return sys.getrefcount(obj)


# The refcount tests below never let a ScatteringPower die before its Atom,
# so that they fail cleanly (instead of crashing pytest) without the fix.


def test_constructor_keeps_scattering_power_alive():
    sp = ScatteringPowerAtom("Ni", "Ni")
    before = _refcount(sp)
    atom = Atom(0, 0, 0, "Ni", sp)
    delta = _refcount(sp) - before
    del atom
    assert delta >= 1


def test_init_keeps_scattering_power_alive():
    sp1 = ScatteringPowerAtom("Ni", "Ni")
    sp2 = ScatteringPowerAtom("O", "O")
    atom = Atom(0, 0, 0, "Ni", sp1)
    before = _refcount(sp2)
    atom.Init(0.1, 0.2, 0.3, "O", sp2)
    delta = _refcount(sp2) - before
    symbol = atom.GetScatteringPower().GetSymbol()
    del atom
    assert delta >= 1
    assert symbol == "O"


def test_copy_constructor_keeps_original_alive():
    sp = ScatteringPowerAtom("Ni", "Ni")
    atom = Atom(0, 0, 0, "Ni", sp)
    before = _refcount(atom)
    copy = Atom(atom)
    delta = _refcount(atom) - before
    symbol = copy.GetScatteringPower().GetSymbol()
    del copy, atom
    assert delta >= 1
    assert symbol == "Ni"


@pytest.mark.parametrize(
    "script",
    [
        # Atom constructed from a temporary ScatteringPower.
        """
        atom = Atom(0, 0, 0, "Ni", ScatteringPowerAtom("Ni", "Ni"))
        gc.collect()
        assert atom.GetScatteringPower().GetSymbol() == "Ni"
        del atom
        """,
        # Atom re-initialized with a temporary ScatteringPower.
        """
        atom = Atom(0, 0, 0, "Ni", ScatteringPowerAtom("Ni", "Ni"))
        atom.Init(0, 0, 0, "O", ScatteringPowerAtom("O", "O"))
        gc.collect()
        assert atom.GetScatteringPower().GetSymbol() == "O"
        del atom
        """,
        # Copy outliving both the original Atom and the ScatteringPower.
        """
        sp = ScatteringPowerAtom("Ni", "Ni")
        atom = Atom(0, 0, 0, "Ni", sp)
        copy = Atom(atom)
        del sp, atom
        gc.collect()
        assert copy.GetScatteringPower().GetSymbol() == "Ni"
        del copy
        """,
        # Dummy atom (no ScatteringPower) must still work.
        """
        atom = Atom(0, 0, 0, "dummy", None)
        assert atom.IsDummy()
        assert atom.GetScatteringPower() is None
        atom.Init(0, 0, 0, "dummy2", None)
        assert atom.IsDummy()
        del atom
        """,
    ],
)
def test_scattering_power_outlived_by_atom(script):
    # Run in a subprocess so that a use-after-free crash is reported as a
    # test failure rather than killing pytest. Freed memory is scribbled
    # (macOS / glibc) to make such a crash deterministic.
    code = (
        "import gc\n"
        "from pyobjcryst.atom import Atom\n"
        "from pyobjcryst.scatteringpower import ScatteringPowerAtom\n"
        + textwrap.dedent(script)
        + "gc.collect()\n"
    )
    env = dict(os.environ, MallocScribble="1", MALLOC_PERTURB_="85")
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )
    assert result.returncode == 0, result.stderr
