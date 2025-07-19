"""Unit tests for __version__.py."""

import pyobjcryst  # noqa


def test_package_version():
    """Ensure the package version is defined and not set to the initial
    placeholder."""
    assert hasattr(pyobjcryst, "__version__")
    assert pyobjcryst.__version__ != "0.0.0"


def test_init_api():
    # Remove this if gTopRefinableObjRegistry import is removed from __init__.py
    assert pyobjcryst.gTopRefinableObjRegistry is not None
