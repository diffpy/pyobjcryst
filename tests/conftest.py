import json
from pathlib import Path

import pytest

from pyobjcryst.crystal import create_crystal_from_cif


@pytest.fixture
def user_filesystem(tmp_path):
    base_dir = Path(tmp_path)
    home_dir = base_dir / "home_dir"
    home_dir.mkdir(parents=True, exist_ok=True)
    cwd_dir = base_dir / "cwd_dir"
    cwd_dir.mkdir(parents=True, exist_ok=True)

    home_config_data = {"username": "home_username", "email": "home@email.com"}
    with open(home_dir / "diffpyconfig.json", "w") as f:
        json.dump(home_config_data, f)

    yield tmp_path


@pytest.fixture
def datafile():
    """Fixture to dynamically load any test file."""

    def _load(filename):
        return "tests/testdata/" + filename

    return _load


@pytest.fixture
def loadcifdata():
    """Fixture to load CIF data files for testing."""

    def _load(filename):
        return create_crystal_from_cif("tests/testdata/" + filename)

    return _load
