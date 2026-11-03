"""Unit tests for pyobjcryst.reflectionprofile bindings."""

import gc
import unittest
import weakref

import numpy as np
import pytest

from pyobjcryst.powderpattern import PowderPattern
from pyobjcryst.radiation import RadiationType, WavelengthType
from pyobjcryst.refinableobj import RefinableObj
from pyobjcryst.reflectionprofile import (
    ReflectionProfileDoubleExponentialPseudoVoigt,
    ReflectionProfilePseudoVoigt,
    ReflectionProfilePseudoVoigtAnisotropic,
    ReflectionProfilePseudoVoigtTCH,
)


class TestReflectionProfile(unittest.TestCase):
    """Tests for ReflectionProfile methods."""

    @pytest.fixture(autouse=True)
    def prepare_fixture(self, loadcifdata):
        self.loadcifdata = loadcifdata

    def setUp(self):
        """Set up a ReflectionProfile instance for testing."""
        x = np.linspace(0, 40, 1000)
        c = self.loadcifdata("paracetamol.cif")

        self.pp = PowderPattern()
        self.pp.SetWavelength(0.7)
        self.pp.SetPowderPatternX(np.deg2rad(x))
        self.pp.SetPowderPatternObs(np.ones_like(x))

        self.ppd = self.pp.AddPowderPatternDiffraction(c)

        self.profile = self.ppd.GetProfile()

    def test_get_computed_profile(self):
        """Sample a profile slice and verify broadening lowers the peak
        height."""
        x = self.pp.GetPowderPatternX()
        hkl = (1, 0, 0)
        window = x[100:200]
        xcenter = float(window[len(window) // 2])

        prof_default = self.profile.GetProfile(window, xcenter, *hkl)
        self.assertEqual(len(prof_default), len(window))
        self.assertGreater(prof_default.max(), 0)

        # broaden and ensure the peak height drops while shape changes
        self.profile.GetPar("W").SetValue(0.05)
        prof_broader = self.profile.GetProfile(window, xcenter, *hkl)

        self.assertFalse(np.allclose(prof_default, prof_broader))
        self.assertLess(prof_broader.max(), prof_default.max())
        self.assertEqual(len(prof_default), len(prof_broader))

    def test_get_profile_width(self):
        """Ensure full-width increases when W increases."""
        xcenter = float(
            self.pp.GetPowderPatternX()[len(self.pp.GetPowderPatternX()) // 4]
        )
        width_default = self.profile.GetFullProfileWidth(0.5, xcenter, 1, 0, 0)
        self.assertGreater(width_default, 0)

        self.profile.GetPar("W").SetValue(0.05)
        width_broader = self.profile.GetFullProfileWidth(0.5, xcenter, 1, 0, 0)
        self.assertGreater(width_broader, width_default)

    def test_create_copy(self):
        """Ensure copy returns an independent profile with identical
        initial params."""
        copy = self.profile.CreateCopy()

        self.assertIsNot(copy, self.profile)
        self.assertEqual(copy.GetClassName(), self.profile.GetClassName())

        eta0_original = self.profile.GetPar("Eta0").GetValue()
        eta0_copy = copy.GetPar("Eta0").GetValue()
        self.assertAlmostEqual(eta0_copy, eta0_original)

        self.profile.GetPar("Eta0").SetValue(eta0_original + 0.1)
        copy.GetPar("Eta0").SetValue(eta0_copy + 0.2)

        self.assertAlmostEqual(
            copy.GetPar("Eta0").GetValue(), eta0_original + 0.2
        )
        self.assertAlmostEqual(
            self.profile.GetPar("Eta0").GetValue(), eta0_original + 0.1
        )

    def test_xml_input(self):
        """Ensure XMLInput restores parameters previously serialized
        with xml()."""
        xml_state = self.profile.xml()
        eta0_original = self.profile.GetPar("Eta0").GetValue()

        self.profile.GetPar("Eta0").SetValue(eta0_original + 0.3)
        self.assertNotAlmostEqual(
            self.profile.GetPar("Eta0").GetValue(), eta0_original
        )

        RefinableObj.XMLInput(self.profile, xml_state)
        self.assertAlmostEqual(
            self.profile.GetPar("Eta0").GetValue(), eta0_original
        )

    def test_xml_output(self):
        """Ensure XMLOutput emits parameter tags and the expected root
        element."""
        xml_state = self.profile.xml()

        self.assertIn("<ReflectionProfile", xml_state)
        for par_name in ("U", "V", "W", "Eta0"):
            self.assertIn(f'Name="{par_name}"', xml_state)

        import io

        buf = io.StringIO()
        RefinableObj.XMLOutput(self.profile, buf, 0)
        xml_from_stream = buf.getvalue()
        self.assertTrue(xml_from_stream.startswith("<ReflectionProfile"))

    def test_is_anisotropic(self):
        """IsAnisotropic returns False for isotropic default profile."""
        self.assertFalse(self.profile.IsAnisotropic())

    def test_concrete_pseudo_voigt_profiles(self):
        """ReflectionProfilePseudoVoigt and Anisotropic can be
        constructed and installed on a PowderPatternDiffraction."""
        iso = ReflectionProfilePseudoVoigt()
        self.assertFalse(iso.IsAnisotropic())
        self.assertTrue(
            {"U", "V", "W", "Eta0", "Eta1"}.issubset(
                {iso.GetPar(i).GetName() for i in range(iso.GetNbPar())}
            )
        )
        iso.SetProfilePar(fwhmCagliotiW=1e-4, eta0=0.7)
        self.assertAlmostEqual(iso.GetPar("Eta0").GetValue(), 0.7)

        self.ppd.SetProfile(iso)
        installed = self.ppd.GetProfile()
        self.assertIsInstance(installed, ReflectionProfilePseudoVoigt)
        self.assertAlmostEqual(installed.GetPar("Eta0").GetValue(), 0.7)

        aniso = ReflectionProfilePseudoVoigtAnisotropic()
        self.assertTrue(aniso.IsAnisotropic())
        aniso.SetProfilePar(fwhmCagliotiW=1e-4)
        self.ppd.SetProfile(aniso)
        installed_aniso = self.ppd.GetProfile()
        self.assertIsInstance(
            installed_aniso, ReflectionProfilePseudoVoigtAnisotropic
        )

    def test_tch_profile_width_and_limits(self):
        """TCH derives one common FWHM and reaches both pure limits."""
        profile = ReflectionProfilePseudoVoigtTCH()
        self.assertFalse(profile.IsAnisotropic())
        self.assertEqual(
            {"U", "V", "W", "X", "Y", "Z", "P", "LGmix"},
            {profile.GetPar(i).GetName() for i in range(profile.GetNbPar())},
        )

        center = np.deg2rad(30.0)
        hg = np.deg2rad(0.08)
        hl = np.deg2rad(0.04)
        profile.SetProfilePar(hg**2, fwhmLorentzZ=hl)
        h = (
            hg**5
            + 2.69269 * hg**4 * hl
            + 2.42843 * hg**3 * hl**2
            + 4.47163 * hg**2 * hl**3
            + 0.07842 * hg * hl**4
            + hl**5
        ) ** 0.2
        self.assertAlmostEqual(
            profile.GetFullProfileWidth(0.5, center, 1, 0, 0), h
        )

        x = np.array([center - h / 2, center, center + h / 2])
        y = profile.GetProfile(x, center, 1, 0, 0)
        np.testing.assert_allclose(y[[0, 2]] / y[1], 0.5, rtol=1e-6)

        profile.SetProfilePar(hg**2)
        self.assertAlmostEqual(
            profile.GetFullProfileWidth(0.5, center, 1, 0, 0), hg
        )
        profile.SetProfilePar(0, fwhmLorentzZ=hl)
        self.assertAlmostEqual(
            profile.GetFullProfileWidth(0.5, center, 1, 0, 0), hl
        )

        size_width = np.deg2rad(0.03) / np.cos(center / 2)
        profile.SetProfilePar(
            0, fwhmScherrerP=np.deg2rad(0.03), scherrerLGmix=1
        )
        self.assertAlmostEqual(
            profile.GetFullProfileWidth(0.5, center, 1, 0, 0), size_width
        )
        profile.SetProfilePar(
            0, fwhmScherrerP=np.deg2rad(0.03), scherrerLGmix=0
        )
        self.assertAlmostEqual(
            profile.GetFullProfileWidth(0.5, center, 1, 0, 0), size_width
        )

        self.ppd.SetProfile(profile)
        installed = self.ppd.GetProfile()
        self.assertIsInstance(installed, ReflectionProfilePseudoVoigtTCH)
        self.assertAlmostEqual(
            installed.GetPar("P").GetValue(), np.deg2rad(0.03)
        )
        self.assertAlmostEqual(installed.GetPar("LGmix").GetValue(), 0)

        restored = ReflectionProfilePseudoVoigtTCH()
        RefinableObj.XMLInput(restored, profile.xml())
        self.assertAlmostEqual(
            restored.GetPar("P").GetValue(), np.deg2rad(0.03)
        )
        self.assertAlmostEqual(restored.GetPar("LGmix").GetValue(), 0)

    def test_anisotropic_set_profile_par(self):
        """SetProfilePar assigns widths and retains symmetric default
        asymmetry."""
        aniso = ReflectionProfilePseudoVoigtAnisotropic()
        aniso.SetProfilePar(
            fwhmCagliotiW=1e-4,
            fwhmLorentzGammaHH=1e-5,
            fwhmLorentzGammaKK=2e-5,
        )
        self.assertAlmostEqual(aniso.GetPar("G_HH").GetValue(), 1e-5)
        self.assertAlmostEqual(aniso.GetPar("G_KK").GetValue(), 2e-5)


def build_tof_pattern(crystal, profile):
    """Return a TOF neutron powder pattern of crystal using profile."""
    pattern = PowderPattern()
    pattern.SetRadiationType(RadiationType.RAD_NEUTRON)
    pattern.GetRadiation().SetWavelengthType(WavelengthType.WAVELENGTH_TOF)
    # TOF decreases with sin(theta)/lambda, so x is in descending order.
    x = np.linspace(120000, 20000, 2001)
    pattern.SetPowderPatternX(x)
    pattern.SetPowderPatternObs(np.ones_like(x))
    diffraction = pattern.AddPowderPatternDiffraction(crystal)
    diffraction.SetProfile(profile)
    return pattern


def test_set_profile_par_tof():
    # Test that SetProfilePar assigns the ten TOF profile parameters
    # Expected: Each refinable parameter has the value passed for it.
    profile = ReflectionProfileDoubleExponentialPseudoVoigt()
    profile.SetProfilePar(
        instrumentAlpha0=0.01,
        instrumentAlpha1=0.1,
        instrumentBeta0=0.02,
        instrumentBeta1=0.004,
        gaussianSigma0=1.0,
        gaussianSigma1=7.0,
        gaussianSigma2=0.5,
        lorentzianGamma0=0.1,
        lorentzianGamma1=0.2,
        lorentzianGamma2=0.4,
    )
    actual_values = {
        profile.GetPar(i).GetName(): profile.GetPar(i).GetValue()
        for i in range(profile.GetNbPar())
    }
    expected_values = {
        "Alpha0": 0.01,
        "Alpha1": 0.1,
        "Beta0": 0.02,
        "Beta1": 0.004,
        "GaussianSigma0": 1.0,
        "GaussianSigma1": 7.0,
        "GaussianSigma2": 0.5,
        "LorentzianGamma0": 0.1,
        "LorentzianGamma1": 0.2,
        "LorentzianGamma2": 0.4,
    }
    assert actual_values == pytest.approx(expected_values)


def test_set_profile_tof_without_unit_cell(loadcifdata):
    # Test installing a TOF profile constructed without a unit cell
    # Expected: SetProfile gives the profile the crystal's unit cell, so
    # the strongest peak is the Si (220) reflection at
    # TOF = DIFC * d + DIFA * d^2.
    crystal = loadcifdata("Si.cif")
    pattern = build_tof_pattern(
        crystal, ReflectionProfileDoubleExponentialPseudoVoigt()
    )
    calc = pattern.GetPowderPatternCalc()
    x = pattern.GetPowderPatternX()
    actual_peak_tof = x[np.argmax(calc)]
    d_220 = crystal.a / np.sqrt(8)
    expected_peak_tof = (
        pattern.GetPar("DIFC").GetValue() * d_220
        + pattern.GetPar("DIFA").GetValue() * d_220**2
    )
    tof_step = abs(x[1] - x[0])
    assert actual_peak_tof == pytest.approx(expected_peak_tof, abs=tof_step)


@pytest.mark.parametrize(
    "parameter_name, new_value",
    [
        # Test that changing a parameter of the installed TOF profile
        # recalculates the pattern
        # C1: The rising-edge exponential coefficient Alpha1 is changed.
        # Expected: The pattern equals one calculated with Alpha1 set
        # before the profile was installed.
        ("Alpha1", 0.2),
        # C2: The falling-edge exponential coefficient Beta0 is changed.
        # Expected: The pattern equals one calculated with Beta0 set
        # before the profile was installed.
        ("Beta0", 0.05),
        # C3: The Gaussian width coefficient GaussianSigma1 is changed.
        # Expected: The pattern equals one calculated with GaussianSigma1
        # set before the profile was installed.
        ("GaussianSigma1", 50.0),
        # C4: The Lorentzian width coefficient LorentzianGamma2 is changed.
        # Expected: The pattern equals one calculated with
        # LorentzianGamma2 set before the profile was installed.
        ("LorentzianGamma2", 2.0),
    ],
)
def test_get_powder_pattern_calc_after_tof_profile_change(
    loadcifdata, parameter_name, new_value
):
    crystal = loadcifdata("Si.cif")
    pattern = build_tof_pattern(
        crystal, ReflectionProfileDoubleExponentialPseudoVoigt()
    )
    pattern.GetPowderPatternCalc()
    profile = pattern.GetPowderPatternComponent(0).GetProfile()
    profile.GetPar(parameter_name).SetValue(new_value)
    actual_calc = pattern.GetPowderPatternCalc()
    expected_profile = ReflectionProfileDoubleExponentialPseudoVoigt(crystal)
    expected_profile.GetPar(parameter_name).SetValue(new_value)
    expected_calc = build_tof_pattern(
        crystal, expected_profile
    ).GetPowderPatternCalc()
    assert np.allclose(actual_calc, expected_calc)


@pytest.mark.parametrize("copy_method", ["CreateCopy", "copy_constructor"])
def test_tof_profile_copy_keeps_original_alive(loadcifdata, copy_method):
    # Test that a copy of a TOF profile keeps its source and unit cell alive
    # Expected: The original profile and crystal survive until the copy is
    # deleted, since both profiles reference the same unit cell.
    crystal = loadcifdata("Si.cif")
    original = ReflectionProfileDoubleExponentialPseudoVoigt(crystal)
    if copy_method == "CreateCopy":
        copied = original.CreateCopy()
    else:
        copied = ReflectionProfileDoubleExponentialPseudoVoigt(original)
    original_ref = weakref.ref(original)
    crystal_ref = weakref.ref(crystal)
    del original, crystal
    gc.collect()
    assert original_ref() is not None
    assert crystal_ref() is not None
    del copied
    gc.collect()
    assert original_ref() is None
    assert crystal_ref() is None


if __name__ == "__main__":
    unittest.main()
