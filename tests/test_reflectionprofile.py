"""Unit tests for pyobjcryst.reflectionprofile bindings."""

import unittest

import numpy as np
import pytest

from pyobjcryst.powderpattern import PowderPattern
from pyobjcryst.refinableobj import RefinableObj
from pyobjcryst.reflectionprofile import (
    ReflectionProfile,
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
        self.crystal = self.loadcifdata("paracetamol.cif")

        self.pp = PowderPattern()
        self.pp.SetWavelength(0.7)
        self.pp.SetPowderPatternX(np.deg2rad(x))
        self.pp.SetPowderPatternObs(np.ones_like(x))

        self.ppd = self.pp.AddPowderPatternDiffraction(self.crystal)

        self.profile = self.ppd.GetProfile()

    def test_concrete_pseudo_voigt_profiles(self):
        """Concrete isotropic and anisotropic profiles are
        constructible."""
        isotropic = ReflectionProfilePseudoVoigt()
        anisotropic = ReflectionProfilePseudoVoigtAnisotropic()

        self.assertIsInstance(isotropic, ReflectionProfile)
        self.assertIsInstance(anisotropic, ReflectionProfile)
        self.assertFalse(isotropic.IsAnisotropic())
        self.assertTrue(anisotropic.IsAnisotropic())

        expected = {
            "U",
            "V",
            "W",
            "P",
            "X",
            "Y",
            "G_HH",
            "G_KK",
            "G_LL",
            "G_HK",
            "G_HL",
            "G_KL",
            "Eta0",
            "Eta1",
            "Asym0",
            "Asym1",
            "Asym2",
        }
        self.assertEqual(
            expected,
            {
                anisotropic.GetPar(i).GetName()
                for i in range(anisotropic.GetNbPar())
            },
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
        profile = ReflectionProfilePseudoVoigtAnisotropic()
        profile.SetProfilePar(1e-6, fwhmLorentzX=2e-6)
        profile.GetPar("P").SetValue(3e-6)

        self.assertAlmostEqual(profile.GetPar("W").GetValue(), 1e-6)
        self.assertAlmostEqual(profile.GetPar("X").GetValue(), 2e-6)
        self.assertAlmostEqual(profile.GetPar("P").GetValue(), 3e-6)
        self.assertAlmostEqual(profile.GetPar("Asym0").GetValue(), 1.0)

    def test_anisotropic_profile_depends_on_hkl(self):
        """Anisotropic Lorentz coefficients produce direction
        dependence."""
        profile = ReflectionProfilePseudoVoigtAnisotropic()
        profile.SetProfilePar(
            1e-6,
            fwhmLorentzGammaHH=1e-3,
            fwhmLorentzGammaKK=2e-3,
            pseudoVoigtEta0=1,
        )
        center = 0.5
        x = np.linspace(center - 0.1, center + 0.1, 401)

        h00 = profile.GetProfile(x, center, 1, 0, 0)
        zero_k0 = profile.GetProfile(x, center, 0, 1, 0)

        self.assertFalse(np.allclose(h00, zero_k0))

    def test_set_profile_copies_reusable_template(self):
        """One profile template can safely initialize multiple
        phases."""
        second = self.pp.AddPowderPatternDiffraction(self.crystal)
        template = ReflectionProfilePseudoVoigtAnisotropic()
        template.GetPar("W").SetValue(1e-6)
        template.GetPar("G_HH").SetValue(2e-6)

        for pdiff in (self.ppd, second):
            pdiff.SetProfile(template)

        first_profile = self.ppd.GetProfile()
        second_profile = second.GetProfile()
        self.assertIsInstance(
            first_profile, ReflectionProfilePseudoVoigtAnisotropic
        )
        self.assertIsInstance(
            second_profile, ReflectionProfilePseudoVoigtAnisotropic
        )
        self.assertAlmostEqual(first_profile.GetPar("W").GetValue(), 1e-6)
        self.assertAlmostEqual(second_profile.GetPar("G_HH").GetValue(), 2e-6)

        first_profile.GetPar("W").SetValue(3e-6)
        self.assertAlmostEqual(template.GetPar("W").GetValue(), 1e-6)
        self.assertAlmostEqual(second_profile.GetPar("W").GetValue(), 1e-6)

    def test_set_profile_accepts_temporary(self):
        """A temporary concrete profile is safely copied into the
        phase."""
        self.ppd.SetProfile(ReflectionProfilePseudoVoigt())
        profile = self.ppd.GetProfile()

        self.assertIsInstance(profile, ReflectionProfilePseudoVoigt)
        self.assertFalse(profile.IsAnisotropic())

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


if __name__ == "__main__":
    unittest.main()
