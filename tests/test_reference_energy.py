"""Unit tests for the pure titration-fitting helpers in reference_energy.

These exercise the real functions with synthetic titration data (no mocks, no
simulation), covering the Hill fit and its degenerate-data fallbacks.
"""

import numpy as np
import pytest

from molecular_simulations.simulate.constantph.reference_energy import (
    fit_titration_midpoint,
    hill_equation,
)

pytestmark = pytest.mark.unit


class TestHillEquation:
    def test_midpoint_is_half(self):
        assert hill_equation(4.3, 4.3, 1.0) == pytest.approx(0.5)

    def test_low_pH_protonated_high_pH_deprotonated(self):
        assert hill_equation(0.0, 4.3, 1.0) > 0.99
        assert hill_equation(9.0, 4.3, 1.0) < 0.01

    def test_accepts_array(self):
        out = hill_equation(np.array([0.0, 4.3, 9.0]), 4.3, 1.0)
        assert out.shape == (3,)
        assert out[0] > out[1] > out[2]


class TestFitTitrationMidpoint:
    def test_recovers_clean_curve(self):
        pH = np.array([1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3])
        frac = hill_equation(pH, 4.3, 1.0)
        mid, n = fit_titration_midpoint(pH, frac, pKa0=7.0)
        assert mid == pytest.approx(4.3, abs=1e-3)
        assert n == pytest.approx(1.0, abs=1e-3)

    def test_recovers_shifted_midpoint(self):
        pH = np.arange(7.5, 13.6, 1.0)
        frac = hill_equation(pH, 10.5, 1.0)
        mid, _n = fit_titration_midpoint(pH, frac, pKa0=7.0)
        assert mid == pytest.approx(10.5, abs=1e-2)

    def test_noisy_curve_is_close(self):
        rng = np.random.default_rng(0)
        pH = np.array([1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3])
        frac = np.clip(hill_equation(pH, 4.3, 1.0) + rng.normal(0, 0.02, pH.size), 0, 1)
        mid, _ = fit_titration_midpoint(pH, frac, pKa0=7.0)
        assert mid == pytest.approx(4.3, abs=0.3)

    def test_fully_deprotonated_curve_does_not_raise(self):
        pH = np.array([1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3])
        mid, _n = fit_titration_midpoint(pH, np.zeros_like(pH), pKa0=7.0)
        # Midpoint lies below the sampled ladder; a finite value is returned.
        assert np.isfinite(mid)
        assert mid <= pH.min() + 6.0
