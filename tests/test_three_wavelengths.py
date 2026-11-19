"""Sanity checks for the explicitly hypothetical multi-wavelength report."""
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.compare_three_wavelengths import (
    ABSORPTION_MULTIPLIERS, BASE, CONFIGS, material_scenario,
)
from eggtrans.core import MATERIALS_800


def test_800_nm_is_exact_existing_reference():
    assert np.array_equal(material_scenario(800), MATERIALS_800)


def test_other_scenarios_change_absorption_only():
    for wavelength in (580, 1200):
        model = material_scenario(wavelength)
        assert np.array_equal(model[:, 1:], MATERIALS_800[:, 1:])
        assert np.array_equal(model[0], MATERIALS_800[0])
        for row, name in enumerate(('Shell', 'Albumen', 'Yolk', 'Blood spot'), 1):
            assert np.isclose(model[row, 0], MATERIALS_800[row, 0] * ABSORPTION_MULTIPLIERS[wavelength][name])


def test_collimation_changes_only_input_angular_distribution():
    data = {key: (mode, params) for key, _label, mode, params in CONFIGS}
    mode, params = data['collimated']
    assert mode == 'Transverse line'
    assert 'line_width_mm' not in params
    assert 'line_length_mm' not in params
    assert params['divergence_x_deg'] < BASE.divergence_x_deg
    assert params['divergence_y_deg'] < BASE.divergence_y_deg


def test_only_documented_scenarios_allowed():
    import pytest
    with pytest.raises(ValueError):
        material_scenario(1450)