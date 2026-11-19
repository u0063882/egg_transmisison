"""Yolk presence/position falsification tests for the Monte Carlo core."""
from dataclasses import replace
import numpy as np
from eggtrans.core import Geometry, Optics, MATERIALS_800, projected_masks, simulate


def test_no_yolk_projection_empty():
    g = Geometry(include_yolk=False, include_blood=False)
    x = np.linspace(-25, 25, 65)
    y = np.linspace(-33, 33, 71)
    assert not projected_masks(g, x, y)['yolk'].any()


def test_yolk_shift_changes_only_ground_truth_position():
    g = Geometry(include_blood=False, yolk_offset_x_mm=0, yolk_offset_y_mm=0)
    right = replace(g, yolk_offset_x_mm=5, yolk_offset_y_mm=-7)
    x = np.linspace(-25, 25, 101)
    y = np.linspace(-33, 33, 111)
    center = projected_masks(g, x, y)['yolk']
    shifted = projected_masks(right, x, y)['yolk']
    assert not np.array_equal(center, shifted)
    assert np.abs(center.sum() - shifted.sum()) <= 12  # integer-pixel rasterisation changes area slightly


def test_homogeneous_no_scatter_yolk_is_optically_invisible():
    """Critical control: a fictitious homogeneous phantom must not show a yolk."""
    m = np.zeros_like(MATERIALS_800)
    m[:,3] = 1.0
    o = Optics(photons_per_line=1000,scan_lines=12,camera_pixels=48,
               divergence_x_deg=0,divergence_y_deg=0,
               max_scattering_events=30,seed=921)
    g = Geometry(include_blood=False, include_yolk=True, yolk_offset_x_mm=5, yolk_offset_y_mm=-7)
    no_yolk = replace(g, include_yolk=False)
    a=simulate(g,o,m)['image']
    b=simulate(no_yolk,o,m)['image']
    np.testing.assert_array_equal(a,b)


def test_yolk_coefficients_affect_observed_signal():
    o=Optics(photons_per_line=1000,scan_lines=18,camera_pixels=48,seed=271)
    base=Geometry(include_blood=False, yolk_offset_x_mm=0, yolk_offset_y_mm=0)
    present=simulate(base,o)['image'].sum()
    absent=simulate(replace(base,include_yolk=False),o)['image'].sum()
    assert absent > present * 1.2
