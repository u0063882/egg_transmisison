import numpy as np
import pytest
from dataclasses import replace
from eggtrans.core import (Geometry, Optics, MATERIALS_800, _geometry_vector, _medium,
                           _ellipsoid_hit, _refract_or_reflect, projected_masks, simulate, validate)


def _fast():
    return Optics(photons_per_line=400, scan_lines=12, camera_pixels=24,
                  camera_half_angle_deg=60, camera_line_width_mm=6)


def test_geometry_medium_labels():
    q=_geometry_vector(Geometry())
    assert _medium(0,0,100,q) == 0
    assert _medium(0,0,21.82,q) == 1
    assert _medium(0,0,19,q) == 2
    assert _medium(0,0,0,q) == 3
    assert _medium(6,13,9,q) == 4


def test_exact_intersections():
    d=_ellipsoid_hit(0,0,-30,0,0,1,0,0,0,22,29.5,22)
    assert np.isclose(d,8)


def test_no_refraction_if_indices_matched():
    # Both phases have the same index: the direction must be unchanged.
    from eggtrans.core import _refract_or_reflect
    d=np.array([.6, 0.,.8])
    # RNG returns 'transmitted' (R=0)
    a,b,c,crossed=_refract_or_reflect(*d,0.,0.,1.,1.35,1.35)
    assert crossed
    assert np.allclose([a,b,c],d)


def test_projections():
    x=np.linspace(-24,24,50); y=np.linspace(-33,33,70)
    m=projected_masks(Geometry(),x,y)
    assert m['yolk'].sum()<m['egg'].sum()
    assert m['blood'].sum()>0


def test_input_validation():
    with pytest.raises(ValueError): validate(Geometry(yolk_radius_z_mm=30),_fast(),MATERIALS_800)
    with pytest.raises(ValueError): validate(Geometry(),_fast(),np.zeros((3,4)))
    with pytest.raises(ValueError): validate(Geometry(),replace(_fast(),camera_half_angle_deg=0),MATERIALS_800)


def test_simulation_reproducible_and_nonnegative():
    o=_fast()
    a=simulate(optics=o)
    b=simulate(optics=o)
    assert np.array_equal(a['image'],b['image'])
    assert np.isfinite(a['image']).all()
    assert (a['image']>=0).all()
    assert a['diagnostics']['launched_photons']==o.scan_lines*o.photons_per_line
    assert a['diagnostics']['weighted_ballistic_signal']<=a['diagnostics']['weighted_camera_signal']


def test_modes_run():
    for mode in ('Spot','Transverse line','Longitudinal line','Broad backlight'):
        a=simulate(optics=replace(_fast(),mode=mode))
        assert a['image'].shape == (12,24)