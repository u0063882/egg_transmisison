"""Independent analytic reference of real MCX: no scatter + matched n => Beer-Lambert.

Run on an NVIDIA CUDA machine:
    RUN_MCX_PHYSICS=1 python -m pytest -q tests/test_mcx_physics.py  # Linux
    $env:RUN_MCX_PHYSICS='1'; python -m pytest -q tests/test_mcx_physics.py  # PowerShell

This test is skipped on ordinary CI/no-GPU machines rather than falsely claiming
that GPU transport has been validated here.
"""
import os
import numpy as np
import pytest
from eggtrans.mcx_engine import backend_status,run_slab_reference

@pytest.mark.gpu
def test_photon_transport_matches_analytical_beer_lambert():
    if os.environ.get('RUN_MCX_PHYSICS')!='1':
        pytest.skip('Requires explicit RUN_MCX_PHYSICS=1 and NVIDIA GPU.')
    ok,message=backend_status()
    if not ok:
        pytest.skip(message)
    for mu_a in (0.0,.03,.10):
        result=run_slab_reference(mu_a_mm1=mu_a,thickness_mm=12,nphotons=50_000)
        assert result['detected_packets']>47_500
        assert result['relative_error']<.045,result

@pytest.mark.gpu
def test_mcx_native_sources_reproduce_expected_exit_footprints():
    if os.environ.get('RUN_MCX_PHYSICS')!='1':
        pytest.skip('Requires explicit RUN_MCX_PHYSICS=1 and NVIDIA GPU.')
    ok,message=backend_status()
    if not ok: pytest.skip(message)
    from eggtrans.mcx_engine import run_source_shape_reference
    r=run_source_shape_reference()
    assert 0.85 < r['Spot']['aspect_x_over_y'] < 1.15,r
    assert r['Transverse line']['aspect_x_over_y'] > 10.,r

@pytest.mark.gpu
def test_mcx_area_camera_preserves_known_spot_vs_line_footprint():
    """Known-answer image from actual MCX in a transparent slab, not stubbed."""
    if os.environ.get('RUN_MCX_PHYSICS')!='1':
        pytest.skip('Requires explicit RUN_MCX_PHYSICS=1 and NVIDIA GPU.')
    ok, message=backend_status()
    if not ok: pytest.skip(message)
    from eggtrans.mcx_engine import run_area_camera_source_reference
    r=run_area_camera_source_reference()
    assert .85 < r['Spot']['aspect_x_over_y'] < 1.15, r
    assert r['Transverse line']['aspect_x_over_y'] > 10., r
    assert all(v['weighted_flux']>.95 for v in r.values()), r

@pytest.mark.gpu
def test_mcx_rejects_real_air_bypass_photons_when_egg_present():
    """GPU test: direct air paths are present with broad illumination and excluded by history gate."""
    if os.environ.get('RUN_MCX_PHYSICS') != '1':
        pytest.skip('Requires explicit RUN_MCX_PHYSICS=1 and NVIDIA GPU.')
    ready,message=backend_status()
    if not ready:
        pytest.skip(message)
    from eggtrans.core import Geometry, Optics, MATERIALS_800
    from eggtrans.mcx_engine import simulate_mcx
    o=Optics(camera_mode='Area',mode='Broad backlight',line_length_mm=68.,
             voxel_mm=1.,photons_per_line=100000,camera_pixels=24,area_rows=24,
             camera_path_mode='Traversed egg',camera_half_angle_deg=60.)
    out=simulate_mcx(Geometry(include_yolk=False,include_blood=False),o,MATERIALS_800)
    diag=out['diagnostics']
    assert np.sum(out['image_unfiltered']) >= np.sum(out['image'])
    assert diag['camera_rejected_bypass_packets'] > 0
    assert diag['camera_accepted_photons'] > 0
