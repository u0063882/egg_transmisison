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
