"""Check that stationary 2D acquisition cannot accidentally become a scan.

These checks use synthetic MCX response packets for adapter correctness;
actual GPU/MCX physical validation is in tests/test_mcx_physics.py.
"""
from dataclasses import replace
import numpy as np
import pytest
from eggtrans.core import Geometry, Optics, MATERIALS_800, validate
from eggtrans.mcx_engine import (build_egg_volume, camera_bin_area,
                                 simulate_mcx, source_config)


class StaticBackend:
    calls = []

    @staticmethod
    def mcxlab(cfg):
        StaticBackend.calls.append(cfg)
        # Synthetic exit-face positions in MCX grid coordinates. One centered
        # photon and one out of the egg's area-camera FOV.
        size = np.asarray(cfg['vol'].shape)
        p = np.array([[size[0]/2, size[1]/2, size[2]],
                      [size[0]-1, size[1]-1, size[2]]], dtype=float)
        v = np.array([[0, 0, 1], [0, 0, 1]], dtype=float)
        return {'detp': {'p': p, 'v': v, 'ppath': np.tile([0.,.5,2.,0.,0.],(len(p),1))}}

    @staticmethod
    def detweight(d, prop,unitinmm=None):
        return np.array([0.6, 0.8])


def test_area_camera_accepts_2d_positions_without_a_line_slit():
    g = Geometry()
    o = Optics(camera_mode='Area', voxel_mm=2., camera_pixels=10,
               area_rows=14, camera_half_angle_deg=30,
               camera_line_width_mm=.5)
    d = build_egg_volume(g, o, MATERIALS_800)
    p = d.mm_to_voxel(np.array([[0, -4, 30], [3, 5, 30],
                                [0, 4, 30], [20, 50, 30]]))
    v = np.array([[0, 0, 1], [0, 0, 1],
                  [1, 0, 1], [0, 0, 1]], dtype=float)
    w = np.array([.2, .3, .8, .9])
    im, n = camera_bin_area(p, v, w, d, o,
                            np.linspace(-20, 20, 15),
                            np.linspace(-10, 10, 11))
    assert im.shape == (14, 10)
    assert n == 2
    assert im.sum() == pytest.approx(.5)
    assert np.count_nonzero(im) == 2   # two different y positions


def test_area_mode_one_mcx_call_and_spot_is_disk():
    g = Geometry()
    o = Optics(camera_mode='Area', mode='Spot', spot_diameter_mm=8,
               voxel_mm=2, scan_lines=6, area_rows=14,
               camera_pixels=12, photons_per_line=100)
    StaticBackend.calls = []
    r = simulate_mcx(g, o, MATERIALS_800, StaticBackend)
    assert len(StaticBackend.calls) == 1
    assert StaticBackend.calls[0]['srctype'] == 'disk'
    assert r['image'].shape == (14, 12)
    assert r['diagnostics']['mcx_runs'] == 1
    assert r['diagnostics']['launched_photons'] == 100
    assert r['camera_mode'] == 'Area'
    assert r['image'].sum() == pytest.approx(.006)


def test_line_camera_multiple_calls_with_same_spot_source():
    g = Geometry()
    o = Optics(camera_mode='Pushbroom', mode='Spot', voxel_mm=2,
               scan_lines=6, area_rows=14, camera_pixels=12,
               photons_per_line=100)
    StaticBackend.calls = []
    r = simulate_mcx(g, o, MATERIALS_800, StaticBackend)
    assert len(StaticBackend.calls) == 6
    assert all(c['srctype'] == 'disk' for c in StaticBackend.calls)
    assert r['image'].shape == (6, 12)
    assert r['diagnostics']['mcx_runs'] == 6
    assert r['diagnostics']['launched_photons'] == 600
    # Scan positions are different, unlike the one-exposure area camera.
    assert len({round(c['srcpos'][1], 3) for c in StaticBackend.calls}) == 6


def test_area_native_source_selection_is_correct_for_every_source():
    g = Geometry()
    o = Optics(camera_mode='Area', voxel_mm=2, photons_per_line=50)
    d = build_egg_volume(g, o, MATERIALS_800)
    expected = {'Spot': ('disk', 8, 8),
                'Transverse line': ('planar', 48, 2),
                'Longitudinal line': ('planar', 2, 48),
                'Broad backlight': ('planar', 48, 48)}
    for mode, (type_, sx, sy) in expected.items():
        selected = replace(o, mode=mode, spot_diameter_mm=8)
        c = source_config(d, selected, 0.)
        assert c['srctype'] == type_
        if mode == 'Spot':
            assert c['srcparam1'][0]*d.grid_mm == sx/2
        else:
            assert c['srcparam1'][0]*d.grid_mm == sx
            assert c['srcparam2'][1]*d.grid_mm == sy


def test_reject_invalid_area_rows_and_camera_mode():
    with pytest.raises(ValueError, match='camera mode'):
        validate(Geometry(), Optics(camera_mode='Unknown'), MATERIALS_800)
    with pytest.raises(ValueError, match='positive'):
        validate(Geometry(), Optics(camera_mode='Area', area_rows=0), MATERIALS_800)


def test_interactive_schematic_shows_selected_camera_geometry():
    from eggtrans.schematic import apparatus_figure
    g = Geometry()
    f_area = apparatus_figure(g, Optics(camera_mode='Area', mode='Spot'))
    names_area = {trace.name for trace in f_area.data if trace.name}
    assert 'Area camera FOV (2D)' in names_area
    assert 'Detector line' not in names_area
    assert 'Spot · MCX emitter' in names_area
    f_line = apparatus_figure(g, Optics(camera_mode='Pushbroom', mode='Spot'))
    names_line = {trace.name for trace in f_line.data if trace.name}
    assert 'Detector line' in names_line
    assert 'Area camera FOV (2D)' not in names_line
