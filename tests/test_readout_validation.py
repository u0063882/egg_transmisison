"""Readout equivalence must not be mistaken for full MCX physics validation."""
from dataclasses import replace
import numpy as np

from eggtrans.core import Geometry, Optics, MATERIALS_800
from eggtrans.mcx_engine import build_egg_volume, camera_bin, camera_bin_area
from eggtrans.readout_validation import compare_fixed_scene_readouts


def test_row_readout_matches_area_histogram_for_same_exits():
    g = Geometry()
    o = Optics(voxel_mm=2., camera_pixels=12, area_rows=14,
               camera_half_angle_deg=30.)
    d = build_egg_volume(g, o, MATERIALS_800)
    xedges = np.linspace(-g.egg_width_mm*.56, g.egg_width_mm*.56, o.camera_pixels+1)
    yedges = np.linspace(-g.egg_length_mm*.54, g.egg_length_mm*.54, o.area_rows+1)
    yc = .5*(yedges[:-1]+yedges[1:])
    # Deterministic exit coordinates, all interior to the selected camera bins.
    xyz = np.array([[-12., float(yc[2]), 30.], [3., float(yc[6]),30.],
                    [9., float(yc[11]),30.], [4.,float(yc[5]),30.]])
    p = d.mm_to_voxel(xyz)
    v = np.tile([0.,0.,1.], (4,1))
    w = np.array([.2,.4,.6,.8])
    area, count = camera_bin_area(p,v,w,d,o,yedges,xedges)
    assert count == 4
    lo = replace(o,camera_line_width_mm=float(yedges[1]-yedges[0]))
    rows = np.stack([camera_bin(p,v,w,d,lo,float(y),xedges)[0] for y in yc])
    np.testing.assert_allclose(area, rows, atol=1e-12)


class StubPMCX:
    calls = 0

    @staticmethod
    def mcxlab(cfg):
        StubPMCX.calls += 1
        # Coordinates in voxel units, centred in the simulation box.
        nx,ny,nz = cfg['vol'].shape
        p = np.array([[nx*.5,ny*.5,nz-1.],
                      [nx*.5+1.,ny*.5+2.,nz-1.],
                      [nx*.5-1.,ny*.5-2.,nz-1.]])
        return {'detp':{'p':p,'v':np.tile([0.,0.,1.],(3,1))}}

    @staticmethod
    def detweight(d,p):
        return np.ones(3)


def test_diagnostic_reuses_one_backend_run_without_transport_shortcuts():
    StubPMCX.calls=0
    g=Geometry()
    o=Optics(camera_mode='Pushbroom',voxel_mm=2.,
             camera_pixels=24,area_rows=24,photons_per_line=100)
    r=compare_fixed_scene_readouts(g,o,MATERIALS_800,backend=StubPMCX)
    assert StubPMCX.calls==1
    assert r['area_image'].shape==(24,24)
    assert r['diagnostics']['actual_egg_motion'] is False
    np.testing.assert_allclose(r['area_image'],r['line_readout_image'],atol=1e-12)
    assert r['diagnostics']['relative_l1_difference'] < 1e-12
