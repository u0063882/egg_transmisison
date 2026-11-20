"""Preflight/domain sizing is deterministic; MCX itself is tested on GPU separately."""
from dataclasses import replace
import numpy as np
import pytest

from eggtrans.core import Geometry, Optics, MATERIALS_800, MODE_NAMES
from eggtrans.mcx_engine import (domain_plan, build_egg_volume, source_config,
                                 scan_positions_mm)


def test_inactive_source_dimensions_do_not_affect_spot_domain():
    g=Geometry()
    base=Optics(mode='Spot',camera_mode='Area',voxel_mm=.35)
    a=domain_plan(g,replace(base,line_length_mm=10.,line_width_mm=.5))
    b=domain_plan(g,replace(base,line_length_mm=90.,line_width_mm=15.))
    assert (a.dims,a.origin_mm)==(b.dims,b.origin_mm)
    assert a.nvoxels<7_000_000


def test_source_specific_line_extent_and_scan_extent():
    g=Geometry()
    o=Optics(mode='Transverse line',camera_mode='Pushbroom',voxel_mm=.35,
             line_length_mm=60)
    scan=domain_plan(g,o)
    area=domain_plan(g,replace(o,camera_mode='Area'))
    assert scan.dims[1]>area.dims[1]
    assert scan.nvoxels<12_000_000
    assert scan.dims[0]>domain_plan(g,replace(o,mode='Spot')).dims[0]
    assert scan.voxel_size_mm==.35


@pytest.mark.parametrize('camera', ('Pushbroom','Area'))
@pytest.mark.parametrize('source', MODE_NAMES)
def test_sources_and_egg_inside_all_domain_edges(source,camera):
    g=Geometry()
    o=Optics(mode=source,camera_mode=camera,line_length_mm=72,
             source_offset_y_mm=10,voxel_mm=.5)
    d=build_egg_volume(g,o,MATERIALS_800)
    for sy in (scan_positions_mm(g,o)[0],scan_positions_mm(g,o)[-1]):
        cfg=source_config(d,o,float(sy))
        p=np.asarray(cfg['srcpos'],dtype=float)
        assert 2<=p[2]<=d.dims[2]-2
        if cfg['srctype']=='disk':
            r=cfg['srcparam1'][0]
            assert p[0]-r>0 and p[0]+r<d.dims[0]
            assert p[1]-r>0 and p[1]+r<d.dims[1]
        else:
            assert 0<p[0] and 0<p[1]
            assert p[0]+cfg['srcparam1'][0]<d.dims[0]
            assert p[1]+cfg['srcparam2'][1]<d.dims[1]
    x,y,z=d.origin_mm
    assert x<=-g.egg_width_mm/2 and x+d.dims[0]*o.voxel_mm>=g.egg_width_mm/2
    assert y<=-g.egg_length_mm/2 and y+d.dims[1]*o.voxel_mm>=g.egg_length_mm/2


def test_padding_expands_and_budget_fails_before_allocation():
    g=Geometry()
    o=Optics(voxel_mm=.35,domain_padding_mm=8)
    small=domain_plan(g,o)
    large=domain_plan(g,replace(o,domain_padding_mm=18))
    assert large.nvoxels>small.nvoxels
    assert small.host_build_estimate_mib>small.label_memory_mib
    with pytest.raises(ValueError,match='configured .* safety budget'):
        build_egg_volume(g,replace(o,domain_padding_mm=18,domain_build_budget_mib=32),MATERIALS_800)


def test_reject_unphysical_padding_or_budget():
    g=Geometry()
    with pytest.raises(ValueError,match='Domain padding'):
        domain_plan(g,Optics(domain_padding_mm=0))
    with pytest.raises(ValueError,match='budget'):
        domain_plan(g,Optics(domain_build_budget_mib=0))


def test_memory_plan_is_exact_before_rasterization():
    g=Geometry();o=Optics(voxel_mm=1,mode='Spot',camera_mode='Area')
    plan=domain_plan(g,o)
    volume=build_egg_volume(g,o,MATERIALS_800)
    assert volume.plan == plan
    assert volume.dims == plan.dims
    assert volume.vol.dtype==np.uint8 and volume.vol.nbytes==plan.nvoxels


def test_preflight_launch_parameters_equal_real_domain_without_allocating_for_ui():
    g=Geometry();o=Optics(mode='Spot',camera_mode='Area',voxel_mm=1.,spot_diameter_mm=6.)
    plan=domain_plan(g,o)
    vol=build_egg_volume(g,o,MATERIALS_800)
    for y in scan_positions_mm(g,o):
        assert source_config(plan,o,float(y)) == source_config(vol,o,float(y))
