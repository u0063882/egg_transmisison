"""Backend-independent geometry, source and detected-camera unit tests."""
from dataclasses import replace
import numpy as np
import pytest
from eggtrans.core import Geometry,Optics,MATERIALS_800,validate,projected_masks
from eggtrans.mcx_engine import (build_egg_volume,source_footprint_mm,source_config,
                                 mcx_config,camera_bin,simulate_mcx,homogeneous_slab_config)

@pytest.fixture
def sample():
    g=Geometry()
    o=Optics(voxel_mm=2.,scan_lines=3,camera_pixels=10,photons_per_line=10)
    return g,o


def test_native_mcx_source_types_and_extents(sample):
    g,o=sample
    d=build_egg_volume(g,o,MATERIALS_800)
    spot=source_config(d,replace(o,mode='Spot',spot_diameter_mm=8),0)
    assert spot['srctype']=='disk'
    assert spot['srcparam1'][0]*d.grid_mm==4 # radius, not line length
    assert spot['srcparam2']==[0.,0.,0.,0.]
    xy=source_footprint_mm(replace(o,mode='Spot'),3)
    assert xy['kind']=='circle' and xy['cy']==3
    trans=source_config(d,replace(o,mode='Transverse line'),0)
    longitudinal=source_config(d,replace(o,mode='Longitudinal line'),0)
    broad=source_config(d,replace(o,mode='Broad backlight'),0)
    assert all(c['srctype']=='planar' for c in (trans,longitudinal,broad))
    assert trans['srcparam1'][0]*d.grid_mm==o.line_length_mm
    assert trans['srcparam2'][1]*d.grid_mm==o.line_width_mm
    assert longitudinal['srcparam1'][0]*d.grid_mm==o.line_width_mm
    assert longitudinal['srcparam2'][1]*d.grid_mm==o.line_length_mm
    assert broad['srcparam1'][0]*d.grid_mm==broad['srcparam2'][1]*d.grid_mm==o.line_length_mm
    assert trans['srcparam1'][1]==longitudinal['srcparam1'][1]==0


def test_source_offset_moves_y_but_not_x(sample):
    g,o=sample
    o=replace(o,mode='Spot',source_offset_y_mm=5)
    d=build_egg_volume(g,o,MATERIALS_800)
    a=source_config(d,o,2)
    b=source_config(d,o,-4)
    assert a['srctype']=='disk'
    assert (a['srcpos'][1]-b['srcpos'][1])*d.grid_mm==pytest.approx(6)
    assert a['srcpos'][0]==b['srcpos'][0]


def test_voxel_material_labels_and_yolk_shift(sample):
    g,o=sample
    d=build_egg_volume(g,o,MATERIALS_800)
    center=np.round(d.mm_to_voxel([g.yolk_offset_x_mm,g.yolk_offset_y_mm,g.yolk_offset_z_mm])-.5).astype(int)
    assert d.vol[tuple(center)]==4
    no=build_egg_volume(replace(g,include_yolk=False),o,MATERIALS_800)
    assert not np.any(no.vol==4)
    moved=build_egg_volume(replace(g,yolk_offset_x_mm=5.),o,MATERIALS_800)
    xc=d.origin_mm[0]+(np.arange(d.dims[0])+.5)*d.grid_mm
    c0=np.mean(xc[np.any(d.vol==4,axis=(1,2))])
    c1=np.mean(xc[np.any(moved.vol==4,axis=(1,2))])
    assert c1>c0


def test_mcx_upper_face_detector_not_fluence(sample):
    g,o=sample
    d=build_egg_volume(g,o,MATERIALS_800)
    c=mcx_config(d,o,0,42)
    assert c['bc']=='______000001'
    assert 'X' in c['savedetflag'] and 'V' in c['savedetflag']
    assert c['issave2pt']==0
    assert c['vol'].shape==d.dims


def test_camera_acceptance_and_slit(sample):
    g,o=sample
    d=build_egg_volume(g,o,MATERIALS_800)
    p=d.mm_to_voxel(np.asarray([[0,0,30],[2,0,30],[4,8,30]]))
    v=np.asarray([[0,0,1],[1,0,0],[0,0,1.]])
    weights=np.array([.5,.6,.7])
    line,n=camera_bin(p,v,weights,d,replace(o,camera_half_angle_deg=20,camera_line_width_mm=4),0,
                      np.linspace(-10,10,11))
    assert n==1 and np.sum(line)==pytest.approx(.5)


def test_invalid_inputs(sample):
    g,o=sample
    with pytest.raises(ValueError):validate(replace(g,yolk_radius_z_mm=50),o,MATERIALS_800)
    with pytest.raises(ValueError):validate(g,o,np.ones((2,4)))
    with pytest.raises(ValueError):validate(g,replace(o,camera_half_angle_deg=0),MATERIALS_800)


def test_phantom_source_is_pencil_and_optical_interfaces_matched():
    c=homogeneous_slab_config(mu_a_mm1=.05,thickness_mm=10,nphotons=1000)
    assert c['srctype']=='pencil'
    assert c['bc']=='______000001'
    assert np.allclose(c['prop'][:,3],1)
    assert np.allclose(c['prop'][:,1],0)
    assert np.all(c['vol'][:,:,12:22]==2)
    assert np.all(c['vol'][:,:,11]==1)


class StubPMCX:
    """Simulated PMCX RESPONSE for adapter unit testing; NOT optical validation."""
    calls=[]
    @staticmethod
    def mcxlab(cfg):
        StubPMCX.calls.append(cfg)
        sy=cfg['srcpos'][1]
        p=np.array([[cfg['vol'].shape[0]/2,sy,cfg['vol'].shape[2]],
                    [cfg['vol'].shape[0]/2,sy+20,cfg['vol'].shape[2]]])
        v=np.array([[0.,0.,1.],[0.,0.,1.]])
        return {'detp':{'p':p,'v':v,'ppath':np.tile([0.,.5,2.,0.,0.],(len(p),1))}}
    @staticmethod
    def detweight(d,p,unitinmm=None):
        return np.ones(2)


def test_end_to_end_adapter_uses_real_mode_in_mcx_call(sample):
    g,o=sample
    StubPMCX.calls=[]
    result=simulate_mcx(g,replace(o,mode='Spot',camera_line_width_mm=4),MATERIALS_800,StubPMCX)
    assert len(StubPMCX.calls)==o.scan_lines
    assert all(c['srctype']=='disk' for c in StubPMCX.calls)
    assert result['image'].shape==(o.scan_lines,o.camera_pixels)
    assert result['diagnostics']['source_type']=='disk'
    StubPMCX.calls=[]
    simulate_mcx(g,replace(o,mode='Transverse line'),MATERIALS_800,StubPMCX)
    assert all(c['srctype']=='planar' for c in StubPMCX.calls)


def test_masks_separate_truth_from_detected_signal(sample):
    g,o=sample
    m=projected_masks(g,np.linspace(-22,22,100),np.linspace(-29,29,100))
    assert m['yolk'].sum()<m['egg'].sum()
