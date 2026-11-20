"""Synthetic MCX records test path-gate indexing and voxel-unit forwarding.

These tests validate the detector adapter, NOT the MCX transport physics.
"""
from dataclasses import replace
import numpy as np
import pytest
from eggtrans.core import Geometry, Optics, MATERIALS_800, validate
from eggtrans.mcx_engine import photon_history_mask, top_face_photons, simulate_mcx


def histories():
    return {'ppath': np.array([
        [60.,0.,0.,0.,0.],
        [50.,.5,0.,0.,0.],
        [20.,1.,10.,0.,0.],
        [20.,1.,0.,5.,0.],
        [20.,1.,0.,0.,3.],
    ],dtype=float)}


def test_history_gate_uses_correct_mcx_medium_indices_and_mode():
    d=histories()
    o=Optics(camera_path_mode='Traversed egg')
    np.testing.assert_array_equal(photon_history_mask(d,5,o,.5),[False,True,True,True,True])
    np.testing.assert_array_equal(
        photon_history_mask(d,5,replace(o,camera_path_mode='Traversed interior'),.5),
        [False,False,True,True,True])
    np.testing.assert_array_equal(
        photon_history_mask(d,5,replace(o,camera_path_mode='All exits'),.5),
        [True]*5)


def test_minimum_material_path_is_measured_in_mm_not_voxels():
    o=Optics(camera_path_mode='Traversed egg',min_egg_path_mm=.3)
    np.testing.assert_array_equal(photon_history_mask(histories(),5,o,.5),
                                  [False,False,True,True,True])
    assert photon_history_mask(histories(),5,replace(o,min_egg_path_mm=.1),.5)[1]


def test_history_gate_fails_if_photon_record_cannot_support_the_claim():
    o=Optics()
    with pytest.raises(RuntimeError,match='ppath'):
        photon_history_mask({},2,o,.5)
    with pytest.raises(RuntimeError,match='ppath dimensions'):
        photon_history_mask({'ppath':np.ones((2,2))},2,o,.5)
    with pytest.raises(RuntimeError,match='invalid'):
        photon_history_mask({'ppath':np.full((2,5),-1.)},2,o,.5)
    assert not photon_history_mask({},0,o,.5).any()


def test_weight_reconstruction_receives_explicit_mcx_voxel_size():
    class WeightBackend:
        unit=None
        @classmethod
        def detweight(cls,detp,prop,unitinmm=None):
            cls.unit=unitinmm
            return np.ones(len(detp['p']))
    detp={'p':np.array([[1.,2.,3.]]),'v':np.array([[0.,0.,1.]]),
          'ppath':np.ones((1,5))}
    p,v,w=top_face_photons(detp,np.ones((6,4)),WeightBackend,unitinmm=.35)
    assert WeightBackend.unit==pytest.approx(.35)
    assert w[0]==pytest.approx(1)


class MixedPathBackend:
    calls=[]
    @classmethod
    def mcxlab(cls,cfg):
        cls.calls.append(cfg)
        nx,ny,nz=cfg['vol'].shape
        pos=np.tile([nx/2,cfg['srcpos'][1]+cfg['srcparam2'][1]/2,nz-.5],(3,1))
        velocities=np.tile([0.,0.,1.],(3,1))
        path=np.array([[50.,0.,0.,0.,0.],
                       [50.,.4,0.,0.,0.],
                       [10.,1.,20.,0.,0.]],dtype=float)
        return {'detp':{'p':pos,'v':velocities,'ppath':path}}

    @staticmethod
    def detweight(detp,prop,unitinmm=None):
        assert unitinmm is not None
        return np.ones(len(detp['p']))


@pytest.mark.parametrize('camera_mode',['Area','Pushbroom'])
def test_same_mcx_exit_packets_produce_raw_and_gated_camera_results(camera_mode):
    MixedPathBackend.calls=[]
    o=Optics(camera_mode=camera_mode,mode='Transverse line',voxel_mm=2.,
             camera_pixels=16,area_rows=12,scan_lines=3,
             photons_per_line=100,camera_path_mode='Traversed egg')
    r=simulate_mcx(Geometry(),o,MATERIALS_800,run_backend=MixedPathBackend)
    n=(1 if camera_mode=='Area' else 3)
    assert len(MixedPathBackend.calls)==n
    assert r['diagnostics']['camera_accepted_before_history_gate']==3*n
    assert r['diagnostics']['camera_accepted_photons']==2*n
    assert r['diagnostics']['camera_rejected_bypass_packets']==n
    assert np.sum(r['image_unfiltered'])==pytest.approx(3*n/100)
    assert np.sum(r['image'])==pytest.approx(2*n/100)
    rr=simulate_mcx(Geometry(),replace(o,camera_path_mode='Traversed interior'),
                    MATERIALS_800,run_backend=MixedPathBackend)
    assert rr['diagnostics']['camera_accepted_photons']==n


def test_camera_history_mode_validated():
    with pytest.raises(ValueError,match='camera path mode'):
        validate(Geometry(),replace(Optics(),camera_path_mode='invalid'),MATERIALS_800)
    with pytest.raises(ValueError,match='Minimum egg'):
        validate(Geometry(),replace(Optics(),min_egg_path_mm=-1),MATERIALS_800)
