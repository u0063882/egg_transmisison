"""Transport is stubbed only for deterministic adapter checks; physics remains PMCX."""
import numpy as np
import pytest
from eggtrans.core import Geometry,Optics,MATERIALS_800
from eggtrans.acquisition_comparison import compare_matched_acquisitions


class FixedPackets:
    configs=[]
    @classmethod
    def mcxlab(cls,cfg):
        cls.configs.append(cfg)
        nx,ny,nz=cfg['vol'].shape
        # source-independent *detected* coordinate proxy in MCX units
        xs=np.linspace(-20,20,40)
        ys=np.linspace(-30,30,40)
        X,Y=np.meshgrid(xs,ys)
        p=np.column_stack([X.ravel()/cfg['unitinmm']+nx/2,
                           Y.ravel()/cfg['unitinmm']+ny/2,
                           np.ones(X.size)*(nz-.5)])
        v=np.tile([0.,0.,1.],(len(p),1))
        return {'detp':{'p':p,'v':v,'ppath':np.tile([0.,.5,2.,0.,0.],(len(p),1))}}
    @staticmethod
    def detweight(d,props,unitinmm=None):
        return np.ones(len(d['p']))


def test_matched_broad_source_has_identical_launch_positions_per_line():
    FixedPackets.configs=[]
    g=Geometry();o=Optics(voxel_mm=2.,camera_pixels=20,photons_per_line=2000)
    r=compare_matched_acquisitions(g,o,MATERIALS_800,rows=12,backend=FixedPackets)
    assert len(FixedPackets.configs)==13
    assert r['area_image'].shape==(12,20)
    assert r['virtual_pushbroom_image'].shape==(12,20)
    assert all(cfg['srctype']=='planar' for cfg in FixedPackets.configs)
    first=FixedPackets.configs[0]
    for cfg in FixedPackets.configs[1:]:
        assert cfg['srcpos']==first['srcpos']
        assert cfg['srcparam1']==first['srcparam1']
        assert cfg['srcparam2']==first['srcparam2']
    assert np.sum(r['area_image']) > 0
    assert r['diagnostics']['accepted_area_photons'] > 0
    assert r['diagnostics']['actual_moving_egg_scanner'] is False
    assert r['diagnostics']['number_of_runs']==13
    # deterministic packets cover camera rows, so 2D/line histogram should match
    assert r['diagnostics']['relative_l1_image_discrepancy']<1e-12


def test_row_count_validation():
    with pytest.raises(ValueError,match='two'):
        compare_matched_acquisitions(Geometry(),Optics(),rows=1,backend=FixedPackets)
