import json
import numpy as np
from eggtrans.core import Geometry, Optics
from eggtrans.mcx_export import voxelize, export


def test_voxel_materials_and_size():
    v, extent=voxelize(Geometry(),Optics(),voxel_mm=1.)
    assert v.ndim==3
    assert set(np.unique(v))=={0,1,2,3,4}
    assert v.shape==tuple(int(n) for n in extent)


def test_mcx_export(tmp_path):
    paths=export(output=tmp_path/'m',voxel_mm=1.,scan_lines=3)
    assert len(paths)==3
    meta=json.loads((tmp_path/'m'/'voxel_meta.json').read_text())
    arr=np.fromfile(tmp_path/'m'/'egg_labels.bin',dtype=np.uint8).reshape(meta['shape_xyz'],order='F')
    assert (arr==3).sum()>0
    a=json.loads(paths[0].read_text())
    b=json.loads(paths[-1].read_text())
    assert a['Domain']['Media'][3]['mus']==5.35
    assert a['Optode']['Source']['Type']=='planar'
    assert a['Optode']['Source']['Pos'][1] < b['Optode']['Source']['Pos'][1]