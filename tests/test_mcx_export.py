import json
import numpy as np
from eggtrans.core import Optics
from eggtrans.mcx_export import export


def test_export_uses_same_spot_native_source_as_interactive_mcx(tmp_path):
    paths=export(o=Optics(mode='Spot',voxel_mm=2,scan_lines=3,photons_per_line=500),output=tmp_path)
    assert len(paths)==3
    c=json.loads(paths[0].read_text())
    assert c['srctype']=='disk'
    assert c['bc']=='______000001'
    assert c['volume_npz']=='mcx_domain.npz'
    vol=np.load(tmp_path/'mcx_domain.npz')['vol']
    assert set(np.unique(vol))=={1,2,3,4,5}
    assert c['srcpos'][1]<json.loads(paths[-1].read_text())['srcpos'][1]
