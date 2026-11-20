import numpy as np
from eggtrans.core import Geometry,Optics,MATERIALS_800
from eggtrans.mcx_engine import build_egg_volume


def test_optically_identical_yolk_labels_are_distinct_but_same_properties():
    g=Geometry()
    o=Optics(voxel_mm=2)
    d=build_egg_volume(g,o,MATERIALS_800)
    assert np.any(d.vol==4)
    m=MATERIALS_800.copy();m[3]=m[2]
    e=build_egg_volume(g,o,m)
    assert np.array_equal(d.vol,e.vol)
    assert np.array_equal(e.props[3],e.props[4])
    # Physical indistinguishability must be tested on real MCX, not fabricated here.
