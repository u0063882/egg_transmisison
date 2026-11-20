"""Spectral import correctness; actual MCX spectra require measured curves."""
import numpy as np
from eggtrans.core import MATERIALS_800

def test_reference_is_single_wavelength_only():
    assert MATERIALS_800.shape==(5,4)
    assert MATERIALS_800[3,1]>MATERIALS_800[2,1]
