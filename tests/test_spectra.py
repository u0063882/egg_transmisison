import pytest
from eggtrans.spectra import load_spectra_csv, at_wavelength


def _csv():
    head='wavelength_nm,material,mu_a_mm_1,mu_s_mm_1,g,n\n'
    for name in ('Shell','Albumen','Yolk','Blood spot'):
        head+=f'800,{name},0.01,0.1,0.8,1.4\n'
        head+=f'1000,{name},0.02,0.2,0.9,1.5\n'
    return head


def test_read_and_interpolate():
    table=load_spectra_csv(_csv())
    m=at_wavelength(table,900.)
    assert m[1,0]==pytest.approx(.015)
    assert m[3,1]==pytest.approx(.15)
    assert m[2,2]==pytest.approx(.85)


def test_no_spectral_extrapolation():
    table=load_spectra_csv(_csv())
    with pytest.raises(ValueError):
        at_wavelength(table,1200.)


def test_missing_material():
    with pytest.raises(ValueError):
        load_spectra_csv(_csv().replace('800,Blood spot,0.01,0.1,0.8,1.4\n','').replace('1000,Blood spot,0.02,0.2,0.9,1.5\n',''))