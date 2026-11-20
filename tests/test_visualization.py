"""Test plot geometry independently of GPU physics and MCX simulation."""
from dataclasses import replace
import numpy as np
import pytest

from eggtrans.core import Geometry, Optics
from eggtrans.visualization import (
    camera_image_figure, camera_scene_3d, outline_xy, source_outline_xy,
)


@pytest.fixture
def case():
    g = Geometry(yolk_offset_x_mm=5, yolk_offset_y_mm=-6)
    o = Optics(camera_mode="Area", mode="Spot", spot_diameter_mm=8,
               source_offset_y_mm=3)
    x = np.linspace(-26, 26, 24)
    y = np.linspace(-33, 33, 32)
    image = np.outer(np.exp(-y*y/180), np.exp(-x*x/110))
    return g, o, x, y, image


def test_truth_outlines_are_correct_and_opt_in(case):
    g,o,x,y,image=case
    fig=camera_image_figure(image,x,y,g,o)
    assert len(fig.data)==1
    fig=camera_image_figure(image,x,y,g,o,show_yolk=True,show_egg=True)
    assert len(fig.data)==3
    xx, yy = outline_xy(g,"yolk")
    assert np.mean((np.min(xx),np.max(xx))) == pytest.approx(g.yolk_offset_x_mm,abs=.005)
    assert np.mean((np.min(yy),np.max(yy))) == pytest.approx(g.yolk_offset_y_mm,abs=.005)
    assert np.max(fig.data[1].x) == pytest.approx(g.egg_width_mm/2)
    assert 'truth' in fig.data[1].name and 'truth' in fig.data[2].name
    assert np.array_equal(fig.data[0].z,image)


def test_selected_illumination_footprint_spot_versus_line(case):
    g,o,x,y,image=case
    sx,sy=source_outline_xy(o,scan_y_mm=0)
    assert 0.5*(np.min(sx)+np.max(sx)) == pytest.approx(0,abs=.002)
    assert 0.5*(np.min(sy)+np.max(sy)) == pytest.approx(3,abs=.002)
    assert np.ptp(sx)==pytest.approx(8,rel=1e-3)
    fig=camera_image_figure(image,x,y,g,o,show_source=True)
    assert len(fig.data)==2
    assert 'not transmitted' in fig.data[1].name
    line=replace(o,camera_mode="Pushbroom",mode="Transverse line",
                 line_length_mm=42,line_width_mm=3)
    lx,ly=source_outline_xy(line,scan_y_mm=7)
    assert np.ptp(lx)==pytest.approx(42)
    assert np.ptp(ly)==pytest.approx(3)
    assert (min(ly)+max(ly))/2==pytest.approx(10)


def test_disabled_yolk_has_no_outline_or_surface(case):
    g,o,x,y,image=case
    g=replace(g,include_yolk=False)
    f=camera_image_figure(image,x,y,g,o,show_yolk=True)
    assert len(f.data)==1
    f3=camera_scene_3d(image,x,y,g,o,show_source=False,show_egg=False)
    assert len(f3.data)==1


def test_three_dimensional_projection_plane_uses_actual_data_and_coordinates(case):
    g,o,x,y,image=case
    f=camera_scene_3d(image,x,y,g,o)
    assert len(f.data)==5  # egg, yolk, source mesh + outline, image surface
    im=f.data[-1]
    assert im.type=='surface'
    assert np.asarray(im.surfacecolor).shape==image.shape
    np.testing.assert_allclose(np.asarray(im.surfacecolor),image)
    assert np.all(np.asarray(im.z)==pytest.approx(g.egg_height_mm/2+o.camera_plane_gap_mm+2*o.voxel_mm))
    assert np.asarray(im.x).shape == image.shape
    assert np.asarray(im.y).shape == image.shape
    source_mesh=f.data[2]
    assert source_mesh.type=='mesh3d'
    assert np.ptp(source_mesh.x)==pytest.approx(8,rel=.001)
    assert np.ptp(source_mesh.y)==pytest.approx(8,rel=.001)


def test_pushbroom_3d_uses_selected_single_source_not_fake_projected_light(case):
    g,o,x,y,image=case
    o=replace(o,camera_mode="Pushbroom",mode="Spot")
    f=camera_scene_3d(image,x,y,g,o,scan_y_mm=-5)
    source=f.data[2]
    assert np.mean((np.min(source.y),np.max(source.y))) == pytest.approx(-2,abs=.005)
    assert 'assembled over time' in f.layout.title.text


def test_mismatched_camera_image_shape_is_rejected(case):
    g,o,x,y,image=case
    with pytest.raises(ValueError,match="image must be"):
        camera_image_figure(image.T,x,y,g,o)
    with pytest.raises(ValueError,match="image must be"):
        camera_scene_3d(image.T,x,y,g,o)
