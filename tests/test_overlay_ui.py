"""Protect the saved-run geometry and explicit 2D/3D controls in Streamlit.

This is an interface wiring check; it does not replace a browser or GPU test.
"""
from pathlib import Path


def test_overlays_use_saved_simulation_geometry_not_current_widgets():
    app = (Path(__file__).resolve().parents[1] / 'app.py').read_text(encoding='utf-8')
    assert "run_g=Geometry(**r['geometry'])" in app
    assert "run_o=Optics(**r['optics'])" in app
    assert "show_egg=show_egg,show_yolk=show_yolk" in app
    assert "show_source=show_source,scan_y_mm=selected_y" in app
    assert "camera_scene_3d(image,x,y,run_g,run_o" in app
    assert "show_3d_egg" in app and "show_3d_yolk" in app
    assert "show_3d_source" in app and "show_3d_plane" in app
    assert "(show_source or show_3d)" in app
