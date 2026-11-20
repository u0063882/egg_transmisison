"""Guard against burying the area-camera selector in the collapsed sidebar.

This is a source-level regression test, not a replacement for a
Streamlit browser test or GPU MCX physics validation.
"""
from pathlib import Path


def test_camera_type_radio_is_visible_before_sidebar():
    content = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    assert content.count("camera_mode=st.radio(") == 1
    assert content.index("camera_mode=st.radio(") < content.index("with st.sidebar:")
    assert "key='camera_mode'" in content
    assert "App v0.6.0" in content
    assert "'Normal camera (2D area)'" in content
