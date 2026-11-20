"""Guard against hiding camera selector in a collapsed Streamlit sidebar."""
from pathlib import Path

def test_selector_on_main_page():
    s = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    assert s.count("camera_mode=st.radio(") == 1
    assert s.index("camera_mode=st.radio(") < s.index("with st.sidebar:")
    assert "App v0.9.0" in s
