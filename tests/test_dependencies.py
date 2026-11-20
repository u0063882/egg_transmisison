"""Catch dependency version typos and mismatches before publishing."""
from pathlib import Path
import tomllib


def test_pmcx_install_requirement_agrees_with_project_metadata():
    root = Path(__file__).resolve().parents[1]
    requirements = (root / "requirements.txt").read_text(encoding="utf-8").splitlines()
    metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    pmcx_req = [line.strip() for line in requirements if line.strip().startswith("pmcx")]
    pmcx_meta = [dep for dep in metadata["project"]["dependencies"] if dep.startswith("pmcx")]
    assert pmcx_req == ["pmcx>=0.8.1,<0.9"]
    assert pmcx_meta == pmcx_req
