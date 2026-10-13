"""Compile research entry points with the interpreter running the test matrix.

This checks syntax without importing CARLA or CUDA packages. Run it under each
supported Python version; a newer parser's feature_version is not sufficient to
reject every newer f-string construct.
"""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_research_sources_compile_on_current_python():
    sources = sorted((ROOT / "sdv").rglob("*.py")) + sorted(ROOT.glob("*.py"))
    assert sources
    for source in sources:
        compile(source.read_text(encoding="utf-8"), str(source), "exec")
