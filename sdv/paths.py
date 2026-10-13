"""Single project root, independent of caller working directory."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / ".runtime"
