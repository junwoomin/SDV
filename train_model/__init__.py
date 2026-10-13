"""Legacy import path retained for external scripts/checkpoint references."""
from pathlib import Path
__path__ = [str(Path(__file__).resolve().parents[1] / 'sdv/training/modeling')]
