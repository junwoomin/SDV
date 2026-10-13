"""Historical import namespace; active collection lives in sdv.collection."""
from pathlib import Path
__path__ = [str(Path(__file__).resolve().parents[1] / "legacy/carla_data_gan")]
