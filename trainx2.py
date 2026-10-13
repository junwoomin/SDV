"""Compatibility entry point; implementation: experiments.training.detection."""
import runpy

if __name__ == "__main__":
    from sdv.cli import help_if_requested
    help_if_requested('미검증 객체 검출 실험 · experiments/training/detection.py')
    runpy.run_module('experiments.training.detection', run_name="__main__")
