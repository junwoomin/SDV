"""Compatibility entry point; implementation: sdv.ui.app."""
import runpy

if __name__ == "__main__":
    from sdv.cli import help_if_requested
    help_if_requested('SDV 한국어 실험 UI · ./start.sh')
    runpy.run_module('sdv.ui.app', run_name="__main__")
