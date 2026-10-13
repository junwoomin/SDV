"""Compatibility entry point; implementation: sdv.training.engine."""
import runpy

if __name__ == "__main__":
    runpy.run_module('sdv.training.engine', run_name="__main__")
