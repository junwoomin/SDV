"""Compatibility entry point; implementation: sdv.training.runner."""
import runpy

if __name__ == "__main__":
    runpy.run_module('sdv.training.runner', run_name="__main__")
