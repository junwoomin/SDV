"""Compatibility entry point; implementation: sdv.collection.runner."""
import runpy

if __name__ == "__main__":
    from sdv.cli import help_if_requested
    help_if_requested('검증된 data/sensor_config.yaml로 CARLA 수집 실행')
    runpy.run_module('sdv.collection.runner', run_name="__main__")
