"""Import-safe entry point for sdv.collection.workflow."""
import runpy

def main():
    from sdv.cli import help_if_requested
    help_if_requested('data/sensor_config.yaml을 읽는 CARLA 수집 실행')
    runpy.run_module('sdv.collection.workflow', run_name="__main__")

if __name__ == "__main__":
    main()
