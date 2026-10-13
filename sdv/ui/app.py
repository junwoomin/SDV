"""Import-safe entry point for sdv.ui.workflow."""
import runpy

def main():
    from sdv.cli import help_if_requested
    help_if_requested('SDV 한국어 실험 UI · ./start.sh')
    runpy.run_module('sdv.ui.workflow', run_name="__main__")

if __name__ == "__main__":
    main()
