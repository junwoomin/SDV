"""Import-safe entry point for sdv.training.experiment."""
import runpy

def main():
    runpy.run_module('sdv.training.experiment', run_name="__main__")

if __name__ == "__main__":
    main()
