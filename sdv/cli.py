"""Help for compatibility launchers without importing an active workload."""
import argparse
import sys
def help_if_requested(description):
    if any(arg in ('-h','--help') for arg in sys.argv[1:]):
        parser=argparse.ArgumentParser(description=description)
        parser.parse_args()
