"""Compatibility entry point for the historical cp1_report launcher.

Current H200 screened training uses run_screened_training instead.
"""
from .legacy.launchers.cp1_report import *  # noqa: F401,F403
from .legacy.launchers.cp1_report import main

if __name__ == '__main__':
    main()
