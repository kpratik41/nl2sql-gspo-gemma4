"""Compatibility entry point for the historical smoke_cp1 launcher.

Current H200 screened training uses run_screened_training instead.
"""
from .legacy.launchers.smoke_cp1 import *  # noqa: F401,F403
from .legacy.launchers.smoke_cp1 import main

if __name__ == '__main__':
    main()
