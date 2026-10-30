"""Compatibility entry point for the historical manage_cp1 launcher.

Current H200 screened training uses run_screened_training instead.
"""
from .legacy.launchers.manage_cp1 import *  # noqa: F401,F403
from .legacy.launchers.manage_cp1 import main

if __name__ == '__main__':
    main()
