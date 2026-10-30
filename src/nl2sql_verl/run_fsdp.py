"""Compatibility entry point for the historical run_fsdp launcher.

Current H200 screened training uses run_screened_training instead.
"""
from .legacy.launchers.run_fsdp import *  # noqa: F401,F403
from .legacy.launchers.run_fsdp import main

if __name__ == '__main__':
    main()
