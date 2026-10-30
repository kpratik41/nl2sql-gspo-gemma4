#!/usr/bin/env python3
"""Compatibility entry point for the shared historical replay renderer."""
if __package__:
    from .layer_replay_report import main
else:
    from layer_replay_report import main

if __name__ == '__main__':
    main()
